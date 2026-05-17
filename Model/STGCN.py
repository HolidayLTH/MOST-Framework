from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from Model.TFT_GAT import InputChannelEmbedding, PINN
try:
    from config import historical_windows, future_windows
except Exception:
    historical_windows = None
    future_windows = None

class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        pad = (0, 0, self.kernel_size - 1, 0)
        x = F.pad(x, pad)
        x = self.conv(x)
        return self.dropout(F.relu(x))


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N], adj: [N, N]
        x = torch.einsum("nm,bctm->bctn", adj, x)
        return F.relu(self.proj(x))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.tconv1 = TemporalConv(in_channels, out_channels, kernel_size, dropout)
        self.gconv = GraphConv(out_channels, out_channels)
        self.tconv2 = TemporalConv(out_channels, out_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.tconv1(x)
        x = self.gconv(x, adj)
        x = self.tconv2(x)
        return x

class STGCN_Benchmark(nn.Module):
    """
    STGCN baseline compatible with TFT_GAT training pipeline.
    Uses edge_index for spatial graph and causal temporal convolutions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        data_props = config['data_props']
        self.num_static_numeric = data_props.get('num_static_numeric', 0)
        self.num_static_categorical = data_props.get('num_static_categorical', 0)
        self.static_categorical_cardinalities = data_props.get('static_categorical_cardinalities', [])

        self.num_historical_numeric = data_props.get('num_historical_numeric', 0)
        self.num_historical_categorical = data_props.get('num_historical_categorical', 0)
        self.historical_categorical_cardinalities = data_props.get('historical_categorical_cardinalities', [])

        self.num_future_numeric = data_props.get('num_future_numeric', 0)
        self.num_future_categorical = data_props.get('num_future_categorical', 0)
        self.future_categorical_cardinalities = data_props.get('future_categorical_cardinalities', [])

        self.num_target = data_props.get('num_target', 1)
        self.num_auxiliary_target = data_props.get('num_auxiliary_target', 1)

        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 \
            else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 \
            else 'future_ts_categorical'

        self.task_type = config.task_type
        self.dropout = config.model.dropout
        self.state_size = config.model.state_size
        self.attention_heads = config.model.attention_heads
        self.lstm_layers = config.model.lstm_layers
        self.target_window_start_idx = (config.target_window_start - 1) if config.target_window_start is not None else 0

        self.Hour_ts_index = config.Hour_ts_index
        self.Day_index = config.Day_index
        self.SL_ts_index = config.SL_ts_index
        self.TT_ts_index = config.TT_ts_index
        self.W_ts_index = config.W_ts_index
        self.O_index_categorical = config.O_index_categorical
        self.O_index_numeric = config.O_index_numeric
        self.D_index_categorical = config.D_index_categorical
        self.D_index_numeric = config.D_index_numeric
        self.OD_index_categorical = config.OD_index_categorical
        self.OD_index_numeric = config.OD_index_numeric

        if self.task_type == 'regression':
            self.output_quantiles = config.model.output_quantiles
            self.num_outputs = len(self.output_quantiles) * self.num_target
            self.num_auxiliary_outputs = len(self.output_quantiles) * self.num_auxiliary_target
        elif self.task_type == 'classification':
            self.output_quantiles = None
            self.num_outputs = 1
            self.num_auxiliary_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")

        self.static_transform = InputChannelEmbedding(state_size=self.state_size,
                                                      num_numeric=self.num_static_numeric,
                                                      num_categorical=self.num_static_categorical,
                                                      categorical_cardinalities=self.static_categorical_cardinalities,
                                                      time_distribute=False)

        self.historical_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_historical_numeric,
            num_categorical=self.num_historical_categorical,
            categorical_cardinalities=self.historical_categorical_cardinalities,
            time_distribute=True)

        self.future_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_future_numeric,
            num_categorical=self.num_future_categorical,
            categorical_cardinalities=self.future_categorical_cardinalities,
            time_distribute=True)

        self.hist_feat_dim = 1 + self.num_historical_numeric + self.num_historical_categorical
        self.future_feat_dim = self.num_future_numeric + self.num_future_categorical
        self.static_rep_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)
        self.future_rep_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)

        if self.static_rep_dim > 0:
            self.static_ctx_proj = nn.Linear(self.static_rep_dim, self.state_size)
            static_ctx_dim = self.state_size
        else:
            self.static_ctx_proj = None
            static_ctx_dim = 0

        kernel_size = 3
        self.stgcn1 = STGCNBlock(self.hist_feat_dim + static_ctx_dim, self.state_size, kernel_size, self.dropout)
        self.stgcn2 = STGCNBlock(self.state_size, self.state_size, kernel_size, self.dropout)
        if historical_windows is not None and future_windows is not None:
            pred_kernel = max(1, int(historical_windows) - int(future_windows) + 1)
        else:
            pred_kernel = 1
        self.end_conv_1 = nn.Conv2d(self.state_size, self.state_size, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(self.state_size, self.state_size, kernel_size=(pred_kernel, 1))
        self.output_proj = nn.Linear(self.state_size, self.num_outputs)

        if self.future_rep_dim > 0:
            self.future_mlp = nn.Sequential(
                nn.Linear(self.future_rep_dim, self.state_size),
                nn.ReLU(),
                nn.Linear(self.state_size, self.state_size)
            )
        else:
            self.future_mlp = None

        self.out_act = nn.Sigmoid()

        self.pinn = PINN(num_W_inputs=len(self.W_ts_index),
                         num_static_station_inputs=len(self.O_index_numeric) + len(self.O_index_categorical),
                         num_static_OD_inputs=len(self.OD_index_numeric) + len(self.OD_index_categorical),
                         num_Hour_inputs=len(self.Hour_ts_index),
                         num_Day_inputs=len(self.Day_index),
                         state_size=self.state_size,
                         dropout=self.dropout)

    @staticmethod
    def _build_adj(edge_index: Optional[torch.Tensor], n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return torch.eye(n, device=device, dtype=dtype)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        adj = torch.zeros(n, n, device=device, dtype=dtype)
        ones = torch.ones_like(src, dtype=dtype)
        adj.index_put_((src, dst), ones, accumulate=True)
        adj.index_put_((dst, src), ones, accumulate=True)
        adj = adj + torch.eye(n, device=device, dtype=dtype)

        deg = adj.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        d_mat = torch.diag(deg_inv_sqrt)
        return d_mat @ adj @ d_mat

    def transform_inputs(self, batch):
        empty_tensor = torch.empty((0, 0))
        static_rep = self.static_transform(x_numeric=batch.get('static_feats_numeric', empty_tensor),
                                           x_categorical=batch.get('static_feats_categorical', empty_tensor))
        historical_ts_rep = self.historical_ts_transform(x_numeric=batch.get('historical_ts_numeric', empty_tensor),
                                                         x_categorical=batch.get('historical_ts_categorical', empty_tensor))
        future_ts_rep = self.future_ts_transform(x_numeric=batch.get('future_ts_numeric', empty_tensor),
                                                 x_categorical=batch.get('future_ts_categorical', empty_tensor))
        return future_ts_rep, historical_ts_rep, static_rep

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC):
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        n = batch['historical_target'].shape[0]
        device = batch['historical_target'].device
        dtype = batch['historical_target'].dtype
        adj = self._build_adj(edge_index, n, device, dtype)

        hist_flow = batch['historical_target']  # [N, Th, 1]
        hist_num = batch.get('historical_ts_numeric', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_cat = batch.get('historical_ts_categorical', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_feat = torch.cat([hist_flow, hist_num, hist_cat], dim=-1)  # [N, Th, C_hist]

        if self.static_ctx_proj is not None and static_rep.numel() > 0:
            static_ctx = self.static_ctx_proj(static_rep)  # [N, C_static_ctx]
            static_ctx = static_ctx.unsqueeze(1).expand(-1, self.num_historical_steps, -1)  # [N, Th, C_static_ctx]
            hist_feat = torch.cat([hist_feat, static_ctx], dim=-1)  # [N, Th, C_hist + C_static_ctx]

        x = hist_feat.permute(2, 1, 0).unsqueeze(0)  # [1, C_in, Th, N]
        x = self.stgcn1(x, adj)
        x = self.stgcn2(x, adj)
        # x: [1, C, Th, N]

        out = F.relu(self.end_conv_1(x))  # [1, C, Th, N]
        out = self.end_conv_2(out)  # [1, C, Tf, N]
        if out.size(2) != self.num_future_steps:
            raise ValueError(f"STGCN output length {out.size(2)} does not match future steps {self.num_future_steps}")
        base = out.permute(0, 3, 2, 1).squeeze(0)  # [N, Tf, C]

        if self.future_mlp is not None and future_ts_rep.numel() > 0:
            adj_out = self.future_mlp(future_ts_rep)  # [N, Tf, C]
        else:
            adj_out = torch.zeros_like(base)

        fused = base + adj_out  # [N, Tf, C]
        predicted_quantiles_IU = self.out_act(self.output_proj(fused))  # [N, Tf, Q]
        predicted_quantiles_IU = predicted_quantiles_IU[:target_N]

        batch_indices = batch['ODindex'][:target_N]

        TT_ts = batch['future_ts_numeric'][:target_N, :, self.TT_ts_index]
        SL_ts = batch['future_ts_numeric'][:target_N, :, self.SL_ts_index]

        if not IS_EVAL:
            phi_ts = batch['future_target_OFlow'][:target_N, :, :]
        else:
            phi_ts = None

        maxphi = batch['maxinflow'][:target_N, :]

        if not IS_EVAL:
            I_ts = batch['future_target'][:target_N, :]
        else:
            I_ts = None

        W_idx_tensor = torch.tensor(list(map(lambda x: x + self.num_future_numeric, self.W_ts_index)))
        W_idx_tensor = (W_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        W_ts_rep = future_ts_rep[:target_N, :, W_idx_tensor]

        O_idx_tensor = torch.tensor(self.O_index_numeric + list(map(lambda x: x + self.num_static_numeric, self.O_index_categorical)))
        O_idx_tensor = (O_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        O_rep = static_rep[:target_N, O_idx_tensor]

        OD_idx_tensor = torch.tensor(self.OD_index_numeric + list(map(lambda x: x + self.num_static_numeric, self.OD_index_categorical)))
        OD_idx_tensor = (OD_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        OD_rep = static_rep[:target_N, OD_idx_tensor]

        Hour_idx_tensor = torch.tensor(list(map(lambda x: x + self.num_future_numeric, self.Hour_ts_index)))
        Hour_idx_tensor = (Hour_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        Hour_ts_rep = future_ts_rep[:target_N, :, Hour_idx_tensor]

        Hour_ts = batch['future_ts_categorical'][:target_N, :, self.Hour_ts_index]

        Day_idx_tensor = torch.tensor(list(map(lambda x: x + self.num_static_numeric, self.Day_index)))
        Day_idx_tensor = (Day_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        Day_rep = static_rep[:target_N, Day_idx_tensor]

        phi, s, alpha, beta, eta, rb, r_physics, congestion = self.pinn(
            SL_raw=SL_ts,
            TT_raw=TT_ts,
            I_pred=predicted_quantiles_IU,
            I_raw=I_ts,
            phi_raw=phi_ts,
            maxphi=maxphi,
            W=W_ts_rep,
            Hour=Hour_ts_rep,
            Hour_raw=Hour_ts,
            Day=Day_rep,
            O=O_rep,
            OD=OD_rep,
            batch_indices=batch_indices,
            IS_EVAL=IS_EVAL
        )

        device = predicted_quantiles_IU.device
        static_inputs = self.num_static_numeric + self.num_static_categorical
        hist_inputs = self.num_historical_numeric + self.num_historical_categorical
        fut_inputs = self.num_future_numeric + self.num_future_categorical
        output_sequence_length = self.num_future_steps - self.target_window_start_idx

        attention_scores = torch.zeros(
            target_N,
            output_sequence_length,
            self.num_historical_steps + self.num_future_steps,
            device=device
        )
        static_weights = torch.zeros(target_N, static_inputs, device=device)
        historical_selection_weights = torch.zeros(target_N, self.num_historical_steps, hist_inputs, device=device)
        future_selection_weights = torch.zeros(target_N, self.num_future_steps, fut_inputs, device=device)

        if edge_index is not None:
            past_attention_alpha = torch.zeros(edge_index.shape[1], self.attention_heads, device=device)
            future_attention_alpha = torch.zeros(edge_index.shape[1], self.attention_heads, device=device)
            past_edgeindex = edge_index
            future_edgeindex = edge_index
        else:
            past_attention_alpha = None
            future_attention_alpha = None
            past_edgeindex = None
            future_edgeindex = None

        return {
            'predicted_quantiles_IU': predicted_quantiles_IU,
            'predicted_quantiles_phi': phi,
            'predicted_R': r_physics,
            'predicted_S': s,
            'predicted_Alpha': alpha,
            'predicted_Beta': beta,
            'predicted_Eta': eta,
            'predicted_Rb': rb,
            'predicted_congestion': congestion,
            'attention_scores': attention_scores,
            'static_weights': static_weights,
            'historical_selection_weights': historical_selection_weights,
            'future_selection_weights': future_selection_weights,
            'pastgat_attention': past_attention_alpha,
            'futuregat_attention': future_attention_alpha,
            'past_edgeindex': past_edgeindex,
            'future_edgeindex': future_edgeindex
        }
