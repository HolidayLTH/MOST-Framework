from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
from Model.TFT_GAT import InputChannelEmbedding, PINN

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    # True means masked
    idx = torch.arange(length, device=device)
    return (idx[None, :] > idx[:, None])


class EncoderLayer(nn.Module):
    """STTN-style encoder layer: temporal self-attn + spatial self-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, spatial_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [N, Th, D]
        res = x
        y, _ = self.temporal_attn(x, x, x, need_weights=False)
        x = self.temporal_norm(res + y)

        # spatial attn per time: [Th, N, D]
        x_s = x.transpose(0, 1)
        res_s = x_s
        y_s, _ = self.spatial_attn(x_s, x_s, x_s, attn_mask=spatial_mask, need_weights=False)
        x_s = self.spatial_norm(res_s + y_s)
        x = x_s.transpose(0, 1)

        x = self.ffn_norm(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    """STTN-style decoder layer: (causal) temporal self-attn + cross-attn + spatial self-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        temporal_mask: Optional[torch.Tensor],
        spatial_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x: [N, Tf, D], memory: [N, Th, D]
        res = x
        y, _ = self.temporal_attn(x, x, x, attn_mask=temporal_mask, need_weights=False)
        x = self.temporal_norm(res + y)

        res = x
        y, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.cross_norm(res + y)

        # spatial per time
        x_s = x.transpose(0, 1)  # [Tf, N, D]
        res_s = x_s
        y_s, _ = self.spatial_attn(x_s, x_s, x_s, attn_mask=spatial_mask, need_weights=False)
        x_s = self.spatial_norm(res_s + y_s)
        x = x_s.transpose(0, 1)

        x = self.ffn_norm(x + self.ffn(x))
        return x


class STTN_Benchmark(nn.Module):
    """STTN baseline compatible with TFT_GAT training pipeline.

    Adapted to this repo:
    - Node is an OD-pair sample (not a station sensor).
    - Spatial transformer uses `edge_index` (OD similarity graph) as a hard attention mask (optional).
    - Decoder uses causal temporal self-attention (no future-step leakage among predictions).
    - Uses future known covariates through `future_ts_rep`.
    - Keeps PINN module and returns the exact output dict keys expected by `process_batch`.
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

        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 else 'future_ts_categorical'

        self.task_type = config.task_type
        self.dropout = float(config.model.dropout)
        self.state_size = int(config.model.state_size)
        self.attention_heads = int(config.model.attention_heads)
        self.num_layers = int(config.model.lstm_layers)
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

        # Embedding pipeline (same as other baselines; also required for PINN inputs)
        self.static_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_static_numeric,
            num_categorical=self.num_static_categorical,
            categorical_cardinalities=self.static_categorical_cardinalities,
            time_distribute=False,
        )
        self.historical_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_historical_numeric,
            num_categorical=self.num_historical_categorical,
            categorical_cardinalities=self.historical_categorical_cardinalities,
            time_distribute=True,
        )
        self.future_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_future_numeric,
            num_categorical=self.num_future_categorical,
            categorical_cardinalities=self.future_categorical_cardinalities,
            time_distribute=True,
        )

        static_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)
        hist_dim = self.state_size * (self.num_historical_numeric + self.num_historical_categorical)
        fut_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)

        self.flow_proj = nn.Linear(1, self.state_size)
        self.static_proj = nn.Linear(static_dim if static_dim > 0 else 1, self.state_size)
        self.hist_cov_proj = nn.Linear(hist_dim if hist_dim > 0 else 1, self.state_size)
        self.fut_cov_proj = nn.Linear(fut_dim if fut_dim > 0 else 1, self.state_size)

        self.encoder = nn.ModuleList(
            [EncoderLayer(self.state_size, self.attention_heads, self.dropout) for _ in range(self.num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(self.state_size, self.attention_heads, self.dropout) for _ in range(self.num_layers)]
        )

        self.output_layer_IU = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.state_size, self.num_outputs),
            nn.Sigmoid(),
        )

        self.pinn = PINN(
            num_W_inputs=len(self.W_ts_index),
            num_static_station_inputs=len(self.O_index_numeric) + len(self.O_index_categorical),
            num_static_OD_inputs=len(self.OD_index_numeric) + len(self.OD_index_categorical),
            num_Hour_inputs=len(self.Hour_ts_index),
            num_Day_inputs=len(self.Day_index),
            state_size=self.state_size,
            dropout=self.dropout,
        )

    def transform_inputs(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty_tensor = torch.empty((0, 0))
        static_rep = self.static_transform(
            x_numeric=batch.get('static_feats_numeric', empty_tensor),
            x_categorical=batch.get('static_feats_categorical', empty_tensor),
        )
        historical_ts_rep = self.historical_ts_transform(
            x_numeric=batch.get('historical_ts_numeric', empty_tensor),
            x_categorical=batch.get('historical_ts_categorical', empty_tensor),
        )
        future_ts_rep = self.future_ts_transform(
            x_numeric=batch.get('future_ts_numeric', empty_tensor),
            x_categorical=batch.get('future_ts_categorical', empty_tensor),
        )
        return future_ts_rep, historical_ts_rep, static_rep

    @staticmethod
    def _spatial_attn_mask(edge_index: Optional[torch.Tensor], n: int, device: torch.device) -> Optional[torch.Tensor]:
        """Return bool mask [N,N] where True means masked (not allowed)."""
        if edge_index is None or edge_index.numel() == 0:
            return None
        src = edge_index[0].long().to(device)
        dst = edge_index[1].long().to(device)
        allowed = torch.zeros(n, n, device=device, dtype=torch.bool)
        allowed.index_put_((src, dst), torch.ones_like(src, dtype=torch.bool, device=device), accumulate=True)
        allowed.index_put_((dst, src), torch.ones_like(src, dtype=torch.bool, device=device), accumulate=True)
        allowed = allowed | torch.eye(n, device=device, dtype=torch.bool)
        return ~allowed

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC):
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        n = batch['historical_target'].shape[0]
        device = batch['historical_target'].device

        # token construction
        if static_rep.numel() == 0:
            static_ctx = self.static_proj(torch.ones(n, 1, device=device))
        else:
            static_ctx = self.static_proj(static_rep)

        hist_flow = batch['historical_target']  # [N, Th, 1]
        flow_h = self.flow_proj(hist_flow)

        if historical_ts_rep.numel() == 0:
            cov_h = self.hist_cov_proj(torch.ones(n, self.num_historical_steps, 1, device=device))
        else:
            cov_h = self.hist_cov_proj(historical_ts_rep)

        x_h = flow_h + cov_h + static_ctx.unsqueeze(1)  # [N, Th, D]

        if future_ts_rep.numel() == 0:
            cov_f = self.fut_cov_proj(torch.ones(n, self.num_future_steps, 1, device=device))
        else:
            cov_f = self.fut_cov_proj(future_ts_rep)

        # decoder token has no future flow (unknown) -> use zeros
        flow_f = torch.zeros(n, self.num_future_steps, 1, device=device)
        x_f = self.flow_proj(flow_f) + cov_f + static_ctx.unsqueeze(1)  # [N, Tf, D]

        spatial_mask = self._spatial_attn_mask(edge_index, n, device)

        # encoder
        memory = x_h
        for layer in self.encoder:
            memory = layer(memory, spatial_mask)

        # decoder (causal temporal self-attn)
        tmask = _causal_mask(self.num_future_steps, device=device)
        out = x_f
        for layer in self.decoder:
            out = layer(out, memory, tmask, spatial_mask)

        predicted_quantiles_IU = self.output_layer_IU(out)
        predicted_quantiles_IU = predicted_quantiles_IU[:, self.target_window_start_idx:, :]
        predicted_quantiles_IU = predicted_quantiles_IU[:target_N]

        # =========== PINN inputs (same wiring as other baselines) ===========
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

        W_idx_tensor = torch.tensor(
            [x + self.num_future_numeric for x in self.W_ts_index],
            device=device,
            dtype=torch.long,
        )
        W_idx_tensor = (W_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
        W_ts_rep = future_ts_rep[:target_N, :, W_idx_tensor]

        O_idx_tensor = torch.tensor(
            self.O_index_numeric + [x + self.num_static_numeric for x in self.O_index_categorical],
            device=device,
            dtype=torch.long,
        )
        O_idx_tensor = (O_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
        O_rep = static_rep[:target_N, O_idx_tensor]

        OD_idx_tensor = torch.tensor(
            self.OD_index_numeric + [x + self.num_static_numeric for x in self.OD_index_categorical],
            device=device,
            dtype=torch.long,
        )
        OD_idx_tensor = (OD_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
        OD_rep = static_rep[:target_N, OD_idx_tensor]

        Hour_idx_tensor = torch.tensor(
            [x + self.num_future_numeric for x in self.Hour_ts_index],
            device=device,
            dtype=torch.long,
        )
        Hour_idx_tensor = (Hour_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
        Hour_ts_rep = future_ts_rep[:target_N, :, Hour_idx_tensor]

        Hour_ts = batch['future_ts_categorical'][:target_N, :, self.Hour_ts_index]

        Day_idx_tensor = torch.tensor(
            [x + self.num_static_numeric for x in self.Day_index],
            device=device,
            dtype=torch.long,
        )
        Day_idx_tensor = (Day_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
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
            IS_EVAL=IS_EVAL,
        )

        static_inputs = self.num_static_numeric + self.num_static_categorical
        hist_inputs = self.num_historical_numeric + self.num_historical_categorical
        fut_inputs = self.num_future_numeric + self.num_future_categorical
        output_sequence_length = self.num_future_steps - self.target_window_start_idx

        attention_scores = torch.zeros(
            target_N,
            output_sequence_length,
            self.num_historical_steps + self.num_future_steps,
            device=device,
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
            'future_edgeindex': future_edgeindex,
        }
