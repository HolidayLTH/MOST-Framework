from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from Model.TFT_GAT import InputChannelEmbedding, PINN

class DiffusionConv(nn.Module):
    """Graph WaveNet diffusion graph convolution.

    x: [B, C, T, N]
    supports: list of [N, N]
    returns: [B, C_out, T, N]
    """

    def __init__(self, in_channels: int, out_channels: int, k_order: int, num_supports: int, dropout: float):
        super().__init__()
        self.k_order = k_order
        self.num_supports = num_supports
        self.dropout = nn.Dropout(dropout)

        # concat: x0 + sum_{sup}(x1..xK) => (1 + num_supports*k_order) * in_channels
        mix_in = (1 + num_supports * k_order) * in_channels
        self.mlp = nn.Conv2d(mix_in, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        b, c, t, n = x.shape
        out = [x]
        if self.k_order > 0 and len(supports) > 0:
            if len(supports) != self.num_supports:
                raise ValueError(f"Expected {self.num_supports} supports, got {len(supports)}")
            for sup in supports:
                xk = x
                for _ in range(self.k_order):
                    # [N,N] x [B,C,T,N] -> [B,C,T,N]
                    xk = torch.einsum("nm,bctm->bctn", sup, xk)
                    out.append(xk)

        h = torch.cat(out, dim=1)  # [B, mix_in, T, N]
        h = self.mlp(h)
        return self.dropout(h)


class GraphWaveNet_Benchmark(nn.Module):
    """Graph WaveNet baseline compatible with TFT_GAT training pipeline.

    Adapted to this repo:
    - Node is an OD-pair sample (not a station sensor).
    - Spatial supports are built from `edge_index` (OD similarity graph) with random-walk normalization.
    - Adds an adaptive adjacency derived from static embeddings (paper's adaptive adjacency analogue) when available.
    - Uses future known covariates via `future_ts_rep` (incl. categorical embedding) through a future-MLP branch.
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
        self.num_blocks = int(max(1, config.model.lstm_layers))
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

        # Graph WaveNet hyperparams (kept small as a baseline)
        self.kernel_size = 2
        self.layers_per_block = 2
        self.dilation_base = 2
        self.k_order = 2

        self.residual_channels = self.state_size
        self.skip_channels = self.state_size

        # Input channels: historical flow + raw known covariates (numeric + categorical as float IDs)
        # We do *not* embed these for the temporal conv trunk to keep it lightweight.
        self.in_channels = 1 + self.num_historical_numeric + self.num_historical_categorical

        self.start_conv = nn.Conv2d(self.in_channels, self.residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.num_supports = 2  # forward + backward random walk
        self.use_adaptive_adj = True
        self.adp_dim = 16

        static_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)
        self.node_emb_net = nn.Linear(static_dim, self.adp_dim) if static_dim > 0 else None

        receptive_field = 1
        for _ in range(self.num_blocks):
            dilation = 1
            for _ in range(self.layers_per_block):
                self.filter_convs.append(
                    nn.Conv2d(
                        self.residual_channels,
                        self.residual_channels,
                        kernel_size=(self.kernel_size, 1),
                        dilation=(dilation, 1),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        self.residual_channels,
                        self.residual_channels,
                        kernel_size=(self.kernel_size, 1),
                        dilation=(dilation, 1),
                    )
                )
                self.residual_convs.append(nn.Conv2d(self.residual_channels, self.residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(self.residual_channels, self.skip_channels, kernel_size=(1, 1)))
                self.graph_convs.append(
                    DiffusionConv(
                        in_channels=self.residual_channels,
                        out_channels=self.residual_channels,
                        k_order=self.k_order,
                        num_supports=self.num_supports + (1 if self.use_adaptive_adj else 0),
                        dropout=self.dropout,
                    )
                )
                self.batch_norms.append(nn.BatchNorm2d(self.residual_channels))

                receptive_field += (self.kernel_size - 1) * dilation
                dilation *= self.dilation_base

        self.receptive_field = receptive_field

        # End convolutions (paper-style): produce per-time-step outputs
        self.end_conv_1 = nn.Conv2d(self.skip_channels, self.skip_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(self.skip_channels, self.num_outputs, kernel_size=(1, 1))

        fut_emb_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)
        if fut_emb_dim > 0:
            self.future_mlp = nn.Sequential(
                nn.Linear(fut_emb_dim, self.state_size),
                nn.ReLU(),
                nn.Linear(self.state_size, self.num_outputs),
            )
        else:
            self.future_mlp = None

        # Future conditioning for trunk (inject into each dilated layer)
        if fut_emb_dim > 0:
            self.future_ctx_proj = nn.Sequential(
                nn.Linear(fut_emb_dim, self.residual_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
        else:
            self.future_ctx_proj = None

        self.out_act = nn.Sigmoid()

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
    def _build_supports(edge_index: Optional[torch.Tensor], n: int, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        if edge_index is None or edge_index.numel() == 0:
            ident = torch.eye(n, device=device, dtype=dtype)
            return [ident, ident]

        src = edge_index[0].long().to(device)
        dst = edge_index[1].long().to(device)

        a = torch.zeros(n, n, device=device, dtype=dtype)
        ones = torch.ones_like(src, dtype=dtype)
        a.index_put_((src, dst), ones, accumulate=True)
        a.index_put_((dst, src), ones, accumulate=True)
        a = a + torch.eye(n, device=device, dtype=dtype)

        def rw(mat: torch.Tensor) -> torch.Tensor:
            deg = mat.sum(dim=1)
            deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
            return deg_inv.unsqueeze(1) * mat

        p_fwd = rw(a)
        p_bwd = rw(a.transpose(0, 1))
        return [p_fwd, p_bwd]

    def _adaptive_support(self, static_rep: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        if (not self.use_adaptive_adj) or (self.node_emb_net is None) or static_rep.numel() == 0:
            return None
        node_emb = self.node_emb_net(static_rep)  # [N, adp_dim]
        a = F.relu(node_emb @ node_emb.transpose(0, 1))
        return torch.softmax(a, dim=-1)

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC):
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, _, static_rep = self.transform_inputs(batch)

        n = batch['historical_target'].shape[0]
        device = batch['historical_target'].device
        dtype = batch['historical_target'].dtype

        supports = self._build_supports(edge_index, n, device, dtype)
        adp = self._adaptive_support(static_rep, device)
        if adp is not None:
            supports = supports + [adp]

        # trunk conditioning from future known covariates (batch-wise node context)
        if self.future_ctx_proj is not None and future_ts_rep.numel() > 0:
            future_ctx = self.future_ctx_proj(future_ts_rep.mean(dim=1))  # [N, C]
            future_ctx = future_ctx.transpose(0, 1).unsqueeze(0).unsqueeze(2)  # [1, C, 1, N]
        else:
            future_ctx = None

        # trunk inputs (raw historical signals)
        hist_flow = batch['historical_target']  # [N, Th, 1]
        hist_num = batch.get('historical_ts_numeric', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_cat = batch.get('historical_ts_categorical', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_feat = torch.cat([hist_flow, hist_num, hist_cat], dim=-1)  # [N, Th, C]

        # [B, C, T, N]
        x = hist_feat.permute(1, 2, 0).unsqueeze(0)  # [1, Th, C, N]
        x = x.permute(0, 2, 1, 3).contiguous()  # [1, C, Th, N]

        # pad to receptive field if needed
        if x.size(2) < self.receptive_field:
            pad_len = self.receptive_field - x.size(2)
            x = F.pad(x, (0, 0, pad_len, 0))

        x = self.start_conv(x)
        skip = None

        layer_idx = 0
        for _ in range(self.num_blocks):
            dilation = 1
            for _ in range(self.layers_per_block):
                residual = x

                if future_ctx is not None:
                    residual = residual + future_ctx

                filt = torch.tanh(self.filter_convs[layer_idx](residual))
                gate = torch.sigmoid(self.gate_convs[layer_idx](residual))
                x = filt * gate

                s = self.skip_convs[layer_idx](x)
                skip = s if skip is None else skip[..., -s.size(2):, :] + s

                x = self.graph_convs[layer_idx](x, supports)
                x = self.residual_convs[layer_idx](x)
                x = x + residual[..., -x.size(2):, :]
                x = self.batch_norms[layer_idx](x)

                dilation *= self.dilation_base
                layer_idx += 1

        if skip is None:
            # very small model fallback
            skip = x

        # End convs to produce per-time-step logits
        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # [1, Q, T', N]

        # Ensure we have enough steps to slice future horizon
        if out.size(2) < self.num_future_steps:
            out = F.pad(out, (0, 0, self.num_future_steps - out.size(2), 0))

        base_logits = out[:, :, -self.num_future_steps:, :]  # [1, Q, Tf, N]
        base_logits = base_logits.permute(0, 3, 2, 1).squeeze(0)  # [N, Tf, Q]

        if self.future_mlp is not None and future_ts_rep.numel() > 0:
            cov_logits = self.future_mlp(future_ts_rep)  # [N, Tf, Q]
        else:
            cov_logits = torch.zeros_like(base_logits)

        predicted_quantiles_IU = self.out_act(base_logits + cov_logits)
        predicted_quantiles_IU = predicted_quantiles_IU[:, self.target_window_start_idx:, :]
        predicted_quantiles_IU = predicted_quantiles_IU[:target_N]

        # ===== PINN inputs (match other baselines) =====
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

        W_idx_tensor = torch.tensor([x + self.num_future_numeric for x in self.W_ts_index], device=device, dtype=torch.long)
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

        Hour_idx_tensor = torch.tensor([x + self.num_future_numeric for x in self.Hour_ts_index], device=device, dtype=torch.long)
        Hour_idx_tensor = (Hour_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
        Hour_ts_rep = future_ts_rep[:target_N, :, Hour_idx_tensor]

        Hour_ts = batch['future_ts_categorical'][:target_N, :, self.Hour_ts_index]

        Day_idx_tensor = torch.tensor([x + self.num_static_numeric for x in self.Day_index], device=device, dtype=torch.long)
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
