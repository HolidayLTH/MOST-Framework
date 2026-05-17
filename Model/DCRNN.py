from __future__ import annotations
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from Model.TFT_GAT import InputChannelEmbedding, PINN

class DiffusionGraphConv(nn.Module):
    """Diffusion graph convolution used by DCRNN.

    Given supports (random walk matrices), computes:
      concat([X, P1 X, ..., P1^K X, P2 X, ..., P2^K X]) @ W

    Shapes:
      X: [B, N, Fin]
      supports: list of [N, N]
      out: [B, N, Fout]
    """

    def __init__(self, in_dim: int, out_dim: int, k_order: int, num_supports: int, bias: bool = True):
        super().__init__()
        if k_order < 0:
            raise ValueError("k_order must be >= 0")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_order = k_order
        self.num_supports = num_supports

        num_mats = 1 + num_supports * k_order
        self.weight = nn.Parameter(torch.empty(in_dim * num_mats, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        bsz, n, fin = x.shape
        if fin != self.in_dim:
            raise ValueError(f"DiffusionGraphConv expected in_dim={self.in_dim}, got {fin}")

        feats = [x]
        if self.k_order > 0:
            if len(supports) != self.num_supports:
                raise ValueError(f"Expected {self.num_supports} supports, got {len(supports)}")
            for sup in supports:
                xk = x
                for _ in range(self.k_order):
                    # [N,N] x [B,N,Fin] -> [B,N,Fin]
                    xk = torch.einsum("nm,bmf->bnf", sup, xk)
                    feats.append(xk)

        x_cat = torch.cat(feats, dim=-1)  # [B, N, Fin * num_mats]
        out = x_cat @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU cell."""

    def __init__(self, input_dim: int, hidden_dim: int, k_order: int, num_supports: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        gate_in_dim = input_dim + hidden_dim
        self.gate_conv = DiffusionGraphConv(
            in_dim=gate_in_dim,
            out_dim=2 * hidden_dim,
            k_order=k_order,
            num_supports=num_supports,
            bias=True,
        )
        self.cand_conv = DiffusionGraphConv(
            in_dim=gate_in_dim,
            out_dim=hidden_dim,
            k_order=k_order,
            num_supports=num_supports,
            bias=True,
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        # x_t: [B, N, input_dim], h_prev: [B, N, hidden_dim]
        xh = torch.cat([x_t, h_prev], dim=-1)
        gates = torch.sigmoid(self.gate_conv(xh, supports))
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=-1)

        xh_cand = torch.cat([x_t, reset_gate * h_prev], dim=-1)
        cand = torch.tanh(self.cand_conv(xh_cand, supports))
        cand = self.dropout(cand)

        h_new = (1.0 - update_gate) * h_prev + update_gate * cand
        return h_new


class DCRNN_Benchmark(nn.Module):
    """DCRNN baseline compatible with TFT_GAT training pipeline.

    - Graph is the OD-pair similarity graph given by `edge_index` (same as STGCN baseline).
    - Encoder-decoder (seq2seq) with diffusion convolutional GRU.
    - Uses future known covariates in the decoder (causal w.r.t. targets).
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

        # Embedding pipeline (for PINN inputs)
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

        self.static_rep_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)
        if self.static_rep_dim > 0:
            self.static_ctx_proj = nn.Linear(self.static_rep_dim, self.state_size)
            self.static_ctx_dim = self.state_size
        else:
            self.static_ctx_proj = None
            self.static_ctx_dim = 0

        # DCRNN core
        self.k_order = 2
        self.num_supports = 2  # forward + backward random walk

        enc_input_dim = 1 + self.num_historical_numeric + self.num_historical_categorical + self.static_ctx_dim
        # decoder covariates use embedded future features (incl. categorical embeddings)
        fut_emb_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)
        self.dec_cov_emb_dim = self.state_size if fut_emb_dim > 0 else 0
        self.dec_cov_proj = nn.Linear(fut_emb_dim, self.dec_cov_emb_dim) if fut_emb_dim > 0 else None
        dec_input_dim = 1 + self.dec_cov_emb_dim  # prev_y + embedded future covariates

        self.encoder_cell = DCGRUCell(
            input_dim=enc_input_dim,
            hidden_dim=self.state_size,
            k_order=self.k_order,
            num_supports=self.num_supports,
            dropout=self.dropout,
        )
        self.decoder_cell = DCGRUCell(
            input_dim=dec_input_dim,
            hidden_dim=self.state_size,
            k_order=self.k_order,
            num_supports=self.num_supports,
            dropout=self.dropout,
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

    @staticmethod
    def _build_supports(
        edge_index: Optional[torch.Tensor],
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
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

        def random_walk(mat: torch.Tensor) -> torch.Tensor:
            deg = mat.sum(dim=1)
            deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
            return deg_inv.unsqueeze(1) * mat

        p_fwd = random_walk(a)
        p_bwd = random_walk(a.transpose(0, 1))
        return [p_fwd, p_bwd]

    def transform_inputs(self, batch):
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

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC):
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        # Build graph supports for all nodes in this batch (targets + neighbors)
        n = batch['historical_target'].shape[0]
        device = batch['historical_target'].device
        dtype = batch['historical_target'].dtype
        supports = self._build_supports(edge_index, n, device, dtype)

        # =========== Encoder ===========
        hist_flow = batch['historical_target']  # [N, Th, 1]
        hist_num = batch.get('historical_ts_numeric', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_cat = batch.get('historical_ts_categorical', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_feat = torch.cat([hist_flow, hist_num, hist_cat], dim=-1)  # [N, Th, F_hist]

        if self.static_ctx_proj is not None and static_rep.numel() > 0:
            static_ctx = self.static_ctx_proj(static_rep)  # [N, C_static]
            static_ctx = static_ctx.unsqueeze(1).expand(-1, self.num_historical_steps, -1)  # [N, Th, C_static]
            hist_feat = torch.cat([hist_feat, static_ctx], dim=-1)  # [N, Th, F_hist + C_static]

        h = torch.zeros(1, n, self.state_size, device=device, dtype=dtype)
        for t in range(self.num_historical_steps):
            x_t = hist_feat[:, t, :].unsqueeze(0)  # [1, N, F_hist + C_static]
            h = self.encoder_cell(x_t, h, supports)

        # =========== Decoder ===========
        # Use embedded future covariates (incl. categorical embeddings)
        if self.dec_cov_proj is None:
            fut_cov_emb = torch.zeros(n, self.num_future_steps, 0, device=device, dtype=dtype)  # [N, Tf, 0]
        else:
            fut_cov_emb = self.dec_cov_proj(future_ts_rep)  # [N, Tf, dec_cov_emb_dim]

        # previous y (scalar) per node
        prev_y = hist_flow[:, -1, 0]  # [N]

        q_mid = 0
        if self.output_quantiles is not None and len(self.output_quantiles) > 0:
            q_mid = len(self.output_quantiles) // 2

        preds = []
        for t in range(self.num_future_steps):
            cov_t = fut_cov_emb[:, t, :]  # [N, dec_cov_emb_dim]
            dec_in = torch.cat([prev_y.unsqueeze(-1), cov_t], dim=-1).unsqueeze(0)  # [1, N, 1 + dec_cov_emb_dim]
            h = self.decoder_cell(dec_in, h, supports)
            step_out = self.output_layer_IU(h.squeeze(0))  # [N, Q]
            preds.append(step_out)
            prev_y = step_out[:, q_mid]

        predicted_quantiles_IU = torch.stack(preds, dim=1)  # [N, Tf, Q]
        predicted_quantiles_IU = predicted_quantiles_IU[:, self.target_window_start_idx:, :]
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

        # PINN uses embedded signals (match TFT_GAT pattern), ensure index tensors are on the right device
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
