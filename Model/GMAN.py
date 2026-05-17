from __future__ import annotations
from typing import List, Optional, Tuple
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


class STAttentionBlock(nn.Module):
    """A lightweight GMAN-style block: Temporal self-attn (per node) + Spatial self-attn (per time)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_ffn = FeedForward(d_model, dropout)
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_norm2 = nn.LayerNorm(d_model)

        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_msg = nn.Linear(d_model, d_model)
        self.spatial_self = nn.Linear(d_model, d_model)
        self.spatial_ffn = FeedForward(d_model, dropout)
        self.spatial_norm1 = nn.LayerNorm(d_model)
        self.spatial_norm2 = nn.LayerNorm(d_model)

    def _dense_spatial_attn_chunked(
        self,
        x_s: torch.Tensor,
        spatial_attn_mask: Optional[torch.Tensor],
        chunk_size: int = 4,
    ) -> torch.Tensor:
        """Dense spatial attention with chunking over time-batch dimension to reduce peak memory."""
        t, n, d = x_s.shape
        out = torch.empty_like(x_s)
        for start in range(0, t, chunk_size):
            end = min(start + chunk_size, t)
            x_chunk = x_s[start:end]
            y_chunk, _ = self.spatial_attn(
                x_chunk,
                x_chunk,
                x_chunk,
                attn_mask=spatial_attn_mask,
                need_weights=False,
            )
            out[start:end] = y_chunk
        return out

    def _sparse_spatial_aggregate(self, x_s: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Sparse edge_index-based spatial propagation, complexity ~ O(T * E)."""
        t, n, d = x_s.shape
        device = x_s.device

        orig_src = edge_index[0].long().to(device)
        orig_dst = edge_index[1].long().to(device)

        # Remove out-of-range edges defensively.
        valid = (orig_src >= 0) & (orig_src < n) & (orig_dst >= 0) & (orig_dst < n)
        orig_src = orig_src[valid]
        orig_dst = orig_dst[valid]

        # Make graph undirected and add self-loops.
        src = torch.cat([orig_src, orig_dst, torch.arange(n, device=device)], dim=0)
        dst = torch.cat([orig_dst, orig_src, torch.arange(n, device=device)], dim=0)

        out = torch.empty_like(x_s)
        ones = torch.ones(dst.shape[0], 1, device=device, dtype=x_s.dtype)
        for i in range(t):
            x_t = x_s[i]  # [N, D]
            msg = self.spatial_msg(x_t[src])
            agg = torch.zeros(n, d, device=device, dtype=x_s.dtype)
            deg = torch.zeros(n, 1, device=device, dtype=x_s.dtype)
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, ones)
            agg = agg / deg.clamp_min(1.0)
            out[i] = self.spatial_self(x_t) + agg
        return out

    def forward(
        self,
        x: torch.Tensor,
        spatial_attn_mask: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        temporal_causal: bool = False,
    ) -> torch.Tensor:
        """x: [N, T, D]. spatial_attn_mask: [N, N] bool (True=masked). edge_index: [2, E]."""
        # Temporal self-attention (batch=N, seq=T)
        res = x
        temporal_attn_mask = None
        if temporal_causal:
            t = x.size(1)
            temporal_attn_mask = torch.triu(
                torch.ones(t, t, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        y, _ = self.temporal_attn(x, x, x, attn_mask=temporal_attn_mask, need_weights=False)
        x = self.temporal_norm1(res + y)
        x = self.temporal_norm2(x + self.temporal_ffn(x))

        # Spatial self-attention (batch=T, seq=N)
        x_s = x.transpose(0, 1)  # [T, N, D]
        res_s = x_s
        if edge_index is not None and edge_index.numel() > 0:
            y_s = self._sparse_spatial_aggregate(x_s, edge_index)
        else:
            y_s = self._dense_spatial_attn_chunked(x_s, spatial_attn_mask=spatial_attn_mask, chunk_size=4)
        x_s = self.spatial_norm1(res_s + y_s)
        x_s = self.spatial_norm2(x_s + self.spatial_ffn(x_s))
        return x_s.transpose(0, 1)  # [N, T, D]


class TransformAttention(nn.Module):
        """GMAN Transform-Attention (paper style).

        Uses spatio-temporal embeddings (STE) to align encoder outputs to future steps:
            Q = STE_fut, K = STE_his, V = encoder_memory

        Shapes:
            memory:  [N, Th, D]
            ste_his: [N, Th, D]
            ste_fut: [N, Tf, D]
            out:     [N, Tf, D]
        """

        def __init__(self, d_model: int, n_heads: int, dropout: float):
                super().__init__()
                if d_model % n_heads != 0:
                        raise ValueError("d_model must be divisible by n_heads")
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                self.scale = self.head_dim ** 0.5

                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(dropout)

        def forward(self, memory: torch.Tensor, ste_his: torch.Tensor, ste_fut: torch.Tensor) -> torch.Tensor:
                n, th, _ = memory.shape
                _, tf, _ = ste_fut.shape

                q = self.q_proj(ste_fut).view(n, tf, self.n_heads, self.head_dim).transpose(1, 2)  # [N,H,Tf,hd]
                k = self.k_proj(ste_his).view(n, th, self.n_heads, self.head_dim).transpose(1, 2)  # [N,H,Th,hd]
                v = self.v_proj(memory).view(n, th, self.n_heads, self.head_dim).transpose(1, 2)   # [N,H,Th,hd]

                attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)  # [N,H,Tf,Th]
                attn = self.dropout(attn)
                out = torch.matmul(attn, v)  # [N,H,Tf,hd]
                out = out.transpose(1, 2).contiguous().view(n, tf, self.d_model)  # [N,Tf,D]
                return self.out_proj(out)


class GMAN_Benchmark(nn.Module):
    """GMAN baseline compatible with TFT_GAT training pipeline.

    Design choices for compatibility and simplicity:
    - Treat each OD-pair sample as a node; uses optional `edge_index` to mask spatial attention.
    - Encoder: ST-attention blocks over historical steps.
    - Decoder: cross-attention from future queries to historical memory.
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
        self.dropout = float(config.model.dropout)
        self.state_size = int(config.model.state_size)
        self.attention_heads = int(config.model.attention_heads)
        self.lstm_layers = int(config.model.lstm_layers)
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

        self.static_rep_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)
        self.hist_feat_dim = 1 + self.num_historical_numeric + self.num_historical_categorical
        self.future_rep_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)

        if self.hist_feat_dim <= 0:
            raise ValueError("historical features are empty; GMAN_Benchmark requires historical temporal features")

        if self.static_rep_dim > 0:
            self.static_ctx_proj = nn.Linear(self.static_rep_dim, self.state_size)
            branch_static_dim = self.state_size
        else:
            self.static_ctx_proj = None
            branch_static_dim = 0

        self.hist_in_proj = nn.Linear(self.hist_feat_dim + branch_static_dim, self.state_size)
        if self.future_rep_dim > 0:
            self.future_in_proj = nn.Linear(self.future_rep_dim, self.state_size)
        else:
            self.future_in_proj = None

        self.hist_blocks = nn.ModuleList(
            [STAttentionBlock(self.state_size, self.attention_heads, self.dropout) for _ in range(self.lstm_layers)]
        )
        self.transform_attn = TransformAttention(
            d_model=self.state_size,
            n_heads=self.attention_heads,
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

    def _append_static_context(self, seq_rep: torch.Tensor, static_rep: torch.Tensor) -> torch.Tensor:
        if self.static_ctx_proj is None or static_rep.numel() == 0:
            return seq_rep
        static_ctx = self.static_ctx_proj(static_rep)
        static_ctx = static_ctx.unsqueeze(1).expand(-1, seq_rep.size(1), -1)
        return torch.cat([seq_rep, static_ctx], dim=-1)

    def _run_st_blocks(
        self,
        seq_in: torch.Tensor,
        blocks: nn.ModuleList,
        spatial_attn_mask: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        temporal_causal: bool,
    ) -> torch.Tensor:
        x = seq_in
        for blk in blocks:
            x = blk(x, spatial_attn_mask, edge_index, temporal_causal=temporal_causal)
        return x

    def _run_hist_branch(
        self,
        hist_feat: torch.Tensor,
        static_rep: torch.Tensor,
        spatial_attn_mask: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hist_rep = self._append_static_context(hist_feat, static_rep)  # [N, Th, C_hist + C_static]
        hist_in = self.hist_in_proj(hist_rep)  # [N, Th, C]
        return self._run_st_blocks(hist_in, self.hist_blocks, spatial_attn_mask, edge_index, temporal_causal=True)

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC, mode: int = 2):
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        n = batch[self.historical_ts_representative_key].shape[0]
        device = batch[self.historical_ts_representative_key].device

        s_mask = None if (edge_index is not None and edge_index.numel() > 0) else self._spatial_attn_mask(edge_index, n, device)

        hist_flow = batch['historical_target']  # [N, Th, 1]
        hist_num = batch.get('historical_ts_numeric', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_cat = batch.get('historical_ts_categorical', torch.zeros(n, self.num_historical_steps, 0, device=device)).float()
        hist_feat = torch.cat([hist_flow, hist_num, hist_cat], dim=-1)  # [N, Th, C_hist]

        h_hist = self._run_hist_branch(hist_feat, static_rep, s_mask, edge_index)  # [N, Th, C]

        if self.future_in_proj is None or future_ts_rep.numel() == 0:
            raise ValueError("future_ts_rep is empty but decoder requires future branch")
        ste_fut = self.future_in_proj(future_ts_rep)  # [N, Tf, C]
        ste_his = self.hist_in_proj(self._append_static_context(hist_feat, static_rep))  # [N, Th, C]

        # Transform-attention decoder (GMAN style): Q=STE_fut, K=STE_his, V=h_hist
        attn_out = self.transform_attn(h_hist, ste_his, ste_fut)  # [N, Tf, C]

        predicted_quantiles_IU = self.output_layer_IU(attn_out)  # [N, Tf, Q]
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

        # =========== placeholders (match other baselines) ===========
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
