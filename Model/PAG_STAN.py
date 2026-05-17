from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from Model.TFT_GAT import InputChannelEmbedding, PINN

class DynamicODCompression(nn.Module):
    """Select high-demand OD pairs per origin to reduce sparsity.

    This follows the spirit of PAG-STAN dynamic compression in OD-sample format.
    """

    def __init__(self, pfp: float = 0.7):
        super().__init__()
        self.pfp = float(pfp)

    @staticmethod
    def _build_inverse_indices(batch_indices, device: torch.device, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_indices is None:
            origin_ids = torch.arange(n, device=device, dtype=torch.long)
        else:
            ids = []
            for k in batch_indices[:n]:
                try:
                    ids.append(int(str(k).split("@")[0]))
                except Exception:
                    ids.append(len(ids))
            origin_ids = torch.tensor(ids, device=device, dtype=torch.long)
        _, inverse = torch.unique(origin_ids, return_inverse=True)
        return origin_ids, inverse

    def forward(self, seq: torch.Tensor, batch_indices) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq: [N, T, C]
        n, t, _ = seq.shape
        device = seq.device
        _, inverse = self._build_inverse_indices(batch_indices, device, n)
        num_origins = int(inverse.max().item()) + 1 if n > 0 else 0

        # Demand proxy uses sequence energy.
        score = seq.abs().mean(dim=(1, 2))  # [N]
        mask = torch.zeros(n, device=device, dtype=seq.dtype)

        for o in range(num_origins):
            idx = torch.where(inverse == o)[0]
            if idx.numel() == 0:
                continue
            vals = score[idx]
            sorted_vals, _ = torch.sort(vals, descending=True)
            # cumsum on CUDA is non-deterministic; temporarily relax determinism locally.
            det_enabled = torch.are_deterministic_algorithms_enabled()
            if det_enabled:
                torch.use_deterministic_algorithms(False)
            try:
                csum = torch.cumsum(sorted_vals, dim=0)
            finally:
                if det_enabled:
                    torch.use_deterministic_algorithms(True)
            total = csum[-1].clamp_min(1e-6)
            keep_k = int((csum / total <= self.pfp).sum().item())
            keep_k = max(1, min(int(idx.numel()), keep_k))
            top_local = torch.topk(vals, k=keep_k, largest=True).indices
            chosen = idx[top_local]
            mask[chosen] = 1.0

        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, t, 1)
        compressed = seq * mask
        return compressed, mask


class AdaptiveGraphConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.msg_linear = nn.Linear(channels, channels)
        self.self_linear = nn.Linear(channels, channels)
        self.node_emb = nn.Parameter(torch.randn(32, channels) * 0.05)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _edge_agg(self, x: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [N, C]
        if edge_index is None or edge_index.numel() == 0:
            return self.self_linear(x)

        n, c = x.shape
        device = x.device
        src = edge_index[0].long().to(device)
        dst = edge_index[1].long().to(device)
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        src = src[valid]
        dst = dst[valid]

        msg = self.msg_linear(x[src])
        agg = torch.zeros(n, c, device=device, dtype=x.dtype)
        deg = torch.zeros(n, 1, device=device, dtype=x.dtype)
        ones = torch.ones(src.shape[0], 1, device=device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        deg.index_add_(0, dst, ones)
        return self.self_linear(x) + agg / deg.clamp_min(1.0)

    def _adaptive_agg(self, x: torch.Tensor) -> torch.Tensor:
        # Build adaptive adjacency from trainable station embeddings.
        n = x.size(0)
        if n <= self.node_emb.size(0):
            emb = self.node_emb[:n]
        else:
            repeat = int((n + self.node_emb.size(0) - 1) / self.node_emb.size(0))
            emb = self.node_emb.repeat(repeat, 1)[:n]
        a = torch.softmax(F.relu(emb @ emb.transpose(0, 1)), dim=-1)
        return a @ x

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        edge_part = self._edge_agg(x, edge_index)
        adp_part = self._adaptive_agg(x)
        w = torch.sigmoid(self.alpha)
        return w * edge_part + (1.0 - w) * adp_part


class AGCLSTMCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gconv_x = AdaptiveGraphConv(hidden_dim)
        self.gconv_h = AdaptiveGraphConv(hidden_dim)

        self.w_i = nn.Linear(hidden_dim, hidden_dim)
        self.u_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_f = nn.Linear(hidden_dim, hidden_dim)
        self.u_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        self.u_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_g = nn.Linear(hidden_dim, hidden_dim)
        self.u_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, edge_index: Optional[torch.Tensor]):
        gx = self.gconv_x(x_t, edge_index)
        gh = self.gconv_h(h_prev, edge_index)

        i = torch.sigmoid(self.w_i(gx) + self.u_i(gh))
        f = torch.sigmoid(self.w_f(gx) + self.u_f(gh))
        o = torch.sigmoid(self.w_o(gx) + self.u_o(gh))
        g = torch.tanh(self.w_g(gx) + self.u_g(gh))

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class AGCLSTM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cell = AGCLSTMCell(hidden_dim)

    def forward(self, seq: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        # seq: [N, T, C]
        n, t, c = seq.shape
        h = torch.zeros(n, c, device=seq.device, dtype=seq.dtype)
        cc = torch.zeros_like(h)
        outs = []
        for i in range(t):
            h, cc = self.cell(seq[:, i, :], h, cc, edge_index)
            outs.append(h)
        return torch.stack(outs, dim=1)


class MPCAttention(nn.Module):
    """Multi-period cross attention over weekly/daily/recent branches."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, weekly: torch.Tensor, daily: torch.Tensor, recent: torch.Tensor) -> torch.Tensor:
        # Inputs: [N, T, C]
        n, t, c = recent.shape
        q = recent.reshape(n * t, 1, c)
        kv = torch.stack([weekly, daily, recent], dim=2).reshape(n * t, 3, c)
        out, _ = self.attn(q, kv, kv)
        out = self.norm(out + q)
        return out.reshape(n, t, c)


class HIFB(nn.Module):
    """Heterogeneous information fusion block for external factors."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.ext_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, core_seq: torch.Tensor, ext_seq: torch.Tensor) -> torch.Tensor:
        e = self.ext_proj(ext_seq)
        g = self.gate(torch.cat([core_seq, e], dim=-1))
        return core_seq + g * e


class PAGSTAN_Benchmark(nn.Module):
    """PAG-STAN baseline.

    - Ignores real-time OD estimation module as requested.
    - Keeps dynamic OD compression + complete spatial-temporal backbone.
    - Adds MPG-related mask outputs for masked physics-guided loss.
    - Keeps training interface and output dict compatible with existing process_batch.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        data_props = config["data_props"]
        self.num_static_numeric = data_props.get("num_static_numeric", 0)
        self.num_static_categorical = data_props.get("num_static_categorical", 0)
        self.static_categorical_cardinalities = data_props.get("static_categorical_cardinalities", [])

        self.num_historical_numeric = data_props.get("num_historical_numeric", 0)
        self.num_historical_categorical = data_props.get("num_historical_categorical", 0)
        self.historical_categorical_cardinalities = data_props.get("historical_categorical_cardinalities", [])

        self.num_future_numeric = data_props.get("num_future_numeric", 0)
        self.num_future_categorical = data_props.get("num_future_categorical", 0)
        self.future_categorical_cardinalities = data_props.get("future_categorical_cardinalities", [])

        self.num_target = data_props.get("num_target", 1)
        self.num_auxiliary_target = data_props.get("num_auxiliary_target", 1)

        self.historical_ts_representative_key = "historical_ts_numeric" if self.num_historical_numeric > 0 else "historical_ts_categorical"
        self.future_ts_representative_key = "future_ts_numeric" if self.num_future_numeric > 0 else "future_ts_categorical"

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

        if self.task_type == "regression":
            self.output_quantiles = config.model.output_quantiles
            self.num_outputs = len(self.output_quantiles) * self.num_target
            self.num_auxiliary_outputs = len(self.output_quantiles) * self.num_auxiliary_target
        elif self.task_type == "classification":
            self.output_quantiles = None
            self.num_outputs = 1
            self.num_auxiliary_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")

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

        self.hist_feat_dim = 1 + self.num_historical_numeric + self.num_historical_categorical
        self.future_rep_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)
        self.static_rep_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)

        if self.hist_feat_dim <= 0:
            raise ValueError("historical features are empty; PGGSTAN_Benchmark requires historical temporal features")

        self.static_ctx_proj = nn.Linear(self.static_rep_dim, self.state_size) if self.static_rep_dim > 0 else None
        branch_static = self.state_size if self.static_ctx_proj is not None else 0

        self.hist_in_proj = nn.Linear(self.hist_feat_dim + branch_static, self.state_size)
        self.fut_in_proj = nn.Linear(self.future_rep_dim, self.state_size) if self.future_rep_dim > 0 else None

        self.compression = DynamicODCompression(pfp=0.7)

        self.weekly_encoder = AGCLSTM(self.state_size)
        self.daily_encoder = AGCLSTM(self.state_size)
        self.recent_encoder = AGCLSTM(self.state_size)

        self.mpc_attn = MPCAttention(self.state_size, max(1, self.attention_heads), self.dropout)
        self.decoder = nn.LSTM(
            input_size=self.state_size,
            hidden_size=self.state_size,
            num_layers=max(1, self.lstm_layers),
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_proj = nn.Linear(2 * self.state_size, self.state_size)

        self.hifb = HIFB(self.state_size, self.dropout)

        # Decoder + HIFB are used for history/future fusion (closer to original PGG-STAN).

        self.output_layer_IU = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.state_size, self.num_outputs),
            nn.Sigmoid(),
        )

        self.phi_head = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.state_size, self.num_auxiliary_outputs),
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

        # MPG loss weights from paper-style objective.
        self.mpg_w_od = nn.Parameter(torch.tensor(1.0))
        self.mpg_w_in = nn.Parameter(torch.tensor(1.0))

    def transform_inputs(self, batch):
        empty_tensor = torch.empty((0, 0))
        static_rep = self.static_transform(
            x_numeric=batch.get("static_feats_numeric", empty_tensor),
            x_categorical=batch.get("static_feats_categorical", empty_tensor),
        )
        historical_ts_rep = self.historical_ts_transform(
            x_numeric=batch.get("historical_ts_numeric", empty_tensor),
            x_categorical=batch.get("historical_ts_categorical", empty_tensor),
        )
        future_ts_rep = self.future_ts_transform(
            x_numeric=batch.get("future_ts_numeric", empty_tensor),
            x_categorical=batch.get("future_ts_categorical", empty_tensor),
        )
        return future_ts_rep, historical_ts_rep, static_rep

    def _append_static_context(self, seq_rep: torch.Tensor, static_rep: torch.Tensor) -> torch.Tensor:
        if self.static_ctx_proj is None or static_rep.numel() == 0:
            return seq_rep
        static_ctx = self.static_ctx_proj(static_rep).unsqueeze(1).expand(-1, seq_rep.size(1), -1)
        return torch.cat([seq_rep, static_ctx], dim=-1)

    @staticmethod
    def _split_periodic(seq: torch.Tensor):
        # Split historical sequence into weekly/daily/recent branches.
        n, t, c = seq.shape
        if t < 3:
            return seq, seq, seq
        k = t // 3
        if k == 0:
            return seq, seq, seq
        weekly = seq[:, :k, :]
        daily = seq[:, k : 2 * k, :]
        recent = seq[:, 2 * k :, :]
        if weekly.size(1) == 0:
            weekly = recent
        if daily.size(1) == 0:
            daily = recent
        if recent.size(1) == 0:
            recent = daily
        target_len = recent.size(1)

        def _resize(x):
            if x.size(1) == target_len:
                return x
            if x.size(1) > target_len:
                return x[:, -target_len:, :]
            rep = int((target_len + x.size(1) - 1) / x.size(1))
            return x.repeat(1, rep, 1)[:, :target_len, :]

        return _resize(weekly), _resize(daily), recent

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC, mode: int = 2):
        del mode  # PAG-STAN backbone is fixed fusion mode.

        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        hist_flow = batch["historical_target"]  # [N, Th, 1]
        hist_num = batch.get("historical_ts_numeric", torch.zeros(hist_flow.size(0), self.num_historical_steps, 0, device=hist_flow.device)).float()
        hist_cat = batch.get("historical_ts_categorical", torch.zeros(hist_flow.size(0), self.num_historical_steps, 0, device=hist_flow.device)).float()
        hist_feat = torch.cat([hist_flow, hist_num, hist_cat], dim=-1)  # [N, Th, C_hist]

        hist_feat = self._append_static_context(hist_feat, static_rep)  # [N, Th, C_hist + C_static]
        hist_rep = self.hist_in_proj(hist_feat)  # [N, Th, C]

        # Dynamic compression from historical complete OD information.
        hist_comp, mpg_mask = self.compression(hist_rep, batch.get("ODindex", None))

        weekly_in, daily_in, recent_in = self._split_periodic(hist_comp)
        h_week = self.weekly_encoder(weekly_in, edge_index)
        h_day = self.daily_encoder(daily_in, edge_index)
        h_recent = self.recent_encoder(recent_in, edge_index)

        h_mpc = self.mpc_attn(h_week, h_day, h_recent)  # [N, Th, C]

        # Encoder output (historical context): [N, Th, C]
        h_hist = h_mpc

        if self.fut_in_proj is None or future_ts_rep.numel() == 0:
            raise ValueError("future_ts_rep is empty but decoder requires future branch")
        fut_feat = self.fut_in_proj(future_ts_rep)  # [N, Tf, C]

        # Decode future sequence and fuse with historical context using HIFB.
        dec_out, _ = self.decoder(fut_feat)  # [N, Tf, 2C]
        dec_out = self.decoder_proj(dec_out)  # [N, Tf, C]
        hist_ctx = h_hist[:, -dec_out.size(1):, :]  # [N, Tf, C]
        if hist_ctx.size(1) != dec_out.size(1):
            # Align history context length with decoder output length.
            target_len = dec_out.size(1)
            if hist_ctx.size(1) == 0:
                hist_ctx = dec_out.new_zeros(dec_out.size(0), target_len, dec_out.size(2))
            elif hist_ctx.size(1) > target_len:
                hist_ctx = hist_ctx[:, -target_len:, :]
            else:
                repeat = int((target_len + hist_ctx.size(1) - 1) / hist_ctx.size(1))
                hist_ctx = hist_ctx.repeat(1, repeat, 1)[:, :target_len, :]
        attn_out = self.hifb(dec_out, hist_ctx)  # [N, Tf, C]

        predicted_quantiles_IU = self.output_layer_IU(attn_out)  # [N, Tf, Q]
        predicted_quantiles_phi = self.phi_head(attn_out)  # [N, Tf, Q_aux]

        # Align MPG mask to future length.
        if mpg_mask.size(1) >= self.num_future_steps:
            mpg_mask_future = mpg_mask[:, -self.num_future_steps :, :]
        else:
            repeat = int((self.num_future_steps + mpg_mask.size(1) - 1) / mpg_mask.size(1))
            mpg_mask_future = mpg_mask.repeat(1, repeat, 1)[:, : self.num_future_steps, :]

        predicted_quantiles_IU = predicted_quantiles_IU[:, self.target_window_start_idx :, :]
        predicted_quantiles_phi = predicted_quantiles_phi[:, self.target_window_start_idx :, :]
        mpg_mask_future = mpg_mask_future[:, self.target_window_start_idx :, :]

        if mpg_mask_future.size(1) != predicted_quantiles_IU.size(1):
            # Keep MPG mask aligned with prediction horizon.
            target_len = predicted_quantiles_IU.size(1)
            if mpg_mask_future.size(1) == 0:
                mpg_mask_future = predicted_quantiles_IU.new_zeros(
                    predicted_quantiles_IU.size(0), target_len, 1
                )
            elif mpg_mask_future.size(1) > target_len:
                mpg_mask_future = mpg_mask_future[:, -target_len:, :]
            else:
                repeat = int((target_len + mpg_mask_future.size(1) - 1) / mpg_mask_future.size(1))
                mpg_mask_future = mpg_mask_future.repeat(1, repeat, 1)[:, :target_len, :]

        predicted_quantiles_IU = predicted_quantiles_IU[:target_N]
        predicted_quantiles_phi = predicted_quantiles_phi[:target_N]
        mpg_mask_future = mpg_mask_future[:target_N]

        batch_indices = batch["ODindex"][:target_N]
        device = predicted_quantiles_IU.device

        TT_ts = batch["future_ts_numeric"][:target_N, :, self.TT_ts_index]
        SL_ts = batch["future_ts_numeric"][:target_N, :, self.SL_ts_index]

        if not IS_EVAL:
            phi_ts = batch["future_target_OFlow"][:target_N, :, :]
            I_ts = batch["future_target"][:target_N, :]
        else:
            phi_ts = None
            I_ts = None

        maxphi = batch["maxinflow"][:target_N, :]

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

        Hour_ts = batch["future_ts_categorical"][:target_N, :, self.Hour_ts_index]

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

        predicted_quantiles_phi = torch.clamp(0.5 * predicted_quantiles_phi + 0.5 * phi, 0.0, 1.0)

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
            past_attention_alpha = torch.zeros(1, self.attention_heads, device=device)
            future_attention_alpha = torch.zeros(1, self.attention_heads, device=device)
            past_edgeindex = torch.zeros(2, 1, dtype=torch.long, device=device)
            future_edgeindex = torch.zeros(2, 1, dtype=torch.long, device=device)

        return {
            "predicted_quantiles_IU": predicted_quantiles_IU,
            "predicted_quantiles_phi": predicted_quantiles_phi,
            "predicted_R": r_physics,
            "predicted_S": s,
            "predicted_Alpha": alpha,
            "predicted_Beta": beta,
            "predicted_Eta": eta,
            "predicted_Rb": rb,
            "predicted_congestion": congestion,
            "attention_scores": attention_scores,
            "static_weights": static_weights,
            "historical_selection_weights": historical_selection_weights,
            "future_selection_weights": future_selection_weights,
            "pastgat_attention": past_attention_alpha,
            "futuregat_attention": future_attention_alpha,
            "past_edgeindex": past_edgeindex,
            "future_edgeindex": future_edgeindex,
            "mpg_mask": mpg_mask_future,
            "use_mpg_loss": torch.tensor(1.0, device=device),
            "mpg_w_od": F.softplus(self.mpg_w_od),
            "mpg_w_in": F.softplus(self.mpg_w_in),
        }
