from __future__ import annotations
import importlib
from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from Model.TFT_GAT import InputChannelEmbedding, PINN

try:
    torchdiffeq_odeint = importlib.import_module("torchdiffeq").odeint
except Exception:
    torchdiffeq_odeint = None


class NeuralDiffusionODEFunc(nn.Module):
    """Neural diffusion field dynamics used by PI-MPN baseline.

    dz/dt = D * Laplace(z) + forcing(context_t)
    where z is a 2-channel field (inflow/outflow latent state).
    """

    def __init__(self, context_dim: int):
        super().__init__()
        self.context_to_force = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, 2),
        )
        self.log_diffusion = nn.Parameter(torch.tensor(-0.7))

        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("laplace_kernel", lap.view(1, 1, 3, 3))

        self._context_seq: Optional[torch.Tensor] = None

    def set_context(self, context_seq: torch.Tensor) -> None:
        # context_seq: [N, Tf, C]
        self._context_seq = context_seq

    def _context_at(self, t: torch.Tensor) -> torch.Tensor:
        if self._context_seq is None:
            raise RuntimeError("ODE context is not set before integration")

        tf = self._context_seq.size(1)
        t_idx = int(torch.clamp(torch.round(t).long(), min=0, max=tf - 1).item())
        return self._context_seq[:, t_idx, :]

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: [N, 2, H, W]
        ctx_t = self._context_at(t)  # [N, C]
        force = self.context_to_force(ctx_t).unsqueeze(-1).unsqueeze(-1)  # [N, 2, 1, 1]

        kernel = self.laplace_kernel.expand(2, 1, 3, 3)
        lap = F.conv2d(z, kernel, padding=1, groups=2)

        diffusion_coeff = F.softplus(self.log_diffusion)
        return diffusion_coeff * lap + force


class NeuralDiffusionFieldHead(nn.Module):
    """Physics-inspired neural diffusion field + ODE correction head.

    Input: fused sequence representation [N, Tf, C]
    Output:
      - flow correction delta [N, Tf, 1]
      - latent io proxy [N, Tf, 1]
    """

    def __init__(self, state_size: int, field_size: int = 10, method: str = "euler"):
        super().__init__()
        self.state_size = state_size
        self.field_size = field_size
        self.method = method

        self.init_proj = nn.Linear(state_size, 2 * field_size * field_size)
        self.context_proj = nn.Linear(state_size, state_size)
        self.ode_func = NeuralDiffusionODEFunc(context_dim=state_size)

        self.delta_head = nn.Sequential(
            nn.Linear(2, state_size // 2),
            nn.ReLU(),
            nn.Linear(state_size // 2, 1),
            nn.Tanh(),
        )
        self.io_head = nn.Sequential(
            nn.Linear(2, state_size // 2),
            nn.ReLU(),
            nn.Linear(state_size // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _fallback_euler(func: nn.Module, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        states = [z0]
        z = z0
        for i in range(1, t.numel()):
            dt = t[i] - t[i - 1]
            dz = func(t[i - 1], z)
            z = z + dt * dz
            states.append(z)
        return torch.stack(states, dim=0)

    def _integrate(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if torchdiffeq_odeint is not None:
            return torchdiffeq_odeint(self.ode_func, z0, t, method=self.method, atol=1e-5, rtol=1e-4)
        return self._fallback_euler(self.ode_func, z0, t)

    def forward(self, fused_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, tf, c = fused_seq.shape
        device = fused_seq.device

        context_seq = self.context_proj(fused_seq)  # [N, Tf, C]
        self.ode_func.set_context(context_seq)

        z0 = self.init_proj(fused_seq[:, -1, :]).view(n, 2, self.field_size, self.field_size)
        t = torch.arange(tf, dtype=fused_seq.dtype, device=device)
        z_traj = self._integrate(z0, t)  # [Tf, N, 2, H, W]
        z_traj = z_traj.permute(1, 0, 2, 3, 4).contiguous()  # [N, Tf, 2, H, W]

        io_latent = z_traj.mean(dim=(-1, -2))  # [N, Tf, 2]
        delta = self.delta_head(io_latent) * 0.05
        io_proxy = self.io_head(io_latent)
        return delta, io_proxy

class PIMPN_Benchmark(nn.Module):
    """PI-MPN baseline compatible with TFT_GAT training pipeline.

    Design principles:
    - Use PI-MPN encoder for history and cross-attention decoding for future.
    - Keep static_rep involved in both historical and future branches.
    - Add explicit physics-inspired module: neural diffusion field + ODE.
    - Keep output dictionary keys aligned with process_batch requirements.
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
            raise ValueError("historical features are empty; PIMPN_Benchmark requires historical temporal features")

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

        # Encoder backbone (same protocol as other baselines)
        self.hist_gru = nn.GRU(
            input_size=self.state_size,
            hidden_size=self.state_size,
            num_layers=max(1, self.lstm_layers),
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
        )

        self.spatial_msg = nn.Linear(self.state_size, self.state_size)
        self.spatial_self = nn.Linear(self.state_size, self.state_size)

        self.future_gru = nn.GRU(
            input_size=self.state_size,
            hidden_size=self.state_size,
            num_layers=max(1, self.lstm_layers),
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
        )

        # Physics-inspired neural diffusion field + ODE
        self.physics_field = NeuralDiffusionFieldHead(state_size=self.state_size, field_size=10, method="euler")

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
        static_ctx = self.static_ctx_proj(static_rep)
        static_ctx = static_ctx.unsqueeze(1).expand(-1, seq_rep.size(1), -1)
        return torch.cat([seq_rep, static_ctx], dim=-1)

    def _spatial_aggregate(self, seq: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        # seq: [N, T, C]
        if edge_index is None or edge_index.numel() == 0:
            return seq

        n, t, c = seq.shape
        device = seq.device
        src = edge_index[0].long().to(device)
        dst = edge_index[1].long().to(device)
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        src = src[valid]
        dst = dst[valid]

        src_orig = src
        dst_orig = dst

        src = torch.cat([src_orig, dst_orig, torch.arange(n, device=device)], dim=0)
        dst = torch.cat([dst_orig, src_orig, torch.arange(n, device=device)], dim=0)

        out = torch.empty_like(seq)
        ones = torch.ones(dst.shape[0], 1, device=device, dtype=seq.dtype)
        for i in range(t):
            x_t = seq[:, i, :]
            msg = self.spatial_msg(x_t[src])
            agg = torch.zeros(n, c, device=device, dtype=seq.dtype)
            deg = torch.zeros(n, 1, device=device, dtype=seq.dtype)
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, ones)
            agg = agg / deg.clamp_min(1.0)
            out[:, i, :] = self.spatial_self(x_t) + agg
        return out

    def _run_hist_branch(self, hist_feat: torch.Tensor, static_rep: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        hist_rep = self._append_static_context(hist_feat, static_rep)  # [N, Th, C_hist + C_static]
        hist_in = self.hist_in_proj(hist_rep)  # [N, Th, C]
        h_hist, _ = self.hist_gru(hist_in)  # [N, Th, C]
        return self._spatial_aggregate(h_hist, edge_index)  # [N, Th, C]

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC, mode: int = 2):
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        hist_flow = batch["historical_target"]  # [N, Th, 1]
        hist_num = batch.get("historical_ts_numeric", torch.zeros(hist_flow.size(0), self.num_historical_steps, 0, device=hist_flow.device)).float()
        hist_cat = batch.get("historical_ts_categorical", torch.zeros(hist_flow.size(0), self.num_historical_steps, 0, device=hist_flow.device)).float()
        hist_feat = torch.cat([hist_flow, hist_num, hist_cat], dim=-1)  # [N, Th, C_hist]

        h_hist = self._run_hist_branch(hist_feat, static_rep, edge_index)  # [N, Th, C]

        if self.future_in_proj is None or future_ts_rep.numel() == 0:
            raise ValueError("future_ts_rep is empty but decoder requires future branch")
        h_fut = self.future_in_proj(future_ts_rep)  # [N, Tf, C]

        # Future decoder initialized by historical context (closer to original PI-MPN usage).
        h0 = h_hist[:, -1:, :].transpose(0, 1).contiguous()  # [1, N, C]
        attn_out, _ = self.future_gru(h_fut, h0)  # [N, Tf, C]

        base_i = self.output_layer_IU(attn_out)  # [N, Tf, Q]
        delta_i, io_proxy = self.physics_field(attn_out)  # [N, Tf, 1]
        delta_i = delta_i.repeat(1, 1, self.num_outputs)
        predicted_quantiles_IU = torch.clamp(base_i + delta_i, 0.0, 1.0)

        predicted_quantiles_IU = predicted_quantiles_IU[:, self.target_window_start_idx :, :]
        predicted_quantiles_IU = predicted_quantiles_IU[:target_N]

        batch_indices = batch["ODindex"][:target_N]
        device = predicted_quantiles_IU.device

        TT_ts = batch["future_ts_numeric"][:target_N, :, self.TT_ts_index]
        SL_ts = batch["future_ts_numeric"][:target_N, :, self.SL_ts_index]

        if not IS_EVAL:
            phi_ts = batch["future_target_OFlow"][:target_N, :, :]
        else:
            phi_ts = None

        maxphi = batch["maxinflow"][:target_N, :]

        if not IS_EVAL:
            I_ts = batch["future_target"][:target_N, :]
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

        Hour_ts = batch["future_ts_categorical"][:target_N, :, self.Hour_ts_index]

        Day_idx_tensor = torch.tensor(
            [x + self.num_static_numeric for x in self.Day_index],
            device=device,
            dtype=torch.long,
        )
        Day_idx_tensor = (Day_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size, device=device)).reshape(-1)
        Day_rep = static_rep[:target_N, Day_idx_tensor]

        # Keep existing PINN outputs for compatibility with current process_batch.
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

        # Blend ODE io proxy into station-flow proxy channel as a lightweight physical signal.
        io_proxy = io_proxy[:target_N, self.target_window_start_idx :, :]
        predicted_quantiles_phi = torch.clamp(0.5 * phi + 0.5 * io_proxy, 0.0, 1.0)

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
        }
