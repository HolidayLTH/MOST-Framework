from typing import Dict, Tuple
import torch
from torch import nn
from omegaconf import DictConfig
from Model.TFT_GAT import InputChannelEmbedding, PINN


class LSTM_Benchmark(nn.Module):
    """
    Benchmark LSTM model matching TFT_GAT input/output interfaces.

    - Uses the same embedding pipeline as TFT_GAT via `transform_inputs`.
    - No attention, no GAT, no static enrichment.
    - Static features are treated as covariates and concatenated to temporal inputs.
    - Keeps the PINN module for physics extensibility.
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        # ============
        # data props
        # ============
        data_props = config['data_props']

        # -- static variables --
        self.num_static_numeric = data_props.get('num_static_numeric', 0)
        self.num_static_categorical = data_props.get('num_static_categorical', 0)
        self.static_categorical_cardinalities = data_props.get('static_categorical_cardinalities', [])

        # -- historical variables --
        self.num_historical_numeric = data_props.get('num_historical_numeric', 0)
        self.num_historical_categorical = data_props.get('num_historical_categorical', 0)
        self.historical_categorical_cardinalities = data_props.get('historical_categorical_cardinalities', [])

        # -- future variables --
        self.num_future_numeric = data_props.get('num_future_numeric', 0)
        self.num_future_categorical = data_props.get('num_future_categorical', 0)
        self.future_categorical_cardinalities = data_props.get('future_categorical_cardinalities', [])

        # -- targets --
        self.num_target = data_props.get('num_target', 1)
        self.num_auxiliary_target = data_props.get('num_auxiliary_target', 1)

        # -- representative keys --
        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 \
            else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 \
            else 'future_ts_categorical'

        self.is_mask = data_props.get('is_mask', True)

        # ============
        # model props
        # ============
        self.task_type = config.task_type
        self.dropout = config.model.dropout
        self.lstm_layers = config.model.lstm_layers
        self.state_size = config.model.state_size
        self.attention_heads = config.model.attention_heads
        self.target_window_start_idx = (config.target_window_start - 1) if config.target_window_start is not None else 0

        ## -- H related --
        self.Hour_ts_index = config.Hour_ts_index

        ## -- Day related --
        self.Day_index = config.Day_index

        ## -- R related --
        self.SL_ts_index = config.SL_ts_index
        self.TT_ts_index = config.TT_ts_index

        ## -- W related --
        self.W_ts_index = config.W_ts_index

        ## -- O related --
        self.O_index_categorical = config.O_index_categorical
        self.O_index_numeric = config.O_index_numeric

        ## -- D related --
        self.D_index_categorical = config.D_index_categorical
        self.D_index_numeric = config.D_index_numeric

        ## -- OD related --
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

        # =====================
        # Input Transformation: embeddings (same as TFT_GAT)
        # =====================
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

        # =============================
        # LSTM Encoder/Decoder
        # =============================
        static_dim = self.state_size * (self.num_static_numeric + self.num_static_categorical)
        hist_dim = self.state_size * (self.num_historical_numeric + self.num_historical_categorical)
        fut_dim = self.state_size * (self.num_future_numeric + self.num_future_categorical)

        self.encoder_lstm = nn.LSTM(
            input_size=hist_dim + static_dim,
            hidden_size=self.state_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=fut_dim + static_dim,
            hidden_size=self.state_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout,
            batch_first=True
        )

        # =============================
        # Output layer
        # =============================
        self.output_layer_IU = nn.Sequential(
            nn.Linear(self.state_size, self.state_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.state_size, self.num_outputs),
            nn.Sigmoid()
        )

        # =============================
        # PINN Output layer
        # =============================
        self.pinn = PINN(num_W_inputs=len(self.W_ts_index),
                         num_static_station_inputs=len(self.O_index_numeric) + len(self.O_index_categorical),
                         num_static_OD_inputs=len(self.OD_index_numeric) + len(self.OD_index_categorical),
                         num_Hour_inputs=len(self.Hour_ts_index),
                         num_Day_inputs=len(self.Day_index),
                         state_size=self.state_size,
                         dropout=self.dropout)

    def transform_inputs(self, batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, ...]:
        """
        Same embedding pipeline as TFT_GAT.
        """
        empty_tensor = torch.empty((0, 0))

        static_rep = self.static_transform(x_numeric=batch.get('static_feats_numeric', empty_tensor),
                                           x_categorical=batch.get('static_feats_categorical', empty_tensor))

        historical_ts_rep = self.historical_ts_transform(x_numeric=batch.get('historical_ts_numeric', empty_tensor),
                                                         x_categorical=batch.get('historical_ts_categorical', empty_tensor))

        future_ts_rep = self.future_ts_transform(x_numeric=batch.get('future_ts_numeric', empty_tensor),
                                                 x_categorical=batch.get('future_ts_categorical', empty_tensor))
        return future_ts_rep, historical_ts_rep, static_rep

    @staticmethod
    def replicate_along_time(static_signal: torch.tensor, time_steps: int) -> torch.tensor:
        return static_signal.unsqueeze(1).repeat(1, time_steps, 1)

    def forward(self, batch, edge_index, target_N, IS_EVAL, IS_STATION, IS_PHYSIC):
        # infer batch structure
        self.num_historical_steps = batch[self.historical_ts_representative_key].shape[1]
        self.num_samples, self.num_future_steps, _ = batch[self.future_ts_representative_key].shape

        # =========== Transform all input channels ==============
        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

        # =========== Concatenate static covariates ==============
        static_rep_hist = self.replicate_along_time(static_rep, self.num_historical_steps)
        static_rep_fut = self.replicate_along_time(static_rep, self.num_future_steps)

        historical_input = torch.cat([historical_ts_rep, static_rep_hist], dim=-1)
        future_input = torch.cat([future_ts_rep, static_rep_fut], dim=-1)

        # Only keep target samples (neighbors excluded)
        historical_input = historical_input[:target_N]
        future_input = future_input[:target_N]
        future_ts_rep = future_ts_rep[:target_N]
        static_rep = static_rep[:target_N]

        # =========== LSTM Encoder/Decoder ==============
        _, hidden_state = self.encoder_lstm(historical_input)
        decoder_output, _ = self.decoder_lstm(future_input, hidden_state)

        # output sequence length
        output_sequence_length = self.num_future_steps - self.target_window_start_idx

        # =========== output projection for IU ==============
        predicted_quantiles_IU = self.output_layer_IU(
            decoder_output[:, self.target_window_start_idx:, :]
        )

        # =========== PINN inputs ==============
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
        W_ts_rep = future_ts_rep[:, :, W_idx_tensor]

        O_idx_tensor = torch.tensor(self.O_index_numeric + list(map(lambda x: x + self.num_static_numeric, self.O_index_categorical)))
        O_idx_tensor = (O_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        O_rep = static_rep[:, O_idx_tensor]

        OD_idx_tensor = torch.tensor(self.OD_index_numeric + list(map(lambda x: x + self.num_static_numeric, self.OD_index_categorical)))
        OD_idx_tensor = (OD_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        OD_rep = static_rep[:, OD_idx_tensor]

        Hour_idx_tensor = torch.tensor(list(map(lambda x: x + self.num_future_numeric, self.Hour_ts_index)))
        Hour_idx_tensor = (Hour_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        Hour_ts_rep = future_ts_rep[:, :, Hour_idx_tensor]

        Hour_ts = batch['future_ts_categorical'][:target_N, :, self.Hour_ts_index]

        Day_idx_tensor = torch.tensor(list(map(lambda x: x + self.num_static_numeric, self.Day_index)))
        Day_idx_tensor = (Day_idx_tensor[:, None] * self.state_size + torch.arange(self.state_size)).reshape(-1)
        Day_rep = static_rep[:, Day_idx_tensor]

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

        # =========== zero placeholders for unused outputs ==============
        device = predicted_quantiles_IU.device
        static_inputs = self.num_static_numeric + self.num_static_categorical
        hist_inputs = self.num_historical_numeric + self.num_historical_categorical
        fut_inputs = self.num_future_numeric + self.num_future_categorical

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
