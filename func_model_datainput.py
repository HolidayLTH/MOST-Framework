# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 20:13:37 2025

@author: 78632
"""
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from config import feature_cols, hour_cols, travel_cols, weather_cols, static_attrs, historical_windows, future_windows, BASE_DIR
import torch
from typing import List
from functools import reduce
import pickle
from datetime import datetime, timedelta

def reciprocal(x):
    return 1/(1+(x+5)**(2/3))

def load_and_cache_neighbormap(network_features_folder):
    files = os.listdir(network_features_folder)
    neighbor_files = sorted([f for f in files if f.startswith('similar_od')])
    network_periods = []

    for f in neighbor_files:
        try:
            parts = f.replace('similar_od_', '').replace('.csv', '').split('-')
            start_date = pd.to_datetime(parts[0], format='%Y%m%d')
            end_date = pd.to_datetime(parts[1], format='%Y%m%d')
            
            file_path = os.path.join(network_features_folder, f)
            neighbormap = pd.read_csv(file_path,index_col=0)
            
            network_periods.append({
                'start': start_date, 'end': end_date,
                'neighbor_map': neighbormap
            })

        except Exception as e:
            print(f"Error processing file {f}: {e}")
    print("--- Network feature loading complete ---")
    return network_periods
    
class SubwayODDataset(Dataset):
    def __init__(self, data_files, scalers):

        super().__init__()
        self.data_files = data_files
        self.scalers = scalers

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):

        file_path = self.data_files[idx]
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # --- 1. load csv file ---
        try:
            with open(rf'{BASE_DIR}/Cache/{filename}.pkl', 'rb') as f:
                day_dict = pickle.load(f)
                print(f'Read the processed file: {file_path}')
                return day_dict
        except:
            pass

        try:
            df = pd.read_csv(file_path)
            df = df[[col  for col  in feature_cols if col in df.columns]]
            df = df.set_index('ODindex',drop=False)

            for col in hour_cols + travel_cols + weather_cols:
                if col in df:
                    df[col] = df[col].map(lambda x: np.fromstring(x.strip("[]"), sep=','))
                    if 'TravelTime' in col or 'FreeFlowTravelTime' in col:
                        df[col] = df[col].map(lambda x: reciprocal(x).round(3))

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return {}
        
        # --- 2. Add inbound station flow ---
        for timeloc in ['@D-1','@D','@D+1']:
            if f'TravelFlow{timeloc}' in df.columns:
                df[f'OFlow{timeloc}'] = df['Oindex'].map(df.groupby('Oindex')[f'TravelFlow{timeloc}'].apply(lambda x: np.sum(np.vstack(x.values), axis=0)))
                scaler = self.scalers['numeric']['TravelFlow']
                df[f'OFlow{timeloc}'] = df[f'OFlow{timeloc}'].map(lambda x: scaler.transform(x.reshape(-1, 1)).flatten().astype(np.float32))
        
        ## --- 2.1. Add station max inflow ---
        maxinflow = pd.read_csv(rf'{BASE_DIR}/Source data/Station_Historical_Max_Inflow.csv',index_col=0)
        maxinflow.rename(columns={'历史最大进站流量':'MaxInflow'},inplace=True)
        date = filename.split('#')[0]
        maxinflow = maxinflow[maxinflow['Date'].str.replace('-','') == date]
        scaler = self.scalers['numeric']['TravelFlow']
        maxinflow['MaxInflow'] = scaler.transform(maxinflow['MaxInflow'].values.reshape(-1, 1)).flatten().astype(np.float32)
        df = pd.merge(df,maxinflow[['Oindex','MaxInflow']],on='Oindex',how='inner')
        df = df.set_index('ODindex',drop=False)
        
        ## -- 2.2. Add bidirectional travel-time asymmetry ---
        congestion_df = pd.read_csv(rf'{BASE_DIR}/Source data/Station_Congestion_Metrics_Adjusted.csv')
        congestion_df['deltaT'] = 2*congestion_df['deltaT']/(congestion_df['T_AX']+congestion_df['T_XA'])
        
        current_date = pd.to_datetime(date, format="%Y%m%d").strftime("%Y-%m-%d")
        df = add_congestion_features(df, congestion_df, current_date)
        df = df.set_index('ODindex',drop=False)
        
        # --- 3. Transform Features ---
        for col in df.columns:
            if col in static_attrs:
                if col in self.scalers['categorical']:
                    encoder = self.scalers['categorical'][col]
                    le_dict = {val: i for i, val in enumerate(encoder.classes_)}
                    unknown_label = len(encoder.classes_)
                    df[col] = df[col].map(le_dict).fillna(unknown_label).astype(np.int32)
                elif col in self.scalers['numeric']:
                    scaler = self.scalers['numeric'][col]
                    df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten().astype(np.float32)
            elif (col == 'MaxInflow') or ('Congestion' in col):
                continue
            else:
                attr = col.split('@')[0]
                
                if attr in self.scalers['categorical']:
                    encoder = self.scalers['categorical'][attr]
                    le_dict = {val: i for i, val in enumerate(encoder.classes_)}
                    unknown_label = len(encoder.classes_)
                    df[col] = df[col].map(lambda x: np.array([le_dict.get(val, unknown_label) for val in x],dtype=np.int32))
                elif attr in self.scalers['numeric']:
                    scaler = self.scalers['numeric'][attr]
                    df[col] = df[col].map(lambda x: scaler.transform(x.reshape(-1, 1)).flatten().astype(np.float32))

                else: # Some variables do not need standardization.
                    df[col] = df[col].map(lambda x: x.astype(np.int32))
        
        # --- 4. Create New Columns ---
        for col in df.columns:
            if 'FreeFlowTravelTime' in col:
                timeloc = col.split('@')[1]
                df[col] = df[col]-df[f'TravelTime@{timeloc}']
                df[col] = df[col].apply(lambda x: np.clip(x, 0, None))
                df.rename(columns={col:f'SupplyLoss@{timeloc}'}, inplace=True)
        
        # --- 5. add day info ---
        is_ewe = 1 if 'E' in filename else 0
        date = filename.split('#')[0]
        even_ID = filename.split('#')[1]
        day_shape = filename.split('#')[2]

        day_info = {
            'Date': date,
            'is_extreme_day': is_ewe,
            'Event_ID': even_ID,
            'DayShape': day_shape
        }
        
        day_dict = {
            "day_info": day_info,
            "data": df
        }
        
        # --- 6. Cache data ---
        with open(rf'{BASE_DIR}/Cache/{filename}.pkl', 'wb') as f:
            pickle.dump(day_dict, f)
            print(f'The file at this path is processed: {file_path}')
 
        return day_dict

def add_congestion_features(df_input, congestion_df, current_date_str):
    """
    Efficiently add Congestion@D, @D-1, @D+1 features to df.
    
    Args:
    df_input: Intermediate dataframe for model input (includes Oindex).
    congestion_df: Dataframe with all congestion info (columns: station, Date, Hour, deltaT).
    current_date_str: Date for the current df (string 'YYYY-MM-DD').
    """
    df = df_input.copy()
    
    # 1. Compute date strings for three days.
    curr_date = datetime.strptime(current_date_str, '%Y-%m-%d')
    
    # ================= 1. Adaptive column detection (core change) =================
    # Define possible suffixes.
    possible_suffixes = ['@D-1', '@D', '@D+1']
    active_suffixes = []
    
    # Scan existing column names.
    existing_cols = set(df.columns)
    
    for suffix in possible_suffixes:
        # Check whether any column ends with the suffix (e.g., TravelTime@D-1).
        # If any column matches, this time step is required.
        if any(col.endswith(suffix) for col in existing_cols):
            active_suffixes.append(suffix)
   
    full_date_map = {
            '@D-1': (curr_date - timedelta(days=1)).strftime('%Y-%m-%d'),
            '@D':   curr_date.strftime('%Y-%m-%d'),
            '@D+1': (curr_date + timedelta(days=1)).strftime('%Y-%m-%d')
        }
    target_date_map = {k: v for k, v in full_date_map.items() if k in active_suffixes}
    
    # 2. Preprocess congestion data (only process the three relevant days).
    relevant_dates = list(target_date_map.values())
    sub_cong = congestion_df[congestion_df['Date'].isin(relevant_dates)].copy()
    
    # 3. Core speedup: pivot table.
    # Convert long table to wide: Index=[Date, Station], Columns=0..23.
    # fill_value=0 auto-fills missing hours (including 0-5).
    cong_pivot = sub_cong.pivot_table(
        index=['Date', 'Oindex'], 
        columns='Hour', 
        values='deltaT', 
        fill_value=0
    )
    cong_pivot = cong_pivot.round(2)
    
    # Force fill 0-23 hours (avoid missing columns when a day has no data).
    cong_pivot = cong_pivot.reindex(columns=range(24), fill_value=0)
    
    # 4. Build lookup dict: (Date, Oindex) -> array.
    # Precompute arrays to avoid repeated work in map.
    lookup_dict = {}
    
    # Iterate each row in the pivot table.
    for idx, row in cong_pivot.iterrows():
        # idx is (Date, station).
        # row.values is a numpy array.
        lookup_dict[idx] = row.values.copy()

    # Default value for missing keys (all-zero array).
    default_zero_arr = np.zeros(24, dtype=np.float64)

    # 5. Map back to the original DataFrame.
    # Handle each day (@D-1, @D, @D+1) separately.
    for suffix, target_date in target_date_map.items():
        col_name = f'Congestion{suffix}'
        
        # Build lookup key: combine df Oindex with target date.
        # Use map + dict.get.
        
        # Small closure for "return zeros if key not found".
        def get_value(o_idx):
            return lookup_dict.get((target_date, o_idx), default_zero_arr)
            
        # Apply mapping.
        df[col_name] = df['Oindex'].map(get_value)
        
    return df

def custom_collate_fn(batch):

    batch = [item for item in batch if item]
    if not batch:
        return {}

    collated_batch = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        collated_batch[key] = values
            
    return collated_batch

def ConcatMultiDays(df, col, a, b):
    res = np.concatenate([
                            np.stack(df[f'{col}@D-1'].values) if f'{col}@D-1' in df.columns else np.empty((len(df), 0)), 
                            np.stack(df[f'{col}@D'].values), 
                            np.stack(df[f'{col}@D+1'].values) if f'{col}@D+1' in df.columns else np.empty((len(df), 0))
                            ], axis=1)[:, a:b]
    return res

def _get_partitioned_indices(sorted_series: pd.Series, thresholds: List[float]) -> List[np.ndarray]:
    """
    Helper function: split a sorted Series into multiple index arrays based on thresholds.

    Example: thresholds = [0, 0.4, 0.8] returns two interval indices:
    1. (0, 0.4]
    2. (0.4, 0.8]
    """
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) < 2:
        raise ValueError(
            f"Thresholds (TS) must be a list with at least two elements (e.g., [0, 0.8]), got: {thresholds}"
        )
        
    partitions = []
    cumperc = sorted_series.cumsum() / sorted_series.sum()
    
    # Iterate thresholds to form (lower, upper] intervals.
    for i in range(len(thresholds) - 1):
        lower_bound = thresholds[i]
        upper_bound = thresholds[i+1]
        
        # Core logic: select entries with cumulative percent in (lower_bound, upper_bound].
        mask = (cumperc > lower_bound) & (cumperc <= upper_bound)
        
        # Get indices for the selected entries.
        indices = sorted_series[mask].index.to_numpy()
        partitions.append(indices)
        
    return partitions

def OD_Filter(df:pd.DataFrame, 
              X:int, 
              FlowTS: List[float] = [0, 0.8], 
              ChangeTS: List[float] = [0, 0.8], 
              SupplyTS: List[float] = [0, 0.8], 
              FlowOnly: bool = False):
    """
    Filter OD pairs by cumulative percentage intervals of flow, flow change, and supply change.

    Returns an N-dimensional nested list of index intersections for each bin combination.
    - FlowOnly=True:  returns 2D list L[flow_bin][change_bin]
    - FlowOnly=False: returns 3D list L[flow_bin][change_bin][supply_bin]
    """
    dt = df.copy()

    # High flow
    TravelFlow = ConcatMultiDays(dt, 'TravelFlow', X+historical_windows, X+historical_windows+future_windows)
    dt['TravelFlowSum'] = TravelFlow.sum(axis=1)
    s_flow = dt['TravelFlowSum'].sort_values(ascending=False,kind='mergesort')
    flow_partitions = _get_partitioned_indices(s_flow, FlowTS)
    
    # High flow change
    TemplateFlow = ConcatMultiDays(dt, 'TemplateFlow', X+historical_windows, X+historical_windows+future_windows)
    dt['FlowChangeSum'] = np.abs(TemplateFlow - TravelFlow).sum(axis=1)
    s_change = dt['FlowChangeSum'].sort_values(ascending=False, kind='mergesort')
    change_partitions = _get_partitioned_indices(s_change, ChangeTS)
    
    if FlowOnly:
        output = [
                    [np.intersect1d(idx_flow, idx_change).tolist()
                        for idx_change in change_partitions
                        ] for idx_flow in flow_partitions
                    ]
    else:
        # High supply change
        SupplyLoss = ConcatMultiDays(dt, 'SupplyLoss', X+historical_windows, X+historical_windows+future_windows)
        dt['SupplyChangeSum'] = SupplyLoss.sum(axis=1)
        s_supply = dt['SupplyChangeSum'].sort_values(ascending=False, kind='mergesort')
        supply_partitions = _get_partitioned_indices(s_supply, SupplyTS)
        
        output = [
                    [
                        [reduce(np.intersect1d, (idx_flow, idx_change, idx_supply)).tolist()
                             for idx_supply in supply_partitions
                             ] for idx_change in change_partitions
                        ] for idx_flow in flow_partitions
                    ]
    
    return output


def sample_neighbors(df, k=5, random_state=None):
    return (
        df.groupby("TargetOD", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), k), random_state=random_state))
          .reset_index(drop=True)
    )

def prepare_tftgat_inputs(od_batch_df: pd.DataFrame, X: int ,BATCH_SIZE: int, feature_map: dict, DEVICE, is_normal:bool, is_spatial:bool, target_N: int):
    """
    Core helper: prepare TFT model input tensors for a given OD batch and window X.

    Args:
        od_batch_df (pd.DataFrame): DataFrame with all features for one OD batch.
        X (int): Current known future window size.

    Returns:
        dict: Dictionary of input tensors matching the TFT forward parameters.
    """
    # --- Static Features ---
    static_numeric = od_batch_df[feature_map['static_feats_numeric']].values
    static_categorical = od_batch_df[feature_map['static_feats_categorical']].values
    
    # --- Historical Features ---
    hist_numeric_list = [
                                ConcatMultiDays(od_batch_df, col, X, X + historical_windows)
                                for col in feature_map['historical_ts_numeric']
                            ]
    
    hist_categorical_list = [ 
                                    # If not Hour, concatenate across days normally
                                    ConcatMultiDays(od_batch_df, col, X, X + historical_windows)
                                    if col != 'Hour'
                                    # Otherwise, concatenate within the same column
                                    else np.tile(np.stack(od_batch_df[col].values), (1, 3))[:, X : X + historical_windows]
                                    for col in feature_map['historical_ts_categorical']
                                ]
 
    # --- Future Features ---
    future_numeric_list = [
                                ConcatMultiDays(od_batch_df, 'TravelTime', X + historical_windows, X + historical_windows + future_windows) + ConcatMultiDays(od_batch_df, 'SupplyLoss', X + historical_windows, X + historical_windows + future_windows)
                                if col == 'TravelTime' and is_normal # In normal mode, set travel time to ideal
                                else np.zeros((BATCH_SIZE, future_windows))
                                if col == 'SupplyLoss' and is_normal # In normal mode, set SupplyLoss to 0
                                else ConcatMultiDays(od_batch_df, col, X + historical_windows, X + historical_windows + future_windows)
                                for col in feature_map['future_ts_numeric']
                            ]
    
    future_categorical_list = [
                                    ConcatMultiDays(od_batch_df, col, X + historical_windows, X + historical_windows + future_windows)
                                    if col != 'Hour'
                                    else np.tile(np.stack(od_batch_df[col].values), (1, 3))[:, X + historical_windows  : X + historical_windows + future_windows]
                                    for col in feature_map['future_ts_categorical']
                                ]
   
    # --- Target ---
    past_target = ConcatMultiDays(od_batch_df, 'TravelFlow', X, X + historical_windows)
    past_target = torch.tensor(past_target, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    future_target = ConcatMultiDays(od_batch_df, 'TravelFlow', X + historical_windows, X + historical_windows + future_windows)
    future_target = torch.tensor(future_target, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    
    # --- Target Auxiliary ---
    batch_indices = od_batch_df.index.tolist()
    
    ## Extract group IDs
    origin_ids = [int(s.split('@')[0]) for s in batch_indices]
    origin_tensor = torch.tensor(origin_ids, device = DEVICE)
    
    ## Get unique stations and inverse indices
    unique_origins, inverse_indices = torch.unique(origin_tensor, return_inverse=True)
    num_origins = len(unique_origins)

    ## Prepare scatter_add indices (key step)
    index_phi_past = inverse_indices.view(-1, 1, 1).expand(-1, past_target.shape[1], past_target.shape[2])
    index_phi_future = inverse_indices.view(-1, 1, 1).expand(-1, future_target.shape[1], future_target.shape[2])
    
    denominator_past = torch.zeros(num_origins, past_target.shape[1], past_target.shape[2], device=DEVICE)
    denominator_future = torch.zeros(num_origins, future_target.shape[1], future_target.shape[2], device=DEVICE)
 
    ### Create a mask to mark neighbors
    is_neighbor_mask = torch.ones_like(past_target)
    if is_spatial: # Neighbors exist only in spatial mode
        is_neighbor_mask[target_N:] = 0 # Zero out later neighbors
    
    ### Keep only target flows for STA label aggregation
    past_target_for_agg = past_target * is_neighbor_mask
    future_target_for_agg = future_target * is_neighbor_mask

    ## Broadcast backfill
    denominator_past.scatter_add_(0, index_phi_past, past_target_for_agg)
    labels_phi_past = denominator_past[inverse_indices]
    
    denominator_future.scatter_add_(0, index_phi_future, future_target_for_agg)
    labels_phi_future = denominator_future[inverse_indices]
    
    # --- Auxiliary Target (Unknown Historical) ---
    past_target_OFlow = ConcatMultiDays(od_batch_df, 'OFlow', X, X + historical_windows)
    # past_target_DFlow = ConcatMultiDays(od_batch_df, 'DFlow', X, X + historical_windows)
    
    # --- Auxiliary Target ---
    future_target_OFlow = ConcatMultiDays(od_batch_df, 'OFlow', X + historical_windows, X + historical_windows + future_windows)
    # future_target_DFlow = ConcatMultiDays(od_batch_df, 'DFlow', X + historical_windows, X + historical_windows + future_windows)   
    
    # --- MaxInflow ---
    maxinflow = od_batch_df[['MaxInflow']].values
    
    # --- Congestion ---
    past_congestion = ConcatMultiDays(od_batch_df, 'Congestion', X, X + historical_windows)
    past_congestion = torch.tensor(past_congestion, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    
    future_congestion = ConcatMultiDays(od_batch_df, 'Congestion', X + historical_windows, X + historical_windows + future_windows)
    future_congestion= torch.tensor(future_congestion, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    
    # --- Convert to Tensors ---
    input_dict = {
        'static_feats_numeric': torch.tensor(static_numeric, dtype=torch.float32).to(DEVICE),
        'static_feats_categorical': torch.tensor(static_categorical, dtype=torch.long).to(DEVICE),
        'historical_ts_numeric': torch.tensor(np.stack(hist_numeric_list, axis=-1), dtype=torch.float32).to(DEVICE),
        'historical_ts_categorical': torch.tensor(np.stack(hist_categorical_list, axis=-1), dtype=torch.long).to(DEVICE),
        'future_ts_numeric': torch.tensor(np.stack(future_numeric_list, axis=-1), dtype=torch.float32).to(DEVICE),
        'future_ts_categorical': torch.tensor(np.stack(future_categorical_list, axis=-1), dtype=torch.long).to(DEVICE),
        'historical_target': past_target,
        'future_target': future_target,
        'historical_target_auxiliary':labels_phi_past, # Sum of same-station OD flows in batch
        'future_target_auxiliary':labels_phi_future, # Sum of same-station OD flows in batch
        'historical_target_OFlow':torch.tensor(past_target_OFlow, dtype=torch.float32).unsqueeze(-1).to(DEVICE), # Actual inbound flow
        'future_target_OFlow':torch.tensor(future_target_OFlow, dtype=torch.float32).unsqueeze(-1).to(DEVICE), # Actual inbound flow
        'maxinflow': torch.tensor(maxinflow, dtype=torch.float32).to(DEVICE), # Max inbound flow
        'historical_congestion':past_congestion, # Past congestion info
        'future_congestion':future_congestion # Future congestion info
    }
    
    # --- detach neighbor ---
    if is_spatial:
        for key, value in input_dict.items():
            input_dict[key] = torch.cat([
                                            value[:target_N], 
                                            value[target_N:].detach()
                                        ], dim=0)
    
    return input_dict

#%% Add perturbations
def perturb_A_keep_sum(
    X,
    idx_A,
    idx_B,
    perturb_level='sample',     # 'none' | 'sample' | 'time' | 'sample_time'
    ratio_range=(0.05, 0.5),
    fixed_ratio=None, 
    eps=1e-6
):
    """
    Apply proportional perturbation to A while keeping A + B unchanged.
    X: Tensor [B, T, N]
    """
    if perturb_level == 'none':
        return X

    X_new = X.clone()
    A = X_new[:, :, idx_A]
    B = X_new[:, :, idx_B]

    Bsz, T, _ = A.shape
    device, dtype = X.device, X.dtype

    # ---------- Generate perturbation ratio r ----------
    if fixed_ratio is not None:
        r = torch.full((1, 1, 1), fixed_ratio, device=device, dtype=dtype)

    else:
        low, high = ratio_range

        if perturb_level == 'sample':
            r = torch.empty((Bsz, 1, 1), device=device, dtype=dtype)

        elif perturb_level == 'time':
            r = torch.empty((1, T, 1), device=device, dtype=dtype)

        elif perturb_level == 'sample_time':
            r = torch.empty((Bsz, T, 1), device=device, dtype=dtype)

        else:
            raise ValueError(f"Unknown perturb_level: {perturb_level}")

        r.uniform_(low, high)

    # ---------- Perturb ----------
    delta = A * r
    A_new = A + delta
    B_new = B - delta

    # ---------- Numerical safety ----------
    B_new = torch.clamp(B_new, min=eps)
    A_new = (A + B) - B_new

    # ---------- Write back ----------
    X_new[:, :, idx_A] = A_new
    X_new[:, :, idx_B] = B_new

    return X_new


def _build_level_tensor(shape_bt, perturb_level, device, dtype, fill_value=1.0):
    """Create a broadcastable [B, T, 1] tensor pattern according to perturb level."""
    bsz, tsz = shape_bt
    if perturb_level == 'sample':
        base = torch.full((bsz, 1, 1), fill_value, device=device, dtype=dtype)
    elif perturb_level == 'time':
        base = torch.full((1, tsz, 1), fill_value, device=device, dtype=dtype)
    elif perturb_level == 'sample_time':
        base = torch.full((bsz, tsz, 1), fill_value, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown perturb_level: {perturb_level}")
    return base.expand(bsz, tsz, 1)


def _shift_1d_with_edge_pad(x_1d: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift a 1D tensor with edge-value padding."""
    if shift == 0:
        return x_1d

    t = x_1d.shape[0]
    if shift > 0:
        k = min(shift, t)
        return torch.cat([x_1d[:1].expand(k), x_1d[:-k]], dim=0)

    k = min(-shift, t)
    return torch.cat([x_1d[k:], x_1d[-1:].expand(k)], dim=0)


def perturb_ordered_pair_weather(
    X,
    idx_A,
    idx_B,
    max_A,
    max_B,
    perturb_level='sample',      # 'none' | 'sample' | 'time' | 'sample_time'
    mutation_prob=0.0, 
    shift=0, 
):
    """
    Apply consistent perturbations to two ordered categorical weather variables:
    1) Intensity mutation: with probability p, shift level +1 / -1 (clipped to bounds).
    2) Time shift: shift along time axis, pad gaps with edge values.
    """
    if perturb_level == 'none':
        return X

    X_new = X.clone()
    A = X_new[:, :, idx_A].to(torch.float32)
    B = X_new[:, :, idx_B].to(torch.float32)

    bsz, tsz = A.shape
    device = X.device

    # --- 1) Intensity mutation ---
    if mutation_prob > 0:
        prob_tensor = _build_level_tensor((bsz, tsz), perturb_level, device, torch.float32, fill_value=mutation_prob)
        trigger = (torch.rand((bsz, tsz, 1), device=device) < prob_tensor)

        sign_base = _build_level_tensor((bsz, tsz), perturb_level, device, torch.float32, fill_value=0.0)
        sign_base = torch.where(torch.rand_like(sign_base) < 0.5, -torch.ones_like(sign_base), torch.ones_like(sign_base))
        delta = (trigger.to(torch.float32) * sign_base).squeeze(-1)

        A = torch.clamp(A + delta, min=0.0, max=float(max_A))
        B = torch.clamp(B + delta, min=0.0, max=float(max_B))

    # --- 2) Time shift ---
    shift_int = int(np.clip(int(shift), -5, 5))
    if shift_int != 0:
        if perturb_level == 'sample_time':
            # sample_time has no natural phase shift on time axis; fall back to per-sample shift
            effective_level = 'sample'
        else:
            effective_level = perturb_level

        if effective_level == 'sample':
            for b in range(bsz):
                A[b] = _shift_1d_with_edge_pad(A[b], shift_int)
                B[b] = _shift_1d_with_edge_pad(B[b], shift_int)
        elif effective_level == 'time':
            # time mode shares one shift across the batch
            for b in range(bsz):
                A[b] = _shift_1d_with_edge_pad(A[b], shift_int)
                B[b] = _shift_1d_with_edge_pad(B[b], shift_int)
        else:
            raise ValueError(f"Unknown perturb_level: {perturb_level}")

    X_new[:, :, idx_A] = A.to(X_new.dtype)
    X_new[:, :, idx_B] = B.to(X_new.dtype)
    return X_new


