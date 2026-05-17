# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:45:55 2025

@author: 78632
"""
import os
from typing import Dict,List,Tuple, Optional, Union
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.init as init
import tft_loss
import torch.nn.functional as F
from torch.utils.data import Dataset 
from config import historical_windows, future_windows, config_known_stats
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

#%% Data preparation
def recycle(iterable):
    while True:
        for x in iterable:
            yield x

#%% Model initialization
class QueueAggregator(object):
    def __init__(self, max_size):
        self._queued_list = []
        self.max_size = max_size

    def append(self, elem):
        self._queued_list.append(elem)
        if len(self._queued_list) > self.max_size:
            self._queued_list.pop(0)

    def get(self):
        return self._queued_list    

def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[:n // 3].fill_(-1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)    

#%% Model training
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)    

#%% Loss computation
def sliding_average_max_with_boundary(x, window_size):
    """
    Apply a sliding window: compute mean and max in each window, then average them.
    At boundaries, only the available elements are used.
    """
    B, T, C = x.shape
    x = x.permute(0, 2, 1)  # (B, C, T)
    
    # Create a ones tensor to compute per-position weights.
    ones = torch.ones_like(x)
    
    # Use average pooling for mean values.
    avg_pool = nn.AvgPool1d(window_size, stride=1, padding=window_size//2)
    # Use max pooling for max values.
    max_pool = nn.MaxPool1d(window_size, stride=1, padding=window_size//2)
    
    # Apply average pooling to values and ones to get sums and counts.
    summed = avg_pool(x * window_size)  # Multiply because avg_pool divides by kernel_size.
    counts = avg_pool(ones * window_size)
    
    # Compute true mean using counts as denominator.
    avg_result = summed / counts.clamp(min=1)  # Avoid division by zero.
    
    # Compute max values.
    max_result = max_pool(x)
    
    # Average the mean and max.
    result = (avg_result + max_result) / 2
    
    return result.permute(0, 2, 1)  # Back to (B, T, C).

def process_batch(batch_original: Dict[str,torch.tensor],
                  batch_normal: Dict[str,torch.tensor],
                  label: torch.tensor,
                  label_U: torch.tensor,
                  label_phi: torch.tensor,
                  label_congestion: torch.tensor,
                  label_supply: torch.tensor,
                  quantiles_tensor: torch.tensor,
                  device:torch.device,
                  IS_CUDA: bool,
                  IS_PHYSIC: bool,
                  IS_STATION: bool,
                  IS_LABEL:bool,
                  IS_EVAL: bool,
                  batch_indices: List[str],
                  tw: int, # tw is the buffer length for the Ohm constraint
                  phy_w: float,
                  hour_ts: torch.tensor,
                  events_str: str,
                  station_weight_override: Optional[float] = None
                  ):  
    
    if IS_CUDA:
        for k in list(batch_original.keys()):
            batch_original[k] = batch_original[k].to(device)
            batch_normal[k] = batch_normal[k].to(device)
    
    predicted_I = batch_original['predicted_quantiles_IU']
    predicted_phi = batch_original['predicted_quantiles_phi']
    predicted_R = batch_original['predicted_R']
    predicted_U = batch_normal['predicted_quantiles_IU']
    predicted_R_normal = batch_normal['predicted_R']
    predicted_congestion = batch_original['predicted_congestion']
    predicted_supply = batch_original['predicted_S']
    
    labels_I = label.squeeze(-1)
    label_U = label_U.squeeze(-1)
    labels_phi = label_phi.squeeze(-1)
    labels_ohm =  labels_I if IS_LABEL else predicted_I[...,1]
    label_congestion = F.relu(label_congestion)
    
    ###### Count OD pairs ######
    ## 1. Extract group IDs (origin stations)
    origin_ids = [int(s.split('@')[0]) for s in batch_indices]
    origin_tensor = torch.tensor(origin_ids, device=device)

    ## 2. Get unique stations and inverse indices
    unique_origins, inverse_indices = torch.unique(origin_tensor, return_inverse=True)

    ## 3. Count OD per station
    origin_counts = torch.bincount(inverse_indices)

    # ## 4. Assign each OD its origin count
    od_counts_per_origin = origin_counts[inverse_indices]
        
    ###### Loss Data #####
    loss_data, q_risk_data, loss_data_array = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_I,
                                                                                     targets=labels_I,
                                                                                     desired_quantiles=quantiles_tensor)
    loss_data = (labels_I.sum(dim=1) * loss_data_array).sum() / (labels_I.sum() + 1e-6)
    
    ###### Loss Data OFlow #####
    loss_data_OFlow, q_risk_data_OFlow, loss_data_OFlow_array = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_phi,
                                                                                                       targets=labels_phi,
                                                                                                       desired_quantiles=quantiles_tensor)
    loss_data_OFlow = (loss_data_OFlow_array / od_counts_per_origin).mean()

    ###### MPG Loss (just for PAG-STAN) #####
    loss_mpg = torch.tensor(0.0, device=device)
    use_mpg = batch_original.get('use_mpg_loss', torch.tensor(0.0, device=device))
    if torch.is_tensor(use_mpg) and float(use_mpg.detach().mean().item()) > 0.5:
        mpg_mask = batch_original.get('mpg_mask', None)

        if mpg_mask is None:
            mpg_mask_2d = torch.ones_like(labels_I[..., 0], device=device)
        else:
            if mpg_mask.dim() == 3:
                mpg_mask_2d = mpg_mask[..., 0]
            elif mpg_mask.dim() == 2:
                mpg_mask_2d = mpg_mask
            else:
                mpg_mask_2d = torch.ones_like(labels_I[..., 0], device=device)
            mpg_mask_2d = mpg_mask_2d.to(device)

        # Use median quantile channel by default for MPG scalar demand.
        if predicted_I.size(-1) > 1:
            pred_od_scalar = predicted_I[..., 1]
        else:
            pred_od_scalar = predicted_I.squeeze(-1)
        label_od_scalar = labels_I if labels_I.dim() == 2 else labels_I.squeeze(-1)

        valid_count = mpg_mask_2d.sum().clamp_min(1.0)
        loss_mpg_od = (((pred_od_scalar - label_od_scalar) ** 2) * mpg_mask_2d).sum() / valid_count

        # Physics quantity relation: sum_j x(i,j) ~= inbound(i)
        if predicted_phi.size(-1) > 1:
            pred_phi_scalar = predicted_phi[..., 1]
        else:
            pred_phi_scalar = predicted_phi.squeeze(-1)

        label_phi_scalar = labels_phi if labels_phi.dim() == 2 else labels_phi.squeeze(-1)

        num_origins = origin_counts.shape[0]
        t_steps = pred_phi_scalar.shape[1]

        pred_sum_origin = torch.zeros(num_origins, t_steps, device=device, dtype=pred_phi_scalar.dtype)
        pred_sum_origin.index_add_(0, inverse_indices, pred_phi_scalar)

        inbound_sum_origin = torch.zeros(num_origins, t_steps, device=device, dtype=label_phi_scalar.dtype)
        inbound_sum_origin.index_add_(0, inverse_indices, label_phi_scalar)

        origin_sample_counts = torch.zeros(num_origins, 1, device=device, dtype=label_phi_scalar.dtype)
        origin_sample_counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=label_phi_scalar.dtype).unsqueeze(-1))
        inbound_origin = inbound_sum_origin / origin_sample_counts.clamp_min(1.0)

        origin_mask_sum = torch.zeros(num_origins, t_steps, device=device, dtype=mpg_mask_2d.dtype)
        origin_mask_sum.index_add_(0, inverse_indices, mpg_mask_2d)
        origin_valid = (origin_mask_sum > 0).float()

        loss_mpg_in = (((pred_sum_origin - inbound_origin) ** 2) * origin_valid).sum() / origin_valid.sum().clamp_min(1.0)

        mpg_w_od = batch_original.get('mpg_w_od', torch.tensor(1.0, device=device))
        mpg_w_in = batch_original.get('mpg_w_in', torch.tensor(1.0, device=device))
        mpg_w_od = mpg_w_od.mean() if torch.is_tensor(mpg_w_od) else torch.tensor(float(mpg_w_od), device=device)
        mpg_w_in = mpg_w_in.mean() if torch.is_tensor(mpg_w_in) else torch.tensor(float(mpg_w_in), device=device)

        loss_mpg = mpg_w_od * loss_mpg_od + mpg_w_in * loss_mpg_in
        loss_mpg = torch.nan_to_num(loss_mpg, nan=0.0)

    ###### Loss Ohm #####
    valid_mask = (hour_ts > 5).float() 
    
    loss_ohm, q_risk_ohm, loss_ohm_array = tft_loss.get_quantiles_loss_and_q_risk(outputs = predicted_U / sliding_average_max_with_boundary(predicted_R * valid_mask + 1-valid_mask,tw),
                                                                                  targets= labels_ohm,  # * valid_mask[...,0],
                                                                                  desired_quantiles=quantiles_tensor)
    loss_ohm = (labels_I.sum(dim=1) * loss_ohm_array).sum() / (labels_I.sum() + 1e-6)
    loss_ohm = torch.nan_to_num(loss_ohm, nan=0.0)
    
    loss_ohm_base, q_risk_ohm_base, loss_ohm_base_array = tft_loss.get_quantiles_loss_and_q_risk(outputs = predicted_R_normal,
                                                                                                 targets= torch.tensor(1, dtype=torch.float32).to(device), #(predicted_I * (1-valid_mask))[...,1], 
                                                                                                 desired_quantiles=quantiles_tensor)
    loss_ohm_base = torch.nan_to_num(loss_ohm_base, nan=0.0)

    ###### Loss U>I #####
    loss_U_gt_I_array = F.softplus(10*(predicted_I * valid_mask - predicted_U * valid_mask)/(predicted_I * valid_mask + 1e-4)).mean(dim=-1).mean(dim=-1) # + torch.sigmoid(1e5 * (labels_I - predicted_U[...,1])).mean(dim=-1) 
    loss_U_gt_I_array = torch.nan_to_num(loss_U_gt_I_array, nan=0.0)
    loss_U_gt_I = loss_U_gt_I_array.mean()
    
    ###### loss U<normal #####
    loss_U_sm_N_array = F.softplus(10*(predicted_U.mean(dim=-1).mean(dim=-1) - 2 * label_U.mean(dim=-1))/(predicted_U.mean(dim=-1).mean(dim=-1) + 1e-4))
    loss_U_sm_N_array = torch.nan_to_num(loss_U_sm_N_array, nan=0.0)
    loss_U_sm_N = loss_U_sm_N_array.mean()
    
    #### Loss congestion reality #####
    mask_high = (label_congestion > 0.1).float()
    loss_high = torch.tensor(0.0, device=device)
    if mask_high.sum() > 0:
        penalty = F.relu(1 - predicted_congestion)
        loss_high = torch.sum(mask_high * penalty) / (mask_high.sum() + 1e-6)
        loss_high = torch.nan_to_num(loss_high, nan=0.0)
    
    loss_corr = torch.tensor(0.0, device=device)
    std_obs = torch.std(label_congestion)
    std_pred = torch.std(predicted_congestion)
    if std_obs > 1e-4 and std_pred > 1e-4:
        vx = predicted_congestion - torch.mean(predicted_congestion)
        vy = label_congestion - torch.mean(label_congestion)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8) + 1e-6)
        loss_corr = 1.0 - cost
        loss_corr = torch.nan_to_num(loss_corr, nan=0.0)
    loss_cong_reality = 0.4*loss_corr + 0.6*loss_high
    
    #### Loss supply reality #####
    is_zero_supply = (label_supply < 1e-3).float()
    loss_zero = torch.sum(is_zero_supply * predicted_supply ) / (is_zero_supply.sum() + 1e-6)
    loss_zero = torch.nan_to_num(loss_zero, nan=0.0)
    
    is_full_supply = (label_supply > 1-1e-3).float()
    loss_full = torch.sum(is_full_supply * (1-predicted_supply) ) / (is_full_supply.sum() + 1e-6)
    loss_full = torch.nan_to_num(loss_full, nan=0.0)
    
    mask_nocongestion = (predicted_congestion < 0.1).float()
    mask_valid = mask_nocongestion * (label_supply > 1e-3).float()
    loss_corr1 = torch.tensor(0.0, device=device)
    loss_corr1 = torch.nan_to_num(loss_corr1, nan=0.0)
    if mask_valid.sum() > 3:
        # Extract valid points and flatten to 1D
        pred_flat = predicted_supply[mask_valid.bool()]
        label_flat = label_supply[mask_valid.bool()]
        
        # Compute Pearson correlation
        vx = pred_flat - torch.mean(pred_flat)
        vy = label_flat - torch.mean(label_flat)
        
        # Add epsilon to avoid zero std
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-8) * torch.sqrt(torch.sum(vy ** 2) + 1e-8) + 1e-6)
        
        # Target correlation -> 1, so loss = 1 - correlation
        loss_corr1 = 1.0 - cost

    loss_supp_reality = 0.4*loss_full + 0.4*loss_zero + 0.2*loss_corr1
    
    loss_reality = loss_supp_reality + loss_cong_reality + loss_mpg

    # Overall loss weights
    ODWeight = config_known_stats['TravelFlow']['max']
    if IS_PHYSIC:
        if events_str == 'E1_E2_E3_E4_E5':
            StationWeight = ODWeight / 10    # 10
        elif events_str == 'E6_E7_E8_E9_E10':
            StationWeight = ODWeight / 33  # 100
        elif events_str == 'E9_E10_E11_E12':
            StationWeight = ODWeight / 30  # 100
    else:
        if events_str == 'E1_E2_E3_E4_E5':
            StationWeight = ODWeight / 15 # 6
        elif events_str == 'E6_E7_E8_E9_E10':
            StationWeight = ODWeight / 10 # 5
        elif events_str == 'E9_E10_E11_E12':
            StationWeight = ODWeight / 10  # 20

    if IS_PHYSIC and station_weight_override is not None:
        StationWeight = station_weight_override
    if not IS_STATION:
        StationWeight = 0.0
    # StationWeight = ODWeight / 10
    OhmWeight = ODWeight
    
    # Total physics
    loss_physic = loss_reality*10 + loss_U_gt_I*10  + loss_U_sm_N*10 # + loss_diversity*10
    
    if not IS_PHYSIC and not IS_EVAL:  # Train: no-physics mode, data accuracy only
        if IS_STATION:
            if phy_w == 0:
                q_loss = loss_data * ODWeight
            else:
                q_loss = loss_data * ODWeight + loss_data_OFlow * StationWeight * phy_w
        else:
            q_loss = loss_data * ODWeight
        q_risk = q_risk_data
        return q_loss, q_risk, (loss_data * ODWeight, loss_data_OFlow * StationWeight, loss_ohm * OhmWeight, loss_ohm_base * OhmWeight, loss_U_gt_I*10, loss_U_sm_N*10, loss_reality*10 )
    
    elif IS_PHYSIC and not IS_EVAL: # Train: physics-rich mode with U > I and Ohm loss
        if IS_STATION:
            if phy_w == 0:
                q_loss = loss_data * ODWeight + loss_ohm * OhmWeight
            else:
                q_loss = loss_data * ODWeight + loss_ohm * OhmWeight + (loss_data_OFlow * StationWeight + loss_physic) * phy_w
        else:
            if phy_w == 0:
                q_loss = loss_data * ODWeight + loss_ohm * OhmWeight
            else:
                q_loss = loss_data * ODWeight + loss_ohm * OhmWeight + loss_physic * phy_w
        q_risk = q_risk_data
        return q_loss, q_risk, (loss_data * ODWeight, loss_data_OFlow * StationWeight, loss_ohm * OhmWeight, loss_ohm_base * OhmWeight, loss_U_gt_I*10, loss_U_sm_N*10, loss_reality*10 )
    
    elif IS_EVAL:  # Evaluation
        q_loss_array = loss_data_array * ODWeight + loss_data_OFlow_array * StationWeight + loss_ohm_array * OhmWeight + loss_U_gt_I_array*10 + loss_U_sm_N_array*10
        q_risk = q_risk_data
        
        # Eval metrics
        target = labels_I.cpu().detach().numpy()
        pred_data = predicted_I.cpu().detach().numpy()
        valid_mask = valid_mask.cpu().detach().numpy()
        pred_combine = (predicted_I.cpu().detach().numpy() + (predicted_U / predicted_R).cpu().detach().numpy())/2
        
        eval_mape = mape(target, (pred_data*valid_mask)[...,1],axis=1)
        eval_smape = smape(target, (pred_data*valid_mask)[...,1],axis=1)
        eval_mae = mae(target, (pred_data*valid_mask)[...,1],axis=1)*config_known_stats['TravelFlow']['max']
        eval_mse = mse(target, (pred_data*valid_mask)[...,1],axis=1)*config_known_stats['TravelFlow']['max']**2
        eval_rmse = rmse(target, (pred_data*valid_mask)[...,1],axis=1)*config_known_stats['TravelFlow']['max']
        eval_picp = picp(target, pred_data,axis=1)
        eval_mpiw = mpiw(target, pred_data,axis=1)*config_known_stats['TravelFlow']['max']
        
        eval_mape_physic = mape(target, (pred_combine*valid_mask)[...,1],axis=1)
        eval_smape_physic = smape(target, (pred_combine*valid_mask)[...,1],axis=1)
        eval_mae_physic = mae(target, (pred_combine*valid_mask)[...,1],axis=1)*config_known_stats['TravelFlow']['max']
        eval_mse_physic = mse(target, (pred_combine*valid_mask)[...,1],axis=1)*config_known_stats['TravelFlow']['max']**2
        eval_rmse_physic = rmse(target, (pred_combine*valid_mask)[...,1],axis=1)*config_known_stats['TravelFlow']['max']
        eval_picp_physic = picp(target, pred_combine,axis=1)
        eval_mpiw_physic = mpiw(target, pred_combine,axis=1)*config_known_stats['TravelFlow']['max']
        
        train_info = (loss_data_array * ODWeight, loss_data_OFlow_array * StationWeight, loss_ohm_array * OhmWeight, loss_ohm_base_array * OhmWeight, loss_U_gt_I_array * 10, loss_U_sm_N*10, loss_reality*10 )
        eval_info = (eval_mape,eval_smape,eval_mae,eval_mse,eval_rmse,eval_picp,eval_mpiw)
        eval_info_physic = (eval_mape_physic,eval_smape_physic,eval_mae_physic,eval_mse_physic,eval_rmse_physic,eval_picp_physic,eval_mpiw_physic)
        
        return q_loss_array, q_risk, train_info, eval_info, eval_info_physic
    
def scale_back(scaler_obj,signal):
    inv_trans = scaler_obj.inverse_transform(copy.deepcopy(signal))
    return inv_trans

#%% Metric computation
# 1. MAPE: Mean Absolute Percentage Error
def mape(y_true, y_pred,axis=None):
    if axis is not None:
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-3)),axis=axis) * 100
    else:
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-3))) * 100

# 2. SMAPE: Symmetric Mean Absolute Percentage Error
def smape(y_true, y_pred,axis=None):
    if axis is not None:
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-3),axis=axis) * 100
    else:
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-3)) * 100

# 3. MAE: Mean Absolute Error
def mae(y_true, y_pred,axis=None):
    if axis is not None:
        return np.mean(np.abs(y_true - y_pred),axis=axis)
    else:
        return np.mean(np.abs(y_true - y_pred))

# 4. MSE: Mean Squared Error
def mse(y_true, y_pred,axis=None):
    if axis is not None:
        return np.mean((y_true - y_pred) ** 2,axis=axis)
    else:
        return np.mean((y_true - y_pred) ** 2)

# 5. RMSE: Root Mean Squared Error
def rmse(y_true, y_pred,axis=None):
    if axis is not None:
        return np.sqrt(mse(y_true, y_pred,axis=axis))
    else:
        return np.sqrt(mse(y_true, y_pred))

# 6. PICP: Prediction Interval Coverage Probability
def picp(y_true, y_pred, axis=None):
    lower = y_pred[..., 0]-1e-4
    upper = y_pred[..., 2]+1e-4
    # y_true = np.squeeze(y_true, axis=-1)
    
    inside = (y_true >= lower) & (y_true <= upper)
    if axis is not None:
        return np.mean(inside, axis=axis) * 100

    else:
        return np.mean(inside) * 100

# 7. MPIW: Mean Prediction Interval Width
def mpiw(y_true, y_pred, axis=None):
    lower = y_pred[..., 0]
    upper = y_pred[..., 2]
    width = upper - lower

    if axis is not None:
        return np.mean(width, axis=axis)
    else:
        return np.mean(width)


#%% Visualization
def plotsample(daywindow_data,batch_indices,input_tensors_original,input_tensors_normal,od_batch_info_original,od_batch_info_normal,chosen_idx,window,date_str,day_shape,feature_map,return_fig=False):
    hist_len = historical_windows
    fut_len = future_windows
    flowscale = config_known_stats['TravelFlow']['max']
    
    # --- 1. Data preparation ---
    # Ground truth
    singlesampledt = daywindow_data.loc[batch_indices[chosen_idx]]
    FlowSeries = np.concatenate((singlesampledt['TravelFlow@D-1'] if 'TravelFlow@D-1' in singlesampledt  else np.array([]), 
                                 singlesampledt['TravelFlow@D'], 
                                 singlesampledt['TravelFlow@D+1'] if 'TravelFlow@D+1' in singlesampledt  else np.array([])
                                 ))*flowscale
    hist_gt = FlowSeries[window:window+hist_len]
    fut_gt = FlowSeries[window+hist_len:window+hist_len+fut_len]
    
    # Model inputs
    input_hist = input_tensors_original['historical_target'][chosen_idx,:,0].cpu().detach().numpy()*flowscale
    input_fut  = input_tensors_original['future_target'][chosen_idx,:,0].cpu().detach().numpy()*flowscale
    
    # Model outputs (12 future steps x 3 quantiles)
    pred_act = od_batch_info_original['predicted_quantiles_IU'][chosen_idx,:,:].cpu().detach().numpy()*flowscale
    pred_ideal = od_batch_info_normal['predicted_quantiles_IU'][chosen_idx,:,:].cpu().detach().numpy()*flowscale
    
    # --- Other features (weather, template, etc.) ---
    t_actual_future = input_tensors_original['future_ts_numeric'][chosen_idx,:,feature_map['future_ts_numeric'].index('TravelTime')].cpu().detach().numpy()
    t_ideal_future = input_tensors_normal['future_ts_numeric'][chosen_idx,:,feature_map['future_ts_numeric'].index('TravelTime')].cpu().detach().numpy()
    t_actual_history = input_tensors_original['historical_ts_numeric'][chosen_idx,:,feature_map['historical_ts_numeric'].index('TravelTime')].cpu().detach().numpy()
    t_ideal_history = input_tensors_normal['historical_ts_numeric'][chosen_idx,:,feature_map['historical_ts_numeric'].index('TravelTime')].cpu().detach().numpy()
    
    # Template flow
    tf_future = input_tensors_original['future_ts_numeric'][chosen_idx,:,feature_map['future_ts_numeric'].index('TemplateFlow')].cpu().detach().numpy()*flowscale
    tf_history = input_tensors_original['historical_ts_numeric'][chosen_idx,:,feature_map['historical_ts_numeric'].index('TemplateFlow')].cpu().detach().numpy()*flowscale
    
    # Future wind
    wind_future = input_tensors_original['future_ts_categorical'][chosen_idx,:,feature_map['future_ts_categorical'].index('WindStatus')].cpu().detach().numpy()
    # Future rain
    rain_future = input_tensors_original['future_ts_categorical'][chosen_idx,:,feature_map['future_ts_categorical'].index('RainStatus')].cpu().detach().numpy()
    # Historical wind
    wind_history = input_tensors_original['historical_ts_categorical'][chosen_idx,:,feature_map['future_ts_categorical'].index('WindStatus')].cpu().detach().numpy()
    # Historical rain
    rain_history = input_tensors_original['historical_ts_categorical'][chosen_idx,:,feature_map['future_ts_categorical'].index('RainStatus')].cpu().detach().numpy()
    # Historical typhoon alert
    typhoonalert_history = input_tensors_original['historical_ts_categorical'][chosen_idx,:,feature_map['historical_ts_categorical'].index('TyphoonAlert')].cpu().detach().numpy()
    # Historical rainstorm alert
    rainstormalert_history = input_tensors_original['historical_ts_categorical'][chosen_idx,:,feature_map['historical_ts_categorical'].index('RainstormAlert')].cpu().detach().numpy()
    # Future typhoon alert
    typhoonalert_future = input_tensors_original['future_ts_categorical'][chosen_idx,:,feature_map['future_ts_categorical'].index('TyphoonAlert')].cpu().detach().numpy()
    # Future rainstorm alert
    rainstormalert_future = input_tensors_original['future_ts_categorical'][chosen_idx,:,feature_map['future_ts_categorical'].index('RainstormAlert')].cpu().detach().numpy()  
    
    # Inbound flow
    OFlow_history_batch = input_tensors_original['historical_target_auxiliary'][chosen_idx,:,0].cpu().detach().numpy() * flowscale
    OFlow_future_batch = input_tensors_original['future_target_auxiliary'][chosen_idx,:,0].cpu().detach().numpy() * flowscale
    OFlow_history_real = input_tensors_original['historical_target_OFlow'][chosen_idx,:,0].cpu().detach().numpy() * flowscale
    OFlow_future_real = input_tensors_original['future_target_OFlow'][chosen_idx,:,0].cpu().detach().numpy() * flowscale
    OFlow_pred = od_batch_info_original['predicted_quantiles_phi'][chosen_idx,:,:].cpu().detach().numpy() * flowscale
    OFlow_max = input_tensors_original['maxinflow'][chosen_idx].cpu().detach().numpy() * flowscale
    OFlow_max = np.full(fut_len, OFlow_max)
    
    # Time axis
    time_hist = np.arange(-hist_len, 0)
    time_fut = np.arange(0, fut_len)
    
    # --- Extract physics parameters (new) ---
    # 1. Final impedance R (existing)
    pred_R_act = od_batch_info_original['predicted_R']
    
    hour_ts = input_tensors_original['future_ts_categorical'][...,[feature_map['future_ts_categorical'].index('Hour')]][[0]]
    valid_mask = (hour_ts > 5).float() 
    pred_R_act_pool = sliding_average_max_with_boundary(pred_R_act * valid_mask + 1-valid_mask,3)
    
    pred_R_act = pred_R_act[chosen_idx, :, 0].cpu().detach().numpy()
    pred_R_act_pool = pred_R_act_pool[chosen_idx, :, 0].cpu().detach().numpy()
    
    pred_R_ideal = od_batch_info_normal['predicted_R'][chosen_idx, :, 0].cpu().detach().numpy()
    
    # 2. Base impedance Rb (new) - assumed in od_batch_info_original
    if 'predicted_Rb' in od_batch_info_original:
        pred_Rb = od_batch_info_original['predicted_Rb'][chosen_idx, :, 0].cpu().detach().numpy()
    else:
        pred_Rb = np.zeros_like(pred_R_act) # Fallback

    # 3. Station capacity Eta (new) - assumed present and denormalized
    if 'predicted_Eta' in od_batch_info_original:
        pred_Eta = od_batch_info_original['predicted_Eta'][chosen_idx, :, 0].cpu().detach().numpy() * flowscale
    else:
        pred_Eta = np.zeros_like(pred_act[:,1]) # Fallback

    # 4. Latent supply S (new)
    if 'predicted_S' in od_batch_info_original:
        pred_S = od_batch_info_original['predicted_S'][chosen_idx, :, 0].cpu().detach().numpy()
    else:
        pred_S = np.zeros_like(pred_act[:,1])
    
    SL_ts_index = feature_map['future_ts_numeric'].index('SupplyLoss')
    TT_ts_index = feature_map['future_ts_numeric'].index('TravelTime')
    
    label_S =  input_tensors_original['future_ts_numeric'][chosen_idx,:,TT_ts_index]/ (input_tensors_original['future_ts_numeric'][chosen_idx,:,SL_ts_index] + input_tensors_original['future_ts_numeric'][chosen_idx,:,TT_ts_index] + 1e-6)
    # label_S = 1 - input_tensors_original['future_ts_numeric'][chosen_idx,:,SL_ts_index] / (input_tensors_normal['future_ts_numeric'][chosen_idx,:,TT_ts_index] + 1e-6)
    label_S = label_S.cpu().detach().numpy()
    # 5. BPR params Alpha, Beta (new) - shape [Batch, 1] or [Batch, Time, 1]
    # Handle as time series for plotting (broadcast if scalar)

    pred_Alpha = od_batch_info_original['predicted_Alpha'][chosen_idx].cpu().detach().numpy()
    pred_Alpha = np.full(fut_len, pred_Alpha) if pred_Alpha.ndim == 0 or pred_Alpha.size == 1 else pred_Alpha.flatten()

    pred_Beta = od_batch_info_original['predicted_Beta'][chosen_idx].cpu().detach().numpy()
    pred_Beta = np.full(fut_len, pred_Beta) if pred_Beta.ndim == 0 or pred_Beta.size == 1 else pred_Beta.flatten()

        
    # 6. Congestion term
    pred_congestion = od_batch_info_original['predicted_congestion'][chosen_idx, :, 0].cpu().detach().numpy()
    label_congestion = F.relu(input_tensors_original['future_congestion'][chosen_idx,:,0]).cpu().detach().numpy() * 10
    
    # ---------------------------
    # Plotting
    # ---------------------------
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1.2, 1.5, 1.5, 1.2]})
    ax1, ax2, ax3, ax4, ax5 = axes
    
    # ===============================
    # Panel 1: weather and alerts (unchanged)
    # ===============================
    ax1.plot(time_hist, wind_history, color='darkgreen', linestyle='-', label='Wind (History)')
    ax1.plot(time_fut, wind_future, color='green', linestyle='-', label='Wind (Future)')
    ax1.plot(time_hist, rain_history, color='darkblue', linestyle='-', label='Rain (History)')
    ax1.plot(time_fut, rain_future, color='blue', linestyle='-', label='Rain (Future)')
    ax1.plot(time_hist, typhoonalert_history, color='darkgreen', linestyle='--', label='Typhoon Alert (History)')
    ax1.plot(time_fut, typhoonalert_future, color='green', linestyle='--', label='Typhoon Alert (Future)' )
    ax1.plot(time_hist, rainstormalert_history, color='darkblue', linestyle='--', label='Rainstorm Alert (History)')
    ax1.plot(time_fut, rainstormalert_future, color='blue', linestyle='--', label='Rainstorm Alert (Future)')
    
    ax1.set_ylabel("Weather & Alerts")
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    
    ylim = ax1.get_ylim()
    ax1.text(time_hist[0] + 1, ylim[1]*0.9, 'Past', fontsize=20, color='gray')
    ax1.text(time_fut[2], ylim[1]*0.9, 'Future', fontsize=20, color='gray')

    # ===============================
    # Panel 2: travel time & impedance (add Rb)
    # ===============================
    
    # Travel time
    ax2.plot(time_hist, t_actual_history, color='darkred', linestyle='--', label='T Actual (History)')
    ax2.plot(time_fut, t_actual_future, color='red', linestyle='--', label='T Actual (Future)')
    ax2.plot(time_hist, t_ideal_history, color='darkblue', linestyle='--', label='T Ideal (History)')
    ax2.plot(time_fut, t_ideal_future, color='blue', linestyle='--', label='T Ideal (Future)')
    
    ylim = ax2.get_ylim()
    ax2.text(time_hist[0] + 1, ylim[0]+(ylim[1]-ylim[0])*0.9, 'Past', fontsize=20, color='gray')
    ax2.text(time_fut[2], ylim[0]+(ylim[1]-ylim[0])*0.9, 'Future', fontsize=20, color='gray')
    
    # R values
    ax2_ = ax2.twinx()
    ax2_.plot(time_fut, pred_R_act, color='red', linestyle='-', label='R Actual')
    ax2_.plot(time_fut, pred_R_ideal, color='blue', linestyle='-', label='R Ideal')
    ax2_.plot(time_fut, pred_Rb, color='black', linestyle='-.', linewidth=2, label='Rb (Base)')
    
    ax2.set_ylabel("Travel Time & R")
    ax2.grid(alpha=0.3)
    
    # Merge legends
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax2.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    
    # ===============================
    # Panel 3: OD flow prediction (unchanged)
    # ===============================
    # Ground truth
    ax3.plot(time_hist, hist_gt, color='black', linestyle='-', label='True Flow (History)')
    ax3.plot(time_fut, fut_gt, color='black', linestyle='-', label='True Flow (Future)')
    
    # Model inputs
    ax3.plot(time_hist, input_hist, color='darkgray', linestyle='-', alpha=0.7, label='Model Input (History)')
    ax3.plot(time_fut, input_fut, color='gray', linestyle='-', alpha=0.7, label='Model Input (Future)')

    # Model prediction (actual)
    ax3.plot(time_fut, pred_act[:, 1], color='orange', label='Pred Median (Actual)', linewidth=2)
    ax3.fill_between(time_fut, pred_act[:, 0], pred_act[:, 2], color='orange', alpha=0.2)
    
    # Model prediction (ideal)
    ax3.plot(time_fut, pred_ideal[:, 1], color='green', label='Pred Median (Ideal)', linewidth=2)
    ax3.fill_between(time_fut, pred_ideal[:, 0], pred_ideal[:, 2], color='green', alpha=0.2)
    
    # Model prediction (Ohm)
    ax3.plot(time_fut, pred_ideal[:, 1]/pred_R_act_pool, color='red', label='Pred Median (Ohm)', linewidth=2)
    ax3.fill_between(time_fut, pred_ideal[:, 0]/pred_R_act_pool, pred_ideal[:, 2]/pred_R_act_pool, color='red', alpha=0.2)    
    
    # Template flow
    ax3.plot(time_hist, tf_history, color='purple', linestyle='-', label='Template Flow')
    ax3.plot(time_fut, tf_future, color='purple', linestyle='-', label='Template Flow')

    ax3.set_ylabel("Passenger Flow")
    ax3.set_xlabel("Time (hours relative to current)")
    ax3.grid(alpha=0.3)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    
    ylim = ax3.get_ylim()
    ax3.text(time_hist[0] + 1, ylim[1]*0.9, 'Past', fontsize=20, color='gray')
    ax3.text(time_fut[2], ylim[1]*0.9, 'Future', fontsize=20, color='gray')
    
    # ===============================
    # Panel 4: station inflow & capacity (add Eta)
    # ===============================
    # Ground truth
    ax4.plot(time_hist, OFlow_history_batch, color='black', linestyle='-', label='Batch OFlow (History)')
    ax4.plot(time_fut, OFlow_future_batch, color='black', linestyle='-', label='Batch OFlow (Future)')
    ax4.plot(time_hist, OFlow_history_real, color='black', linestyle='--', label='True OFlow (History)')
    ax4.plot(time_fut, OFlow_future_real, color='black', linestyle='--', label='True OFlow (Future)')
    ax4.plot(time_fut, OFlow_max, color='green', linestyle='--', linewidth=3, label='Max OFlow')

    # Model prediction (OFlow)
    ax4.plot(time_fut, OFlow_pred[:, 1], color='blue', label='Pred Median for OFlow', linewidth=2)
    ax4.fill_between(time_fut, OFlow_pred[:, 0], OFlow_pred[:, 2], color='blue', alpha=0.2)
    
    # # Model prediction (DFlow)
    # ax4.plot(time_fut, DFlow_pred[:, 1], color='red', linestyle='--', label='Pred Median for DFlow', linewidth=2)
    # ax4.fill_between(time_fut, DFlow_pred[:, 0], DFlow_pred[:, 2], color='red', alpha=0.2)
    
    # Add capacity Eta (thick red dashed line)
    ax4.plot(time_fut, pred_Eta, color='red', linestyle='--', linewidth=2.5, label='Capacity (Eta)')

    ax4.set_ylabel("Passenger Entry/Exit Flow")
    ax4.set_xlabel("Time (hours relative to current)")
    ax4.grid(alpha=0.3)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    ylim = ax4.get_ylim()
    ax4.text(time_hist[0] + 1, ylim[1]*0.9, 'Past', fontsize=20, color='gray')
    ax4.text(time_fut[2], ylim[1]*0.9, 'Future', fontsize=20, color='gray')
    
    # ===============================
    # Panel 5: physics params S, Alpha, Beta (new)
    # ===============================
    # Left axis: latent supply S
    lns0 = ax5.plot(time_hist, pred_congestion, color='purple', linestyle='-', marker='o', markersize=4, linewidth=2, label='Predicted Congestion')
    lns00 = ax5.plot(time_hist, label_congestion, color='green', linestyle='-', marker='o', markersize=4, linewidth=2, label='Label Congestion')
  
    lns1 = ax5.plot(time_fut, pred_S, color='purple', linestyle='-', marker='o', markersize=4, linewidth=2, label='Predicted Supply')
    lns11 = ax5.plot(time_fut, label_S, color='green', linestyle='-', marker='o', markersize=4, linewidth=2, label='Label Supply')
    
    ax5.set_ylabel("Latent Supply/Congestion", color='purple')
    ax5.tick_params(axis='y', labelcolor='purple')
    # Expand S-axis range slightly to avoid clipping
    s_min, s_max = pred_S.min(), pred_S.max()
    if s_max - s_min < 1e-5:
        ax5.set_ylim(s_min - 0.1, s_max + 0.1)
    
    # Right axis: BPR params Alpha, Beta
    ax5_ = ax5.twinx()
    lns2 = ax5_.plot(time_fut, pred_Alpha, color='brown', linestyle='--', linewidth=2, label='Alpha')
    lns3 = ax5_.plot(time_fut, pred_Beta, color='orange', linestyle=':', linewidth=2, label='Beta')
    
    ax5_.set_ylabel("BPR Params (α, β)", color='brown')
    ax5_.tick_params(axis='y', labelcolor='brown')
    
    # Merge legends
    lns = lns0 + lns00 + lns1 + lns11 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax5.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)
    
    ax5.grid(alpha=0.3)
    ax5.set_xlabel("Time (hours relative to current)")
    
    # ---------------------------
    # Style
    # ---------------------------
    for ax in axes:
        ax.axvline(0, color='gray', linestyle='--', alpha=0.7)

        time_all = np.arange(window, window + hist_len + fut_len)
        if day_shape[0] != '-':
            time_all_mod = time_all - 24
        else:
            time_all_mod = time_all
        time_all_mod = ['-'+str(i%24) 
                        if i//24 == -1
                        else '+'+str(i%24) 
                        if i//24 == 1
                        else str(i%24)
                        for i in time_all_mod]

        ax1.set_xticks(ticks=np.concatenate([time_hist,time_fut]),labels=time_all_mod)
    
    plt.suptitle(f'Date: {date_str} | Event Shape: {day_shape} | OD: {batch_indices[chosen_idx]}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    del singlesampledt, FlowSeries
    del hist_gt,fut_gt,input_hist,input_fut,pred_act,pred_ideal,tf_future,tf_history
    del pred_R_act,pred_R_ideal,t_actual_future,t_ideal_future,t_actual_history,t_ideal_history
    del wind_future,rain_future,wind_history,rain_history,typhoonalert_history,rainstormalert_history,typhoonalert_future,rainstormalert_future
    del pred_Rb, pred_Eta, pred_S, pred_Alpha, pred_Beta

    if return_fig:
        # Mode 2: return figure for TensorBoard (writer.add_figure)
        # Caller must call plt.close(fig) after use to free memory
        return fig
    else:
        # Mode 1: show directly and clean up (Interactive Mode)
        plt.show()
        plt.close(fig)
        plt.close('all') # Force cleanup
        del fig,axes,ax1,ax2,ax3,ax4,ax5
        return None
    
def display_sample_wise_selection_stats(weights_arr: np.ndarray,
                                        observation_index: int,
                                        feature_names: List[str],
                                        top_n: Optional[int] = None,
                                        title: Optional[str] = '',
                                        historical: Optional[bool] = True,
                                        rank_stepwise: Optional[bool] = False,
                                        prctiles: List[float] = None,
                                        rc_ts=None,
                                        ax=None,
                                        ax1=None,
                                        fig=None):

    # a-priori assume non-temporal input channel
    num_temporal_steps = None

    # infer number of attributes according to the shape of the weights array
    weights_shape = weights_arr.shape
    num_features = weights_shape[-1]
    # infer whether the input channel is temporal or not
    is_temporal: bool = len(weights_shape) > 2

    # bound maximal number of features to display by the total amount of features available (in case provided)
    top_n = min(num_features, top_n) if top_n else num_features

    sample_weights = weights_arr[observation_index, ...]
    
    if is_temporal:
        # infer number of temporal steps
        num_temporal_steps = weights_shape[1]
        # aggregate the weights (by averaging) across all the time-steps
        sample_weights_trans = sample_weights.T
        weights_df = pd.DataFrame({'weight': sample_weights_trans.mean(axis=1)}, index=feature_names)
    else:
        # in case the input channel is not temporal, just use the weights as is
        weights_df = pd.DataFrame({'weight': sample_weights}, index=feature_names)
    
    # *** Change: create fig and ax only when ax is None ***
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    
    weights_df.sort_values('weight', ascending=False).iloc[:top_n].plot.bar(ax=ax)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(11)
        tick.label1.set_rotation(45)

    ax.grid(True)
    # ax.set_xlabel('Feature Name',fontsize=30)
    ax.set_ylabel('Selection Weight',fontsize=30)
    ax.set_title(title + (" - " if title != "" else "") + \
                "Selection Weights " + ("Aggregation " if is_temporal else "") + \
                (f"- Top {top_n}" if top_n < num_features else ""),fontsize=30)
    ax.tick_params(axis='x', labelsize=25)  # X-axis tick label font size
    ax.tick_params(axis='y', labelsize=25)  # Y-axis tick label font size
    # ax.legend(fontsize=25,loc='upper left', bbox_to_anchor=(1, 1))
    #plt.show() # Keep commented out

    if is_temporal:
        # ========================
        # Temporal Display
        # ========================
        # infer the order of the features, according to the average selection weight across time
        order = sample_weights_trans.mean(axis=1).argsort()[::-1]

        # order the weights sequences as well as their names accordingly
        ordered_weights = sample_weights_trans[order]
        ordered_names = [feature_names[i] for i in order.tolist()]

        if rank_stepwise:
            # the weights are now considered to be the ranking after ordering the features in each time-step separately
            ordered_weights = np.argsort(ordered_weights, axis=0)

        # *** Change: create fig, ax1, ax2 only when ax1 is None ***
        if ax1 is None:
            fig, ax1 = plt.subplots(figsize=(30, 20))
            ax2 = ax1.twiny()
        # *** Change: if ax1 is provided, fig must be provided to create ax2 ***
        elif fig is not None:
            ax2 = ax1.twiny()
        else:
            # If ax1 is provided but fig is not, this is an error
            raise ValueError("`fig` must be provided when `ax1` is provided for temporal stats.")

        # create a corresponding x-axis, going forward/backwards, depending on the configuration
        if historical:
            map_x = {idx: val for idx, val in enumerate(np.arange(- num_temporal_steps, 1))}
        else:
            map_x = {idx: val for idx, val in enumerate(np.arange(1, num_temporal_steps + 1))}

        def format_fn(tick_val, tick_pos):
            if int(tick_val) in map_x:
                return map_x[int(tick_val)]
            else:
                return ''

        # display the weights as images
        im = ax1.pcolor(ordered_weights, edgecolors='gray', linewidths=2)
        # feature names displayed to the left
        ax1.yaxis.set_ticks(np.arange(len(ordered_names)))
        ax1.set_yticklabels(ordered_names,fontsize=25)
        ax2.set_xticks([])
        ax2.xaxis.set_ticks_position('top')
        ax1.set_xlabel(('Historical' if historical else 'Future') + ' Time-Steps',fontsize=30)
        ax2.set_xlabel(('Historical' if historical else 'Future') + ' Time-Steps',fontsize=30)
        ax1.tick_params(axis='x', labelsize=25)  # X-axis tick label font size
        ax1.tick_params(axis='y', labelsize=25)  # Y-axis tick label font size
        ax2.tick_params(axis='x', labelsize=25)  # X-axis tick label font size
        ax2.tick_params(axis='y', labelsize=25)  # Y-axis tick label font size
        ax1.xaxis.set_major_formatter(FuncFormatter(format_fn))
        
        if fig:
            fig.colorbar(im, orientation="horizontal", pad=0.08, ax=ax2)

def display_sample_wise_attention_scores(attention_scores: np.ndarray,
                                         observation_index: int,
                                         horizons: Union[int, List[int]],
                                         unit: Optional[str] = None,
                                         ax = None):
    # if ``horizons`` is provided as int, transform into a list.
    if isinstance(horizons, int):
        horizons = [horizons]

    # take the relevant record from  the provided array, using the specified index
    sample_attn_scores = attention_scores[observation_index, ...]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 10))

    attn_shape = sample_attn_scores.shape
    
    x_axis = np.arange(attn_shape[0] - attn_shape[1], attn_shape[0])

    # for each horizon, plot the associated attention score signal for all the steps
    for step in horizons:
        ax.plot(x_axis, sample_attn_scores[step - 1], marker='o', lw=3, label=f"t+{step}")

    ax.axvline(x=-0.5, lw=2, color='r', linestyle='--')
    ax.grid(True)
    ax.legend(fontsize=20)
    ax.set_xlabel('Relative Time-Step ' + (f"[{unit}]" if unit else ""),fontsize=30)
    ax.set_ylabel('Attention Score',fontsize=30)
    ax.tick_params(axis='x', labelsize=20)  # X-axis tick label font size
    ax.tick_params(axis='y', labelsize=20)  # Y-axis tick label font size
    ax.set_title('Attention Mechanism Scores - Per Horizon',fontsize=40)
    
    return sample_attn_scores

def plotmodel(od_batch_info_original,chosen_idx,feature_map,window,day_shape,return_fig = False):

    # Create a large canvas, e.g., 35 wide and 40 tall
    fig = plt.figure(figsize=(30, 35))

    # Create a 4x2 grid
    gs_main = fig.add_gridspec(nrows=2, ncols=1, 
                               hspace=0.3,  # <--- (key) adjust spacing between row 1 and row 2 (0.5 is example)
                               height_ratios=[1, 3])
    gs_bottom = gs_main[1].subgridspec(nrows=3, ncols=2, 
                                  hspace=0.5, # <--- (key) spacing for other rows (f, b/d, c/e)
                                  wspace=0.25)  # <--- spacing between columns (b/d, c/e)
    
    ax_a = fig.add_subplot(gs_main[0])
    ax_f = fig.add_subplot(gs_bottom[0, :])
    ax_b = fig.add_subplot(gs_bottom[1, 0])
    ax_d = fig.add_subplot(gs_bottom[1, 1])
    ax_c = fig.add_subplot(gs_bottom[2, 0])
    ax_e = fig.add_subplot(gs_bottom[2, 1])
    
    hist_len = historical_windows
    fut_len = future_windows
    time_hist = np.arange(-hist_len, 0)
    time_fut = np.arange(0, fut_len)
    time_all = np.arange(window, window + hist_len + fut_len)
    if day_shape[0] != '-':
        time_all_mod = time_all - 24
    else:
        time_all_mod = time_all
    time_all_mod = ['-'+str(i%24) 
                    if i//24 == -1
                    else '+'+str(i%24) 
                    if i//24 == 1
                    else str(i%24)
                    for i in time_all_mod]
    
    # --- Global font settings ---
    # Choose an installed serif font to avoid repeated findfont warnings.
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    preferred_fonts = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    selected_font = next((name for name in preferred_fonts if name in installed_fonts), 'serif')
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

    display_sample_wise_selection_stats(
        weights_arr=od_batch_info_original['static_weights'].cpu().detach().numpy(),
        observation_index=chosen_idx,
        feature_names=feature_map['static_feats_numeric'] + feature_map['static_feats_categorical'],
        top_n=10,
        title='Static Features',
        ax=ax_a,  # Pass the corresponding ax
        fig=fig   # Pass the fig
    )
    
    # --- Panel f (row 2) ---

    display_sample_wise_attention_scores(
        attention_scores=od_batch_info_original['attention_scores'].cpu().detach().numpy(),
        observation_index=chosen_idx, # Not in the original call, but required by the function
        horizons=[1, 3, 5, 7, 9, 11],
        unit='Hours',
        ax=ax_f  # Pass the corresponding ax
    )
    
    # --- Panels b and c (row 3 col 1, row 4 col 1) ---

    display_sample_wise_selection_stats(
        weights_arr=od_batch_info_original['historical_selection_weights'].cpu().detach().numpy(),
        observation_index=chosen_idx,
        feature_names=feature_map['historical_ts_numeric'] + feature_map['historical_ts_categorical'],
        top_n=20,
        title='Historical Features',
        rank_stepwise=True,
        ax=ax_b,    # Panel b (bar chart) on ax_b
        ax1=ax_c,   # Panel c (heatmap) on ax_c
        fig=fig     # Pass fig (needed for ax2 and colorbar)
    )
    
    # --- Panels d and e (row 3 col 2, row 4 col 2) ---

    display_sample_wise_selection_stats(
        weights_arr=od_batch_info_original['future_selection_weights'].cpu().detach().numpy(),
        observation_index=chosen_idx,
        feature_names=feature_map['future_ts_numeric'] + feature_map['future_ts_categorical'],
        top_n=20,
        title='Future Features',
        historical=False,
        rank_stepwise=False,
        ax=ax_d,    # Panel d (bar chart) on ax_d
        ax1=ax_e,   # Panel e (heatmap) on ax_e
        fig=fig     # Pass fig (needed for ax2 and colorbar)
    )

    ax_f.set_xticks(ticks=np.concatenate([time_hist,time_fut]),labels=time_all_mod)
    ax_c.set_xticks(ticks=time_fut,labels=time_all_mod[:len(time_hist)])
    ax_e.set_xticks(ticks=time_fut,labels=time_all_mod[len(time_hist):])
    
    fig.tight_layout() 
    
    if return_fig:
        # Mode 2: return figure for TensorBoard (writer.add_figure)
        # Caller must call plt.close(fig) after use to free memory
        return fig
    else:
        # Mode 1: show directly and clean up (Interactive Mode)
        plt.show()
        plt.close(fig)
        plt.close('all') # Force cleanup
        del fig,gs_main,gs_bottom,ax_a,ax_f,ax_b,ax_d,ax_c,ax_e
        return None

#%% Logging
from datetime import datetime
def log_evaluation_metrics(
    log_file_path: str,
    epoch_idx: int,
    subset_name: str,
    day_info_list: list,
    metric_dataframes: dict,
    aggregated_metrics: dict,
    quantiles_tensor: torch.Tensor
):
    """
    Format evaluation results (partitioned DataFrames and aggregated metrics) into a log file.

    Args:
        log_file_path (str): Full path to the log file.
        epoch_idx (int): Current epoch index.
        subset_name (str): Dataset subset name ('train' or 'test').
        day_info_list (list): List of day_info dicts in the current evaluation batch.
        metric_dataframes (dict): Dict of partitioned DataFrames (e.g., {'df_loss': df_loss, ...}).
        aggregated_metrics (dict): Dict of aggregated scalar metrics (e.g., {'eval_loss': val, ...}).
        quantiles_tensor (torch.Tensor): Quantile tensor from model outputs.
    """
    
    # Get current time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, 'a', encoding='utf-8') as f:
        # --- 1. Epoch and subset boundary ---
        f.write("\n" + "=" * 120 + "\n")
        f.write(f"🚀 EPOCH: {epoch_idx:03d} | SUBSET: {subset_name.upper()} | TIME: {timestamp}\n")
        f.write("=" * 120 + "\n\n")

        # --- 2. Record Day Info used in this evaluation batch ---
        f.write("📅 评估中使用的 Day Batches:\n")
        
        # Convert day_info_list to DataFrame for pretty printing
        day_info_df = pd.DataFrame(day_info_list)
        if not day_info_df.empty:
            f.write(day_info_df.to_string(index=False))
        else:
            f.write("   (no valid day info)\n")
        f.write("\n" + "-" * 120 + "\n")

        # --- 3. Record aggregated metrics ---
        f.write("📈 聚合指标 (加权平均):\n")
        
        # 3.1 Loss metrics
        f.write("   - 损失 (Losses):\n")
        f.write(f"     -> Quantile Loss (q_loss):    {aggregated_metrics['eval_loss']:.6f}\n")
        f.write(f"     -> Data Loss (loss_data):     {aggregated_metrics['eval_loss_data']:.6f}\n")
        f.write(f"     -> Data Loss (loss_data_OFlow):     {aggregated_metrics['eval_loss_data_OFlow']:.6f}\n")
        f.write(f"     -> OHM Loss (loss_ohm):       {aggregated_metrics['eval_loss_ohm']:.6f}\n")
        f.write(f"     -> Upper > Lower Loss (U_gt_I): {aggregated_metrics['eval_loss_U_gt_I']:.6f}\n")

        # 3.2 Accuracy and interval metrics
        f.write("   - 精度 (Accuracy) & 区间 (Intervals):\n")
        f.write(f"     -> MAPE:  {aggregated_metrics['eval_mape']:.6f} | sMAPE: {aggregated_metrics['eval_smape']:.6f}\n")
        f.write(f"     -> MAE:   {aggregated_metrics['eval_mae']:.6f} | MSE:   {aggregated_metrics['eval_mse']:.6f} | RMSE: {aggregated_metrics['eval_rmse']:.6f}\n")
        f.write(f"     -> PICP:  {aggregated_metrics['eval_picp']:.6f} | MPIW:  {aggregated_metrics['eval_mpiw']:.6f}\n")
        
        f.write(f"     -> MAPE_PHYSIC:  {aggregated_metrics['eval_mape_physic']:.6f} | sMAPE_PHYSIC: {aggregated_metrics['eval_smape_physic']:.6f}\n")
        f.write(f"     -> MAE_PHYSIC:   {aggregated_metrics['eval_mae_physic']:.6f} | MSE_PHYSIC:   {aggregated_metrics['eval_mse_physic']:.6f} | RMSE_PHYSIC: {aggregated_metrics['eval_rmse_physic']:.6f}\n")
        f.write(f"     -> PICP_PHYSIC:  {aggregated_metrics['eval_picp_physic']:.6f} | MPIW_PHYSIC:  {aggregated_metrics['eval_mpiw_physic']:.6f}\n")

        # --- 4. Record partitioned metrics (DataFrames) ---
        f.write("📋 分区指标 (Flow Bin x Change Bin) 报告:\n\n")

        for metric_key, df in metric_dataframes.items():
            f.write(f"  {metric_key.upper()} (partitioned metrics):\n")
            
            if df.empty or df.isnull().all().all():
                 f.write("   (no data)\n\n")
                 continue
                 
            # For df_q_risk (DataFrame of lists), special handling is needed
            if metric_key == 'df_q_risk':
                # Build a table for each quantile
                for i, q in enumerate(quantiles_tensor.tolist()):
                    df_q = df.applymap(lambda x: x[i] if isinstance(x, list) and len(x) > i else None)
                    f.write(f"    - Quantile Risk (q={q:.1f}):\n")
                    f.write(df_q.to_string() + "\n\n")
            else:
                # Print all other DataFrames (scalars or NumPy arrays)
                f.write(df.to_string() + "\n\n")

        f.write("=" * 120 + "\n")

#%% Log parsing
import re
from typing import List, Dict, Any, Optional

def safe_float(value_str: str) -> Optional[float]:
    """
    Safely convert a string to float, handling 'NaN'.
    """
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return None

def parse_agg_metrics(text: str) -> Dict[str, Any]:
    """
    Parse the "aggregated metrics" section.
    """
    metrics = {}
    current_section = None
    for line in text.split('\n'):
        if line.strip().startswith('- '):
            # Detect a new metric category
            section_name_raw = line.strip().strip('- ').split('(')[0].strip()
            # Create a Python-friendly key name
            current_section = section_name_raw.replace(' ', '_').replace('&', 'and').lower()
            metrics[current_section] = {}
        elif line.strip().startswith('->'):
            # Parse key-value pairs
            if not current_section:
                continue
            pairs = line.strip('-> ').split('|')
            for pair in pairs:
                if ':' not in pair:
                    continue
                key, value = pair.split(':', 1)
                # Normalize key name
                key_clean = key.strip().replace(' ', '_').replace('(', '').replace(')', '')
                metrics[current_section][key_clean] = safe_float(value.strip())
    return metrics

def parse_partitioned_table(table_text: str) -> Dict[str, Any]:
    """
    Parse a partitioned metrics table (e.g., DF_LOSS).
    """
    lines = table_text.strip().split('\n')
    data = {'columns': [], 'index': [], 'data': []}
    if not lines:
        return data

    header_line = lines[0]
    # Split by 2+ spaces using regex to keep column names with spaces
    headers = re.split(r'\s{2,}', header_line.strip())
    data['columns'] = [h.strip() for h in headers if h.strip()]

    for line in lines[1:]:
        if not line.strip() or not line.strip().startswith('Flow Bin'):
            continue
        try:
            # Split index and data at the first ']'
            index_part, data_part = line.split(']', 1)
            data['index'].append(index_part.strip() + ']')
            
            values = [safe_float(v) for v in data_part.split()]
            
            # Fill missing values (handle trailing gaps from 'NaN')
            values.extend([None] * (len(data['columns']) - len(values)))
            data['data'].append(values)
        except ValueError:
            # Skip malformed lines
            continue
    return data

def parse_partitioned_metrics(text: str) -> Dict[str, Any]:
    """
    Parse the full partitioned-metrics report, including all DF_ tables.
    """
    metrics = {}
    # Split by the start of each DF_ table
    tables = re.split(r'\n\s*(?=DF_)', text.strip())
    
    for table_block in tables:
        if not table_block.strip():
            continue
        
        df_name_match = re.match(r'DF_([\w_]+)\s*\(分区指标\):', table_block.strip())
        if not df_name_match:
            continue
        df_name = f"DF_{df_name_match.group(1)}"
        
        if df_name == "DF_Q_RISK":
            # DF_Q_RISK contains sub-tables
            metrics[df_name] = {}
            # Split by sub-tables ('- Quantile Risk')
            sub_tables = re.split(r'\n\s*(?=-\s*Quantile Risk)', table_block)
            for sub_block in sub_tables:
                if not sub_block.strip() or not sub_block.startswith('- Quantile Risk'):
                    continue
                sub_name_match = re.match(r'-\s*(.*?):', sub_block)
                if not sub_name_match:
                    continue
                sub_name = sub_name_match.group(1).strip()
                # Extract sub-table text and parse it
                table_content = sub_block.split(':', 1)[-1]
                metrics[df_name][sub_name] = parse_partitioned_table(table_content)
        else:
            # All other DF_ tables
            table_content = table_block.split(':', 1)[-1]
            metrics[df_name] = parse_partitioned_table(table_content)
            
    return metrics

def parse_log_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Main function: parse the full log file.

    It parses evaluation_log.txt into a structured list of dicts,
    each dict representing one evaluation entry.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: file not found {file_path}")
        return [{"error": f"File not found: {file_path}"}]
    except Exception as e:
        print(f"Error: failed to read file {e}")
        return [{"error": f"Error reading file: {e}"}]

    # Use positive lookahead to split logs,
    # so '🚀 EPOCH:' stays at the start of each block.
    blocks = re.split(r'(?=🚀 EPOCH:)', content)
    all_logs = []

    for block in blocks:
        if not block.strip() or not block.startswith('🚀 EPOCH:'):
            continue
        
        log_entry = {}
        
        # --- 1. Parse entry header ---
        header_match = re.search(r'🚀 EPOCH: (\d+) \| SUBSET: (.*?) \| TIME: (.*)', block)
        if not header_match:
            continue
        log_entry['epoch'] = int(header_match.group(1))
        log_entry['subset'] = header_match.group(2).strip()
        log_entry['time'] = header_match.group(3).strip()

        # --- 2. Parse Day Batches ---
        db_match = re.search(r'📅 评估中使用的 Day Batches:\n\s*Date\s+is_extreme_day\s+Event_ID\s+DayShape\n(.*?)\n---', block, re.S)
        log_entry['day_batches'] = []
        if db_match:
            data_lines = db_match.group(1).strip().split('\n')
            for line in data_lines:
                parts = line.split()
                if len(parts) == 4:
                    log_entry['day_batches'].append({
                        "Date": parts[0],
                        "is_extreme_day": int(parts[1]),
                        "Event_ID": parts[2],
                        "DayShape": parts[3]
                    })

        # --- 3. Parse aggregated metrics ---
        agg_match = re.search(r'📈 聚合指标 \(加权平均\):\n(.*?)(?=\n📋|\Z)', block, re.S)
        log_entry['aggregated_metrics'] = {}
        if agg_match:
            log_entry['aggregated_metrics'] = parse_agg_metrics(agg_match.group(1))

        # --- 4. Parse partitioned metrics ---
        # Match until the next '===' separator or EOF
        part_match = re.search(r'📋 分区指标 \(Flow Bin x Change Bin\) 报告:\n(.*?)(?=\n========================================================================================================================\n\n========================================================================================================================|\Z)', block, re.S)
        log_entry['partitioned_metrics'] = {}
        if part_match:
            log_entry['partitioned_metrics'] = parse_partitioned_metrics(part_match.group(1))

        all_logs.append(log_entry)

    # Note: if you want reverse order (latest first),
    # just return all_logs[::-1]
    return all_logs

#%% Model comparison
def _flatten_metrics(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Recursively flatten nested dictionaries.
    Example: {'losses': {'q_loss': 5.1}} -> {'losses_q_loss': 5.1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_metrics(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def create_aggregated_comparison_table(
    log_A: List[Dict], 
    log_B: List[Dict],
    model_A_name: str = "WithPhysics", 
    model_B_name: str = "WithoutPhysics"
) -> pd.DataFrame:
    """
    Create and return a DataFrame comparing aggregated metrics from two models.
    """
    data = []
    for log_list, model_name in [(log_A, model_A_name), (log_B, model_B_name)]:
        for entry in log_list:
            subset = entry.get('subset')
            if not entry.get('day_batches'):
                continue # Skip entries without day_batch
            
            # Assume each entry evaluates one day_batch
            day_batch = entry['day_batches'][0]
            date = day_batch.get('Date')
            day_shape = day_batch.get('DayShape')
            
            flat_metrics = _flatten_metrics(entry.get('aggregated_metrics', {}))
            
            row = {
                'model': model_name,
                'subset': subset, 
                'Date': date, 
                'DayShape': day_shape
            }
            row.update(flat_metrics)
            data.append(row)
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    
    # 获取所有指标列
    metric_cols = [col for col in df.columns if col not in ['model', 'subset', 'Date', 'DayShape']]
    
    # 创建数据透视表，索引为(Date, subset, DayShape)，列为模型
    try:
        df_pivot = df.pivot_table(
            index=['Date', 'subset', 'DayShape'],
            columns='model',
            values=metric_cols
        )
    except Exception as e:
        print(f"创建透视表失败: {e}")
        return df # 返回未透视的DF

    # 计算差异列 (B - A)
    for col in metric_cols:
        col_A = (col, model_A_name)
        col_B = (col, model_B_name)
        
        # 确保两个模型的列都存在
        if col_A in df_pivot.columns and col_B in df_pivot.columns:
            df_pivot[(col, f'Difference ({model_B_name}-{model_A_name})')] = \
                df_pivot[col_B] - df_pivot[col_A]

    # 按指标名称排序，使 A, B, Diff 在一起
    df_pivot = df_pivot.sort_index(axis=1, level=0)
    return df_pivot
