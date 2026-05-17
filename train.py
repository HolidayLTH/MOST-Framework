# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 23:49:32 2025

@author: 78632
"""
# --- import packages
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
print(os.getcwd())

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import gc
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from Model.TFT_GAT import TFT_GAT
from Model.LSTM import LSTM_Benchmark
from Model.Transformer import Transformer_Benchmark
from Model.STGCN import STGCN_Benchmark
from Model.DCRNN import DCRNN_Benchmark
from Model.GMAN import GMAN_Benchmark
from Model.STTN import STTN_Benchmark
from Model.GraphWaveNet import GraphWaveNet_Benchmark
from Model.PI_MPN import PIMPN_Benchmark
from Model.PAG_STAN import PAGSTAN_Benchmark

# --- import functions ---
from func_model_datainput import SubwayODDataset, custom_collate_fn, prepare_tftgat_inputs, OD_Filter, load_and_cache_neighbormap, sample_neighbors, perturb_A_keep_sum, perturb_ordered_pair_weather
from func_model_process import weight_init, QueueAggregator, process_batch, recycle, plotsample, plotmodel, log_evaluation_metrics

# --- import profile for data processing---
with open(r'./Profile/profile.pkl','rb') as fp:
    profile = pickle.load(fp)   

feature_map = profile['feature_map']
scalers = profile['scalers']
cardinalities_map = profile['categorical_cardinalities']   

# --- OD-centered local graph / OD neighbors ---
network_features_folder = r'./Network Features'
neighbor_maps = load_and_cache_neighbormap(network_features_folder)

# --- 17-event pool ---
event_pool = ['E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15','E16','E17'] 

# --- key parameters of the experiment / just for introduction & actually input by argparse ---
IS_SPATIAL = True # is GAT used (always True)
IS_PHYSIC =  True # is Ohm's law used
IS_STATION = True # is KCL used
IS_LABEL = True # is real flow used to supervise Ohm's loss (always True)
train_events_input = ['E1','E2','E3','E4','E5']
only_eval = True
perturb_config = {
    'level': 'none',
    'range': (-0.5, 0.5),
    'fixed': None
}
model_name = 'TFT-GAT'
station_weight_override=None

# --- function for fixing random seed ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# --- functions for robustness testing ---
def _build_perturb_suffix(perturb_config: dict) -> str:
    level = perturb_config.get('level', 'none')
    pattern = perturb_config.get('pattern', 'none')
    if level == 'none' or pattern == 'none':
        return ""
    
    suffix = f"_PT-{pattern}-{level}"

    if pattern in {'travel_time', 'all'}:
        if perturb_config.get('fixed') is not None:
            suffix += f"_Fix{perturb_config['fixed']}"
        else:
            suffix += f"_R{perturb_config['range'][0]}to{perturb_config['range'][1]}"

    if pattern in {'wind', 'rain', 'all'}:
        suffix += f"_WP{perturb_config.get('weather_prob', 0.0)}"
        suffix += f"_WS{perturb_config.get('weather_shift', 0)}"

    return suffix

# --- main experiment function ---
def run_experiment(IS_SPATIAL, IS_PHYSIC, IS_STATION, IS_LABEL, train_events_input, only_eval, perturb_config, model_name, station_weight_override=None):

    # --- fix random seed ---
    GLOBAL_SEED = 2025
    seed_everything(GLOBAL_SEED)
    data_logic_rng = random.Random(GLOBAL_SEED)

    ewe_files = sorted([os.path.join('EWE_data', f) for f in os.listdir('EWE_data') if f.endswith('.csv')])
    
    typical_train_events = train_events_input
    typical_test_events = sorted(list(set(event_pool) - set(typical_train_events)),key = lambda x: int(x[1:]))
    
    print(f"Current Training Events: {typical_train_events}")
    print(f"Current Testing Events: {typical_test_events}")
    
    ewe_train_files = [f for f in ewe_files if any([True if f.split('#')[1] == e else False for e in typical_train_events])]
    ewe_test_files = [f for f in ewe_files if any([True if f.split('#')[1] == e else False for e in typical_test_events])]
    
    # --- data split ---
    train_files = ewe_train_files
    
    train_dataset = SubwayODDataset(data_files = train_files, scalers = profile['scalers'])
    test_dataset = SubwayODDataset(data_files = ewe_test_files, scalers = profile['scalers'])
    
    torch_gen = torch.Generator()
    torch_gen.manual_seed(GLOBAL_SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, generator=torch_gen)
    train_loader_eval = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    train_loader = recycle(train_loader)
    train_loader_eval = recycle(train_loader_eval)
    test_loader = recycle(test_loader)

    # --- file name definition ---
    events_str = "_".join(typical_train_events)
    config_suffix = f'{"SPA" if IS_SPATIAL else "NOSPA"}_{"PHY" if IS_PHYSIC else "NOPHY"}_{"STA" if IS_STATION else "NOSTA"}_{"LAB" if IS_LABEL else "NOLAB"}'
    file_name = f'{config_suffix}_{events_str}'
    if model_name != "TFT-GAT":
        file_name = f"{file_name}_{model_name}"
    
    # suffix for perturbation settings in robustness testing
    pt_suffix = _build_perturb_suffix(perturb_config)
    
    LOG_FILE = rf'./Log/{model_name}/evaluation_log_{file_name}{pt_suffix}.txt'
    model_cache_folder_map = {
        'TFT-GAT': 'TFT-GAT',
        'LSTM': 'LSTM',
        'Transformer': 'Transformer',
        'STGCN': 'STGCN',
        'DCRNN': 'DCRNN',
        'GraphWaveNet': 'GraphWaveNet',
        'STTN': 'STTN',
        'GMAN': 'GMAN',
        'PI-MPN': 'PI-MPN',
        'PAG-STAN': 'PAG-STAN',
    }
    model_cache_folder = model_cache_folder_map[model_name]

    PARAM_FILE = rf'./Weights/{model_cache_folder}/model_params_{file_name}.pth'
    os.makedirs(os.path.dirname(PARAM_FILE), exist_ok=True)
    
    print("==================================================")
    print(f"Start Running: {file_name}")
    print(f"Events: {events_str}")
    print("==================================================")
    
    # --- tensorboard logging ---
    log_dir = os.path.join('Tensorboard', f'{file_name}{pt_suffix}')
    writer = SummaryWriter(log_dir)
    
    # --- global parameter configuration ---
    BATCH_SIZE = 16 # number of origins per training batch
    if model_name in {'STTN', 'GMAN'}:
        BATCH_SIZE_INFER = 2
    else:
        BATCH_SIZE_INFER = 16 # number of origins per inference batch
    LEARNING_RATE = 1e-3
    STATESIZE = 64

    FlowTS=[0,0.6] # filter OD pairs by cumulative flow share
    ChangeTS=[0,0.6] # filter OD pairs by cumulative flow-change share
    SupplyTS=[0,0.6] # filter OD pairs by cumulative supply-change share (unused for FLOW ONLY)

    FlowTS_INFER=[0,0.2,0.4,0.6,0.8,0.9]
    ChangeTS_INFER=[0,0.2,0.4,0.6,0.8,0.9]
    SupplyTS_INFER=[0,0.2,0.4,0.6,0.8,0.9]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    IS_CUDA = torch.cuda.is_available()

    TW = 3 # Buffer window for Ohm's law to control temporal spillover effects
    PHY_W = 0.0 # Initialize physics loss weight
    
    # --- model configuration ---
    configuration = {'optimization':
                 {
                     'batch_size': {'training': BATCH_SIZE, 'inference': BATCH_SIZE_INFER},
                     'learning_rate': LEARNING_RATE,
                     'max_grad_norm': 1.0,
                 }
                 ,
                 'model':
                 {
                     'dropout': 0.05,
                     'state_size': STATESIZE,
                     'output_quantiles': [0.1, 0.5, 0.9],
                     'lstm_layers': 1,
                     'attention_heads': 4
                 },
                 # these arguments are related to possible extensions of the model class
                 'task_type':'regression',
                 'target_window_start': None,
                 
                 'SL_ts_index':[feature_map['future_ts_numeric'].index('SupplyLoss')],
                 
                 'TT_ts_index':[feature_map['future_ts_numeric'].index('TravelTime')],
                 
                 'W_ts_index':[feature_map['future_ts_categorical'].index('WindStatus'),
                               feature_map['future_ts_categorical'].index('RainStatus'),
                               feature_map['future_ts_categorical'].index('TyphoonAlert'),
                               feature_map['future_ts_categorical'].index('RainstormAlert')],
                 
                 'Hour_ts_index':[feature_map['future_ts_categorical'].index('Hour')],
                 
                 'Day_index':[feature_map['static_feats_categorical'].index('Month'),
                              feature_map['static_feats_categorical'].index('DayofWeek')],
                 
                 'O_index_categorical':[feature_map['static_feats_categorical'].index('Oindex')],
                 'O_index_numeric':[feature_map['static_feats_numeric'].index('lines_count#O'),
                                    feature_map['static_feats_numeric'].index('degree_centrality#O'),
                                    feature_map['static_feats_numeric'].index('closeness_centrality#O'),
                                    feature_map['static_feats_numeric'].index('betweenness_centrality#O')],
    
                 'D_index_categorical':[feature_map['static_feats_categorical'].index('Dindex')],
                 'D_index_numeric':[feature_map['static_feats_numeric'].index('lines_count#D'),
                                    feature_map['static_feats_numeric'].index('degree_centrality#D'),
                                    feature_map['static_feats_numeric'].index('closeness_centrality#D'),
                                    feature_map['static_feats_numeric'].index('betweenness_centrality#D')],
    
                 'OD_index_categorical':[feature_map['static_feats_categorical'].index('ODindex')],
                 'OD_index_numeric':[feature_map['static_feats_numeric'].index('min_transfers'),
                                    feature_map['static_feats_numeric'].index('shortest_path_length'),
                                    feature_map['static_feats_numeric'].index('min_station_count'),
                                    feature_map['static_feats_numeric'].index('num_effective_paths'),
                                    feature_map['static_feats_numeric'].index('avg_path_similarity')]
                }
    
    structure = {
        'num_historical_numeric': len(feature_map['historical_ts_numeric']), # known historical numeric features
        'num_historical_categorical': len(feature_map['historical_ts_categorical']), # known historical categorical features
        'num_static_numeric': len(feature_map['static_feats_numeric']), # static numeric features
        'num_static_categorical': len(feature_map['static_feats_categorical']), # static categorical features
        'num_future_numeric': len(feature_map['future_ts_numeric']), # known future numeric features
        'num_future_categorical': len(feature_map['future_ts_categorical']), # known future categorical features
        'historical_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['historical_ts_categorical']],
        'static_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['static_feats_categorical']],
        'future_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['future_ts_categorical']],
        'num_target': 1, 
        'num_auxiliary_target':1,
        'is_mask': False
    }
    
    configuration['data_props'] = structure
    
    # --- initialize model ---
    model_map = {
        "TFT-GAT": TFT_GAT,
        "LSTM": LSTM_Benchmark,
        "Transformer": Transformer_Benchmark,
        "STGCN": STGCN_Benchmark,
        "DCRNN": DCRNN_Benchmark,
        "GMAN": GMAN_Benchmark,
        "STTN": STTN_Benchmark,
        "GraphWaveNet": GraphWaveNet_Benchmark,
        "PI-MPN": PIMPN_Benchmark,
        "PAG-STAN": PAGSTAN_Benchmark,
    }
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")
    model = model_map[model_name](config=OmegaConf.create(configuration))    
    model.apply(weight_init)  
    # nn.init.normal_(model.pinn.b_net[2].weight, mean=0.0, std=0.001)
    # nn.init.constant_(model.pinn.b_net[2].bias, 0.0)
    model.to(DEVICE)
    opt = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                    lr=configuration['optimization']['learning_rate'])
    
    # --- initialize training parameters ---
    # after how many epochs should we quit training
    max_epochs = 20
    # how many training day batches will compose a single training epoch
    epoch_iters = len(train_files)
    # upon completing a training epoch, we perform an evaluation of all the subsets
    # eval_iters will define how many day batches of each set will compose a single evaluation round
    eval_iters = 1
    # during training, on what frequency of od batch should we display the monitored performance
    log_interval = 20 
    # what is the running-window used by our QueueAggregator object for monitoring the training performance
    max_queue_size = 200

    # initialize the loss aggregator for running window performance estimation
    loss_aggregator = QueueAggregator(max_size=max_queue_size)
    
    # initialize counters
    epoch_idx = 0 
    day_batch_idx = 0 
    window_batch_idx = 0 
    od_batch_idx = 0
    log_idx = 0
    
    # quantile output setting
    quantiles_tensor = torch.tensor(configuration['model']['output_quantiles']).to(DEVICE)
    
    # --- initialize model parameters ---
    ckpt_loaded = False
    if os.path.exists(PARAM_FILE):
        model.load_state_dict(torch.load(PARAM_FILE))
        ckpt_loaded = True
        print(rf'Successful: {PARAM_FILE} is loaded!!!')
    else:
        print(rf'Error: {PARAM_FILE} is not found!!!')
        if only_eval:
            raise FileNotFoundError(f"only_eval=True but checkpoint not found: {PARAM_FILE}")

    # --- training ---
    while epoch_idx <= max_epochs:
        
        if only_eval:
            print(f"Skipping training, starting evaluation directly. (Log: {LOG_FILE})")
            break
        
        print(f"Starting Epoch Index {epoch_idx}/{max_epochs}")
    
        # switch to training mode     
        model.train()
        
        day_batch_idx = 0
        for i in range(epoch_iters):
            
            # get training day batch
            day_batch = next(train_loader)
            day_data = day_batch['data'][0]
            day_info = day_batch['day_info'][0]
            date_str = day_info['Date']
            day_shape = day_info['DayShape']
            date = pd.to_datetime(date_str,format='%Y%m%d')

            # get OD neighbor in this period
            if IS_SPATIAL:
                for neighbor_map in neighbor_maps:
                    if (date >= neighbor_map['start']) & (date <= neighbor_map['end']):
                        break
                day_neighbor_map = neighbor_map['neighbor_map']
                allods = pd.unique(day_neighbor_map['TargetOD'])
                self_map = pd.DataFrame({'TargetOD': allods,'SimilarOD': allods})
                day_neighbor_map = pd.concat([day_neighbor_map, self_map], ignore_index=True)
                day_neighbor_map = day_neighbor_map.sort_values(by=['TargetOD','SimilarOD']).reset_index(drop=True)
                day_neighbor_map = day_neighbor_map.drop_duplicates()
            
            # get a reasonable window set
            window_pool = {'000':(0,48),'010':(6,36),'011':(6,48),'110':(0,36),'111':(0,48),'-11':(0,24),'01-':(6,24),'11-':(0,24),'-10':(0,18)}[day_shape]
            window_pool = list(range(*window_pool,6))
            window_batch_idx = 0
            
            # get a specifc window
            for window in window_pool:
                
                bias = data_logic_rng.choice(range(6))
                window = window+bias
                
                # get training day window batch
                daywindow_od_indices = OD_Filter(day_data, window, FlowTS=FlowTS, ChangeTS=ChangeTS, SupplyTS=SupplyTS, FlowOnly=True)[0][0]
                data_logic_rng.shuffle(daywindow_od_indices) 
                daywindow_data = day_data.loc[daywindow_od_indices]
                daywindow_o_indices = daywindow_data['Oindex'].unique()
                
                # get neighbor in this window
                if IS_SPATIAL:
                    daywindow_neighbor_map = day_neighbor_map[day_neighbor_map['TargetOD'].isin(daywindow_od_indices)
                                                        & day_neighbor_map['SimilarOD'].isin(daywindow_od_indices)]
                
                # od_batch_count
                od_batch_idx = 0
                i = 0
                
                # get training od batch
                for i in trange(0, len(daywindow_o_indices), BATCH_SIZE, desc = f'Date:{date}-Window:{window}'):
                    
                    # get neighbor in this batch
                    batch_od_target_indices = daywindow_data[daywindow_data['Oindex'].isin(daywindow_o_indices[i:i+BATCH_SIZE])].index.tolist()
                    batch_od_target_indices = sorted(batch_od_target_indices,key = lambda x:x.split('@')[0])
                    target_N = len(batch_od_target_indices)
                    
                    if target_N < 10:
                        continue
                    
                    if IS_SPATIAL:
                        # sample neighbors for this batch   
                        batch_od_neighbor_indices = sample_neighbors(daywindow_neighbor_map.loc[daywindow_neighbor_map['TargetOD'].isin(batch_od_target_indices)], k=5, random_state=GLOBAL_SEED)['SimilarOD'].values.tolist()
                        batch_od_neighbor_indices = np.setdiff1d(batch_od_neighbor_indices, batch_od_target_indices).tolist()
                        batch_indices = batch_od_target_indices + batch_od_neighbor_indices
                        # construct edge index needed by GAT for this batch
                        batch_od_neighbor_map = daywindow_neighbor_map[daywindow_neighbor_map['TargetOD'].isin(batch_indices) & daywindow_neighbor_map['SimilarOD'].isin(batch_indices)]
                        code_to_idx = pd.Series(batch_indices)
                        code_to_idx = pd.Series(code_to_idx.index, index=code_to_idx.values).to_dict()
                        src = batch_od_neighbor_map['TargetOD'].map(code_to_idx).values
                        dst = batch_od_neighbor_map['SimilarOD'].map(code_to_idx).values
                        edge_index = torch.tensor([dst, src], dtype=torch.long)
                        edge_index = edge_index.to(DEVICE)
                    else:
                        batch_indices = batch_od_target_indices
                        edge_index = None
    
                    # create model input
                    batch_od_data = daywindow_data.loc[batch_indices]
                    input_tensors_original = prepare_tftgat_inputs(batch_od_data, window, len(batch_indices), feature_map, DEVICE, is_normal=False, is_spatial=IS_SPATIAL, target_N=target_N)
                    input_tensors_normal = prepare_tftgat_inputs(batch_od_data, window, len(batch_indices), feature_map, DEVICE, is_normal=True, is_spatial=IS_SPATIAL, target_N=target_N)
                    
                    # extra info
                    input_tensors_original['date'] = date
                    input_tensors_original['start_time'] = window
                    input_tensors_original['ODindex'] = batch_indices
                    input_tensors_normal['ODindex'] = batch_indices
    
                    # get model output
                    od_batch_info_original = model(input_tensors_original, edge_index, target_N, IS_EVAL = False, IS_STATION = IS_STATION, IS_PHYSIC = IS_PHYSIC)
                    od_batch_info_normal = model(input_tensors_normal, edge_index, target_N, IS_EVAL = False, IS_STATION = IS_STATION, IS_PHYSIC = IS_PHYSIC)
    
                    # reset gradients
                    opt.zero_grad()
                    
                    # process batch
                    loss,_,(loss_data,loss_data_OFlow,loss_ohm,loss_ohm_base,loss_U_gt_I,loss_U_sm_N,loss_reality) = process_batch(batch_original = od_batch_info_original,
                                           batch_normal = od_batch_info_normal,
                                           label = input_tensors_original['future_target'][:target_N],
                                           label_U = input_tensors_original['future_ts_numeric'][:target_N,:,[feature_map['future_ts_numeric'].index('TemplateFlow')]],
                                           label_phi = input_tensors_original['future_target_auxiliary'][:target_N],
                                           label_congestion = input_tensors_original['future_congestion'][:target_N],
                                           label_supply = input_tensors_original['future_ts_numeric'][:target_N,:,configuration['TT_ts_index']]/ (input_tensors_original['future_ts_numeric'][:target_N,:,configuration['SL_ts_index']] + input_tensors_original['future_ts_numeric'][:target_N,:,configuration['TT_ts_index']] + 1e-6), # 1 - input_tensors_original['future_ts_numeric'][:target_N,:,configuration['SL_ts_index']] / (input_tensors_normal['future_ts_numeric'][:target_N,:,configuration['TT_ts_index']] + 1e-6), # 
                                           quantiles_tensor = quantiles_tensor,
                                           device = DEVICE,
                                           IS_CUDA = IS_CUDA,
                                           IS_PHYSIC = IS_PHYSIC,
                                           IS_STATION = IS_STATION,
                                           IS_LABEL = IS_LABEL,
                                           IS_EVAL = False,
                                           batch_indices = batch_indices[:target_N],
                                           tw = TW,
                                           phy_w = PHY_W,
                                           hour_ts = input_tensors_original['future_ts_categorical'][...,configuration['Hour_ts_index']][[0]],
                                           events_str = events_str,
                                           station_weight_override = station_weight_override)
                    
                    # compute gradients
                    loss.backward()
        
                    # gradient clipping
                    if configuration['optimization']['max_grad_norm'] > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), configuration['optimization']['max_grad_norm'])
                    
                    # update weights
                    opt.step()
            
                    # accumulate performance
                    loss_aggregator.append(loss.item())
    
                    
                    # log performance
                    if od_batch_idx % log_interval == 0:
                        chosen_idx = data_logic_rng.choice(range(target_N))
                        log_idx += 1
                        writer.add_scalar('Loss/train', loss.item(), log_idx)
                        writer.flush()
                        print('\n')
                        print(f"Epoch: {epoch_idx}, DAY Bacth Index: {day_batch_idx}, Window Batch Index: {window_batch_idx}, OD Batch Index: {od_batch_idx} - Train Loss = {np.mean(loss_aggregator.get())}")
                        print(f'loss_data = {loss_data}, loss_data_OFlow = {loss_data_OFlow}, loss_ohm = {loss_ohm}, loss_ohm_base={loss_ohm_base}, loss_U_gt_I={loss_U_gt_I}, loss_U_sm_N={loss_U_sm_N}, loss_reality={loss_reality}')
                        
                    # memory control
                    del loss, loss_data, loss_ohm, loss_ohm_base, loss_U_gt_I
                    del od_batch_info_original, od_batch_info_normal
                    del input_tensors_original, input_tensors_normal
                    if IS_SPATIAL:
                        del edge_index,code_to_idx,src,dst
                    if IS_CUDA:
                        torch.cuda.empty_cache()
                    plt.close('all')
                    gc.collect()
                    
                    # completed od batch
                    od_batch_idx += 1
                    del batch_od_data, batch_od_neighbor_map, batch_indices, batch_od_target_indices, batch_od_neighbor_indices
    
                # completed day window batch
                window_batch_idx += 1
                del daywindow_data, daywindow_neighbor_map, daywindow_od_indices, daywindow_o_indices
            # completed day batch
    
            day_batch_idx += 1
            del day_batch, day_data, day_neighbor_map, allods, self_map
        # completed epoch
        epoch_idx += 1
        if epoch_idx>=5 and PHY_W < 1:
            PHY_W += 0.1
            print('==============')
            print(f'NEW PHYSIC WEIGHT:{PHY_W}')
            print('==============')
            
        torch.save(model.state_dict(), PARAM_FILE)
        
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Successfully saved the model')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    
    # --- evaluation/inference --- 
    model.eval()
    log_idx = 0
    with torch.no_grad():
        
        # for each subset
        for subset_name, subset_loader in zip(['train','test'],[train_loader_eval,test_loader]):
            
            print(f"Evaluating {subset_name} set")
            eval_iters = {'train':len(train_files),'test':len(ewe_test_files)}[subset_name]
    
            for _ in trange(eval_iters, desc='evaluation: day iteration'):
                
                num_flow_bins = len(FlowTS_INFER) - 1
                num_change_bins = len(ChangeTS_INFER) - 1
                
                # training metric
                q_loss_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                loss_data_OFlow_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                loss_data_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                loss_ohm_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                loss_U_gt_I_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                sample_count_vals = [[0 for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
    
                # evaluation metric
                mape_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                smape_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                mae_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                mse_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                rmse_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                picp_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                mpiw_vals = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                
                # evaluation metric with physic
                mape_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                smape_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                mae_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                mse_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                rmse_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                picp_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                mpiw_vals_physic = [[[] for _ in range(num_change_bins)] for _ in range(num_flow_bins)]
                
                # day information
                day_info_list = []
                
                # get day batch
                day_batch = next(subset_loader)
                day_data = day_batch['data'][0]
                day_info = day_batch['day_info'][0]
                date_str = day_info['Date']
                day_shape = day_info['DayShape']
                date = pd.to_datetime(date_str,format='%Y%m%d')
                day_info_list.append(day_info)
                
                # get neighbor by period
                if IS_SPATIAL:
                    for neighbor_map in neighbor_maps:
                        if (date >= neighbor_map['start']) & (date <= neighbor_map['end']):
                            break
                    day_neighbor_map = neighbor_map['neighbor_map']
                    allods = pd.unique(day_neighbor_map['TargetOD'])
                    self_map = pd.DataFrame({'TargetOD': allods,'SimilarOD': allods})
                    day_neighbor_map = pd.concat([day_neighbor_map, self_map], ignore_index=True)
                    day_neighbor_map = day_neighbor_map.sort_values(by=['TargetOD','SimilarOD']).reset_index(drop=True)
                    day_neighbor_map = day_neighbor_map.drop_duplicates()
                
                # get a specific window pool
                window_pool = {'000':(0,48),'010':(6,36),'011':(6,48),'110':(0,36),'111':(0,48),'-11':(0,24),'01-':(6,24),'11-':(0,24),'-10':(0,18)}[day_info['DayShape']]
                window_pool = list(range(*window_pool,6))
                
                for window in tqdm(window_pool, desc='evaluation: window iteration'):
                    bias = 3
                    window = window + bias
                
                    # get a window sample, binned by FLOWTS and ChangeTS
                    daywindow_bins = OD_Filter(day_data, window, FlowTS=FlowTS_INFER, ChangeTS=ChangeTS_INFER, SupplyTS=SupplyTS_INFER, FlowOnly=True)
                    daywindow_od_indices = sum([
                                                    indices_list 
                                                    for flow_bin in daywindow_bins 
                                                    for indices_list in flow_bin
                                                ],[])
                    daywindow_data = day_data.loc[daywindow_od_indices]
                    daywindow_o_indices = daywindow_data['Oindex'].unique() # unique origins
                    
                    # create a mapping from OD index to its corresponding bin group (flow_bin_idx, change_bin_idx)
                    od_to_group_map = {}
                    for i in range(num_flow_bins):
                        for j in range(num_change_bins):
                            for od_idx in daywindow_bins[i][j]:
                                od_to_group_map[od_idx] = (i, j)
    
                    # filter neighbor by window sample
                    if IS_SPATIAL:
                        daywindow_neighbor_map = day_neighbor_map[day_neighbor_map['TargetOD'].isin(daywindow_od_indices)
                                                                  & day_neighbor_map['SimilarOD'].isin(daywindow_od_indices)]
                        
                    for i in tqdm(range(0, len(daywindow_o_indices), BATCH_SIZE_INFER), desc=f'Date:{date}-Window:{window}'):
                        
                        # get neighbor in this batch
                        batch_od_target_indices = daywindow_data[daywindow_data['Oindex'].isin(daywindow_o_indices[i:i+BATCH_SIZE_INFER])].index.tolist()
                        batch_od_target_indices = sorted(batch_od_target_indices,key = lambda x:x.split('@')[0])
                        target_N = len(batch_od_target_indices)
                            
                        if target_N < 10:
                            continue # skip small od batches for more stable evaluation results
                
                        if IS_SPATIAL:
                            # sample neighbors for this batch
                            batch_od_neighbor_indices = sample_neighbors(daywindow_neighbor_map.loc[daywindow_neighbor_map['TargetOD'].isin(batch_od_target_indices)], k=5, random_state=GLOBAL_SEED)['SimilarOD'].values.tolist()
                            batch_od_neighbor_indices = np.setdiff1d(batch_od_neighbor_indices, batch_od_target_indices).tolist()
                            batch_indices = batch_od_target_indices + batch_od_neighbor_indices
                            # construct edge index needed by GAT for this batch
                            batch_od_neighbor_map = daywindow_neighbor_map[daywindow_neighbor_map['TargetOD'].isin(batch_indices) & daywindow_neighbor_map['SimilarOD'].isin(batch_indices)]
                            code_to_idx = pd.Series(batch_indices)
                            code_to_idx = pd.Series(code_to_idx.index, index=code_to_idx.values).to_dict()
                            src = batch_od_neighbor_map['TargetOD'].map(code_to_idx).values
                            dst = batch_od_neighbor_map['SimilarOD'].map(code_to_idx).values
                            edge_index = torch.from_numpy(np.array([dst, src])).long()
                            edge_index = edge_index.to(DEVICE)
                        else:
                            batch_indices = batch_od_target_indices
                            edge_index = None
    
                        # create model input
                        batch_od_data = daywindow_data.loc[batch_indices]
                        input_tensors_original = prepare_tftgat_inputs(batch_od_data, window, len(batch_indices), feature_map, DEVICE, is_normal=False, is_spatial=IS_SPATIAL, target_N=target_N)
                        input_tensors_normal = prepare_tftgat_inputs(batch_od_data, window, len(batch_indices), feature_map, DEVICE, is_normal=True, is_spatial=IS_SPATIAL, target_N=target_N)
                        input_tensors_original['ODindex'] = batch_indices
                        input_tensors_normal['ODindex'] = batch_indices
                        
                        # apply perturbation if needed
                        pt_level = perturb_config.get('level', 'none')
                        pt_pattern = perturb_config.get('pattern', 'none')

                        if pt_level != 'none' and pt_pattern in {'travel_time', 'all'}:
                            input_tensors_original['future_ts_numeric'] = perturb_A_keep_sum(
                                input_tensors_original['future_ts_numeric'],
                                configuration['SL_ts_index'],
                                configuration['TT_ts_index'],
                                perturb_level=pt_level,
                                ratio_range=perturb_config['range'],
                                fixed_ratio=perturb_config['fixed']
                            )

                        if pt_level != 'none' and pt_pattern in {'wind', 'all'}:
                            wind_idx = feature_map['future_ts_categorical'].index('WindStatus')
                            typhoon_idx = feature_map['future_ts_categorical'].index('TyphoonAlert')
                            input_tensors_original['future_ts_categorical'] = perturb_ordered_pair_weather(
                                input_tensors_original['future_ts_categorical'],
                                idx_A=wind_idx,
                                idx_B=typhoon_idx,
                                max_A=12,
                                max_B=6,
                                perturb_level=pt_level,
                                mutation_prob=perturb_config.get('weather_prob', 0.0),
                                shift=perturb_config.get('weather_shift', 0),
                            )

                        if pt_level != 'none' and pt_pattern in {'rain', 'all'}:
                            rain_idx = feature_map['future_ts_categorical'].index('RainStatus')
                            rainstorm_idx = feature_map['future_ts_categorical'].index('RainstormAlert')
                            input_tensors_original['future_ts_categorical'] = perturb_ordered_pair_weather(
                                input_tensors_original['future_ts_categorical'],
                                idx_A=rain_idx,
                                idx_B=rainstorm_idx,
                                max_A=6,
                                max_B=6,
                                perturb_level=pt_level,
                                mutation_prob=perturb_config.get('weather_prob', 0.0),
                                shift=perturb_config.get('weather_shift', 0),
                            )
    
                        # get model output
                        od_batch_info_original = model(input_tensors_original, edge_index, target_N, IS_EVAL = True, IS_STATION = IS_STATION,IS_PHYSIC = IS_PHYSIC)
                        od_batch_info_normal = model(input_tensors_normal, edge_index, target_N, IS_EVAL = True, IS_STATION = IS_STATION, IS_PHYSIC = IS_PHYSIC)
              
                        # process batch
                        batch_loss, batch_q_risk, train_metrcs, eval_metrics, eval_metrics_physic = process_batch(batch_original = od_batch_info_original,
                            batch_normal = od_batch_info_normal,
                            label = input_tensors_original['future_target'][:target_N],
                            label_U = input_tensors_original['future_ts_numeric'][:target_N,:,[feature_map['future_ts_numeric'].index('TemplateFlow')]],
                            label_phi = input_tensors_original['future_target_auxiliary'][:target_N],
                            label_congestion = input_tensors_original['future_congestion'][:target_N],
                            label_supply =  input_tensors_original['future_ts_numeric'][:target_N,:,configuration['TT_ts_index']]/ (input_tensors_original['future_ts_numeric'][:target_N,:,configuration['SL_ts_index']] + input_tensors_original['future_ts_numeric'][:target_N,:,configuration['TT_ts_index']] + 1e-6), # 1 - input_tensors_original['future_ts_numeric'][:target_N,:,configuration['SL_ts_index']] / (input_tensors_normal['future_ts_numeric'][:target_N,:,configuration['TT_ts_index']] + 1e-6),#
                            quantiles_tensor = quantiles_tensor,
                            device = DEVICE,
                            IS_CUDA = IS_CUDA,
                            IS_PHYSIC = IS_PHYSIC,
                            IS_STATION = IS_STATION,
                            IS_LABEL = IS_LABEL,
                            IS_EVAL = True,
                            batch_indices = batch_indices[:target_N],
                            tw = TW,
                            phy_w = 1,
                            hour_ts = input_tensors_original['future_ts_categorical'][...,configuration['Hour_ts_index']][[0]],
                            events_str = events_str,
                            station_weight_override = station_weight_override)
                        
                        # create a mapping from OD index to its corresponding bin group (flow_bin_idx, change_bin_idx) for this batch
                        batch_group_ids = np.array([od_to_group_map.get(od, (-1, -1)) for od in batch_indices[:target_N]])
                        
                        # accumulate performance for each bin
                        for i in range(num_flow_bins):
                            for j in range(num_change_bins):
                                # find samples belonging to the current bin (i, j)
                                mask = (batch_group_ids[:, 0] == i) & (batch_group_ids[:, 1] == j)
                                if not mask.any():
                                    continue
                        
                                # accumulate performance
                                q_loss_vals[i][j].extend(batch_loss[mask].tolist())
                                # q_risk_vals[i][j].extend(batch_q_risk[mask].tolist())
                                loss_data_vals[i][j].extend(train_metrcs[0][mask].tolist())
                                loss_data_OFlow_vals[i][j].extend(train_metrcs[1][mask].tolist())
                                loss_ohm_vals[i][j].extend(train_metrcs[2][mask].tolist())
                                loss_U_gt_I_vals[i][j].extend(train_metrcs[4][mask].tolist())
                                sample_count_vals[i][j] += mask.sum()
                                
                                mape_vals[i][j].extend(eval_metrics[0][mask].tolist())
                                smape_vals[i][j].extend(eval_metrics[1][mask].tolist())
                                mae_vals[i][j].extend(eval_metrics[2][mask].tolist())
                                mse_vals[i][j].extend(eval_metrics[3][mask].tolist())
                                rmse_vals[i][j].extend(eval_metrics[4][mask].tolist())
                                picp_vals[i][j].extend(eval_metrics[5][mask].tolist())
                                mpiw_vals[i][j].extend(eval_metrics[6][mask].tolist())
                                
                                mape_vals_physic[i][j].extend(eval_metrics_physic[0][mask].tolist())
                                smape_vals_physic[i][j].extend(eval_metrics_physic[1][mask].tolist())
                                mae_vals_physic[i][j].extend(eval_metrics_physic[2][mask].tolist())
                                mse_vals_physic[i][j].extend(eval_metrics_physic[3][mask].tolist())
                                rmse_vals_physic[i][j].extend(eval_metrics_physic[4][mask].tolist())
                                picp_vals_physic[i][j].extend(eval_metrics_physic[5][mask].tolist())
                                mpiw_vals_physic[i][j].extend(eval_metrics_physic[6][mask].tolist())
                        
                        if subset_name == 'test':
                            log_idx += 1
                            if log_idx % 10 == 0:
                                # visualize a sample from the current batch
                                chosen_idx = data_logic_rng.choice(range(target_N))
                                fig1 = plotsample(daywindow_data,batch_indices,input_tensors_original,input_tensors_normal,od_batch_info_original,od_batch_info_normal,chosen_idx,window,date_str,day_shape,feature_map,return_fig=True)
                                writer.add_figure('Prediction/Eval', fig1, log_idx)
                                if only_eval:
                                    fig2 = plotmodel(od_batch_info_original,chosen_idx,feature_map,window,day_shape,return_fig=True)
                                    writer.add_figure('Model/Eval', fig2, log_idx)
                                
                                writer.flush()
                            
                        # memory control: batch level
                        del batch_od_data, batch_od_neighbor_map, batch_indices, batch_od_target_indices, batch_od_neighbor_indices, batch_group_ids
                        del batch_loss, batch_q_risk, train_metrcs, eval_metrics
                        del od_batch_info_original, od_batch_info_normal
                        del input_tensors_original, input_tensors_normal
                        if IS_SPATIAL:
                            del edge_index,code_to_idx,src,dst
                        if IS_CUDA:
                            torch.cuda.empty_cache()
                        plt.close('all')
                        gc.collect()

                    # memory control: window level
                    del daywindow_data, daywindow_bins, daywindow_neighbor_map, daywindow_od_indices, od_to_group_map, daywindow_o_indices
                
                # memory control: day level
                del day_batch, day_data,  day_neighbor_map, allods, self_map
                
                for i in range(num_flow_bins):
                    for j in range(num_change_bins):
                        if q_loss_vals[i][j]:
                        
                            q_loss_vals[i][j] = np.mean(q_loss_vals[i][j]) 
                            loss_data_vals[i][j] = np.mean(loss_data_vals[i][j]) 
                            loss_data_OFlow_vals[i][j] = np.mean(loss_data_OFlow_vals[i][j]) 
                            loss_ohm_vals[i][j] = np.mean(loss_ohm_vals[i][j]) 
                            loss_U_gt_I_vals[i][j] = np.mean(loss_U_gt_I_vals[i][j]) 
                         
                            mape_vals[i][j] = np.mean(mape_vals[i][j]) 
                            smape_vals[i][j] = np.mean(smape_vals[i][j]) 
                            mae_vals[i][j] = np.mean(mae_vals[i][j]) 
                            mse_vals[i][j] = np.mean(mse_vals[i][j]) 
                            rmse_vals[i][j] = np.mean(rmse_vals[i][j]) 
                            picp_vals[i][j] = np.mean(picp_vals[i][j]) 
                            mpiw_vals[i][j] = np.mean(mpiw_vals[i][j]) 
                            
                            mape_vals_physic[i][j] = np.mean(mape_vals_physic[i][j]) 
                            smape_vals_physic[i][j] = np.mean(smape_vals_physic[i][j]) 
                            mae_vals_physic[i][j] = np.mean(mae_vals_physic[i][j]) 
                            mse_vals_physic[i][j] = np.mean(mse_vals_physic[i][j]) 
                            rmse_vals_physic[i][j] = np.mean(rmse_vals_physic[i][j]) 
                            picp_vals_physic[i][j] = np.mean(picp_vals_physic[i][j]) 
                            mpiw_vals_physic[i][j] = np.mean(mpiw_vals_physic[i][j]) 
                        else:
                            q_loss_vals[i][j] = 0
                            loss_data_vals[i][j] = 0
                            loss_data_OFlow_vals[i][j] = 0
                            loss_ohm_vals[i][j] = 0
                            loss_U_gt_I_vals[i][j] = 0
                         
                            mape_vals[i][j] = 0
                            smape_vals[i][j] = 0
                            mae_vals[i][j] = 0
                            mse_vals[i][j] = 0
                            rmse_vals[i][j] = 0
                            picp_vals[i][j] = 0
                            mpiw_vals[i][j] = 0
                            
                            mape_vals_physic[i][j] = 0
                            smape_vals_physic[i][j] = 0
                            mae_vals_physic[i][j] = 0
                            mse_vals_physic[i][j] = 0
                            rmse_vals_physic[i][j] = 0
                            picp_vals_physic[i][j] = 0
                            mpiw_vals_physic[i][j] = 0
    
                flow_index_labels = [
                    f"Flow Bin {i} ({FlowTS_INFER[i]}, {FlowTS_INFER[i+1]}]" 
                    for i in range(num_flow_bins)
                ]
                change_column_labels = [
                    f"Change Bin {j} ({ChangeTS_INFER[j]}, {ChangeTS_INFER[j+1]}]" 
                    for j in range(num_change_bins)
                ]
    
                def create_metric_df(matrix, index_labels, col_labels, converter_func):
                    df = pd.DataFrame(
                        data=matrix, 
                        index=index_labels, 
                        columns=col_labels
                    )
                    return df.applymap(lambda x: converter_func(x) if x is not None else None)
                
                # create metric dataframes and compute weighted average metrics
                df_loss = create_metric_df(q_loss_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_loss_data = create_metric_df(loss_data_vals, flow_index_labels, change_column_labels,  lambda x: x)
                df_loss_data_OFlow = create_metric_df(loss_data_OFlow_vals, flow_index_labels, change_column_labels,  lambda x: x)
                df_loss_ohm = create_metric_df(loss_ohm_vals, flow_index_labels, change_column_labels,  lambda x: x)
                df_loss_U_gt_I = create_metric_df(loss_U_gt_I_vals, flow_index_labels, change_column_labels,  lambda x: x)
                df_number = create_metric_df(sample_count_vals, flow_index_labels, change_column_labels,  lambda x: x)
                
                df_mape = create_metric_df(mape_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_smape = create_metric_df(smape_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_mae = create_metric_df(mae_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_mse = create_metric_df(mse_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_rmse = create_metric_df(rmse_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_picp = create_metric_df(picp_vals, flow_index_labels, change_column_labels, lambda x: x)
                df_mpiw = create_metric_df(mpiw_vals, flow_index_labels, change_column_labels, lambda x: x)
                
                df_mape_physic = create_metric_df(mape_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                df_smape_physic = create_metric_df(smape_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                df_mae_physic = create_metric_df(mae_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                df_mse_physic = create_metric_df(mse_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                df_rmse_physic = create_metric_df(rmse_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                df_picp_physic = create_metric_df(picp_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                df_mpiw_physic = create_metric_df(mpiw_vals_physic, flow_index_labels, change_column_labels, lambda x: x)
                
                eval_loss = np.average(df_loss.fillna(0).values, weights=df_number.fillna(0).values)
                eval_loss_data = np.average(df_loss_data.fillna(0).values, weights=df_number.fillna(0).values)
                eval_loss_data_OFlow = np.average(df_loss_data_OFlow.fillna(0).values, weights=df_number.fillna(0).values)
                eval_loss_ohm = np.average(df_loss_ohm.fillna(0).values, weights=df_number.fillna(0).values)
                eval_loss_U_gt_I = np.average(df_loss_U_gt_I.fillna(0).values, weights=df_number.fillna(0).values)
                
                eval_mape = np.average(df_mape.fillna(0).values, weights=df_number.fillna(0).values)
                eval_smape = np.average(df_smape.fillna(0).values, weights=df_number.fillna(0).values)
                eval_mae = np.average(df_mae.fillna(0).values, weights=df_number.fillna(0).values)
                eval_mse = np.average(df_mse.fillna(0).values, weights=df_number.fillna(0).values)
                eval_rmse = np.average(df_rmse.fillna(0).values, weights=df_number.fillna(0).values)
                eval_picp = np.average(df_picp.fillna(0).values, weights=df_number.fillna(0).values)
                eval_mpiw = np.average(df_mpiw.fillna(0).values, weights=df_number.fillna(0).values)
                
                eval_mape_physic = np.average(df_mape_physic.fillna(0).values, weights=df_number.fillna(0).values)
                eval_smape_physic = np.average(df_smape_physic.fillna(0).values, weights=df_number.fillna(0).values)
                eval_mae_physic = np.average(df_mae_physic.fillna(0).values, weights=df_number.fillna(0).values)
                eval_mse_physic = np.average(df_mse_physic.fillna(0).values, weights=df_number.fillna(0).values)
                eval_rmse_physic = np.average(df_rmse_physic.fillna(0).values, weights=df_number.fillna(0).values)
                eval_picp_physic = np.average(df_picp_physic.fillna(0).values, weights=df_number.fillna(0).values)
                eval_mpiw_physic = np.average(df_mpiw_physic.fillna(0).values, weights=df_number.fillna(0).values)
                
                # log performance
                print('='*100)
                print(f"Epoch: {epoch_idx}" + f" - Eval {subset_name} - \n" + \
                      f"q_loss = {eval_loss:.5f} , "
                      # " , ".join([f"q_risk_{q:.1} = {risk:.5f}" for q,risk in zip(quantiles_tensor,eval_q_risk)])
                      )
                    
                print(f"loss_data: {eval_loss_data:.5f} , " + \
                      f"loss_data_OFlow: {eval_loss_data_OFlow:.5f} , " + \
                      f"loss_ohm: {eval_loss_ohm:.5f} , " + \
                      f"loss_U_gt_I: {eval_loss_U_gt_I:.5f} , ")
                
                print(f"mape: {eval_mape:.5f} , " + \
                      f"smape: {eval_smape:.5f} , " + \
                      f"mae: {eval_mae:.5f} , "+ \
                      f"mse: {eval_mse:.5f} , "+ \
                      f"rmse: {eval_rmse:.5f} , "+ \
                      f"picp: {eval_picp:.5f} , "+ \
                      f"mpiw: {eval_mpiw:.5f}")
                
                print(f"mape_physic: {eval_mape_physic:.5f} , " + \
                      f"smape_physic: {eval_smape_physic:.5f} , " + \
                      f"mae_physic: {eval_mae_physic:.5f} , "+ \
                      f"mse_physic: {eval_mse_physic:.5f} , "+ \
                      f"rmse_physic: {eval_rmse_physic:.5f} , "+ \
                      f"picp_physic: {eval_picp_physic:.5f} , "+ \
                      f"mpiw_physic: {eval_mpiw_physic:.5f}")
                
                print('='*100)
                
                plt.close('all')
                gc.collect()
                
                # organize all dataFrames
                metric_dataframes = {
                    'df_loss': df_loss, 
                    'df_loss_data': df_loss_data,
                    'df_loss_data_OFlow': df_loss_data_OFlow,
                    'df_loss_ohm': df_loss_ohm, 'df_loss_U_gt_I': df_loss_U_gt_I, 'df_number': df_number,
                    
                    'df_mape': df_mape, 'df_smape': df_smape, 'df_mae': df_mae, 'df_mse': df_mse,
                    'df_rmse': df_rmse, 'df_picp': df_picp, 'df_mpiw': df_mpiw,
                    
                    'df_mape_physic': df_mape_physic, 'df_smape_physic': df_smape_physic, 'df_mae_physic': df_mae_physic, 'df_mse_physic': df_mse_physic,
                    'df_rmse_physic': df_rmse_physic, 'df_picp_physic': df_picp_physic, 'df_mpiw_physic': df_mpiw_physic
                }
                
                # organize all metrics
                aggregated_metrics = {
                    'eval_loss': eval_loss, 
                    'eval_loss_data': eval_loss_data,
                    'eval_loss_data_OFlow': eval_loss_data_OFlow,
                    'eval_loss_ohm': eval_loss_ohm, 'eval_loss_U_gt_I': eval_loss_U_gt_I,
                    
                    'eval_mape': eval_mape, 'eval_smape': eval_smape, 'eval_mae': eval_mae,
                    'eval_mse': eval_mse, 'eval_rmse': eval_rmse, 'eval_picp': eval_picp,
                    'eval_mpiw': eval_mpiw,
                    
                    'eval_mape_physic': eval_mape_physic, 'eval_smape_physic': eval_smape_physic, 'eval_mae_physic': eval_mae_physic,
                    'eval_mse_physic': eval_mse_physic, 'eval_rmse_physic': eval_rmse_physic, 'eval_picp_physic': eval_picp_physic,
                    'eval_mpiw_physic': eval_mpiw_physic
                }
                
                log_evaluation_metrics(
                    log_file_path=LOG_FILE,
                    epoch_idx=epoch_idx,
                    subset_name=subset_name,
                    day_info_list=day_info_list,
                    metric_dataframes=metric_dataframes,
                    aggregated_metrics=aggregated_metrics,
                    quantiles_tensor=quantiles_tensor
                )
                
                del day_info_list
                
                del q_loss_vals, loss_data_vals, loss_ohm_vals, loss_data_OFlow_vals, loss_U_gt_I_vals, sample_count_vals
                del mape_vals, smape_vals, mae_vals, mse_vals, rmse_vals, picp_vals, mpiw_vals
                del mape_vals_physic, smape_vals_physic, mae_vals_physic, mse_vals_physic, rmse_vals_physic, picp_vals_physic, mpiw_vals_physic
        
                del df_loss, df_loss_data, df_loss_ohm, df_loss_data_OFlow, df_loss_U_gt_I, df_number
                del df_mape, df_smape, df_mae, df_mse, df_rmse, df_picp, df_mpiw
                del df_mape_physic, df_smape_physic, df_mae_physic, df_mse_physic, df_rmse_physic, df_picp_physic, df_mpiw_physic
                
                del eval_loss, eval_loss_data, eval_loss_ohm, eval_loss_data_OFlow, eval_loss_U_gt_I
                del eval_mape, eval_smape, eval_mae,eval_mse, eval_rmse, eval_picp, eval_mpiw
                del eval_mape_physic, eval_smape_physic, eval_mae_physic, eval_mse_physic, eval_rmse_physic, eval_picp_physic, eval_mpiw_physic
                
                del metric_dataframes, aggregated_metrics
                del flow_index_labels, change_column_labels
                del create_metric_df
    del model, opt, loss_aggregator
    if IS_CUDA:
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Finished: {file_name}\n\n")


if __name__ == '__main__':
    
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(description='Train model with specific events.')

    # Add --events argument; nargs='+' allows multiple values (list)
    parser.add_argument('--events', nargs='+', required=True, help='List of events used for training (e.g., E1 E2 E3 E4 E5)')
    parser.add_argument('--model', type=str, default='TFT-GAT',
                        choices=['TFT-GAT', 'LSTM', 'Transformer', 'ODformer', 'STGCN', 'DCRNN', 'GMAN', 'STTN', 'GraphWaveNet', 'PI-MPN', 'PAG-STAN'],
                        help='Model type: TFT-GAT | LSTM | Transformer | ODformer | STGCN | DCRNN | GMAN | STTN | GraphWaveNet | PI-MPN | PAG-STAN')
    
    # Add ONLY_EVAL switch
    parser.add_argument('--only_eval', action='store_true', help='If True, skip training and run evaluation only')
    
    # Add perturbation params (for log suffix and evaluation control)
    parser.add_argument('--perturb_level', type=str, default='none', choices=['none', 'sample', 'time', 'sample_time'], help='Perturbation level: none | sample | time | sample_time')
    parser.add_argument('--perturb_pattern', type=str, default='none', choices=['none', 'travel_time', 'wind', 'rain', 'all'], help='Perturbation pattern: none | travel_time | wind | rain | all')
    parser.add_argument('--ratio_min', type=float, default=-0.5, help='Min ratio for perturbation')
    parser.add_argument('--ratio_max', type=float, default=0.5, help='Max ratio for perturbation')
    parser.add_argument('--fixed_ratio', type=float, default=None, help='Fixed ratio for perturbation (overrides range)')
    parser.add_argument('--weather_mutation_prob', type=float, default=0.0, help='Mutation probability for weather perturbation, in [0, 1]')
    parser.add_argument('--weather_shift', type=int, default=0, help='Phase shift for weather perturbation, in [-5, 5]')
    parser.add_argument('--station_weights', nargs='*', type=float, default=None,
                        help='Optional StationWeight list for SPA_PHY_STA_LAB runs')
    
    # Parse arguments
    args = parser.parse_args()
    if args.events is None:
        # Fallback defaults for step-by-step testing without CLI args
        args = argparse.Namespace(
            events=train_events_input,
            model='TFT_GAT',
            only_eval=False,
            perturb_level='none',
            perturb_pattern='none',
            ratio_min=-0.5,
            ratio_max=0.5,
            fixed_ratio=None,
            weather_mutation_prob=0.0,
            weather_shift=0,
            station_weights=None
        )
    
    # Get event list from CLI
    input_train_events = args.events
    
    # Pack perturbation config into a dict for passing
    if not (0.0 <= args.weather_mutation_prob <= 1.0):
        raise ValueError(f"weather_mutation_prob must be in [0, 1], got {args.weather_mutation_prob}")
    if not (-5 <= args.weather_shift <= 5):
        raise ValueError(f"weather_shift must be in [-5, 5], got {args.weather_shift}")
    if args.ratio_min > args.ratio_max:
        raise ValueError(f"ratio_min must be <= ratio_max, got ({args.ratio_min}, {args.ratio_max})")

    perturb_config = {
        'level': args.perturb_level,
        'pattern': args.perturb_pattern,
        'range': (args.ratio_min, args.ratio_max),
        'fixed': args.fixed_ratio,
        'weather_prob': args.weather_mutation_prob,
        'weather_shift': args.weather_shift,
    }

    # Simple validation: ensure events are in event_pool
    for e in input_train_events:
        if e not in event_pool:
            raise ValueError(f"Error: Event {e} is not in the predefined event_pool!")

    # Define four configs to run (GAT, Ohm, KCL, Label)
    experiment_configs = [
        # 1. SPA_PHY_STA_LAB
        (True, True, True, True),
        # 2. SPA_PHY_NOSTA_LAB
        (True, True, False, True),
        # 3. SPA_NOPHY_NOSTA_LAB
        (True, False, False, True),
        # 4. SPA_NOPHY_STA_LAB
        (True, False, True, True),
    ]

    station_weights = args.station_weights if args.station_weights else [None]

    for config in experiment_configs:
        if config == (True, True, True, True) and station_weights != [None]:
            for station_weight_override in station_weights:
                run_experiment(*config, input_train_events, args.only_eval, perturb_config, args.model, station_weight_override)
        else:
            run_experiment(*config, input_train_events, args.only_eval, perturb_config, args.model, None)

