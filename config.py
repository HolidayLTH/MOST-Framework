# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:01:51 2025

@author: 78632
"""
import pandas as pd
import numpy as np
import os

def get_attr_name(col_name: str) -> str:
    if '@' in col_name:
        return col_name.split('@')[0]
    return col_name

def load_and_cache_network_features(network_features_folder):
    files = os.listdir(network_features_folder)
    station_files = sorted([f for f in files if f.startswith('station_')])
    network_periods = []
    invalid_stations = ['比亚迪北', '龙背', '自然博物馆西', '未来城', '燕子岭', '综合保税区', '中芯国际', '文化聚落', '站前路东']

    for f in station_files:
        try:
            parts = f.replace('station_', '').replace('.csv', '').split('-')
            start_date = pd.to_datetime(parts[0], format='%Y%m%d')
            end_date = pd.to_datetime(parts[1], format='%Y%m%d')
            station_file_path = os.path.join(network_features_folder, f)
            od_file_path = os.path.join(network_features_folder, f.replace('station_', 'od_'))
            station_df = pd.read_csv(station_file_path,index_col=0)
            station_df = station_df[~station_df['station_name'].isin(invalid_stations)].reset_index(drop=True)
            od_df = pd.read_csv(od_file_path,index_col=0)
            od_df = od_df[(~od_df['Origin'].isin(invalid_stations)) & (~od_df['Destination'].isin(invalid_stations))].reset_index(drop=True)
            network_periods.append({
                'start': start_date, 'end': end_date,
                'station_features': station_df, 'od_features': od_df
            })
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    return network_periods
#%%
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

#%% cols

identifier_cols = [
                        '进站站点', '出站站点', 'Date'
                    ]

event_cols = [
                    'Event_ID', 'Day_Shape'
                ]

index_cols = [
                    'Oindex', 'Dindex', 'ODindex'
                ]

day_cols = [
                'Month', 'DayofWeek'
            ]

hour_cols = ['Hour']

network_cols = [
                    'lines_count#O', 'degree_centrality#O', 'closeness_centrality#O', 'betweenness_centrality#O',
                    'lines_count#D', 'degree_centrality#D', 'closeness_centrality#D', 'betweenness_centrality#D',
                    'min_transfers', 'shortest_path_length', 'min_station_count', 'num_effective_paths', 'avg_path_similarity' 
                ]

travel_cols = [
                    'TravelTime@D', 'TravelTime@D-1', 'TravelTime@D+1',
                    'FreeFlowTravelTime@D', 'FreeFlowTravelTime@D-1', 'FreeFlowTravelTime@D+1',
                    'TravelFlow@D', 'TravelFlow@D-1', 'TravelFlow@D+1',
                    'TemplateFlow@D', 'TemplateFlow@D-1', 'TemplateFlow@D+1'
                ]

weather_cols = [
                    'WindStatus@D', 'WindStatus@D-1', 'WindStatus@D+1',
                    'RainStatus@D', 'RainStatus@D-1', 'RainStatus@D+1',
                    'TyphoonAlert@D', 'TyphoonAlert@D-1', 'TyphoonAlert@D+1',
                    'RainstormAlert@D', 'RainstormAlert@D-1', 'RainstormAlert@D+1'
                ]

#%% attrs
feature_cols = index_cols + day_cols + hour_cols + network_cols + travel_cols + weather_cols
all_attrs = pd.Series([get_attr_name(col) for col in feature_cols]).drop_duplicates().tolist()
all_attrs = ['SupplyLoss' if x=='FreeFlowTravelTime' else x for x in all_attrs] # Convert FreeFlowTravelTime to SupplyLoss.

static_attrs = [
                    'Oindex', 'Dindex', 'ODindex',
                    'lines_count#O', 'degree_centrality#O', 'closeness_centrality#O', 'betweenness_centrality#O',
                    'lines_count#D', 'degree_centrality#D', 'closeness_centrality#D', 'betweenness_centrality#D',
                    'min_transfers', 'shortest_path_length', 'min_station_count', 'num_effective_paths', 'avg_path_similarity',
                    'Month', 'DayofWeek'
                ]

categorical_attrs = [
                        'Oindex', 'Dindex', 'ODindex',
                        'WindStatus', 'RainStatus', 
                        'TyphoonAlert', 'RainstormAlert',
                        'Month','DayofWeek','Hour'
                    ]

numeric_attrs = [
                    'TravelTime', 'SupplyLoss', 'TravelFlow', 'TemplateFlow', 
                    'lines_count#O', 'degree_centrality#O', 'closeness_centrality#O', 'betweenness_centrality#O',
                    'lines_count#D', 'degree_centrality#D', 'closeness_centrality#D', 'betweenness_centrality#D',
                    'min_transfers', 'shortest_path_length', 'min_station_count', 'num_effective_paths', 'avg_path_similarity'
                 ]

known_attrs = [
                    'Hour',
                    'TravelTime', 'SupplyLoss', 'TemplateFlow',
                    'WindStatus', 'RainStatus',
                    'TyphoonAlert', 'RainstormAlert'
               ]

le_attrs = ['Oindex', 'Dindex', 'ODindex', 'Month', 'DayofWeek', 'Hour']


#%% config

NETWORK_FEATURES_FOLDER = rf"{BASE_DIR}/Network Features"
network_periods = load_and_cache_network_features(NETWORK_FEATURES_FOLDER)

global_min = {}
global_max = {}

cols_station = ['lines_count', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality']
cols_od = ['min_transfers', 'shortest_path_length', 'min_station_count', 'num_effective_paths', 'avg_path_similarity']

historical_windows = 12
future_windows = 12

for df in network_periods:
    # station_features
    for col in cols_station:
        val_min = df['station_features'][col].min()
        val_max = df['station_features'][col].max()
        global_min[col] = min(global_min.get(col, np.inf), val_min)
        global_max[col] = max(global_max.get(col, -np.inf), val_max)
    
    # od_features
    for col in cols_od:
        val_min = df['od_features'][col].min()
        val_max = df['od_features'][col].max()
        global_min[col] = min(global_min.get(col, np.inf), val_min)
        global_max[col] = max(global_max.get(col, -np.inf), val_max)

config_known_stats = {
                           'Month': {'categories': np.arange(1, 13).tolist()},
                           'DayofWeek': {'categories': np.arange(0, 7).tolist()},
                           'Hour': {'categories': np.arange(0, 24).tolist()},
                           'Oindex': {'categories': sorted(network_periods[-1]['od_features']['Oindex'].unique().tolist())},
                           'Dindex': {'categories': sorted(network_periods[-1]['od_features']['Dindex'].unique().tolist())},
                           'ODindex': {'categories': sorted(network_periods[-1]['od_features']['ODindex'].unique().tolist())},
                           
                           'lines_count#O': {'min':global_min['lines_count'], 'max': global_max['lines_count']}, 
                           'degree_centrality#O': {'min':global_min['degree_centrality'], 'max': global_max['degree_centrality']}, 
                           'closeness_centrality#O': {'min':global_min['closeness_centrality'], 'max': global_max['closeness_centrality']}, 
                           'betweenness_centrality#O':  {'min':global_min['betweenness_centrality'], 'max': global_max['betweenness_centrality']}, 
                           'lines_count#D': {'min':global_min['lines_count'], 'max': global_max['lines_count']}, 
                           'degree_centrality#D': {'min':global_min['degree_centrality'], 'max': global_max['degree_centrality']}, 
                           'closeness_centrality#D': {'min':global_min['closeness_centrality'], 'max': global_max['closeness_centrality']}, 
                           'betweenness_centrality#D': {'min':global_min['betweenness_centrality'], 'max': global_max['betweenness_centrality']}, 
                           
                           'min_transfers': {'min':global_min['min_transfers'], 'max': global_max['min_transfers']}, 
                           'shortest_path_length': {'min':global_min['shortest_path_length'], 'max': global_max['shortest_path_length']}, 
                           'min_station_count': {'min':global_min['min_station_count'], 'max': global_max['min_station_count']}, 
                           'num_effective_paths':  {'min':global_min['num_effective_paths'], 'max': global_max['num_effective_paths']}, 
                           'avg_path_similarity': {'min':global_min['avg_path_similarity'], 'max': global_max['avg_path_similarity']}, 
                           
                           'WindStatus': {'categories': np.arange(0, 12).tolist()}, 
                           'RainStatus': {'categories': np.arange(0, 6).tolist()},
                           'TyphoonAlert': {'categories': np.arange(0, 6).tolist()},
                           'RainstormAlert': {'categories': np.arange(0, 6).tolist()},
                           
                           'TravelTime': {'min': 0, 'max': 0.2}, # Assume travel time is always above 3 minutes.
                           'SupplyLoss': {'min': 0, 'max': 1},
                           'TravelFlow': {'min': 0, 'max': 4000},
                           'TemplateFlow': {'min': 0, 'max': 4000}
                       }


