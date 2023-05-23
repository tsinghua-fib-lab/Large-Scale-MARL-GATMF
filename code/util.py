import os
import pickle
import pandas as pd
import numpy as np

import constants

def load_data(MSA_name):
    print('Loading data...')
    MSA_NAME_FULL = constants.MSA_NAME_FULL_DICT[MSA_name]
    epic_data_root = '../data'
    data=dict()

    f = open(os.path.join(epic_data_root, MSA_name, '%s_2020-03-01_to_2020-05-02_processed.pkl'%MSA_NAME_FULL), 'rb') 
    poi_cbg_visits_list = pickle.load(f)
    f.close()
    data['poi_cbg_visits_list']=poi_cbg_visits_list

    d = pd.read_csv(os.path.join(epic_data_root,MSA_name, 'parameters_%s.csv' % MSA_name)) 
    poi_areas = d['feet'].values
    poi_dwell_times = d['median'].values
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    data['poi_areas']=poi_areas
    data['poi_times']=poi_dwell_times
    data['poi_dwell_time_correction_factors']=poi_dwell_time_correction_factors

    cbg_ids_msa = pd.read_csv(os.path.join(epic_data_root,MSA_name,'%s_cbg_ids.csv'%MSA_NAME_FULL)) 
    cbg_ids_msa.rename(columns={"cbg_id":"census_block_group"}, inplace=True)

    filepath = os.path.join(epic_data_root,"safegraph_open_census_data/data/cbg_b01.csv")
    cbg_agesex = pd.read_csv(filepath)
    cbg_age_msa = pd.merge(cbg_ids_msa, cbg_agesex, on='census_block_group', how='left')
    del cbg_agesex

    for i in range(3,25+1):
        male_column = 'B01001e'+str(i)
        female_column = 'B01001e'+str(i+24)
        cbg_age_msa[constants.DETAILED_AGE_LIST[i-3]] = cbg_age_msa.apply(lambda x : x[male_column]+x[female_column],axis=1)

    cbg_age_msa.rename(columns={'B01001e1':'Sum'},inplace=True)
    columns_of_interest = ['census_block_group','Sum'] + constants.DETAILED_AGE_LIST
    cbg_age_msa = cbg_age_msa[columns_of_interest].copy()

    cbg_age_msa.fillna(0,inplace=True)
    cbg_age_msa['Sum'] = cbg_age_msa['Sum'].apply(lambda x : x if x!=0 else 1)

    cbg_sizes = cbg_age_msa['Sum'].values
    cbg_sizes = np.array(cbg_sizes,dtype='int32')
    data['cbg_sizes']=cbg_sizes
    data['cbg_ages']=cbg_age_msa.to_numpy()[:,2:]/cbg_sizes[:,np.newaxis]

    cbg_death_rates_original = np.loadtxt(os.path.join(epic_data_root, MSA_name, 'cbg_death_rates_original_'+MSA_name))
    cbg_attack_rates_original = np.ones(cbg_death_rates_original.shape)

    attack_scale = 1
    cbg_attack_rates_scaled = cbg_attack_rates_original * attack_scale
    cbg_death_rates_scaled = cbg_death_rates_original * constants.death_scale_dict[MSA_name]
    data['cbg_attack_rates_scaled']=cbg_attack_rates_scaled
    data['cbg_death_rates_scaled']=cbg_death_rates_scaled

    return data