"""
Filename: stas_plot.py
Author: yellower
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
import seaborn as sns
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve,to_categorical
import pandas as pd
import torch
import scipy.stats as stats

if __name__ == '__main__':
    patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'编号': 'patient_id'}).sort_values(by='入院日期').reset_index(drop=True)
    complication_df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',
                                    index_col=0).drop_duplicates(subset=['patient_id'])
    patients_info_valid = patients_info.iloc[437:,:]
    patients_info_train = patients_info.iloc[:437,:]


    patients_info_complication = patients_info.merge(complication_df[['patient_id', 'is_lung', 'is_Kidney', 'is_Hypertension',
                                                         'is_Hyperglycemia', 'Others', 'disease', 'is_Hyperlipidemia']],
                                        on='patient_id')
    ending_df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0).drop_duplicates(subset=['patient_id'])
    patients_info_complication = patients_info_complication.sort_values(by='入院日期').reset_index(drop=True)
    patients_info_all = patients_info_complication.merge(ending_df[['patient_id', 'ending']], on='patient_id').sort_values(by='入院日期').reset_index(drop=True)

