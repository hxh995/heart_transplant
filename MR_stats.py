"""
Filename: MR_stats.py
Author: yellower
"""
import numpy as np
import torch
from collections import Counter

import pandas as pd
import torch.nn as nn
import scipy.stats as stats


if __name__ == '__main__' :
    sick_names = ['is_lung','disease','is_Kidney','is_Hypertension','is_Hyperlipidemia','is_Hyperglycemia','ending']
    MR_before_bf = pd.read_excel('./data/df_MR_before_ending.xlsx', index_col=0)
    MR_before_columns = MR_before_bf.columns[1:-8]

    for sick_name in sick_names:
        print(sick_name)
        MR_file_pvalue_before = pd.DataFrame(columns=['MR_name', 'pvalue'])
        for df_CF_column in MR_before_columns:
            try:
                MR_file_pvalue_before_index = len(MR_file_pvalue_before)
                MR_file_pvalue_before.loc[MR_file_pvalue_before_index, 'MR_name'] = df_CF_column

                before_operation_is_lung = MR_before_bf.loc[MR_before_bf[sick_name] == 1, df_CF_column].values
                before_operation_is_lung = before_operation_is_lung[np.isfinite(before_operation_is_lung)]
                before_operation_not_lung = MR_before_bf.loc[MR_before_bf[sick_name] != 1, df_CF_column].values
                before_operation_not_lung = before_operation_not_lung[np.isfinite(before_operation_not_lung)]
                compare_static_before_operation = stats.mannwhitneyu(before_operation_is_lung, before_operation_not_lung,
                                                                     alternative='two-sided').pvalue
                MR_file_pvalue_before.loc[MR_file_pvalue_before_index, 'pvalue'] = compare_static_before_operation


            except Exception as e:
                print(df_CF_column)
                print(e)
        MR_file_pvalue_before = MR_file_pvalue_before.sort_values(by='pvalue')
        MR_file_pvalue_before.to_excel('./data/stats_data/MR_{}_bf_stats.xlsx'.format(sick_name))
