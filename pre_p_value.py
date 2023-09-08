"""
Filename: pre_p_value.py
Author: yellower
"""
import math
import pandas as pd
import scipy.ndimage
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import re

if __name__ == '__main__' :
    # sick_names = ['is_Hyperglycemia', 'is_Hyperlipidemia', 'is_Hypertension']
    # py_type = 'bf'
    # for sick_name in sick_names:
    #     stats_df = pd.read_excel('./data/stats_data/CF_{}_{}.xlsx'.format(sick_name,py_type),index_col=0)
    #     stats_df = stats_df.sort_values(by='p_value')
    #     stats_df['rank_value'] = range(1,len(stats_df)+1)
    #     m = len(stats_df)
    #     stats_df['q_value'] = stats_df.apply(lambda x:(x.p_value*m)/x.rank_value,axis=1)
    #     stats_df.to_excel('./data/stats_data/CF_{}_{}_adjust.xlsx'.format(sick_name,py_type))

    # sick_name = 'is_Hyperglycemia'
    # df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',index_col=0)
    # df_operation = df[df.dis_day <= 0]
    # df_after_operation = df[df.dis_day > 0]
    # CF_file_pvalue_before = pd.DataFrame(columns=['CF', 'pvalue','fold_change'])
    #
    # CF_file_pvalue_after = pd.DataFrame(columns=['CF', 'pvalue','fold_change'])
    # # fold_change: 1/0
    # df_CF_columns = df.columns[3:-7]
    # for df_CF_column in df_CF_columns:
    #     try:
    #         CF_file_pvalue_before_index = len(CF_file_pvalue_before)
    #         CF_file_pvalue_after_index = len(CF_file_pvalue_after)
    #         CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'CF'] = df_CF_column
    #         CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'CF'] = df_CF_column
    #
    #
    #         before_operation_is_lung = df_operation.loc[df_operation[sick_name] == 1, df_CF_column].values
    #         before_operation_is_lung = before_operation_is_lung[np.isfinite(before_operation_is_lung)]
    #         before_operation_not_lung = df_operation.loc[df_operation[sick_name] != 1, df_CF_column].values
    #         before_operation_not_lung = before_operation_not_lung[np.isfinite(before_operation_not_lung)]
    #
    #         after_operation_is_lung = df_after_operation.loc[df_after_operation[sick_name] == 1, df_CF_column].values
    #         after_operation_is_lung = after_operation_is_lung[np.isfinite(after_operation_is_lung)]
    #         after_operation_not_lung = df_after_operation.loc[df_after_operation[sick_name] != 1, df_CF_column].values
    #         after_operation_not_lung = after_operation_not_lung[np.isfinite(after_operation_not_lung)]
    #
    #         compare_static_before_operation = stats.mannwhitneyu(before_operation_is_lung, before_operation_not_lung,
    #                                                              alternative='two-sided').pvalue
    #         CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'pvalue'] = compare_static_before_operation
    #         CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'fold_change'] = before_operation_is_lung.mean()/ before_operation_not_lung.mean()
    #
    #         compare_static_after_operation = stats.mannwhitneyu(after_operation_is_lung, after_operation_not_lung,
    #                                                             alternative='two-sided').pvalue
    #         CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'pvalue'] = compare_static_after_operation
    #         CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'fold_change'] = after_operation_is_lung.mean() / after_operation_not_lung.mean()
    #     except Exception as e:
    #         print(df_CF_column)
    #         print(e)
    #
    #
    # CF_file_pvalue_before = CF_file_pvalue_before.sort_values(by='pvalue')
    # CF_file_pvalue_after = CF_file_pvalue_after.sort_values(by='pvalue')
    #
    # CF_file_pvalue_before['rank_value'] = range(1,len(CF_file_pvalue_before)+1)
    # CF_file_pvalue_after['rank_value'] = range(1, len(CF_file_pvalue_after) + 1)
    # m = len(CF_file_pvalue_before)
    # CF_file_pvalue_before['q_value'] = CF_file_pvalue_before.apply(lambda x: (x.pvalue * m) /x.rank_value,axis=1)
    # m = len(CF_file_pvalue_after)
    # CF_file_pvalue_after['q_value'] = CF_file_pvalue_after.apply(lambda x: (x.pvalue * m) / x.rank_value, axis=1)
    #
    # CF_file_pvalue_before.to_excel('./data/stats_data/CF_{}_bf_stats.xlsx'.format(sick_name))
    # CF_file_pvalue_after.to_excel('./data/stats_data/CF_{}_multi_stats.xlsx'.format(sick_name ))


    sick_names = ['is_lung','is_Kidney','ending','disease','is_Hyperglycemia','is_Hyperlipidemia','is_Hypertension']
    py_type = 'bf'
    pred_cytoscope = pd.DataFrame(columns=['name','node','log_q_value','cov'])
    pred_label_cytoscope = pd.DataFrame(columns=['name', 'node', 'log_q_value','cov'])
    for sick_name in sick_names:
        print(sick_name)
        CF_pred_df = pd.read_excel('./data/stats_data/CF_{}_{}.xlsx'.format(sick_name,py_type),index_col=0)
        CF_label_df = pd.read_excel('./data/stats_data/CF_{}_{}_stats.xlsx'.format(sick_name,py_type),index_col=0)

        MR_pred_df = pd.read_excel('./data/stats_data/MR_{}_{}.xlsx'.format(sick_name,py_type),index_col=0)
        MR_label_df = pd.read_excel('./data/stats_data/MR_{}_{}_stats.xlsx'.format(sick_name, py_type), index_col=0)

        NLP_pred_df = pd.read_excel('./data/stats_data/NLP_{}_{}.xlsx'.format(sick_name,py_type),index_col=0)
        NLP_label_df = pd.read_excel('./data/stats_data/NLP_{}_{}_stats.xlsx'.format(sick_name, py_type), index_col=0)

        pred_df = pd.concat([CF_pred_df.rename(columns={'CF_name':'name'}),MR_pred_df.rename(columns={'MR_name':'name'}),NLP_pred_df.rename(columns={'NLP_name':'name'})]).sort_values(by='p_value')
        label_df = pd.concat([CF_label_df.rename(columns={'CF':'name'}),MR_label_df.rename(columns={'MR_name':'name'}),NLP_label_df.rename(columns={'NLP_name':'name'})]).sort_values(by='pvalue')

        #pred_df = pd.concat([CF_pred_df.rename(columns={'CF_name': 'name'}), MR_pred_df.rename(columns={'MR_name': 'name'})]).sort_values(by='p_value')
        #label_df = pd.concat([CF_label_df.rename(columns={'CF': 'name'}), MR_label_df.rename(columns={'MR_name': 'name'})]).sort_values(by='pvalue')



        pred_df['rank_value'] = range(1,len(pred_df)+1)
        label_df['rank_value'] = range(1,len(label_df)+1)
        m = len(pred_df)
        pred_df['q_value'] = pred_df.apply(lambda x: (x.p_value * m) /x.rank_value,axis=1)
        m = len(label_df)
        label_df['q_value'] = label_df.apply(lambda x: (x.pvalue * m) /x.rank_value,axis=1)
        pred_df['log_q_value'] = pred_df.q_value.apply(lambda x:-math.log10(x))

        try :
            pred_df_q_value = pred_df[pred_df.q_value<0.05].copy()
            label_df_q_value = label_df[label_df.q_value<0.05]
            pred_df_q_value.loc[:, 'node'] = sick_name
            pred_cytoscope = pd.concat([pred_cytoscope, pred_df_q_value[['name', 'node', 'log_q_value','cov']]])
        except Exception as e:
            print(len(pred_df_q_value))

        # print(sick_name)
        # print('pred_q_value < 0.05:')
        # print(len(pred_df_q_value))
        # print('label_q_value < 0.05:')
        # print(len(label_df_q_value))
        try:
            pred_label_q_value = pred_df_q_value[pred_df_q_value['name'].isin(label_df_q_value['name'])].copy()
            pred_label_q_value.loc[:,'node'] = sick_name
            print('pred_label')
            print(len(pred_label_q_value))
            pred_label_cytoscope = pd.concat([pred_label_cytoscope,pred_label_q_value[['name','node','log_q_value','cov']]])
        except Exception as e:
            print(e)
            print(sick_name)

    pred_cytoscope['group'] =  pred_cytoscope.apply(lambda x: "group_1" if x['cov']>0 else "group_2",axis=1)
    pred_label_cytoscope['group'] = pred_label_cytoscope.apply(lambda x: "group_1" if x['cov'] > 0 else "group_2", axis=1)

    pred_cytoscope.to_csv('./data/stats_data/pred_cytoscope_{}_all_NLP.txt'.format(py_type),sep='\t',index=False)
    pred_label_cytoscope.to_csv('./data/stats_data/pred_label_cytoscope_{}_all_NLP.txt'.format(py_type),sep='\t',index=False)










