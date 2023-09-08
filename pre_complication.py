import pandas as pd
import scipy.ndimage
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re


def get_df_p_value_figure(df,df_column,figure_path,not_label,is_label):
    df_not = df[df.iloc[:,-1] == 0]
    df_is = df[df.iloc[:,-1] == 1]
    df_not_column = df_not[df_column]
    df_is_column = df_is[df_column]
    df_not_column = df_not_column[np.isfinite(df_not_column)]
    df_is_column = df_is_column[np.isfinite(df_is_column)]
    compare_static = stats.mannwhitneyu(df_not_column, df_is_column,alternative='two-sided').pvalue

    plt.figure(figsize=(6, 6))
    sns.kdeplot(df_not_column, color='r', label= not_label)
    sns.kdeplot(df_is_column, color='g', label= is_label)

    plt.title('{}:{}'.format(df_column, compare_static))
    plt.legend()
    plt.savefig(os.path.join(figure_path, ''.join([df_column, '.png'])))
    plt.close()

    return compare_static

if __name__ == '__main__':
    # patient_info = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',index_col=0)
    # df_CF_columns = patient_info.columns[3:-5]
    # sick_name = 'is_Kidney'
    # CF_file_pvalue_before = pd.DataFrame(columns=['CF','pvalue'])
    #
    # CF_file_pvalue_after = pd.DataFrame(columns=['CF', 'pvalue'])
    #
    # df_before = patient_info[patient_info.dis_day<=0]
    # df_after = patient_info[patient_info.dis_day > 0]
    # for df_CF_column in df_CF_columns:
    #     CF_file_pvalue_before_index = len(CF_file_pvalue_before)
    #     CF_file_pvalue_after_index = len(CF_file_pvalue_after)
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index,'CF'] = df_CF_column
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'CF'] = df_CF_column
    #     before_operation_is_sick = df_before.loc[df_before[sick_name]==1,df_CF_column]
    #     before_operation_is_sick_processed = before_operation_is_sick[np.isfinite(before_operation_is_sick)]
    #     before_operation_not_sick = df_before.loc[df_before[sick_name]!=1,df_CF_column]
    #     before_operation_not_sick_processed = before_operation_not_sick[np.isfinite(before_operation_not_sick)]
    #
    #     after_operation_is_sick = df_after.loc[df_after[sick_name] == 1, df_CF_column]
    #     after_operation_is_sick_processed = after_operation_is_sick[np.isfinite(after_operation_is_sick)]
    #     after_operation_not_sick = df_after.loc[df_after[sick_name] != 1, df_CF_column]
    #     after_operation_not_sick_processed = after_operation_not_sick[np.isfinite(after_operation_not_sick)]
    #
    #     compare_static_before = stats.mannwhitneyu(before_operation_is_sick_processed,before_operation_not_sick_processed,alternative='two-sided').pvalue
    #     compare_static_after = stats.mannwhitneyu(after_operation_is_sick_processed, after_operation_not_sick_processed,
    #                                                alternative='two-sided').pvalue
    #
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'pvalue'] = compare_static_before
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'pvalue'] = compare_static_after
    #
    # CF_file_pvalue_before.sort_values(by='pvalue').to_excel('./data/CF_file_pvalue_{}_Before.xlsx'.format(sick_name))
    # CF_file_pvalue_after.sort_values(by='pvalue').to_excel('./data/CF_file_pvalue_{}_After.xlsx'.format(sick_name))

    patient_info = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',index_col=0)
    patient_info_sick = patient_info[['patient_id','is_Kidney']].drop_duplicates(subset='patient_id')
    df_before_MR = pd.read_excel('./data/df_before_MR.xlsx',index_col=0).merge(patient_info_sick,on='patient_id')
    df_after_MR = pd.read_excel('./data/df_after_MR.xlsx',index_col=0).merge(patient_info_sick,on='patient_id')
    df_MR_columns = df_before_MR.columns[1:-1]
    sick_name = 'is_Kidney'
    MR_file_pvalue_before = pd.DataFrame(columns=['MR','pvalue'])
    MR_file_pvalue_after = pd.DataFrame(columns=['MR','pvalue'])
    for df_MR_column in df_MR_columns:
        MR_file_pvalue_before_index = len(MR_file_pvalue_before)
        MR_file_pvalue_after_index = len(MR_file_pvalue_after)
        MR_file_pvalue_before.loc[MR_file_pvalue_before_index,'MR'] = df_MR_column
        MR_file_pvalue_after.loc[MR_file_pvalue_after_index, 'MR'] = df_MR_column
        before_operation_is_sick = df_before_MR.loc[df_before_MR[sick_name]==1,df_MR_column]
        before_operation_is_sick_processed = before_operation_is_sick[np.isfinite(before_operation_is_sick)]
        before_operation_not_sick = df_before_MR.loc[df_before_MR[sick_name]!=1,df_MR_column]
        before_operation_not_sick_processed = before_operation_not_sick[np.isfinite(before_operation_not_sick)]

        after_operation_is_sick = df_after_MR.loc[df_after_MR[sick_name] == 1, df_MR_column]
        after_operation_is_sick_processed = after_operation_is_sick[np.isfinite(after_operation_is_sick)]
        after_operation_not_sick = df_after_MR.loc[df_after_MR[sick_name] != 1,df_MR_column]
        after_operation_not_sick_processed = after_operation_not_sick[np.isfinite(after_operation_not_sick)]

        compare_static_before = stats.mannwhitneyu(before_operation_is_sick_processed,
                                                   before_operation_not_sick_processed, alternative='two-sided').pvalue
        compare_static_after = stats.mannwhitneyu(after_operation_is_sick_processed, after_operation_not_sick_processed,
                                                       alternative='two-sided').pvalue

        MR_file_pvalue_before.loc[MR_file_pvalue_before_index, 'pvalue'] = compare_static_before
        MR_file_pvalue_after.loc[MR_file_pvalue_after_index, 'pvalue'] = compare_static_after

    MR_file_pvalue_before.sort_values(by='pvalue').to_excel('./data/MR_file_pvalue_{}_Before.xlsx'.format(sick_name))
    MR_file_pvalue_after.sort_values(by='pvalue').to_excel('./data/MR_file_pvalue_{}_After.xlsx'.format(sick_name))









