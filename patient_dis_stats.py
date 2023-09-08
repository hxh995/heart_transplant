import pandas as pd
import numpy as np
import openpyxl
import os
import re
from datetime import datetime
import time
from interval import Interval
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False
def column_process(x):
    if not pd.isnull(x) :
        if type(x) == int or type(x) == float:
            return x
        elif ',' in x:
            try:
               np.mean([float (i) for i in x.split(',')])
            except:
                x = str(x)
                if ('<' in x) or ('>' in x) or (';' in x) or ('&gt;' in x) or ('&lt;' in x):
                    return np.mean([float(i.replace('<', '').replace('>', '').replace('&gt;','').replace('&lt;','').replace(';','')) for i in x.split(',')])
                else:
                    print(x)
        else:
            try:
                x=float(x)
                return x
            except:
                x=str(x)
                if ('<' in x) or ('>' in x) or (';' in x) or ('&gt;' in x) or ('&lt;' in x):
                    return float(x.replace('<','').replace('>','').replace('&gt;','').replace('&lt;','').replace(';',''))
                else:
                    print(x)
    else:
        return np.nan
def is_in_Interval(x,CF_interval):
    if pd.isnull(x):
        return np.nan
    if x not in CF_interval:
        return 1
    else:
        return 0
## 获得指定时间段部分超出正常值的指标
def get_df_discharge_static(df_discharge):
    df_discharge_copy = df_discharge.copy()
    df_discharge_copy['MCHC'] = df_discharge.MCHC.apply(column_process)
    df_CF_columns = df_discharge_copy.columns[3:]
    CF = pd.read_excel('./data/data_processed_format.xlsx')
    CF = CF[~CF['参考区间'].isnull()]
    CF_specific = CF[(CF['参考区间'].str.contains('-')) & (~CF['参考区间'].str.contains('<|>'))]['指标名'].values
    # df_CF_drop=['尿亚硝酸盐','尿比重','尿糖','尿沉渣定量分析','大便颜色','大便性状','大便红细胞','尿酮体','抗甲状腺过氧化物酶抗体','尿维生素C','尿胆原','巨细胞病毒IgM',
    #             '乙肝表面抗体','快速血浆反应素实验','前S1抗原','结核T-SPOT','38KD','快速血浆反应素实验','梅毒特异性抗体',
    #             '抗HIV抗体','HCV-Ab','风疹病毒IgG','糖类抗原CA199','鳞状细胞癌相关抗原','细胞角蛋白19片段','巨细胞病毒IgG',
    #             'C-反应蛋白','C-反应蛋白','尿PH值','尿潜血','弓形虫抗体IgG','乙肝核心抗体','大便吞噬细胞','微量白蛋白','尿蛋白']
    df_CF_drop = ['尿比重', '尿PH值', 'cBase(B)', 'cBase(Ecf)']
    for df_CF_column in df_CF_columns:
        if (df_CF_column in CF_specific) and (df_CF_column not in df_CF_drop):
            # print(df_CF_column)
            df_discharge_copy[df_CF_column] = df_discharge[df_CF_column].apply(column_process)
    df_discharge_static = df_discharge_copy.copy().iloc[:, :3]
    for df_CF_column in df_CF_columns:
        if (df_CF_column in CF_specific) and (df_CF_column not in df_CF_drop):
            CF_reference = CF.loc[CF['指标名'] == df_CF_column, '参考区间'].values[0].split('-')
            CF_interval = Interval(float(CF_reference[0]), float(CF_reference[1]))
            df_discharge_static = pd.concat([df_discharge_static, df_discharge_copy[df_CF_column].apply(lambda x: is_in_Interval(x, CF_interval)).rename(df_CF_column)], axis=1)
if __name__ == '__main__':
    # df=pd.read_excel('./data/patient_dis_day_CF_lung.xlsx',index_col=0)
    # df=df[df.columns[df.isna().sum().values / len(df) < 0.9]]
    # #df_discharge_static = get_df_discharge_static(df[df.dis_day<=0])
    # CF = pd.read_excel('./data/data_processed_format.xlsx',index_col=0)
    # df_copy = df.iloc[:,:3].copy()
    # df_CF_columns = df.columns[3:]
    # for df_CF_column in df_CF_columns:
    #     print(df_CF_column)
    #     df_copy = pd.concat([df_copy,df[df_CF_column].apply(column_process)],axis=1)
    # df_operation = df_copy[df.dis_day<=0]
    # df_after_operation = df_copy[df.dis_day >0]
    # CF_file_median = pd.DataFrame(columns=['CF','Reference_interval','before_operation','after_operation'])
    # CF_file_median_index = len(CF_file_median)
    # CF_file_pvalue = pd.DataFrame(columns=['CF','pvalue'])
    # CF_file_pvalue_index = len(CF_file_pvalue)
    # CF_file_mean = pd.DataFrame(columns=['CF','Reference_interval','before_operation','after_operation'])
    # CF_file_mean_index=len(CF_file_mean)
    # figure_path='./data/figures_lung'
    # for df_CF_column in df_CF_columns:
    #     CF_file_median.loc[CF_file_median_index,'CF']=df_CF_column
    #     CF_file_pvalue.loc[CF_file_pvalue_index,'CF'] = df_CF_column
    #     CF_file_mean.loc[CF_file_median_index, 'CF'] = df_CF_column
    #     if len(CF.loc[CF['指标名'] == df_CF_column, '参考区间']):
    #         CF_interval=CF.loc[CF['指标名'] == df_CF_column, '参考区间'].values[0]
    #     else:
    #         CF_interval=np.nan
    #     CF_file_median.loc[CF_file_median_index,'Reference_interval'] = CF_interval
    #     CF_file_median.loc[CF_file_median_index,'before_operation'] = np.nanmedian(df_operation.loc[:,df_CF_column].values)
    #     CF_file_median.loc[CF_file_median_index,'after_operation'] = np.nanmedian(df_after_operation.loc[:,df_CF_column].values)
    #     CF_file_mean.loc[CF_file_median_index, 'Reference_interval'] = CF_interval
    #     CF_file_mean.loc[CF_file_median_index, 'before_operation'] = np.nanmean(df_operation.loc[:, df_CF_column].values)
    #     CF_file_mean.loc[CF_file_median_index, 'after_operation'] = np.nanmean(df_after_operation.loc[:, df_CF_column].values)
    #     before_operation = df_operation.loc[:,df_CF_column].values
    #     after_operation =df_after_operation.loc[:,df_CF_column].values
    #     before_operation = before_operation[np.isfinite(before_operation)]
    #     after_operation = after_operation[np.isfinite(after_operation)]
    #     compare_static=stats.mannwhitneyu(before_operation, after_operation,alternative='two-sided').pvalue
    #     CF_file_pvalue.loc[CF_file_pvalue_index, 'pvalue'] = compare_static
    #     plt.figure(figsize=(6, 6))
    #     sns.kdeplot(before_operation,color='g',label='before_operation')
    #     sns.kdeplot(after_operation,color='r',label='after_operation')
    #     plt.title('{}:{}'.format(df_CF_column,compare_static))
    #     plt.legend()
    #     plt.savefig(os.path.join(figure_path,''.join([df_CF_column,'.png'])))
    #     plt.close()
    #     CF_file_median_index = CF_file_median_index + 1
    #     CF_file_pvalue_index = CF_file_pvalue_index + 1
    #     CF_file_mean_index = CF_file_mean_index + 1
    # CF_file_median.to_excel('./data/CF_file_median_lung.xlsx')
    # CF_file_pvalue.to_excel('./data/CF_file_pvalue_lung.xlsx')
    # CF_file_mean.to_excel('./data/CF_file_mean_lung.xlsx')

    ## 获取手术前手术后  CF pvalue
    # df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx', index_col=0)
    # df = df[df.columns[df.isna().sum().values / len(df) < 0.9]]
    # # df_discharge_static = get_df_discharge_static(df[df.dis_day<=0])
    # CF = pd.read_excel('./data/data_processed_format.xlsx', index_col=0)
    # df_copy = df.iloc[:, :3].copy()
    # df_CF_columns = df.columns[3:]
    # for df_CF_column in df_CF_columns:
    #     print(df_CF_column)
    #     df_copy = pd.concat([df_copy, df[df_CF_column].apply(column_process)], axis=1)
    # df_operation = df_copy[df.dis_day <= 0]
    # df_after_operation = df_copy[df.dis_day > 0]
    # CF_file_pvalue_before = pd.DataFrame(columns=['CF', 'pvalue'])
    # CF_file_pvalue_before_index = len(CF_file_pvalue_before)
    # CF_file_pvalue_after = pd.DataFrame(columns=['CF', 'pvalue'])
    # CF_file_pvalue_after_index = len(CF_file_pvalue_after)
    # figure_before_path = './data/figures_lung/CF_lung_before'
    # figure_after_path  = './data/figures_lung/CF_lung_after'
    # for df_CF_column in df_CF_columns[:-5]:
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'CF'] = df_CF_column
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'CF'] = df_CF_column
    #     if len(CF.loc[CF['指标名'] == df_CF_column, '参考区间']):
    #         CF_interval = CF.loc[CF['指标名'] == df_CF_column, '参考区间'].values[0]
    #         if type(CF_interval) == str:
    #             if '-' in CF_interval:
    #                 CF_interval_min = float(CF_interval.split('-')[0])
    #                 CF_interval_max = float(CF_interval.split('-')[1])
    #             elif '<' in CF_interval:
    #                 CF_interval_min = 0
    #                 CF_interval_max = float(CF_interval.split('<')[1])
    #             else:
    #                 CF_interval_min = 0
    #                 CF_interval_max = float(CF_interval.split('＜')[1])
    #
    #         else:
    #             CF_interval = np.nan
    #
    #     else:
    #         CF_interval = np.nan
    #
    #     y = np.arange(0.0, 1, 0.01)
    #
    #     before_operation_is_lung = df_operation.loc[df_operation['is_lung'] == 1, df_CF_column].values
    #     before_operation_is_lung = before_operation_is_lung[np.isfinite(before_operation_is_lung)]
    #     before_operation_not_lung = df_operation.loc[df_operation['is_lung'] != 1, df_CF_column].values
    #     before_operation_not_lung  = before_operation_not_lung[np.isfinite(before_operation_not_lung)]
    #
    #     after_operation_is_lung = df_after_operation.loc[df_after_operation['is_lung'] == 1, df_CF_column].values
    #     after_operation_is_lung = after_operation_is_lung[np.isfinite(after_operation_is_lung)]
    #     after_operation_not_lung = df_after_operation.loc[df_after_operation['is_lung'] != 1, df_CF_column].values
    #     after_operation_not_lung = after_operation_not_lung[np.isfinite(after_operation_not_lung)]
    #
    #     compare_static_before_operation = stats.mannwhitneyu(before_operation_is_lung, before_operation_not_lung, alternative='two-sided').pvalue
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'pvalue'] = compare_static_before_operation
    #     plt.figure(figsize=(6, 6))
    #     sns.kdeplot(before_operation_is_lung, color='r', label='before_operation_is_lung')
    #     sns.kdeplot(before_operation_not_lung, color='g', label='before_operation_not_lung')
    #
    #     if CF_interval:
    #         plt.fill_betweenx(y,CF_interval_min,CF_interval_max,facecolor='green',alpha=0.3)
    #     plt.title('{}:{}'.format(df_CF_column, compare_static_before_operation))
    #     plt.legend()
    #     plt.savefig(os.path.join(figure_before_path, ''.join([df_CF_column, '.png'])))
    #     plt.close()
    #
    #     compare_static_after_operation = stats.mannwhitneyu(after_operation_is_lung, after_operation_not_lung,
    #                                                          alternative='two-sided').pvalue
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'pvalue'] = compare_static_after_operation
    #     plt.figure(figsize=(6, 6))
    #     sns.kdeplot(after_operation_is_lung, color='r', label='after_operation_is_lung')
    #     sns.kdeplot(after_operation_not_lung, color='g', label='after_operation_not_lung')
    #
    #     if CF_interval:
    #         plt.fill_betweenx(y,CF_interval_min,CF_interval_max,facecolor='green',alpha=0.3)
    #     plt.title('{}:{}'.format(df_CF_column, compare_static_after_operation))
    #     plt.legend()
    #     plt.savefig(os.path.join(figure_after_path, ''.join([df_CF_column, '.png'])))
    #     plt.close()
    #
    #
    #     CF_file_pvalue_before_index = CF_file_pvalue_before_index + 1
    #     CF_file_pvalue_after_index = CF_file_pvalue_after_index + 1
    #
    #
    #
    # CF_file_pvalue_before.sort_values(by='pvalue').to_excel('./data/CF_file_pvalue_lung_Before.xlsx')
    # CF_file_pvalue_after.sort_values(by='pvalue').to_excel('./data/CF_file_pvalue_lung_After.xlsx')

    df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0)
    df = df[df.columns[df.isna().sum().values / len(df) < 0.9]]
    # df_discharge_static = get_df_discharge_static(df[df.dis_day<=0])
    CF = pd.read_excel('./data/data_processed_format.xlsx', index_col=0)
    df_copy = df.iloc[:, :3].copy()
    df_CF_columns = df.columns[3:]
    for df_CF_column in df_CF_columns:
        print(df_CF_column)
        df_copy = pd.concat([df_copy, df[df_CF_column].apply(column_process)], axis=1)
    df_operation = df_copy[df.dis_day <= 0]
    df_after_operation = df_copy[df.dis_day > 0]
    CF_file_pvalue_before = pd.DataFrame(columns=['CF', 'pvalue'])
    CF_file_pvalue_before_index = len(CF_file_pvalue_before)
    CF_file_pvalue_after = pd.DataFrame(columns=['CF', 'pvalue'])
    CF_file_pvalue_after_index = len(CF_file_pvalue_after)
    figure_before_path = './data/figures_stats/CF_dead_before'
    figure_after_path = './data/figures_stats/CF_dead_after'
    for df_CF_column in df_CF_columns[:-5]:
        CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'CF'] = df_CF_column
        CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'CF'] = df_CF_column
        if len(CF.loc[CF['指标名'] == df_CF_column, '参考区间']):
            CF_interval = CF.loc[CF['指标名'] == df_CF_column, '参考区间'].values[0]
            if type(CF_interval) == str:
                if '-' in CF_interval:
                    CF_interval_min = float(CF_interval.split('-')[0])
                    CF_interval_max = float(CF_interval.split('-')[1])
                elif '<' in CF_interval:
                    CF_interval_min = 0
                    CF_interval_max = float(CF_interval.split('<')[1])
                else:
                    CF_interval_min = 0
                    CF_interval_max = float(CF_interval.split('＜')[1])

            else:
                CF_interval = np.nan

        else:
            CF_interval = np.nan

        y = np.arange(0.0, 1, 0.01)

        before_operation_is_lung = df_operation.loc[df_operation['ending'] == 1, df_CF_column].values
        before_operation_is_lung = before_operation_is_lung[np.isfinite(before_operation_is_lung)]
        before_operation_not_lung = df_operation.loc[df_operation['ending'] != 1, df_CF_column].values
        before_operation_not_lung = before_operation_not_lung[np.isfinite(before_operation_not_lung)]

        after_operation_is_lung = df_after_operation.loc[df_after_operation['ending'] == 1, df_CF_column].values
        after_operation_is_lung = after_operation_is_lung[np.isfinite(after_operation_is_lung)]
        after_operation_not_lung = df_after_operation.loc[df_after_operation['ending'] != 1, df_CF_column].values
        after_operation_not_lung = after_operation_not_lung[np.isfinite(after_operation_not_lung)]

        compare_static_before_operation = stats.mannwhitneyu(before_operation_is_lung, before_operation_not_lung,
                                                             alternative='two-sided').pvalue
        CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'pvalue'] = compare_static_before_operation
        plt.figure(figsize=(6, 6))
        sns.kdeplot(before_operation_is_lung, color='r', label='before_operation_is_dead')
        sns.kdeplot(before_operation_not_lung, color='g', label='before_operation_not_dead')

        if CF_interval:
            plt.fill_betweenx(y, CF_interval_min, CF_interval_max, facecolor='green', alpha=0.3)
        plt.title('{}:{}'.format(df_CF_column, compare_static_before_operation))
        plt.legend()
        plt.savefig(os.path.join(figure_before_path, ''.join([df_CF_column, '.png'])))
        plt.close()

        compare_static_after_operation = stats.mannwhitneyu(after_operation_is_lung, after_operation_not_lung,
                                                            alternative='two-sided').pvalue
        CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'pvalue'] = compare_static_after_operation
        plt.figure(figsize=(6, 6))
        sns.kdeplot(after_operation_is_lung, color='r', label='after_operation_is_dead')
        sns.kdeplot(after_operation_not_lung, color='g', label='after_operation_not_dead')

        if CF_interval:
            plt.fill_betweenx(y, CF_interval_min, CF_interval_max, facecolor='green', alpha=0.3)
        plt.title('{}:{}'.format(df_CF_column, compare_static_after_operation))
        plt.legend()
        plt.savefig(os.path.join(figure_after_path, ''.join([df_CF_column, '.png'])))
        plt.close()

        CF_file_pvalue_before_index = CF_file_pvalue_before_index + 1
        CF_file_pvalue_after_index = CF_file_pvalue_after_index + 1

    CF_file_pvalue_before.sort_values(by='pvalue').to_excel('./data/CF_file_pvalue_dead_Before.xlsx')
    CF_file_pvalue_after.sort_values(by='pvalue').to_excel('./data/CF_file_pvalue_dead_After.xlsx')












