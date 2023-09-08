import pandas as pd
import numpy as np
import openpyxl
import os
import re
from datetime import datetime
import time
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus']=False

def merge_ending(df,patients_info):

    df = df.merge(patients_info[['patient_id', '出院时结局']], on='patient_id')
    df = df[df['出院时结局'].isin(['死亡', '临床治愈', '好转'])]
    df['出院时结局'] = df['出院时结局'].apply(lambda x: 0 if x == '死亡' else 1)
    return df
def merge_sick(df,is_sick_patient_id):
    df['is_sick'] = 0
    df.loc[df.patient_id.isin(is_sick_patient_id),'is_sick'] = 1
    return df
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
    df = pd.DataFrame(columns = ['patient_id','time','item','value'])
    CF_df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx',index_col=0)
    CF_df_before = CF_df[(CF_df.dis_day<=0) & (CF_df.dis_day>-180)]
    CF_df_after = CF_df[(CF_df.dis_day>0) & (CF_df.dis_day<360)]
    patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'编号': 'patient_id'})
    patients_info_is_lung = patients_info[patients_info['术后并发症'].str.contains('肺')].patient_id.unique()
    patients_info_is_dead = CF_df[CF_df.ending==0].patient_id.unique()

    # CF_df_complication = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',index_col=0)
    reg_patient_id = '(\d).*'
    with open('./data/OCR.txt', "r") as f:  # 设置文件对象
        str = f.readlines()  # 可以是随便对文件的操作
        # print(str)
        for line in str:
            # print(line.rsplit())
            first,second,third = line.rsplit()
            print(re.findall('\d+', first)[0])
            df_index = len(df)
            df.loc[df_index,'patient_id'] = re.findall('\d+', first)[0]
            df.loc[df_index,'time'] = re.findall(r".*(术.).*", first)[0]
            df.loc[df_index,'item'] = second
            df.loc[df_index, 'value']  = float(third)

    df['patient_id'] = df.patient_id.apply(lambda x: int(x))
    df_before = df[df.time == '术前'].pivot_table(index=['patient_id'],columns='item',values='value',aggfunc=np.mean).reset_index()
    df_after = df[df.time == '术后'].pivot_table(index=['patient_id'],columns='item',values='value',aggfunc=np.mean).reset_index()
    # df_before = merge_ending(df_before,patients_info)
    # df_after = merge_ending(df_after,patients_info)
    # df_columns = df_before.columns[1:-1]
    # CF_file_pvalue_before = pd.DataFrame(columns=['item','pvalue'])
    # CF_file_pvalue_after = pd.DataFrame(columns=['item', 'pvalue'])

    # figure_before_path = './data/figures_stats/MR_before'
    # figure_after_path = './data/figures_stats/MR_after'
    # for df_column in df_columns:
    #     compare_static_before_operation = get_df_p_value_figure(df_before,df_column,figure_before_path,'dead','cured')
    #     CF_file_pvalue_before_index = len(CF_file_pvalue_before)
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'item'] = df_column
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'pvalue'] = compare_static_before_operation
    #
    #     compare_static_after_operation = get_df_p_value_figure(df_after, df_column, figure_after_path, 'dead', 'cured')
    #     CF_file_pvalue_after_index = len(CF_file_pvalue_after)
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'item'] = df_column
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'pvalue'] = compare_static_after_operation
    #
    # CF_file_pvalue_before.sort_values(by='pvalue').to_excel('./data/MR_file_pvalue_dead_before.xlsx')
    # CF_file_pvalue_after.sort_values(by='pvalue').to_excel('./data/MR_file_pvalue_dead_after.xlsx')

    # df_before = merge_sick(df_before, patients_info_is_lung)
    # df_after = merge_sick(df_after, patients_info_is_lung)
    # df_columns = df_before.columns[1:-1]
    # CF_file_pvalue_before = pd.DataFrame(columns=['item', 'pvalue'])
    # CF_file_pvalue_after = pd.DataFrame(columns=['item', 'pvalue'])
    #
    # figure_before_path = './data/figures_stats/MR_sick_before'
    # figure_after_path = './data/figures_stats/MR_sick_after'
    # for df_column in df_columns:
    #     compare_static_before_operation = get_df_p_value_figure(df_before,df_column,figure_before_path,'not_lung','is_lung')
    #     CF_file_pvalue_before_index = len(CF_file_pvalue_before)
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'item'] = df_column
    #     CF_file_pvalue_before.loc[CF_file_pvalue_before_index, 'pvalue'] = compare_static_before_operation
    #
    #     compare_static_after_operation = get_df_p_value_figure(df_after, df_column, figure_after_path, 'not_lung', 'is_lung')
    #     CF_file_pvalue_after_index = len(CF_file_pvalue_after)
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'item'] = df_column
    #     CF_file_pvalue_after.loc[CF_file_pvalue_after_index, 'pvalue'] = compare_static_after_operation
    #
    # CF_file_pvalue_before.sort_values(by='pvalue').to_excel('./data/MR_file_pvalue_lung_before.xlsx')
    # CF_file_pvalue_after.sort_values(by='pvalue').to_excel('./data/MR_file_pvalue_lung_after.xlsx')



    df_columns = df_before.columns[1:]
    CF_columns = [i for i in CF_df_before.columns[3:-1]]
    MR_columns = [i for i in df_before.columns[1:]]
    DF_before = CF_df_before.groupby('patient_id')[CF_columns].mean().reset_index().merge(df_before, on='patient_id', how='inner')
    DF_after = CF_df_after.groupby('patient_id')[CF_columns].mean().reset_index().merge(df_after, on='patient_id', how='inner')

    df_MR_CF_p_value_before = pd.DataFrame(columns=['MR_item','CF','correlation','p_value'])
    df_MR_CF_p_value_after = pd.DataFrame(columns=['MR_item','CF','correlation','p_value'])
    for df_column in df_columns:
        MR_column_value = DF_before[df_column].values
        for CF_column in CF_columns:
            CF_column_value = DF_before[CF_column].values
            correlation ,p_value = stats.spearmanr(MR_column_value, CF_column_value,nan_policy='omit')
            index = len(df_MR_CF_p_value_before)
            df_MR_CF_p_value_before.loc[index,'MR_item'] = df_column
            df_MR_CF_p_value_before.loc[index,'CF'] = CF_column
            df_MR_CF_p_value_before.loc[index, 'correlation'] = correlation
            df_MR_CF_p_value_before.loc[index, 'p_value'] = p_value

    for df_column in df_columns:
        MR_column_value = DF_after[df_column].values
        for CF_column in CF_columns:
            CF_column_value = DF_after[CF_column].values
            correlation ,p_value = stats.spearmanr(MR_column_value, CF_column_value,nan_policy='omit')
            index = len(df_MR_CF_p_value_after)
            df_MR_CF_p_value_after.loc[index,'MR_item'] = df_column
            df_MR_CF_p_value_after.loc[index,'CF'] = CF_column
            df_MR_CF_p_value_after.loc[index, 'correlation'] = correlation
            df_MR_CF_p_value_after.loc[index, 'p_value'] = p_value






    # df_file_pvalue_ba = pd.DataFrame(columns=['item', 'pvalue'])
    # figure_after_path = './data/figures_stats/item_BA'
    # for df_column in df_columns:
    #
    #     df_column_before = df_before[df_column]
    #     df_column_before = df_column_before[np.isfinite(df_column_before)]
    #
    #     df_column_after = df_after[df_column]
    #     df_column_after = df_column_after[np.isfinite(df_column_after)]
    #
    #     compare_static_before_operation = stats.mannwhitneyu(df_column_before, df_column_after,alternative='two-sided').pvalue
    #
    #     index = len(df_file_pvalue_ba)
    #     df_file_pvalue_ba.loc[index, 'item'] = df_column
    #     df_file_pvalue_ba.loc[index, 'pvalue'] = compare_static_before_operation
    #
    #     plt.figure(figsize=(6, 6))
    #     sns.kdeplot(df_column_after, color='r', label='before_operation')
    #     sns.kdeplot(df_column_before, color='g', label='after_operation')
    #
    #     plt.title('{}:{}'.format(df_column, compare_static_before_operation))
    #     plt.legend()
    #     plt.savefig(os.path.join(figure_after_path, ''.join([df_column, '.png'])))
    #     plt.close()










