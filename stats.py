import pandas as pd
import numpy as np
import math


if __name__ == '__main__':
    CF_save_path = './data/MR_file_pvalue_lung_After_adjust.xlsx'
    CF_file = pd.read_excel('./data/MR_file_pvalue_lung_after.xlsx',index_col=0).reset_index()
    CF_patients = pd.read_excel('./data/patients_dis_day_MR_lung_infection.xlsx',index_col=0)
    CF_patients_after = CF_patients[CF_patients.dis_day>0]
    CF_columns = CF_file.item.values
    # CF_dead = CF_patients_after[CF_patients_after['出院时结局']== 0]
    # CF_cured = CF_patients_after[CF_patients_after['出院时结局'] == 1]
    CF_lung = CF_patients_after[CF_patients_after.is_sick == 1]
    CF_not_lung = CF_patients_after[CF_patients_after.is_sick == 0]
    for CF_column in CF_columns:
        # dead_mean = CF_dead[CF_column].mean()
        # CF_cured_mean = CF_cured[CF_column].mean()
        # CF_file.loc[CF_file.item==CF_column,'fold_change'] = dead_mean/CF_cured_mean
        CF_lung_mean = CF_lung[CF_column].mean()
        CF_not_lung_mean = CF_not_lung[CF_column].mean()
        CF_file.loc[CF_file.item == CF_column, 'fold_change'] = CF_lung_mean / CF_not_lung_mean

    CF_values = CF_file.pvalue.values
    sorted_index = sorted(range(len(CF_values)), key=lambda k: CF_values[k])
    print(sorted_index)
    for index, row in CF_file.iterrows():
        k = sorted_index[index] + 1
        q_value = row.pvalue * (len(CF_file)/k)
        CF_file.loc[index,'q_value'] = q_value
        CF_file.loc[index,'q_value_log'] = - math.log(q_value,10)
    CF_file.sort_values(by='q_value').to_excel(CF_save_path)

    # CF_save_path = './data/NLP_file_pvalue_lung_adjust.xlsx'
    # CF_file = pd.read_excel('./data/NLP_pvalue_lung.xlsx', index_col=0).reset_index()
    # # CF_patients = pd.read_excel('./data/patients_dis_day_MR_ending.xlsx', index_col=0)
    # # CF_patients_after = CF_patients[CF_patients.dis_day <= 0]
    # # CF_columns = CF_file.item.values
    # # CF_dead = CF_patients_after[CF_patients_after['出院时结局'] == 0]
    # # CF_cured = CF_patients_after[CF_patients_after['出院时结局'] == 1]
    # # # CF_lung = CF_patients_after[CF_patients_after.is_lung == 1]
    # # # CF_not_lung = CF_patients_after[CF_patients_after.is_lung == 0]
    # # for CF_column in CF_columns:
    # #     dead_mean = CF_dead[CF_column].mean()
    # #     CF_cured_mean = CF_cured[CF_column].mean()
    # #     CF_file.loc[CF_file.item == CF_column, 'fold_change'] = dead_mean / CF_cured_mean
    # #     # CF_lung_mean = CF_lung[CF_column].mean()
    # #     # CF_not_lung_mean = CF_not_lung[CF_column].mean()
    # #     # CF_file.loc[CF_file.CF == CF_column, 'fold_change'] = CF_lung_mean / CF_not_lung_mean
    #
    # CF_values = CF_file.pvalue.values
    # sorted_index = sorted(range(len(CF_values)), key=lambda k: CF_values[k])
    # print(sorted_index)
    # for index, row in CF_file.iterrows():
    #     k = sorted_index[index] + 1
    #     q_value = row.pvalue * (len(CF_file) / k)
    #     CF_file.loc[index, 'q_value'] = q_value
    #     CF_file.loc[index, 'q_value_log'] = - math.log(q_value, 10)
    # CF_file.sort_values(by='q_value').to_excel(CF_save_path)





