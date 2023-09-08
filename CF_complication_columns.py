import pandas as pd
import numpy as np

CF_complication_pvalue_before = pd.read_excel('./data/CF_file_pvalue_dead_Before.xlsx',index_col=0)
CF_complication_pvalue_after = pd.read_excel('./data/CF_file_pvalue_dead_After.xlsx',index_col=0)

df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx',index_col=0)

CF_complication_pvalue_before_CF = CF_complication_pvalue_before[CF_complication_pvalue_before.pvalue<=0.05].CF.values
CF_complication_pvalue_after_CF = CF_complication_pvalue_after[CF_complication_pvalue_after.pvalue<=0.05].CF.values

df_columns = df.columns[3:-5]

intersection_columns_before = np.intersect1d(df_columns, CF_complication_pvalue_before_CF)
intersection_columns_after = np.intersect1d(df_columns, CF_complication_pvalue_after_CF)
columns_before_after = np.append(intersection_columns_before,intersection_columns_after)

columns_before_after_unique = np.unique(intersection_columns_before)

df_new_columns = df.columns[:3].tolist() + columns_before_after_unique.tolist() + df.columns[-5:].tolist()

df_new = df[df_new_columns]

df_new.to_excel('./data/patient_dis_day_CF_ending_processed_pvalue_before.xlsx')


