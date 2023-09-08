import pandas as pd
import scipy.ndimage
def get_shape_images(images):

    shape_bridge = [64/len(images),1,1]
    images_shaped = scipy.ndimage.zoom(images,shape_bridge,order=1)
    return images_shaped


if __name__ == '__main__':
    # patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    # df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all_pvalue.xlsx', index_col=0)
    # patients_info = patients_info[~patients_info['术后并发症'].isna()]
    # patients_info_is_lung_id = patients_info[patients_info['术后并发症'].str.contains('.*肺.*感染.*',regex=True)]['编号']
    #
    # df['is_lung'] = df.apply(lambda x: 1 if x.patient_id in patients_info_is_lung_id else 0,axis=1)
    #
    # df.to_excel('./data/patient_dis_day_CF_complication_processed_all_pvalue.xlsx')
    # df_CT = pd.read_pickle('./data/df_CT_all_day.pkl')
    # df_CT['images'] = df_CT['images'].apply(lambda x:get_shape_images(x))
    # df_CT.to_pickle('./data/df_CT_all_shaped.pkl')

    # NLP_df = pd.read_excel('./data/NLP_patient_context_concat.xlsx',index_col=0)
    # NLP_dis_day_df = pd.read_excel('./data/NLP_patient_dis_day.xlsx',index_col=0)
    # patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    # patients_info['临床诊断'] = patients_info['临床诊断'].apply(
    #     lambda x: x.replace('、', ' ').replace('，', ' ').replace('：', ' ').replace(',', ' ').replace('\n', ' ').replace(
    #         'Ⅳ', 'IV'))
    # patients_info = patients_info.drop('临床诊断', axis=1).join(
    #     patients_info['临床诊断'].str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('临床诊断'))
    # patients_info['临床诊断'] = patients_info['临床诊断'].apply(lambda x: x.strip())
    # patients_info = patients_info[patients_info['临床诊断'] != '']
    # # patients_info.groupby('临床诊断').size().sort_values(ascending=False).to_csv('./data/clinical_diagnose_sort.xlsx')
    # patients_info = patients_info[patients_info['出院时结局'].isin(['临床治愈', '好转', '死亡'])]
    # patients_info.loc[:, "出院时结局"] = patients_info['出院时结局'].apply(lambda x: 1 if x in ['临床治愈', '好转'] else 0)
    # dict_status = {0: '死亡', 1: '好转'}
    #
    #
    # patients_info = patients_info[patients_info['临床诊断'].str.len() >= 2]
    # df_info = pd.DataFrame(columns=['patient_name','临床诊断'])
    # for name, grouped in patients_info.groupby('编号'):
    #     df_len = len(df_info)
    #     clinical_diagnose = ','.join(grouped['临床诊断'].values.tolist())
    #     clinical_diagnose = clinical_diagnose.replace("\n", "").replace("，", "").replace("。","").replace('“', "").replace('"', "")
    #     clinical_diagnose = clinical_diagnose.replace('型', '性').replace('瓣膜性心脏病', '').replace('20161026', '').replace(
    #         'Ⅳ', 'IV').replace('Ⅲ', 'III').replace('(NYHA分级)', '').replace('很', '').replace('？', '')
    #     df_info.loc[df_len, 'patient_id'] = grouped['编号'].values[0]
    #     df_info.loc[df_len, 'patient_name'] = grouped['姓名'].values[0]
    #     df_info.loc[df_len, '临床诊断'] = clinical_diagnose
    #     df_info.loc[df_len,'ending_label'] = grouped['出院时结局'].values[0]
    #
    # NLP_all_df = NLP_df.merge(NLP_dis_day_df,on='patient_name').merge(df_info[['patient_id','patient_name','临床诊断','ending_label']])
    # NLP_all_df.to_excel('./data/NLP_patient_context_concat_all.xlsx')

    complication_df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx', index_col=0)
    patient_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    patient_info['year'] = patient_info['入院日期'].apply(lambda x: x.split('.')[0])
    patient_info.groupby('year').size()
    patient_info = patient_info.sort_values('入院日期').reset_index(drop=True)
    patient_info_included = patient_info.iloc[-200:,]

    # complication_df['disease'] = 1
    # patient_info_no_disease_patient_id = patient_info[patient_info['术后并发症']=='无']['编号'].values
    # complication_df.loc[complication_df['patient_id'].isin(patient_info_no_disease_patient_id),'disease'] = 0
    # print('disease')
    # print(len(complication_df[complication_df['disease'] == 1].patient_id.unique()))
    #
    #
    # patient_info_is_Hyperlipidemia_patient_id = patient_info[patient_info['术后并发症'].str.contains('高脂血症')]['编号'].values
    # complication_df['is_Hyperlipidemia'] = 0
    # complication_df.loc[complication_df['patient_id'].isin(patient_info_is_Hyperlipidemia_patient_id), 'is_Hyperlipidemia'] = 1
    # print('is_Hyperlipidemia')
    # print(len(complication_df[complication_df['is_Hyperlipidemia']==1].patient_id.unique()))
    #
    #
    # patient_info_is_Hyperglycemia_patient_id = patient_info[patient_info['术后并发症'].str.contains('血糖')]['编号'].values
    # complication_df['is_Hyperglycemia'] = 0
    # complication_df.loc[complication_df['patient_id'].isin(patient_info_is_Hyperglycemia_patient_id), 'is_Hyperglycemia'] = 1
    # print('is_Hyperglycemia')
    # print(len(complication_df[complication_df['is_Hyperglycemia']==1].patient_id.unique()))
    #
    #
    #
    # patient_info_is_Hypertension_patient_id = patient_info[patient_info['术后并发症'].str.contains('血压')][
    #     '编号'].values
    # complication_df['is_Hypertension'] = 0
    # complication_df.loc[complication_df['patient_id'].isin(patient_info_is_Hypertension_patient_id), 'is_Hypertension'] = 1
    # print('is_Hypertension')
    # print(len(complication_df[complication_df['is_Hypertension'] == 1].patient_id.unique()))
    #
    # complication_df.to_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx')






