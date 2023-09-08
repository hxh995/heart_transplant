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

def get_df_p_value_figure(df,df_column,df_outcome,figure_path,not_label,is_label):
    df_not = df[df.loc[:,df_outcome] == 0]
    df_is = df[df.loc[:,df_outcome] == 1]
    df_not_column = df_not[df_column]
    df_is_column = df_is[df_column]
    print(df_not_column,df_is_column)
    df_not_column = df_not_column[np.isfinite(df_not_column)]
    df_is_column = df_is_column[np.isfinite(df_is_column)]
    compare_static = stats.mannwhitneyu(df_not_column,df_is_column,alternative ='two-sided').pvalue

    plt.figure(figsize=(6,6))
    sns.kdeplot(df_not_column,color='r',label = not_label)
    sns.kdeplot(df_is_column,color='g',label = is_label)

    plt.title('{}:{}'.format(df_column,compare_static))
    plt.legend()
    plt.savefig(os.path.join(figure_path,''.join([df_column,'.png'])))
    plt.close()

    return compare_static

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
if __name__ == '__main__':
    patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'编号':'patient_id'})
    # patients_info['临床诊断'] = patients_info ['临床诊断'].apply(lambda x:x.replace('、',' ').replace('，',' ').replace('：',' ').replace(',',' ').replace('\n',' ').replace('Ⅳ','IV'))
    # patients_info = patients_info.drop('临床诊断',axis=1).join(patients_info['临床诊断'].str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('临床诊断'))
    # patients_info['临床诊断'] = patients_info['临床诊断'].apply(lambda x:x.strip())
    # patients_info = patients_info[patients_info['临床诊断']!='']

    patients_info = patients_info[patients_info['出院时结局'].isin(['临床治愈','好转','死亡'])]
    patients_info['出院时结局'] = patients_info['出院时结局'].apply(lambda x:0 if x == '死亡' else 1)
    complication_df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',index_col=0).drop_duplicates(subset=['patient_id'])
    patients_info = patients_info.merge(complication_df[['patient_id','is_lung', 'is_Kidney', 'is_Hypertension','is_Hyperglycemia', 'Others', 'disease', 'is_Hyperlipidemia']],on='patient_id')
    ending_df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx',index_col=0).drop_duplicates(subset=['patient_id'])
    patients_info = patients_info.merge(ending_df[['patient_id','ending']], on='patient_id')

    patients_info['cardiomyopathy_type'] = 6
    patients_info.loc[patients_info['临床诊断'].str.contains('扩张型心肌病|扩张性心肌病|扩张性心肌病[充血性心肌病]'),'cardiomyopathy_type'] = 1
    patients_info.loc[patients_info['临床诊断'].str.contains('缺血性心肌病'),'cardiomyopathy_type'] = 2
    patients_info.loc[patients_info['临床诊断'].str.contains('限制性心肌病|限制型心肌病'), 'cardiomyopathy_type'] = 3
    patients_info.loc[patients_info['临床诊断'].str.contains('肥厚.*心肌病',regex=True), 'cardiomyopathy_type'] = 4
    patients_info.loc[patients_info['临床诊断'].str.contains('.*右室心肌病.*', regex=True), 'cardiomyopathy_type'] = 5
    patients_info.loc[~patients_info['临床诊断'].str.contains('.*心肌病.*', regex=True), 'cardiomyopathy_type'] = 0

    patients_info['临床诊断'] = patients_info['临床诊断'].apply(lambda x: x.replace('Ⅳ', 'IV').replace('Ⅲ','III'))

    patients_info['cardiac_function'] = 0

    patients_info.loc[patients_info['临床诊断'].str.contains('.*心功能(3|III)级.*', regex=True), 'cardiac_function'] = 2
    patients_info.loc[patients_info['临床诊断'].str.contains('.*心功能(IV|4)级.*', regex=True), 'cardiac_function'] = 4
    patients_info.loc[patients_info['临床诊断'].str.contains('.*心功能(III-IV|3-4)级.*', regex=True), 'cardiac_function'] = 3
    patients_info.loc[patients_info['临床诊断'].str.contains('.*心功能II级.*', regex=True), 'cardiac_function'] = 1

    patients_info['is_Hyperuricemia'] = 0
    patients_info.loc[patients_info['临床诊断'].str.contains('高尿酸血症'), 'is_Hyperuricemia'] = 1

    patients_info['高脂血症'] = 0
    patients_info.loc[patients_info['临床诊断'].str.contains('高脂血症'), '高脂血症'] = 1

    patients_info['高胆红素血症'] = 0
    patients_info.loc[patients_info['临床诊断'].str.contains('高胆红素血症'), '高胆红素血症'] = 1



    patients_info['disease'] = 0
    patients_info.loc[patients_info['临床诊断'].str.contains('症'),'disease'] = 1

    # patients_info['Other_disease'] = 0
    #
    #
    # disease_type = pd.read_excel('./data/sick_check.xlsx')
    #
    # patients_info.loc[patients_info['临床诊断'].str.contains('|'.join(disease_type['临床诊断'].values[3:])), 'Other_disease'] = 1
    figure_path = './data/figures/NLP_dead'

    # CF_file_pvalue = pd.DataFrame(columns=['item','pvalue'])
    # for df_column in df_columns:
    #     compare_static_operation = get_df_p_value_figure(patients_info,df_column,'出院时结局',figure_path,'is_dead','is_healthy')
    #     CF_file_pvalue_index = len(CF_file_pvalue)
    #     CF_file_pvalue.loc[CF_file_pvalue_index,'item'] = df_column
    #     CF_file_pvalue.loc[CF_file_pvalue_index,'pvalue'] = compare_static_operation
    #
    # CF_file_pvalue.to_excel('./data/NLP_pvalue_dead.xlsx')
    sick_names = ['is_lung','disease','is_Kidney','is_Hypertension','is_Hyperlipidemia','is_Hyperglycemia','ending']
    # for sick_name in sick_names:
    #     figure_path = './data/figures/NLP_is_Kidney'
    #     df_columns = ['cardiomyopathy_type', 'cardiac_function','is_Hyperuricemia','高脂血症','高胆红素血症','disease']
    #     # CF_file_pvalue = pd.DataFrame(columns=['item', 'pvalue'])
    #     NLP_file_pvalue_before = pd.DataFrame(columns=['NLP_name', 'pvalue'])
    #     for df_column in df_columns:
    #         # compare_static_operation = get_df_p_value_figure(patients_info, df_column, 'is_Kidney', figure_path, 'not_Kidney', 'is_Kidney')
    #         NLP_file_pvalue_index = len(NLP_file_pvalue_before)
    #         NLP_file_pvalue_before.loc[NLP_file_pvalue_index, 'NLP_name'] = df_column
    #         before_operation_is_lung = patients_info.loc[patients_info[sick_name] == 1, df_column].values
    #         before_operation_is_lung = before_operation_is_lung[np.isfinite(before_operation_is_lung)]
    #         before_operation_not_lung = patients_info.loc[patients_info[sick_name] != 1, df_column].values
    #         before_operation_not_lung = before_operation_not_lung[np.isfinite(before_operation_not_lung)]
    #         compare_static_before_operation = stats.mannwhitneyu(before_operation_is_lung, before_operation_not_lung,alternative='two-sided').pvalue
    #         NLP_file_pvalue_before.loc[NLP_file_pvalue_index, 'pvalue'] = compare_static_before_operation
    #     NLP_file_pvalue_before.to_excel('./data/stats_data/NLP_{}_bf_stats.xlsx'.format(sick_name))

    py_type = 'bf'
    T_V = 'valid'
    with torch.no_grad():
        for sick_name in sick_names[:-2]:
            fold_selected = 0
            auc_max = 0
            for fold in range(0,5):
                # print(fold)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                X_train = torch.load('./data/Integration_data/NLP_X_{}_{}_{}-{}.pt'.format('train', sick_name, py_type, str(fold)))
                X_valid = torch.load('./data/Integration_data/NLP_X_{}_{}_{}-{}.pt'.format(T_V, sick_name, py_type, str(fold)))
                # ResNet_transformer_model = torch.load('./data/models/ResNet_transformer_model_MR_{}_{}-{}.pkl'.format(sick_name, py_type,str(fold)))
                # ResNet_transformer_model_lstm = torch.load('./data/models/ResNet_transformer_model_MR_transformer_{}_{}-{}.pkl'.format(sick_name,py_type,str(fold)))

                transformer_model = torch.load(
                    './data/models/Integration_ending/Transformer_NLP_{}_{}-{}.pkl'.format(sick_name, py_type, str(fold)))
                transformer_model.eval()
                # ResNet_transformer_model.eval()
                # ResNet_transformer_model_lstm.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                data_all = X_train + X_valid
                from NLP_lung import collate_func
                loader = DataLoader(dataset=data_all, batch_size=16, collate_fn=collate_func)
                flag = False
                for batch_patient_id, batch_src_pad, batch_src_len, batch_tgt_pad, batch_tgt_len, batch_ending in loader:
                    batch_NLP_hidden, logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model(
                        batch_src_pad.to(device), batch_tgt_pad.to(device), torch.tensor(batch_tgt_len))
                    # batch_ending = to_categorical(batch_ending, 2)

                    # for i in range(len(batch_ending)):
                    #     if batch_ending[i] == 1:
                    #         print(batch_patient_id[i])
                    #         print(batch_ending[i])
                    #         print(F.softmax(logits[i], dim=-1))

                    if flag:
                        output = torch.cat([output, logits], dim=0)
                        output_label = output_label + batch_ending

                        # output_patient_id = torch.cat([output_patient_id,batch_patient_id],dim=0)
                        output_patient_id = output_patient_id + batch_patient_id
                        NLP_hidden = torch.cat([NLP_hidden, batch_NLP_hidden], dim=0)
                    else:
                        output = logits
                        output_label = batch_ending
                        output_patient_id = batch_patient_id
                        NLP_hidden = batch_NLP_hidden
                        flag = True
                    # print(logits)
                    # print(label_batch_valid)
                print(fold)
                print(Counter(output_label))
                output_label = to_categorical(output_label, 2)
                logits = F.softmax(output, dim=-1);
                for i in range(logits.shape[1]):
                    fpr, tpr, roc_auc = caculate_auc(output_label[:, i], logits[:, i].detach().cpu().numpy());
                    print(roc_auc)
                if roc_auc > auc_max:
                    fold_selected = fold
                    pydicom_hidden_df = pd.DataFrame(logits[:,1].detach().cpu().numpy())
                    pydicom_hidden_df.columns = ['NLP_logits']
                    pydicom_hidden_df['patient_id'] = output_patient_id


            person_df = pd.DataFrame(columns=['NLP_name', 'cov', 'p_value'])
            df_columns = ['cardiomyopathy_type', 'cardiac_function','is_Hyperuricemia','高脂血症','高胆红素血症','disease']
            for column in df_columns:
                NLP_file_pvalue_index = len(person_df)
                column_df = patients_info.loc[:, ['patient_id', column]].merge(pydicom_hidden_df,on='patient_id')
                result = stats.pearsonr(column_df.iloc[:, -2], column_df.iloc[:, -1])
                person_df_len = len(person_df)
                person_df.loc[person_df_len, 'NLP_name'] = column
                person_df.loc[person_df_len, 'cov'] = result.statistic
                person_df.loc[person_df_len, 'p_value'] = result.pvalue
            person_df = person_df.sort_values(by='p_value', ascending=True)
            person_df.to_excel('./data/stats_data/NLP_{}_bf.xlsx'.format(sick_name))








