"""
Filename: MR_p_value.py
Author: yellower
"""

import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve,to_categorical
import pandas as pd
import torch.nn as nn
import scipy.stats as stats

if __name__ == '__main__' :
    sick_name = 'is_lung'
    py_type = 'bf'
    T_V = 'valid'

    sick_names = ['is_lung','disease','is_Kidney','is_Hypertension','is_Hyperlipidemia','is_Hyperglycemia','ending']
    for sick_name in sick_names[4:]:
        fold_selected = 0
        auc_max = 0
        for fold in range(0,5):
            # print(fold)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            X_train = torch.load('./data/Integration_data/MR_X_{}_{}_{}-{}.pt'.format('train', sick_name, py_type, str(fold)))
            X_valid = torch.load('./data/Integration_data/MR_X_{}_{}_{}-{}.pt'.format(T_V, sick_name, py_type, str(fold)))
            # ResNet_transformer_model = torch.load('./data/models/ResNet_transformer_model_MR_{}_{}-{}.pkl'.format(sick_name, py_type,str(fold)))
            # ResNet_transformer_model_lstm = torch.load('./data/models/ResNet_transformer_model_MR_transformer_{}_{}-{}.pkl'.format(sick_name,py_type,str(fold)))

            ResNet_transformer_model = torch.load('./data/models/ResNet_transformer_model_MR_encoder_{}_{}-{}.pkl'.format(sick_name, py_type,
                                                                                        str(fold)))
            ResNet_transformer_model.eval()
            # ResNet_transformer_model.eval()
            # ResNet_transformer_model_lstm.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_all = X_train + X_valid
            loader = DataLoader(dataset=data_all, batch_size=56)
            flag = False
            for idx, (batch_patient_id, batch_pydicom_MR, batch_pydicom_len, batch_ending) in enumerate(loader):
                batch_pydicom_pad = torch.from_numpy(np.array(batch_pydicom_MR.squeeze())).float()
                batch_pydicom_hidden, output_logits = ResNet_transformer_model(batch_pydicom_pad.to(device),
                                                                               batch_pydicom_len)
                # batch_pydicom_hidden, output_logits = ResNet_transformer_model_lstm(dec_outputs)
                # output_logits = F.softmax(output_logits, dim=-1);
                # for i in range(len(batch_ending)):
                #     if batch_ending[i] == 1:
                #         print(batch_patient_id[i])
                #         print(F.softmax(output_logits[i], dim=-1))
                if flag:
                    output = torch.cat([output, output_logits], dim=0)
                    output_label = torch.cat([output_label, batch_ending], dim=0)
                    patient_id = torch.cat([patient_id, batch_patient_id], dim=0)
                    pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
                else:
                    output = output_logits
                    output_label = batch_ending
                    patient_id = batch_patient_id
                    pydicom_hidden = batch_pydicom_hidden
                    flag = True
            logits = F.softmax(output, dim=-1);
            output_label_ = to_categorical(output_label.numpy().tolist(), 2)
            print(fold)
            print(Counter(output_label.numpy().tolist()))
            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)
            if roc_auc > auc_max:
                fold_selected = fold
                pydicom_hidden_df = pd.DataFrame(logits[:,1].detach().cpu().numpy())
                pydicom_hidden_df.columns = ['MR_logits']
                pydicom_hidden_df['patient_id'] = patient_id


        person_df = pd.DataFrame(columns=['MR_name', 'cov', 'p_value'])
        df_MR_before = pd.read_excel('./data/df_before_MR.xlsx',index_col=0)
        df_MR_before_columns = df_MR_before.columns[1:]
        df_MR_before = df_MR_before.fillna(df_MR_before.mean(numeric_only=True))
        for column in df_MR_before_columns:
            column_df = df_MR_before.loc[:, ['patient_id', column]].merge(pydicom_hidden_df,on='patient_id')
            result = stats.pearsonr(column_df.iloc[:, -2], column_df.iloc[:, -1])
            person_df_len = len(person_df)
            person_df.loc[person_df_len, 'MR_name'] = column
            person_df.loc[person_df_len, 'cov'] = result.statistic
            person_df.loc[person_df_len, 'p_value'] = result.pvalue
        person_df = person_df.sort_values(by='p_value', ascending=True)
        person_df.to_excel('./data/stats_data/MR_{}_bf.xlsx'.format(sick_name))






