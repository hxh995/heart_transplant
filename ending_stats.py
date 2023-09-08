"""
Filename: ending_stats.py
Author: yellower
"""
from torch.utils.data import DataLoader
import pandas as pd
import scipy.ndimage
import numpy as np
import scipy.stats as stats
import torch
# from CF_CT_NLP_ending import collate_fn
from CF_complication import  collate_fn
from ending_predict import Encoder,Decoder,Seq2Seq
from utils import column_process,to_categorical,caculate_auc
import torch.nn.functional as F

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    py_type = 'bf'
    sick_name = 'ending'
    if py_type == 'multi':
        for fold in range(5):
            Encoder = torch.load('./data/models/Integration_ending/Encoder_multi_CF_ending-{}.pkl'.format(str(fold)))
            Decoder = torch.load('./data/models/Integration_ending/Decoder_multi_CF_ending-{}.pkl'.format(str(fold)))
            Seq2Seq_model = torch.load('./data/models/Integration_ending/Seq2Seq_multi_CF_ending-{}.pkl'.format(str(fold)))
            X_valid = torch.load("./data/Integration_data/X_valid_CF_NLP-{}.pt".format(str(fold)))
            with torch.no_grad():
                data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid), collate_fn=collate_fn)
                patient_id_valid, packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid, \
                    packed_first_trip_dim_valid, patients_first_trip_len_valid, packed_patients_pdf_dim_valid, patients_pdf_word_len_valid, patients_ending_valid = iter(
                    data_loader_valid).__next__()

                Seq2Seq_model.eval()
                CF_hidden_valid, logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),
                                                              patients_CF_len_before_operation_valid
                                                              , packed_patients_CF_after_operation_valid.to(device),
                                                              patients_CF_len_after_operation_valid)
                logits_valid = F.softmax(logits_valid, dim=-1);
                data_label_valid = to_categorical(patients_ending_valid, 2)
                print(fold)
                for i in range(logits_valid.shape[1]):
                    fpr, tpr, roc_auc = caculate_auc(data_label_valid[:, i], logits_valid[:, i].detach().cpu().numpy());
                    print(roc_auc)
        fold = 2
        Encoder = torch.load('./data/models/Integration_ending/Encoder_multi_CF_ending-{}.pkl'.format(str(fold)))
        Decoder = torch.load('./data/models/Integration_ending/Decoder_multi_CF_ending-{}.pkl'.format(str(fold)))
        Seq2Seq_model = torch.load('./data/models/Integration_ending/Seq2Seq_multi_CF_ending-{}.pkl'.format(str(fold)))

        X_valid = torch.load("./data/Integration_data/X_valid_CF_NLP-{}.pt".format(str(fold)))
        X_train = torch.load("./data/Integration_data/X_train_CF_NLP-{}.pt".format(str(fold)))
        data_all = X_train + X_valid
        data_loader_valid = DataLoader(data_all, batch_size=len(data_all), collate_fn=collate_fn)
        patient_id_valid, packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid, \
            packed_first_trip_dim_valid, patients_first_trip_len_valid, packed_patients_pdf_dim_valid, patients_pdf_word_len_valid, patients_ending_valid = iter(data_loader_valid).__next__()

        CF_hidden_valid, logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),
                                                      patients_CF_len_before_operation_valid
                                                      , packed_patients_CF_after_operation_valid.to(device),
                                                      patients_CF_len_after_operation_valid)

        logits = F.softmax(logits_valid, dim=-1);
        CF_logits = logits.detach().cpu().numpy()
        CF_logits_df = pd.DataFrame(CF_logits)
        CF_logits_df.columns = ['CF_hidden_{}'.format(str(i)) for i in range(CF_logits.shape[1])]
        CF_logits_df['patient_id'] = patient_id_valid
        CF_logits_df = CF_logits_df.drop_duplicates(subset=['patient_id'])

        df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0)

        df_after_df = df[df.dis_day > 0]
        df_after_df = df_after_df.fillna(df_after_df.mean(numeric_only=True))
        df_after_df = df_after_df.groupby('patient_id').mean().reset_index()
        df_after_df_column = df_after_df.iloc[:, 3:-1].columns
        person_df = pd.DataFrame(columns=['CF_name', 'cov', 'p_value'])
        for column in df_after_df_column:
            column_df = df_after_df.loc[:, ['patient_id', column]].merge(
                CF_logits_df.loc[:, ['patient_id', 'CF_hidden_1']], on='patient_id')
            print(column)
            person_df_len = len(person_df)
            result = stats.pearsonr(column_df.iloc[:, -2], column_df.iloc[:, -1])
            person_df.loc[person_df_len, 'CF_name'] = column
            person_df.loc[person_df_len, 'cov'] = result.statistic
            person_df.loc[person_df_len, 'p_value'] = result.pvalue

        person_df = person_df.sort_values(by='p_value', ascending=True)
        person_df.to_excel('./data/stats_data/CF_{}_multi.xlsx'.format(sick_name))

    else:
        for fold in range(5):
            Encoder = torch.load('./data/models/Integration_lung/Encoder_bf_CF_{}-{}'.format(sick_name,str(fold)))
            Decoder = torch.load('./data/models/Integration_lung/Decoder_bf_CF_{}-{}'.format(sick_name,str(fold)))
            Seq2Seq_model = torch.load('./data/models/Integration_lung/Seq2Seq_bf_CF_{}-{}'.format(sick_name,str(fold)))
            X_valid = torch.load('./data/Integration_data/X_valid_CF_{}-{}.pt'.format(sick_name,str(fold)))
            with torch.no_grad():
                data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid), collate_fn=collate_fn)
                patient_id_valid, packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid \
                    , patients_ending_valid = iter(
                    data_loader_valid).__next__()

                Seq2Seq_model.eval()
                CF_hidden_valid, logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),
                                                              patients_CF_len_before_operation_valid
                                                              , packed_patients_CF_after_operation_valid.to(device),
                                                              patients_CF_len_after_operation_valid)
                logits_valid = F.softmax(logits_valid, dim=-1);
                data_label_valid = to_categorical(patients_ending_valid, 2)
                print(fold)
                for i in range(logits_valid.shape[1]):
                    fpr, tpr, roc_auc = caculate_auc(data_label_valid[:, i], logits_valid[:, i].detach().cpu().numpy());
                    print(roc_auc)
        fold = 0
        Encoder = torch.load('./data/models/Integration_lung/Encoder_bf_CF_{}-{}'.format(sick_name, str(fold)))
        Decoder = torch.load('./data/models/Integration_lung/Decoder_bf_CF_{}-{}'.format(sick_name, str(fold)))
        Seq2Seq_model = torch.load('./data/models/Integration_lung/Seq2Seq_bf_CF_{}-{}'.format(sick_name, str(fold)))

        X_valid = torch.load('./data/Integration_data/X_valid_CF_{}-{}.pt'.format(sick_name,str(fold)))
        X_train = torch.load('./data/Integration_data/X_train_CF_{}-{}.pt'.format(sick_name,str(fold)))
        data_all = X_train + X_valid
        data_loader_valid = DataLoader(data_all, batch_size=len(data_all), collate_fn=collate_fn)
        patient_id_valid, packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid \
            , patients_ending_valid = iter(data_loader_valid).__next__()

        CF_hidden_valid, logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),
                                                      patients_CF_len_before_operation_valid
                                                      , packed_patients_CF_after_operation_valid.to(device),
                                                      patients_CF_len_after_operation_valid)

        logits = F.softmax(logits_valid, dim=-1);
        CF_logits = logits.detach().cpu().numpy()
        CF_logits_df = pd.DataFrame(CF_logits)
        CF_logits_df.columns = ['CF_hidden_{}'.format(str(i)) for i in range(CF_logits.shape[1])]
        CF_logits_df['patient_id'] = patient_id_valid
        CF_logits_df = CF_logits_df.drop_duplicates(subset=['patient_id'])

        df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0)

        df_after_df = df[df.dis_day <= 0]
        df_after_df = df_after_df.fillna(df_after_df.mean(numeric_only=True))
        df_after_df = df_after_df.groupby('patient_id').mean().reset_index()
        df_after_df_column = df_after_df.iloc[:, 3:-1].columns
        person_df = pd.DataFrame(columns=['CF_name', 'cov', 'p_value'])
        for column in df_after_df_column:
            column_df = df_after_df.loc[:, ['patient_id', column]].merge(
                CF_logits_df.loc[:, ['patient_id', 'CF_hidden_1']], on='patient_id')
            print(column)
            person_df_len = len(person_df)
            result = stats.pearsonr(column_df.iloc[:, -2], column_df.iloc[:, -1])
            person_df.loc[person_df_len, 'CF_name'] = column
            person_df.loc[person_df_len, 'cov'] = result.statistic
            person_df.loc[person_df_len, 'p_value'] = result.pvalue

        person_df = person_df.sort_values(by='p_value', ascending=True)
        person_df.to_excel('./data/stats_data/CF_{}_bf.xlsx'.format(sick_name))