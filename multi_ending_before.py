import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import Multi_layer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.ndimage
import cv2
from torch.utils.data import DataLoader
from CF_CT_NLP_ending import collate_fn
import pandas as pd
from utils import caculate_auc

class Config(object):
    lr = 0.001
    epochs = 100
    num_out = 2

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    for fold in range(0,5):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        NLP_hidden_train_df = pd.read_excel('./data/Integration_data/NLP_hidden_CF_train_ending_clinical-{}.xlsx'.format(str(fold)),index_col=0).drop_duplicates(subset=['patient_id'])
        MR_hidden_df = pd.read_excel('./data/Integration_data/Pydicom_train_df_new_ending_bf-{}.xlsx'.format(str(fold)),index_col=0).drop_duplicates(subset=['patient_id'])
        # CT_hidden_train_df = pd.read_excel('./data/Integration_data/CT_hidden_train_df_ending_all-{}.xlsx'.format(str(fold)),index_col=0).drop_duplicates(subset=['patient_id'])
        CF_hidden_df = pd.read_excel('./data/Integration_data/CF_hidden_train_df_ending_before-{}.xlsx'.format(str(fold)),index_col=0).drop_duplicates(subset=['patient_id'])


        # df_train = CF_hidden_df.drop(columns='label').merge(NLP_hidden_train_df.drop(columns='label'),on='patient_id',how='left').merge(CT_hidden_train_df,on='patient_id',how='left')

        df_train = NLP_hidden_train_df.drop(columns='label').merge(CF_hidden_df,on='patient_id',how='left')
        df_train = df_train.merge(NLP_hidden_train_df[['patient_id','label']],on='patient_id')
        df_train = df_train.fillna(value=0).drop(columns='patient_id')
        opt = Config()
        Multi_model = Multi_layer(df_train.shape[1]-1,opt.num_out).to(device)
        criterion = nn.BCELoss().to(device)
        Multi_model.train()
        optimizer = torch.optim.Adam(Multi_model.parameters(), lr=opt.lr)
        for epoch in tqdm(range(opt.epochs)):
            output_logits_train = Multi_model(torch.FloatTensor(df_train.iloc[:,:-1].values).to(device))
            batch_ending = to_categorical(df_train.iloc[:,-1], opt.num_out)
            loss= criterion(output_logits_train, torch.from_numpy(batch_ending).float().to(device))
            loss.backward();
            optimizer.step();
            optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                print("Multi_model_Epoch {} | Multi_model_Loss {:.4f}".format(epoch + 1, loss.item()));

        with torch.no_grad():
            NLP_hidden_valid_df = pd.read_excel(
                 './data/Integration_data/NLP_hidden_CF_valid_ending_clinical-{}.xlsx'.format(str(fold)),
                index_col=0).drop_duplicates(subset=['patient_id'])
            # CT_hidden_valid_df = pd.read_excel(
            #     './data/Integration_data/CT_hidden_valid_df_ending_all-{}.xlsx'.format(str(fold)),
            #     index_col=0).drop_duplicates(subset=['patient_id'])
            MR_hidden_df = pd.read_excel('./data/Integration_data/Pydicom_valid_df_new_ending_bf-{}.xlsx'.format(str(fold)),index_col=0)
            CF_hidden_df = pd.read_excel(
                './data/Integration_data/CF_hidden_valid_df_ending_before-{}.xlsx'.format(str(fold)),index_col=0)


            df_valid = NLP_hidden_valid_df.drop(columns='label').merge(CF_hidden_df,on='patient_id', how='left')
            df_valid = df_valid.merge(NLP_hidden_valid_df[['patient_id', 'label']], on='patient_id')
            df_valid = df_valid.fillna(value=0).drop(columns='patient_id')
            Multi_model.eval()
            output_logits_valid = Multi_model(torch.FloatTensor(df_valid.iloc[:,:-1].values).to(device))
            output_label_valid = to_categorical(df_valid.iloc[:,-1], 2)

            for  i in range(output_logits_valid.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label_valid[:, i], output_logits_valid[:, i].detach().cpu().numpy());
                print(roc_auc)


            torch.save(output_label_valid[:, 1],
                       './data/Integration_data/patients_ending_valid-{}_inte_bf_ending.pt'.format(str(fold)))
            torch.save(output_logits_valid[:, 1].detach().cpu().numpy(),
                       './data/Integration_data/logits_ending-{}_bf.pt'.format(str(fold)))







