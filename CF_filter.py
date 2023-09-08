import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn.functional as F
from utils import column_process,to_categorical,caculate_auc,acu_curve
from sklearn.model_selection import train_test_split
import torch
from ending_predict import Encoder,Decoder,Seq2Seq
from sklearn.model_selection import StratifiedKFold
from CF_complication import collate_fn

class net(nn.Module):
    def __init__(self,input_dims,hidden_dims,num_out,layers_num=2,batch_first=True,drop_prob = 0.2):
        super(net, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dims,hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims, num_out)
        )

    def forward(self,x):
        return self.dense(x)


class Config():
    CF_lr = 0.001
    epochs = 80

    hidden_dims = 32
    num_out = 2

if __name__ == '__main__':
    # df = pd.read_excel('./data/patient_dis_day_CF_ending_processed_pvalue_before.xlsx', index_col=0)
    #
    #
    # model_input_dims = len(df.columns) - 8
    # df_before_operation = df[df.dis_day < 0].reset_index(drop=True)
    # df_before_operation = df_before_operation.fillna(df_before_operation.mean(numeric_only=True))
    #
    #
    # df_before_operation.iloc[:, 3:-1] = df_before_operation.iloc[:, 3:-1].apply(lambda x: (x - x.mean()) / (x.std()))
    #
    #
    # opt = Config()
    # DNN_model = net(len(df_before_operation.columns[3:-1]),32,opt.num_out)
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(DNN_model.parameters(), lr=opt.CF_lr)
    # labels = to_categorical(df_before_operation.iloc[:, -1], 2)
    # for epoch in tqdm(range(opt.epochs)):
    #     logits = DNN_model(torch.FloatTensor(df_before_operation[df_before_operation.columns[3:-1]].values))
    #     logits = F.softmax(logits,dim=-1)
    #     loss = criterion(F.softmax(logits, dim=-1), torch.FloatTensor(labels))
    #     loss.backward();
    #     optimizer.step();
    #
    # for i in range(logits.shape[1]):
    #     fpr, tpr, roc_auc = caculate_auc(labels[:, i], logits[:, i].detach().cpu().numpy());
    #     print(roc_auc)
    #
    # patients_id_list = []
    #
    # for i in range(len(labels)):
    #     if labels[i][1] == 0:
    #         if logits[i][1] > 0.8:
    #             patients_id_list.append(df_before_operation.loc[i, 'patient_id'])
    #             print(df_before_operation.loc[i, 'patient_id'])
    #             print(df_before_operation.loc[i, 'dis_day'])
    #             print(labels[i])
    #             print(logits[i])
    #             df_after_operation = df_before_operation.drop(i)
    #
    # df_before_operation.to_excel('./data/df_before_operation_processed_ending.xlsx')


    sick_name = 'is_Hypertension'
    if sick_name == 'ending':
        df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0)
        last_name = -1
    else:
        df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx', index_col=0)
        last_name = -7
    manner = 'fold_'
    df_after_operation = df[df.dis_day > 0].reset_index(drop=True)
    df_after_operation = df_after_operation.fillna(df_after_operation.mean(numeric_only=True))
    print(len(df_after_operation['patient_id'].unique()))

    df_after_operation.iloc[:, 3:last_name] = df_after_operation.iloc[:, 3:last_name].apply(lambda x: (x - x.mean()) / (x.std()))

    opt = Config()


    labels = to_categorical(df_after_operation.loc[:, sick_name], 2)
    if manner != 'fold':
        DNN_model = net(len(df_after_operation.columns[3:last_name]), 32, opt.num_out)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(DNN_model.parameters(), lr=opt.CF_lr)
        for epoch in tqdm(range(opt.epochs)):
            logits = DNN_model(torch.FloatTensor(df_after_operation[df_after_operation.columns[3:last_name]].values))
            logits = F.softmax(logits, dim=-1)
            loss = criterion(F.softmax(logits, dim=-1), torch.FloatTensor(labels))
            loss.backward();
            optimizer.step();

        for i in range(logits.shape[1]):
            fpr, tpr, roc_auc = caculate_auc(labels[:, i], logits[:, i].detach().cpu().numpy());
            print(roc_auc)

        patients_id_list = []

        for i in range(len(labels)):
            if labels[i][1] == 1:
                if logits[i][0] > 0.99:
                    patients_id_list.append(df_after_operation.loc[i, 'patient_id'])
                    patient_id_i = df_after_operation.loc[i, 'patient_id']
                    # print(df_after_operation.loc[df_after_operation['patient_id']==patient_id_i,:])
                    # print(labels[i])
                    # print(logits[i])
                    if len(df_after_operation.loc[df_after_operation['patient_id']==patient_id_i,:]) !=1:
                        df_after_operation = df_after_operation.drop(i)
        print(len(df_after_operation['patient_id'].unique()))
        df_after_operation.to_excel('./data/df_after_operation_processed_{}.xlsx'.format(sick_name))
    else:

        for fold in range(2):
            DNN_model = net(len(df_after_operation.columns[3:-7]), 32, opt.num_out)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(DNN_model.parameters(), lr=opt.CF_lr)
            NLP_X_train = torch.load('./data/Integration_data/NLP_X_train_{}_bf-{}.pt'.format(sick_name, str(fold)))
            patients_id_train = []
            DNN_model.train()
            for i in NLP_X_train:
                patients_id_train.append(i[0])
            X_train = df_after_operation[df_after_operation['patient_id'].isin(patients_id_train)]
            X_train_label = to_categorical(X_train.loc[:, sick_name], 2)
            X_valid = df_after_operation[~df_after_operation['patient_id'].isin(patients_id_train)]
            for epoch in tqdm(range(opt.epochs)):
                logits = DNN_model(torch.FloatTensor(X_train[X_train.columns[3:-7]].values))
                logits = F.softmax(logits, dim=-1)
                loss = criterion(F.softmax(logits, dim=-1), torch.FloatTensor(X_train_label))
                loss.backward();
                optimizer.step();
            DNN_model.eval()
            logits = DNN_model(torch.FloatTensor(X_valid[X_valid.columns[3:-7]].values))
            logits = F.softmax(logits, dim=-1)
            X_valid_label = to_categorical(X_valid.loc[:, sick_name], 2)
            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(X_valid_label[:,i] ,logits[:, i].detach().cpu().numpy());
                print(roc_auc)

            patients_id_list = []
            labels = to_categorical(df_after_operation.loc[:, sick_name], 2)
            logits = DNN_model(torch.FloatTensor(df_after_operation[df_after_operation.columns[3:-7]].values))
            logits = F.softmax(logits, dim=-1)
            df_after_operation = df_after_operation.reset_index(drop=True)
            for i in range(len(labels)):
                if labels[i][1] == 1:
                    if logits[i][0] > 0.99:
                        patients_id_list.append(df_after_operation.loc[i, 'patient_id'])
                        # print(df_after_operation.loc[i, 'patient_id'])
                        # print(df_after_operation.loc[i, 'dis_day'])
                        # print(labels[i])
                        # print(logits[i])
                        df_after_operation = df_after_operation.drop(i)
            print(len(df_after_operation['patient_id'].unique()))
            df_after_operation.to_excel('./data/df_after_operation_processed_{}.xlsx'.format(sick_name))
















