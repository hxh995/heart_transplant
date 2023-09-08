"""
Filename: Radiology_MR.py
Author: yellower
"""
import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet,ResNet_transformer_encoder_CT_lstm,ResNet_transformer_encoder_CT
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader


class PYDICOM_series_MR_CT_dataset(Dataset):
    def __init__(self,patients_id,pydicom_MR,pydicom_MR_len,patients_ending_CT):
        self.patients_id = patients_id
        self.pydicom_MR = pydicom_MR
        self.pydicom_MR_len = pydicom_MR_len
        self.label = patients_ending_CT
    def __getitem__(self,item):
        return self.patients_id[item],self.pydicom_MR[item],self.pydicom_MR_len[item],self.label[item]
    def __len__(self):
        return len(self.label)

class Config(object):
    MR_in_channels = 72
    ResNet_lr = 0.01
    batch_size = 64
    epochs = 100
    MR_d_model = 32
    MR_enc_ffc_hidden = 32
    MR_d_k = 32
    MR_d_v = 32
    MR_max_len_decoder = 1
    num_out = 2
    transformer_lr = 0.001


if __name__ == '__main__':

    patients_id_MR = torch.load("./data/patients_id_pydicom.pt")
    patients_pydicom_MR = torch.load("./data/patients_pydicom_pydicom.pt")
    patients_pydicom_len_MR = torch.load("./data/patients_pydicom_len_pydicom.pt")
    patients_ending = []
    sick_name = 'is_Hyperglycemia'
    if sick_name == 'ending':
        complication_df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx',index_col=0).drop_duplicates(subset=['patient_id'])
    else:
        complication_df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',index_col=0).drop_duplicates(subset=['patient_id'])


    py_type = 'bf'
    for i in patients_id_MR:
        patients_ending.append(complication_df.loc[complication_df['patient_id'] == i, sick_name].values[0])


    dataset = PYDICOM_series_MR_CT_dataset(patients_id_MR,patients_pydicom_MR, patients_pydicom_len_MR,patients_ending)
    opt = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Valid_is = False
    if Valid_is:
        ResNet_transformer_encoder = ResNet_transformer_encoder_CT(opt.MR_in_channels, opt.MR_d_model,
                                                                   opt.MR_max_len_decoder, opt.MR_enc_ffc_hidden,
                                                                   opt.MR_d_k,
                                                                   opt.MR_d_v, opt.num_out, device).to(device)
        if py_type == 'bf':
            NLP_X_train = torch.load(
                './data/Integration_data/NLP_X_train_{}_{}-{}.pt'.format(sick_name, py_type, 'VALID'))
            patients_id_train = []
            X_train_copy = []
            for i in NLP_X_train:
                patients_id_train.append(i[0])
            X_train, X_valid = [], []
            for i in dataset:
                if i[0] in patients_id_train:
                    X_train.append(i)
                    if i[-1] == 0:
                        X_train_copy.append(i)
                else:
                    X_valid.append(i)
            # X_train = X_train + X_train_copy

        optimizer = torch.optim.SGD(ResNet_transformer_encoder.parameters(), lr=opt.ResNet_lr)
        # optimizer_transformer = torch.optim.SGD(ResNet_transformer_model_lstm.parameters(), lr=opt.transformer_lr)
        # optimizer_fc = torch.optim.SGD(Project_model.parameters(), lr=opt.transformer_lr)
        dataloader = DataLoader(X_train, batch_size=opt.batch_size)
        criterion = nn.BCELoss().to(device)
        # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([10, 1])).to(device)
        for epoch in tqdm(range(opt.epochs)):
            for idx, (batch_id, batch_pydicom_MR, batch_pydicom_len, batch_ending) in enumerate(dataloader):
                batch_pydicom_pad = torch.from_numpy(np.array(batch_pydicom_MR.squeeze())).float()
                # ResNet_outputs = ResNet_transformer(batch_pydicom_pad.to(device))
                # print(ResNet_outputs.shape)
                __, output_logits = ResNet_transformer_encoder(batch_pydicom_pad.to(device), batch_pydicom_len)
                # __, output_logits = Project_model(ResNet_outputs)
                batch_ending = to_categorical(batch_ending, opt.num_out)
                loss = criterion(F.softmax(output_logits, dim=-1),
                                 torch.from_numpy(batch_ending).float().to(device))
                loss.backward();
                optimizer.step();
                optimizer.zero_grad()
                # optimizer_transformer.step();
                # optimizer_transformer.zero_grad()

            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));


        torch.save(ResNet_transformer_encoder,'./data/models/ResNet_transformer_model_MR_encoder_{}_{}-{}.pkl'.format(sick_name, py_type,'VALID'))
        torch.save(X_train, './data/Integration_data/MR_X_train_{}_{}-{}.pt'.format(sick_name, py_type,'VALID'))
        torch.save(X_valid, './data/Integration_data/MR_X_valid_{}_{}-{}.pt'.format(sick_name, py_type,'VALID'))
        T_V = 'valid'
        with torch.no_grad():
            df_path = './data/Integration_data/MR_hidden_{}_df_{}_{}_{}.xlsx'.format(T_V, sick_name, py_type, 'VALID')
            ResNet_transformer_encoder.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loader = DataLoader(dataset=X_valid, batch_size=52)
            flag = False
            for idx, (batch_patient_id, batch_pydicom_MR, batch_pydicom_len, batch_ending) in enumerate(loader):
                batch_pydicom_pad = torch.from_numpy(np.array(batch_pydicom_MR.squeeze())).float()
                batch_pydicom_hidden, output_logits = ResNet_transformer_encoder(batch_pydicom_pad.to(device),
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
            from collections import Counter
            from utils import caculate_auc
            print(Counter(output_label.numpy().tolist()))
            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)
            # figure_path = './data/figures/figure_test_Radiology_complication.jpg'
            # acu_curve(fpr, tpr, roc_auc,figure_path)
            #
            pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
            pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
            pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
            pydicom_hidden_df['patient_id'] = patient_id
            pydicom_hidden_df.to_excel(df_path)
            if T_V == 'valid':
                torch.save(output_label_[:, 1],
                           './data/Integration_data/patients_valid_MR_{}_{}-{}.pt'.format(sick_name, py_type,'VALID'))
                torch.save(logits[:, 1].detach().cpu().numpy(),
                           './data/Integration_data/logits_MR_{}_{}-{}.pt'.format(sick_name, py_type,'VALID'))



    else:
        for fold in range(0,5):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            # ResNet_transformer_model = ResNet(opt.MR_in_channels, device).to(device)
            # ResNet_transformer_model_lstm = ResNet_transformer_encoder_CT_lstm(opt.MR_d_model,opt.MR_max_len_decoder,opt.MR_enc_ffc_hidden, opt.MR_d_k,
            #                                                                    opt.MR_d_v, opt.num_out, device).to(device)
            ResNet_transformer_encoder = ResNet_transformer_encoder_CT(opt.MR_in_channels,opt.MR_d_model,opt.MR_max_len_decoder,opt.MR_enc_ffc_hidden, opt.MR_d_k,
                                                                    opt.MR_d_v, opt.num_out, device).to(device)
            if py_type == 'bf':
                NLP_X_train = torch.load('./data/Integration_data/NLP_X_train_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
                patients_id_train = []
                X_train_copy = []
                for i in NLP_X_train:
                    patients_id_train.append(i[0])
                X_train, X_valid = [], []
                for i in dataset:
                    if i[0] in patients_id_train:
                        X_train.append(i)
                        if i[-1] == 1:
                            X_train_copy.append(i)
                    else:
                        X_valid.append(i)
                # X_train = X_train + X_train_copy
                X_train = X_train + X_valid
                optimizer = torch.optim.SGD(ResNet_transformer_encoder.parameters(), lr=opt.ResNet_lr)
                # optimizer_transformer = torch.optim.SGD(ResNet_transformer_model_lstm.parameters(), lr=opt.transformer_lr)
                # optimizer_fc = torch.optim.SGD(Project_model.parameters(), lr=opt.transformer_lr)
                dataloader = DataLoader(X_train, batch_size=opt.batch_size)
                criterion = nn.BCELoss().to(device)
                # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([10, 1])).to(device)
                for epoch in tqdm(range(opt.epochs)):
                    for idx, (batch_id, batch_pydicom_MR,batch_pydicom_len,batch_ending) in enumerate(dataloader):
                        batch_pydicom_pad = torch.from_numpy(np.array(batch_pydicom_MR.squeeze())).float()
                        # ResNet_outputs = ResNet_transformer(batch_pydicom_pad.to(device))
                        # print(ResNet_outputs.shape)
                        __ ,output_logits= ResNet_transformer_encoder(batch_pydicom_pad.to(device),batch_pydicom_len)
                        # __, output_logits = Project_model(ResNet_outputs)
                        batch_ending = to_categorical(batch_ending, opt.num_out)
                        loss = criterion(F.softmax(output_logits, dim=-1),
                                         torch.from_numpy(batch_ending).float().to(device))
                        loss.backward();
                        optimizer.step();
                        optimizer.zero_grad()
                        # optimizer_transformer.step();
                        # optimizer_transformer.zero_grad()

                    if (epoch + 1) % 50 == 0:
                        print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));

                # torch.save(ResNet_transformer_model,
                #            './data/models/ResNet_transformer_model_MR_{}_{}-{}.pkl'.format(sick_name, py_type,str(fold)))

                # torch.save(ResNet_transformer_model_lstm,
                #            './data/models/ResNet_transformer_model_MR_transformer_{}_{}-{}.pkl'.format(sick_name,py_type,
                #                                                                                                  str(fold)))

                torch.save(ResNet_transformer_encoder,
                           './data/models/ResNet_transformer_model_MR_encoder_{}_{}-{}.pkl'.format(sick_name, py_type,
                                                                                                       str(fold)))


                torch.save(X_train, './data/Integration_data/MR_X_train_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
                torch.save(X_valid, './data/Integration_data/MR_X_valid_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))











