import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet_transformer_lstm,ResNet_transformer_encoder_CT,ResNet,ResNet_transformer_encoder_CT_lstm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.ndimage
import cv2
from collections import Counter

from torch.utils.data import DataLoader


def get_keys_from_dict(dict,keys):
    from operator import itemgetter
    # need make sure keys in dict_key
    out = itemgetter(*keys)(dict)
    # print(out)
    return out


class PYDICOM_series_MR_CT_dataset(Dataset):
    def __init__(self,patients_id, pydicom_CT_before, patients_pydicom_len_CT_before,
                 pydicom_CT, patients_pydicom_len_CT, patients_ending_CT):
        self.patients_id = patients_id
        self.pydicom_CT_before = pydicom_CT_before
        self.patients_pydicom_len_CT_before = patients_pydicom_len_CT_before
        self.pydicom_CT = pydicom_CT
        self.patients_pydicom_len_CT = patients_pydicom_len_CT
        self.label = patients_ending_CT
    def __getitem__(self,item):
        return self.patients_id[item],self.pydicom_CT_before[item],\
               self.patients_pydicom_len_CT_before[item],self.pydicom_CT[item],self.patients_pydicom_len_CT[item],\
               self.label[item]
    def __len__(self):
        return len(self.label)

def collate_func_CT_MR(dataset):
    patients_id = []
    pydicom_CT_before = []
    patients_pydicom_len_CT_before = []
    patients_pydicom_CT = []
    patients_pydicom_len_CT = []
    patients_ending = []
    for i in dataset:
        patients_id.append(i[0])
        for j in i[1]:
            pydicom_CT_before.append(j)
        patients_pydicom_len_CT_before.append(i[2])
        for j in i[3]:
            patients_pydicom_CT.append(j)
        patients_pydicom_len_CT.append(i[4])
        patients_ending.append(i[5])

    return patients_id,pydicom_CT_before,patients_pydicom_len_CT_before,\
           patients_pydicom_CT,patients_pydicom_len_CT,patients_ending


class Config(object):
    encoder_lr = 0.001
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_lr = 0.01

    CT_d_model = 8
    lstm_hidden_size = 32
    CT_height = 64
    CT_enc_ffc_hidden = 8
    CT_d_k = 8
    CT_d_v = 8
    num_out = 2
    CT_in_channels = 64
    CT_max_len_encoder = 2
    CT_max_len_decoder = 256
    batch_size = 48

if __name__ == '__main__':

    complication_df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',
                                    index_col=0).drop_duplicates(subset=['patient_id'])
    sick_name = 'is_lung'
    py_type = 'multi'
    # patients_id_CT = torch.load("./data/patients_id_pydicom_ending_all.pt")
    # patients_pydicom_CT_bf = torch.load("./data/patients_pydicom_ending_bf_all.pt")
    # patients_pydicom_len_CT_bf = torch.load("./data/patients_pydicom_len_ending_bf_all.pt")
    # patients_pydicom_CT_af = torch.load("./data/patients_pydicom_ending_af_all.pt")
    # patients_pydicom_len_CT_af = torch.load("./data/patients_pydicom_len_ending_af_all.pt")


    # patients_id_CT_bf = torch.load("./data/filter_CT/load_CT_data/patients_id_pydicom_ending_bf_all.pt")
    # patients_pydicom_CT_bf = torch.load("./data/filter_CT/load_CT_data/patients_pydicom_ending_bf_all.pt")
    # patients_pydicom_len_CT_bf = torch.load("./data/filter_CT/load_CT_data/patients_pydicom_len_ending_bf_all.pt")

    patients_id_CT = torch.load("./data/filter_CT/load_CT_data/patients_id_pydicom_ending_af_no_lstm.pt")
    # patients_pydicom_CT_bf = torch.load("./data/filter_CT/load_CT_data/patients_id_pydicom_ending_af_all_bf.pt")
    # patients_pydicom_len_CT_bf = torch.load("./data/filter_CT/load_CT_data/patients_pydicom_len_ending_af_all_bf.pt")
    patients_pydicom_CT_af = torch.load("./data/filter_CT/load_CT_data/patients_pydicom_ending_af_no_lstm.pt")
    patients_pydicom_len_CT_af = torch.load("./data/filter_CT/load_CT_data/patients_pydicom_len_ending_af_no_lstm.pt")

    patients_ending_lung = []
    for i in patients_id_CT:
        patients_ending_lung.append(complication_df.loc[complication_df['patient_id'] == i, sick_name].values[0])

    dataset = PYDICOM_series_MR_CT_dataset(patients_id_CT, patients_pydicom_CT_af, patients_pydicom_len_CT_af,
                                           patients_pydicom_CT_af, patients_pydicom_len_CT_af, patients_ending_lung)
    for fold in range(0,5):
        # fold = 3
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if py_type == 'bf':
            NLP_hidden_train_df = pd.read_excel(
                './data/Integration_data/NLP_hidden_CF_train_lung_{}-{}.xlsx'.format(py_type, str(fold)),
                index_col=0).drop_duplicates(subset=['patient_id'])
            X_train_id = NLP_hidden_train_df.patient_id.unique()
        else:
            NLP_X_train = torch.load('./data/Integration_data/NLP_X_train_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
            X_train_id = []
            for i in NLP_X_train:
                X_train_id.append(i[0])
        X_train = []
        X_valid = []
        for i in dataset:
            if i[0] in X_train_id:
                X_train.append(i)
            else:
                X_valid.append(i)
        torch.save(X_train, './data/Integration_data/CT_X_train_{}_{}-{}.pt'.format(sick_name, py_type,str(fold)))
        # X_train = X_train + X_train_copy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt = Config()
        # ResNet_transformer_model = ResNet_transformer(opt.CT_in_channels, opt.CT_height, opt.CT_d_model,
        #                                               opt.lstm_hidden_size, opt.CT_max_len_encoder,
        #                                               opt.CT_max_len_decoder,
        #                                               opt.CT_enc_ffc_hidden, opt.CT_d_k, opt.CT_d_v, opt.num_out).to(device)


        #
        ResNet_transformer_model = ResNet(opt.CT_in_channels, opt.CT_height,device).to(device)
        ResNet_transformer_model_lstm = ResNet_transformer_encoder_CT_lstm(opt.CT_d_model, opt.lstm_hidden_size, opt.CT_max_len_decoder, opt.CT_enc_ffc_hidden,opt.CT_d_k, opt.CT_d_v,opt.num_out,device).to(device)

        ResNet_transformer_model.train()
        ResNet_transformer_model_lstm.train()
        optimizer = torch.optim.SGD(ResNet_transformer_model.parameters(), lr=opt.encoder_lr)
        optimizer_lstm = torch.optim.SGD(ResNet_transformer_model_lstm.parameters(), lr=opt.lstm_lr)

        dataloader = DataLoader(X_train, batch_size=opt.batch_size)
        criterion = nn.BCELoss().to(device)
        accum_steps = 8
        for epoch in tqdm(range(opt.epochs)):
            for idx, (batch_id, batch_pydicom_bf, batch_bf_len, batch_pydicom_CT, batch_CT_len, batch_ending) in enumerate(
                    dataloader):
                # batch_pydicom_bf_pad = torch.from_numpy(np.array(batch_pydicom_bf)).float()
                batch_pydicom_af_pad = torch.from_numpy(np.array(batch_pydicom_CT)).float()
                ResNet_outputs = ResNet_transformer_model(batch_pydicom_af_pad.to(device))
                __,output_logits = ResNet_transformer_model_lstm(ResNet_outputs,batch_CT_len)
                batch_ending = to_categorical(batch_ending, opt.num_out)
                loss = criterion(F.softmax(output_logits, dim=-1),
                                 torch.from_numpy(batch_ending).float().to(device))
                loss.backward();
                optimizer.step();
                optimizer.zero_grad()
                optimizer_lstm.step();
                optimizer_lstm.zero_grad()

            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));

        torch.save(ResNet_transformer_model,
                   './data/models/Integration_lung/ResNet_transformer_model_CT_{}_{}-{}.pkl'.format(sick_name,py_type,str(fold)))
        torch.save(ResNet_transformer_model_lstm,'./data/models/Integration_lung/ResNet_transformer_model_CT_lstm_{}_{}-{}.pkl'.format(sick_name,py_type,str(fold)))
        torch.save(X_valid, './data/Integration_data/CT_X_valid_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
