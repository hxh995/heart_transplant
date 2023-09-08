import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet_transformer_encoder_CT,ResNet_transformer_decoder,ResNet
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.ndimage
import cv2

def get_keys_from_dict(dict,keys):
    from operator import itemgetter
    # need make sure keys in dict_key
    out = itemgetter(*keys)(dict)
    # print(out)
    return out

from torch.utils.data import DataLoader


class Config(object):
    encoder_lr = 0.001
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CT_d_model = 128
    lstm_hidden_size = 64
    CT_height = 64
    CT_enc_ffc_hidden = 128
    CT_d_k = 64
    CT_d_v = 64
    num_out = 2
    CT_in_channels = 64
    CT_max_len_enco = 3
    batch_size = 64
    neg_ratio = 20
def collate_fn(dataset):
    patients_id = []
    pydicom_CT_before = []
    patients_pydicom_len_CT_before = []
    patients_pydicom_CT = []
    patients_pydicom_len_CT = []
    patients_ending = []
    for i in dataset:
        patients_id.append(i[0])
        for j in i[1]:
            pydicom_CT_before.append(torch.from_numpy(j))
        patients_pydicom_len_CT_before.append(i[2])
        for j in i[3]:
            patients_pydicom_CT.append(torch.from_numpy(j))
        patients_pydicom_len_CT.append(i[4])
        patients_ending.append(i[5])

    return patients_id,pydicom_CT_before,patients_pydicom_len_CT_before,\
           patients_pydicom_CT,patients_pydicom_len_CT,patients_ending

class PYDICOM_series_MR_CT_dataset(Dataset):
    def __init__(self, patients_id,  pydicom_CT_before, patients_pydicom_len_CT_before,
                 pydicom_CT, patients_pydicom_len_CT, patients_ending_CT):
        self.patients_id = patients_id
        self.pydicom_CT_before = pydicom_CT_before
        self.patients_pydicom_len_CT_before = patients_pydicom_len_CT_before
        self.pydicom_CT = pydicom_CT
        self.patients_pydicom_len_CT = patients_pydicom_len_CT
        self.label = patients_ending_CT

    def __getitem__(self, item):
        return self.patients_id[item],  self.pydicom_CT_before[item], \
               self.patients_pydicom_len_CT_before[item], self.pydicom_CT[item], self.patients_pydicom_len_CT[item], \
               self.label[item]

    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    # df_CT = pd.read_pickle('./data/df_CT_all_shaped.pkl')
    # patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    # df_CT_before = df_CT[df_CT.dis_day<0]
    # patients_id = []
    # patients_pydicom = []
    # patients_disday = []
    # patients_pydicom_len = []
    # patients_ending = []
    # patients_info = patients_info[patients_info['出院时结局'].isin(['死亡','临床治愈','好转'])]
    # patients_info['ending'] = patients_info['出院时结局'].apply(lambda x:0 if x=='死亡' else 1)
    # for name,grouped in df_CT_before.groupby('patient_id'):
    #     # print(name)
    #     if name in patients_info['编号'].values:
    #         grouped = grouped.sort_values(by='dis_day')
    #         grouped_disday = []
    #         grouped_images = []
    #
    #         for index, row in grouped.iterrows():
    #             grouped_images.append(np.array(row['images']))
    #             grouped_disday.append(row['dis_day'])
    #         patients_id.append(name)
    #         patients_disday.append(grouped_disday)
    #         patients_pydicom.append(grouped_images)
    #         patients_pydicom_len.append(len(grouped))
    #         patients_ending.append(patients_info.loc[patients_info['编号']==name,'ending'].values.item())
    #     else:
    #         print(name)
    #
    # torch.save(patients_id,"./data/patients_id_pydicom_ending_bf.pt")
    # torch.save(patients_disday, "./data/patients_disday_pydicom_ending_bf.pt")
    # torch.save(patients_pydicom, "./data/patients_pydicom_ending_bf.pt")
    # torch.save(patients_pydicom_len, "./data/patients_pydicom_len_ending_bf.pt")
    # torch.save(patients_ending, "./data/patients_ending_ending_bf.pt")

    patients_id_CT = torch.load("./data/patients_id_pydicom_ending_bf.pt")
    patients_pydicom_CT = torch.load("./data/patients_pydicom_ending_bf.pt")
    patients_pydicom_len_CT = torch.load("./data/patients_pydicom_len_ending_bf.pt")
    patients_ending_CT = torch.load("./data/patients_ending_ending_bf.pt")
    patients_pydicom_disday_CT = torch.load("./data/patients_disday_pydicom_ending_bf.pt")

    dataset = PYDICOM_series_MR_CT_dataset(patients_id_CT,  patients_pydicom_CT,patients_pydicom_len_CT,patients_pydicom_CT, patients_pydicom_len_CT, patients_ending_CT)
    opt = Config()
    weight = [opt.neg_ratio, 1]
    for fold in range(0, 5):
        # fold = 3
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        X_train_NLP = torch.load('./data/Integration_data/NLP_X_train_ending_clinical-{}.pt'.format(str(fold)))
        patients_id_train = []
        X_train = []
        X_train_copy = []
        X_valid = []
        for i in X_train_NLP:
            patients_id_train.append(i[0])
        for i in dataset:
            if i[0] in patients_id_train:
                X_train.append(i)
                if i[-1] == 0:
                    X_train_copy.append(i)
            else:
                X_valid.append(i)

        X_train = X_train + X_train_copy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_loader_train = DataLoader(X_train, batch_size=opt.batch_size, collate_fn=collate_fn,shuffle=True)

        ResNet_transformer_model_encoder = ResNet_transformer_encoder_CT(opt.CT_in_channels, opt.CT_height,
                                                                         opt.CT_d_model, 5,
                                                                         opt.CT_enc_ffc_hidden, opt.CT_d_k, opt.CT_d_v,
                                                                         opt.num_out, opt.device).to(device)
        optimizer_encoder = torch.optim.SGD(ResNet_transformer_model_encoder.parameters(), lr=opt.encoder_lr)
        # criterion = nn.BCELoss().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([opt.neg_ratio, 1])).to(device)


        for epoch in tqdm(range(opt.epochs)):
            for idx, (batch_id,  batch_pydicom_CT_before, batch_pydicom_CT_before_len,batch_pydicom_CT, batch_CT_len ,batch_ending) in enumerate(data_loader_train):
                batch_pydicom_CT_pad = torch.stack(batch_pydicom_CT_before).float()
                encoder_output,encoder_atten,encoder_hidden,encoder_logits = ResNet_transformer_model_encoder(batch_pydicom_CT_pad.to(device),batch_pydicom_CT_before_len)
                batch_ending = to_categorical(batch_ending, opt.num_out)
                loss = criterion(encoder_logits,torch.from_numpy(batch_ending).float().to(device))
                loss.backward();
                optimizer_encoder.step();
                optimizer_encoder.zero_grad()
            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));
        torch.save(ResNet_transformer_model_encoder,'./data/models/Integration_ending/ResNet_transformer_model_encoder_ending_CT_bf-{}.pkl'.format(str(fold)))

        torch.save(X_train, './data/Integration_data/CT_X_train_ending_bf-{}.pt'.format(str(fold)))
        torch.save(X_valid, './data/Integration_data/CT_X_valid_ending_bf-{}.pt'.format(str(fold)))