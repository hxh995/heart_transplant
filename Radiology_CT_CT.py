import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet_transformer_encoder_CT,ResNet_transformer_decoder,ResNet_transformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.ndimage
import cv2

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
    encoder_lr = 0.01
    epochs = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_lr = 0.001
    CT_d_model = 64
    lstm_hidden_size = 32
    CT_height = 64
    CT_enc_ffc_hidden = 32
    CT_d_k = 32
    CT_d_v = 32
    CT_in_channels = 64
    CT_max_len_encoder = 3
    CT_max_len_decoder = 5
    batch_size = 25


if __name__ == '__main__':

    df_CT = pd.read_pickle('./data/df_CT_all_shaped.pkl')
    patients_info = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',
                                    index_col=0).drop_duplicates(subset=['patient_id'])
    df_CT_before = df_CT[df_CT.dis_day<0]
    df_CT_after = df_CT[df_CT.dis_day>=0]
    patients_id = []
    patients_pydicom_bf = []
    patients_pydicom_af = []
    patients_disday_bf = []
    patients_disday_af = []
    patients_pydicom_len_bf = []
    patients_pydicom_len_af = []
    patients_ending = []

    patient_id_before = []
    patient_id_after = []
    for name,grouped in df_CT.groupby('patient_id'):
        # print(name)
        if int(name) in patients_info['patient_id'].values:
            grouped_before = grouped[grouped.dis_day<0]
            grouped_after = grouped[grouped.dis_day>=0]
            patients_id.append(name)
            patients_ending.append(patients_info.loc[patients_info['patient_id'] == name, 'is_lung'].values.item())
            if len(grouped_before)!=0:
                patient_id_before.append(name)
                grouped_before = grouped_before.sort_values(by='dis_day')
                grouped_disday_before = []
                grouped_images_before = []

                for index, row in grouped_before.iterrows():
                    grouped_images_before.append(np.array(row['images']))
                    grouped_disday_before.append(row['dis_day'])

                patients_disday_bf.append(grouped_disday_before)
                patients_pydicom_bf.append(grouped_images_before)
                patients_pydicom_len_bf.append(len(grouped_before))

            if len(grouped_after)!=0:
                patient_id_after.append(name)
                grouped_after = grouped_after.sort_values(by='dis_day')
                grouped_disday_after = []
                grouped_images_after = []

                for index, row in grouped_after.iterrows():
                    grouped_images_after.append(np.array(row['images']))
                    grouped_disday_after.append(row['dis_day'])
                patients_disday_af.append(grouped_disday_after)
                patients_pydicom_af.append(grouped_images_after)
                patients_pydicom_len_af.append(len(grouped_after))

        else:
            print('not')
            print(name)
    #



    # torch.save(patients_id,"./data/patients_id_pydicom_ending_all.pt")
    # torch.save(patients_disday_bf, "./data/patients_disday_pydicom_ending_bf_all.pt")
    # torch.save(patients_disday_af, "./data/patients_disday_pydicom_ending_af_all.pt")
    # torch.save(patients_pydicom_bf, "./data/patients_pydicom_ending_bf_all.pt")
    # torch.save(patients_pydicom_len_bf, "./data/patients_pydicom_len_ending_bf_all.pt")
    # torch.save(patients_pydicom_af, "./data/patients_pydicom_ending_af_all.pt")
    # torch.save(patients_pydicom_len_af, "./data/patients_pydicom_len_ending_af_all.pt")
    # torch.save(patients_ending, "./data/patients_ending_ending_all.pt")


    # patients_id_CT = torch.load("./data/patients_id_pydicom_ending_all.pt")
    # patients_ending_CT = torch.load("./data/patients_ending_ending_all.pt")
    # patients_pydicom_CT_bf = torch.load("./data/patients_pydicom_ending_bf_all.pt")
    # patients_pydicom_len_CT_bf = torch.load("./data/patients_pydicom_len_ending_bf_all.pt")
    # patients_pydicom_CT_af = torch.load("./data/patients_pydicom_ending_af_all.pt")
    # patients_pydicom_len_CT_af = torch.load("./data/patients_pydicom_len_ending_af_all.pt")
    #
    # dataset = PYDICOM_series_MR_CT_dataset(patients_id_CT,patients_pydicom_CT_bf,patients_pydicom_len_CT_bf,patients_pydicom_CT_af,patients_pydicom_len_CT_af,patients_ending_CT)
    # #
    # for fold in range(0,5):
    #     # fold = 3
    #     torch.cuda.empty_cache()
    #     torch.cuda.empty_cache()
    #     torch.cuda.empty_cache()
    #     torch.cuda.empty_cache()
    #     torch.cuda.empty_cache()
    #     torch.cuda.empty_cache()
    #     torch.cuda.empty_cache()
    #     CF_X_train = torch.load('./data/Integration_data/X_train_CF_NLP-{}.pt'.format(str(fold)))
    #     X_train_id = []
    #     for i in CF_X_train:
    #         X_train_id.append(i[0])
    #     X_train = []
    #     X_valid = []
    #     X_train_copy = []
    #
    #     for i in dataset:
    #         if i[0] in X_train_id:
    #             X_train.append(i)
    #             if i[-1] == 0:
    #                 X_train_copy.append(i)
    #         else:
    #             X_valid.append(i)
    #     X_train = X_train + X_train_copy
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     opt = Config()
    #     ResNet_transformer_model = ResNet_transformer(opt.CT_in_channels, opt.CT_height,opt.CT_d_model, opt.lstm_hidden_size,opt.CT_max_len_encoder,opt.CT_max_len_decoder,
    #                                                   opt.CT_enc_ffc_hidden, opt.CT_d_k, opt.CT_d_v, opt.num_out, opt.device).to(device)
    #     ResNet_transformer_model.train()
    #     optimizer = torch.optim.SGD(ResNet_transformer_model.parameters(), lr=opt.encoder_lr)
    #
    #     dataloader = DataLoader(X_train, batch_size=opt.batch_size, collate_fn=collate_func_CT_MR)
    #     criterion = nn.BCELoss().to(device)
    #     accum_steps = 8
    #     for epoch in tqdm(range(opt.epochs)):
    #         for idx, (batch_id, batch_pydicom_bf, batch_bf_len,batch_pydicom_CT, batch_CT_len ,batch_ending) in enumerate(dataloader):
    #             batch_pydicom_bf_pad = torch.FloatTensor(batch_pydicom_bf)
    #             batch_pydicom_af_pad = torch.FloatTensor(batch_pydicom_CT)
    #             __, output_logits = ResNet_transformer_model(batch_pydicom_bf_pad.to(device),batch_pydicom_af_pad.to(device),batch_bf_len,batch_CT_len)
    #             batch_ending = to_categorical(batch_ending, opt.num_out)
    #             loss = criterion(F.softmax(output_logits, dim=-1), torch.from_numpy(batch_ending).float().to(device)) / accum_steps
    #             loss.backward();
    #         if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
    #             optimizer.step();
    #             optimizer.zero_grad()
    #         if (epoch + 1) % 50 == 0:
    #             print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()*accum_steps));
    #
    #     torch.save(ResNet_transformer_model,'./data/models/Integration_lung/ResNet_transformer_model_CT_ending-{}.pkl'.format(str(fold)))
    #
    #     torch.save(X_train,'./data/Integration_data/CT_X_train_ending-{}.pt'.format(str(fold)))
    #     torch.save(X_valid,'./data/Integration_data/CT_X_valid_ending-{}.pt'.format(str(fold)))
