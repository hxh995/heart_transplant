import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet,ResNet_transformer_encoder
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
    def __init__(self,patients_id, patients_age, pydicom_MR, patients_pydicom_len_MR,
                 pydicom_CT, patients_pydicom_len_CT, patients_ending_CT):
        self.patients_id = patients_id
        self.patients_age = patients_age
        self.pydicom_MR = pydicom_MR
        self.patients_pydicom_len_MR = patients_pydicom_len_MR
        self.pydicom_CT = pydicom_CT
        self.patients_pydicom_len_CT = patients_pydicom_len_CT
        self.label = patients_ending_CT
    def __getitem__(self,item):
        return self.patients_id[item],self.patients_age[item],self.pydicom_MR[item],\
               self.patients_pydicom_len_MR[item],self.pydicom_CT[item],self.patients_pydicom_len_CT[item],\
               self.label[item]
    def __len__(self):
        return len(self.label)

def collate_func_CT_MR(dataset):
    patients_id = []
    patients_age = []
    patients_MR = []
    patients_pydicom_len_MR = []
    patients_pydicom_CT = []
    patients_pydicom_len_CT = []
    patients_ending = []
    for i in dataset:
        patients_id.append(i[0])
        patients_age.append(i[1])
        patients_MR.append(torch.from_numpy(i[2]))
        patients_pydicom_len_MR.append(i[3])
        for j in i[4]:
            patients_pydicom_CT.append(j)
        patients_pydicom_len_CT.append(i[5])
        patients_ending.append(i[6])

    return patients_id,patients_age,patients_MR,patients_pydicom_len_MR,\
           patients_pydicom_CT,patients_pydicom_len_CT,patients_ending



class Config(object):
    encoder_lr = 0.001
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MR_d_model = 128
    MR_height = 256
    MR_enc_ffc_hidden = 56
    MR_d_k = 64
    MR_d_v = 64
    num_out = 2
    MR_in_channels = 72
    MR_max_len_enco = 256


    Resnet_lr = 0.001
    decoder_lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CT_d_model = 128
    lstm_hidden_size = 64
    CT_height = 64
    CT_enc_ffc_hidden = 64
    CT_d_k = 64
    CT_d_v = 64
    num_out = 2
    CT_in_channels = 64
    CT_max_len_enco = 5
    batch_size = 36

    decay = 0.7

if __name__ == '__main__':

    patients_id_MR = torch.load("./data/patients_id_pydicom.pt")
    patients_pydicom_MR = torch.load("./data/patients_pydicom_pydicom.pt")
    patients_pydicom_len_MR = torch.load("./data/patients_pydicom_len_pydicom.pt")


    patients_id_CT = torch.load("./data/patients_id_pydicom_complication.pt")
    patients_age_CT = torch.load("./data/patients_age_pydicom_complication.pt")
    patients_pydicom_CT = torch.load("./data/patients_pydicom_complication.pt")
    patients_pydicom_len_CT = torch.load("./data/patients_pydicom_complication_len.pt")
    patients_ending_CT = torch.load("./data/patients_ending_complication.pt")
    patients_pydicom_disday_CT = torch.load("./data/patients_pydicom_complication_dis_day.pt")

    frame_number,__,width,heigth = patients_pydicom_MR[0].shape

    patients_pydicom_MR_encoder = []
    patients_pydicom_MR_len_encoder = []
    patients_id_MR_CT = []
    patients_stop_id = [622,627,565,604,538,478]

    for id_CT in patients_id_CT:
        if id_CT in patients_stop_id:
            print(id_CT)
            CT_index = patients_id_CT.index(id_CT)
            # del patients_id_CT[CT_index]
            # del patients_age_CT[CT_index]
            # del patients_pydicom_CT[CT_index]
            # del patients_pydicom_len_CT[CT_index]
            # del patients_ending_CT[CT_index]
            print(patients_id_CT.pop(CT_index))
            patients_age_CT.pop(CT_index)
            patients_pydicom_CT.pop(CT_index)
            patients_pydicom_len_CT.pop(CT_index)
            patients_ending_CT.pop(CT_index)
    for id_CT in patients_id_CT:
        if id_CT in patients_id_MR :
            MR_index = patients_id_MR.index(id_CT)
            # print(MR_index)
            patients_pydicom_MR_encoder.append(patients_pydicom_MR[MR_index])
            patients_pydicom_MR_len_encoder.append(patients_pydicom_len_MR[MR_index])
        else:
            zero_MR = np.zeros([frame_number,1,width,heigth])
            # zero_MR = np.expand_dims(zero_MR,axis=1)
            # print(zero_MR.shape)
            # breakpoint()
            patients_pydicom_MR_encoder.append(zero_MR)
            patients_pydicom_MR_len_encoder.append(0)


    print(len(patients_id_CT))
    print(len(patients_pydicom_MR_encoder))
    print(len(patients_pydicom_MR_len_encoder))


    #
    #
    opt = Config()

    dataset = PYDICOM_series_MR_CT_dataset(patients_id_CT,patients_age_CT,patients_pydicom_MR_encoder,patients_pydicom_MR_len_encoder,
                                           patients_pydicom_CT,patients_pydicom_len_CT,patients_ending_CT)

    # for fold in range(5):
    fold = 3
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    CF_X_train = pd.read_excel('./data/Integration_data/CF_hidden_train_df_lung-{}.xlsx'.format(str(fold)), index_col=0)
    X_train_id = CF_X_train.patient_id.unique()
    X_train = []
    X_valid = []
    for i in dataset:
        if i[0] in X_train_id:
            X_train.append(i)
        else:
            X_valid.append(i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ResNet_transformer_model_encoder = ResNet_transformer_encoder(opt.MR_in_channels, opt.MR_height, opt.MR_d_model, opt.MR_max_len_enco,
                                                  opt.MR_enc_ffc_hidden, opt.MR_d_k, opt.MR_d_v, opt.num_out, opt.device).to(device)


    ResNet_transformer_model_encoder.train()


    ResNet_model = ResNet(opt.CT_in_channels, opt.CT_height, opt.device).to(device)

    ResNet_transformer_model_decoder = ResNet_transformer_decoder(opt.CT_height, 512, opt.CT_d_model, opt.lstm_hidden_size,opt.CT_max_len_enco,opt.CT_enc_ffc_hidden,opt.CT_d_k,opt.CT_d_v,opt.num_out,device).to(device)

    ResNet_model.train()
    ResNet_transformer_model_decoder.train()

    optimizer_encoder = torch.optim.SGD(ResNet_transformer_model_encoder.parameters(), lr=opt.encoder_lr)
    optimizer_Resnet = torch.optim.SGD(ResNet_model.parameters(), lr=opt.Resnet_lr)
    optimizer_decoder = torch.optim.SGD(ResNet_transformer_model_decoder.parameters(), lr=opt.decoder_lr)


    dataloader = DataLoader(X_train, batch_size=opt.batch_size, collate_fn=collate_func_CT_MR)
    criterion = nn.BCELoss().to(device)



    for epoch in tqdm(range(opt.epochs)):
        for idx, (batch_id, batch_age, batch_pydicom_MR, batch_MR_len,batch_pydicom_CT, batch_CT_len ,batch_ending) in enumerate(dataloader):
            batch_pydicom_MR_pad = torch.stack(batch_pydicom_MR)
            encoder_output,encoder_atten,encoder_output_logits = ResNet_transformer_model_encoder(batch_pydicom_MR_pad.squeeze(dim=2).to(device).float(),batch_MR_len)
            batch_dis_day = get_keys_from_dict(patients_pydicom_disday_CT,batch_id)
            batch_dis_day = [i for j in batch_dis_day for i in j]
            x = ResNet_model(torch.stack(batch_pydicom_CT).to(device),batch_CT_len)
            __, output_logits = ResNet_transformer_model_decoder(encoder_output,x,batch_CT_len,batch_dis_day,batch_age)
            batch_ending = to_categorical(batch_ending, opt.num_out)
            loss_encoder = criterion(encoder_output_logits,torch.from_numpy(batch_ending).float().to(device))
            loss_decoder = criterion(F.softmax(output_logits, dim=-1), torch.from_numpy(batch_ending).float().to(device))
            loss = loss_encoder*(1-opt.decay) + loss_decoder*opt.decay
            loss.backward();
            optimizer_encoder.step();
            optimizer_encoder.zero_grad()
            optimizer_Resnet.step();
            optimizer_Resnet.zero_grad()
            optimizer_decoder.step();
            optimizer_decoder.zero_grad()

        if (epoch + 1) % 50 == 0:
            print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));


    torch.save(ResNet_transformer_model_encoder,'./data/models/Integration_lung/ResNet_transformer_model_encoder_lung-{}.pkl'.format(str(fold)))
    torch.save(ResNet_model,'./data/models/Integration_lung/ResNet_model_lung-{}.pkl'.format(str(fold)))
    torch.save(ResNet_transformer_model_decoder,'./data/models/Integration_lung/ResNet_transformer_model_decoder_lung-{}.pkl'.format(str(fold)))

    torch.save(X_train,'./data/Integration_data/MR_CT_X_train-{}.pt'.format(str(fold)))
    torch.save(X_valid,'./data/Integration_data/MR_CT_X_valid-{}.pt'.format(str(fold)))


