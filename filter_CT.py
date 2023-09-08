import matplotlib.pyplot as plt
import cv2
import pandas as pd
import torch,os
import torch.nn as nn
import torch.utils.data as data
from collections import Counter
import torch.nn.functional as F
from utils import caculate_auc,acu_curve,to_categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet_transformer_filter,Resnet_hidden_lstm_64_channel_filter,ResNet_transformer
from tqdm import tqdm
from utils import caculate_auc
from torch.utils.data import DataLoader
from Radiology_CT_lung import PYDICOM_series_MR_CT_dataset
import numpy as np
from Radiology_CT_lung import PYDICOM_series_MR_CT_dataset



class PYDICOM_series_dataset(Dataset):
    def __init__(self,patients_id,patients_index_id,patients_pydicom,patients_ending):
        self.patients_id = patients_id
        self.patients_index_id = patients_index_id
        self.pydicom_datas = patients_pydicom
        self.label = patients_ending

    def __getitem__(self,item):
        return self.patients_id[item],self.patients_index_id[item],self.pydicom_datas[item],self.label[item]
    def __len__(self):
        return len(self.label)




class Config(object):
    Resnet_lr = 0.001
    lstm_lr = 0.01
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 63
    in_channels = 64
    num_out = 2



def collate_func_CT_MR(dataset):
    patients_id = []
    pydicom_CT_before = []
    patients_ending = []
    for i in dataset:
        patients_id.append(i[0])
        pydicom_CT_before.append(i[1])
        patients_ending.append(i[2])

    return patients_id,pydicom_CT_before,patients_ending

if __name__ == '__main__':

    # df_CT = pd.read_pickle('./data/df_CT.pkl')
    # patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'姓名': 'patient_name', '编号': 'patient_id'})
    #
    # patients_id = []
    # patients_pydicom = []
    # patients_ending = []
    #
    # for index, row in df_CT.iterrows():
    #     patient_id_new = '_'.join([str(row['patient_id']),str(row['dis_day'])])
    #     print(patient_id_new)
    #     patients_id.append(patient_id_new)
    #     # print(row['images'].shape)
    #     patients_pydicom.append(row['images'])
    #     patients_ending.append(row['is_lung'])
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # CF_hidden_train_df = pd.read_excel('./data/Integration_data/CF_hidden_train_df_lung-{}.xlsx'.format(str(3)), index_col=0)
    # X_train_id = CF_hidden_train_df.patient_id.unique()
    # dataset = PYDICOM_series_dataset(patients_id, patients_pydicom, patients_ending)
    # X_train = []
    # X_valid = []
    # X_train_image_enhance = []
    # for i in dataset:
    #     patients_id = int(i[0].split('_')[0])
    #     if patients_id in X_train_id:
    #         X_train.append(i)
    #         # patients_id = i[0]
    #         # image = cv2.flip(i[1], 0)
    #         # ending = i[2]
    #         # X_train_image_enhance.append((patients_id,image,ending))
    #     else:
    #         X_valid.append(i)
    #
    # # X_train = X_train + X_train_image_enhance
    # opt = Config()
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # ResNet_transformer_model = ResNet_transformer_filter(opt.in_channels, opt.height,opt.device).to(device)
    #
    # ResNet_transformer_lstm = Resnet_hidden_lstm_64_channel_filter(opt.height, opt.num_out, device).to(device)
    #
    # optimizer_Resnet = torch.optim.SGD(ResNet_transformer_model.parameters(), lr=opt.Resnet_lr)
    # optimizer_lstm = torch.optim.SGD(ResNet_transformer_lstm.parameters(), lr=opt.lstm_lr)
    #
    # dataloader = DataLoader(X_train, batch_size=opt.batch_size)
    # criterion = nn.BCELoss().to(device)
    #
    # ResNet_transformer_model.train()
    # ResNet_transformer_lstm.train()
    #
    # for epoch in tqdm(range(opt.epochs)):
    #     for idx, (batch_id, batch_pydicom_data, batch_ending) in enumerate(dataloader):
    #         x = ResNet_transformer_model(batch_pydicom_data.to(device))
    #         __, output_logits = ResNet_transformer_lstm(x)
    #         batch_ending = to_categorical(batch_ending, opt.num_out)
    #         loss = criterion(F.softmax(output_logits, dim=-1), torch.from_numpy(batch_ending).float().to(device))
    #         loss.backward();
    #         optimizer_Resnet.step();
    #         optimizer_Resnet.zero_grad()
    #         optimizer_lstm.step();
    #         optimizer_lstm.zero_grad()
    #     if (epoch + 1) % 50 == 0:
    #         print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));
    #
    # with torch.no_grad():
    #     ResNet_transformer_model.eval()
    #     ResNet_transformer_lstm.eval()
    #     flag = False
    #     dataloader = DataLoader(X_valid, batch_size=opt.batch_size)
    #     for batch_patient_id, batch_pydicom_data,batch_ending in dataloader:
    #         x = ResNet_transformer_model(batch_pydicom_data.to(device))
    #         batch_pydicom_hidden, output_logits = ResNet_transformer_lstm(x)
    #         for i in range(len(batch_ending)):
    #             if batch_ending[i] == 1:
    #                 print(batch_patient_id[i])
    #                 print(F.softmax(output_logits[i], dim=-1))
    #         if flag:
    #             output = torch.cat([output, output_logits], dim=0)
    #             output_label = torch.cat([output_label ,batch_ending],dim=0)
    #             patient_id = patient_id + batch_patient_id
    #             pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
    #         else:
    #             output = output_logits
    #             output_label = batch_ending
    #             patient_id = batch_patient_id
    #             pydicom_hidden = batch_pydicom_hidden
    #             flag = True
    #         # print(logits)
    #         # print(label_batch_valid)
    #     logits = F.softmax(output, dim=-1);
    #     output_label_ = to_categorical(output_label.detach().cpu().numpy().tolist(),2)
    #     for i in range(logits.shape[1]):
    #         fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
    #         print(roc_auc)

    complication_df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0).drop_duplicates(subset=['patient_id'])
    sick_name = 'ending'
    py_type = 'bf'
    if sick_name == 'is_lung':
        df_CT = pd.read_pickle('./data/filter_CT/df_CT_all_shaped_lung_2.pkl')
    else:
        df_CT = pd.read_pickle("./data/df_CT_all_shaped.pkl".format(sick_name))
    df_CT['index_id'] = range(0,len(df_CT))
    df_CT_before = df_CT[df_CT.dis_day <= 0]
    df_CT_after = df_CT[df_CT.dis_day > 0]
    print(len(df_CT.patient_id.unique()))

    patients_id = []
    patients_index_id = []
    patients_pydicom = []
    patients_ending_lung = []
    if py_type == 'bf':
        df_CT_processed = df_CT_before
    else:
        df_CT_processed = df_CT_after

    for index, row in df_CT_processed.iterrows():
        patient_id_new = '_'.join([str(row['patient_id']),str(row['dis_day'])])
        print(patient_id_new)
        if row['images'].max()==0:
            continue
        if len(complication_df.loc[complication_df['patient_id'] == row['patient_id']]) == 0:
            continue
        patients_id.append(row['patient_id'])
        patients_index_id.append(row['index_id'])
        # print(row['images'].shape)
        patients_pydicom.append(np.array(row['images']))
        patients_ending_lung.append(complication_df.loc[complication_df['patient_id'] == row['patient_id'], sick_name].values[0])

    # torch.save(patients_id,"./data/filter_CT/patients_id_CT.pt")
    # torch.save(patients_pydicom, "./data/filter_CT/patients_pydicom.pt")
    # torch.save(patients_ending_lung, "./data/filter_CT/patients_ending_lung.pt")

    # patients_id_CT = torch.load("./data/filter_CT/patients_id_CT.pt")[0:100]
    # patients_pydicom = torch.load("./data/filter_CT/patients_pydicom.pt")[0:100]
    # patients_ending_lung = torch.load("./data/filter_CT/patients_ending_lung.pt")[0:100]



    dataset = PYDICOM_series_dataset(patients_id,patients_index_id,patients_pydicom, patients_ending_lung)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    opt = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ResNet_transformer_lstm.train()
    Valid_is = True
    if Valid_is:
        ResNet_transformer_model = ResNet_transformer_filter(opt.in_channels).to(device)

        ResNet_transformer_lstm = Resnet_hidden_lstm_64_channel_filter(opt.num_out, device).to(device)

        optimizer_Resnet = torch.optim.SGD(ResNet_transformer_model.parameters(), lr=opt.Resnet_lr)
        optimizer_lstm = torch.optim.SGD(ResNet_transformer_lstm.parameters(), lr=opt.lstm_lr)
        # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1,10])).to(device)
        criterion = nn.BCELoss().to(device)
        ResNet_transformer_model.train()

        if py_type == 'bf':
            CF_hidden_valid_df = pd.read_excel(
                './data/Integration_data/CF_hidden_valid_df_{}-{}.xlsx'.format(sick_name, 'VALID'))
        else:
            CF_hidden_train_df = pd.read_excel(
                './data/Integration_data/CF_hidden_valid_df_{}-{}_multi.xlsx'.format(sick_name, 'VALID'))

        X_valid_id = CF_hidden_valid_df.patient_id.unique()
        X_train = []
        X_valid = []
        for i in dataset:
            if int(i[0]) in X_valid_id:
                X_valid.append(i)
            else:
                X_train.append(i)
        dataloader = DataLoader(X_train, batch_size=opt.batch_size)
        for epoch in tqdm(range(opt.epochs)):
            for idx, (batch_id, batch_id_index, batch_pydicom_data, batch_ending) in enumerate(dataloader):
                # print(idx)
                # print(batch_id)
                x = ResNet_transformer_model(batch_pydicom_data.float().to(device))
                __, output_logits = ResNet_transformer_lstm(x)
                batch_ending = to_categorical(batch_ending, opt.num_out)
                loss = criterion(F.softmax(output_logits, dim=-1), torch.from_numpy(batch_ending).float().to(device))
                loss.backward();
                optimizer_Resnet.step();
                optimizer_Resnet.zero_grad()
                optimizer_lstm.step();
                optimizer_lstm.zero_grad()

            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));

        with torch.no_grad():

            ResNet_transformer_model.eval()
            ResNet_transformer_lstm.eval()
            flag = False
            for batch_patient_id, batch_index_id, batch_pydicom_data, batch_ending in dataloader:
                x = ResNet_transformer_model(batch_pydicom_data.float().to(device))
                batch_pydicom_hidden, output_logits = ResNet_transformer_lstm(x)
                # for i in range(len(batch_ending)):
                #     if batch_ending[i] == 1:
                #         print(batch_patient_id[i])
                #         print(F.softmax(output_logits[i], dim=-1))
                if flag:
                    output = torch.cat([output, output_logits], dim=0)
                    output_label = torch.cat([output_label, batch_ending], dim=0)
                    patient_id = torch.cat([patient_id, batch_patient_id], dim=0)
                    patients_id_index = torch.cat([patients_id_index, batch_index_id], dim=0)
                    pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
                else:
                    output = output_logits
                    output_label = batch_ending
                    patient_id = batch_patient_id
                    patients_id_index = batch_index_id
                    pydicom_hidden = batch_pydicom_hidden
                    flag = True
            print("Train_Counter")
            print(Counter(output_label.detach().cpu().numpy().tolist()))
            pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
            pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
            pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
            pydicom_hidden_df['patient_id'] = patient_id
            df_path = './data/Integration_data/CT_hidden_train_df_{}_{}_{}.xlsx'.format(sick_name, py_type, 'VALID')
            pydicom_hidden_df.to_excel(df_path)

            flag = False
            dataloader = DataLoader(X_valid, batch_size=opt.batch_size)
            for batch_patient_id, batch_index_id, batch_pydicom_data, batch_ending in dataloader:
                x = ResNet_transformer_model(batch_pydicom_data.float().to(device))
                batch_pydicom_hidden, output_logits = ResNet_transformer_lstm(x)
                # for i in range(len(batch_ending)):
                #     if batch_ending[i] == 1:
                #         print(batch_patient_id[i])
                #         print(F.softmax(output_logits[i], dim=-1))
                if flag:
                    output = torch.cat([output, output_logits], dim=0)
                    output_label = torch.cat([output_label, batch_ending], dim=0)
                    patient_id = torch.cat([patient_id, batch_patient_id], dim=0)
                    patients_id_index = torch.cat([patients_id_index, batch_index_id], dim=0)
                    pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
                else:
                    output = output_logits
                    output_label = batch_ending
                    patient_id = batch_patient_id
                    patients_id_index = batch_index_id
                    pydicom_hidden = batch_pydicom_hidden
                    flag = True
                # print(logits)
                # print(label_batch_valid)
            print("Valid_Counter")
            print(Counter(output_label.detach().cpu().numpy().tolist()))
            logits = F.softmax(output, dim=-1);
            output_label_ = to_categorical(output_label.detach().cpu().numpy().tolist(), 2)
            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)

            pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
            pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
            pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
            pydicom_hidden_df['patient_id'] = patient_id
            df_path = './data/Integration_data/CT_hidden_valid_df_{}_{}_{}.xlsx'.format(sick_name, py_type, 'VALID')
            pydicom_hidden_df.to_excel(df_path)

            torch.save(output_label_[:, 1],
                       './data/Integration_data/patients_valid_CT_{}_{}-{}.pt'.format(sick_name, py_type, 'VALID'))
            torch.save(logits[:, 1].detach().cpu().numpy(),
                       './data/Integration_data/logits_CT_{}_{}-{}.pt'.format(sick_name, py_type, 'VALID'))
            for i in range(len(output_label_)):
                if output_label_[i][1] == 1:
                    if logits[i][0] > 0.8:
                        patient_id_index_drop = patients_id_index[i].item()
                        if len(df_CT.loc[df_CT['index_id'] == patient_id_index_drop, 'patient_id']) == 0:
                            continue
                        else:
                            patient_id_drop = \
                            df_CT.loc[df_CT['index_id'] == patient_id_index_drop, 'patient_id'].values[0]
                            print(df_CT.loc[df_CT['patient_id'] == patient_id_drop, :])
                            df_CT = df_CT[df_CT['index_id'] != patients_id_index[i].item()]

        if sick_name == 'is_lung':
            df_CT.to_pickle("./data/filter_CT/df_CT_all_shaped_lung_2.pkl")
        else:
            df_CT.to_pickle("./data/filter_CT/df_CT_all_shaped_{}_1.pkl".format(sick_name))

    else:
        for fold in range(0,5):
            ResNet_transformer_model = ResNet_transformer_filter(opt.in_channels).to(device)

            ResNet_transformer_lstm = Resnet_hidden_lstm_64_channel_filter(opt.num_out, device).to(device)

            optimizer_Resnet = torch.optim.SGD(ResNet_transformer_model.parameters(), lr=opt.Resnet_lr)
            optimizer_lstm = torch.optim.SGD(ResNet_transformer_lstm.parameters(), lr=opt.lstm_lr)
            # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1,10])).to(device)
            criterion = nn.BCELoss().to(device)
            ResNet_transformer_model.train()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            if py_type == 'bf':
                CF_hidden_train_df = pd.read_excel('./data/Integration_data/CF_hidden_train_df_{}-{}.xlsx'.format(sick_name, str(fold)))
            else:
                CF_hidden_train_df = pd.read_excel('./data/Integration_data/CF_hidden_train_df_{}-{}_multi.xlsx'.format(sick_name, str(fold)))


            X_train_id = CF_hidden_train_df.patient_id.unique()
            X_train = []
            X_valid = []
            for i in dataset:
                if int(i[0]) in X_train_id:
                    X_train.append(i)
                else:
                    X_valid.append(i)
            dataloader = DataLoader(X_train, batch_size=opt.batch_size)
            for epoch in tqdm(range(opt.epochs)):
                for idx, (batch_id, batch_id_index,batch_pydicom_data, batch_ending) in enumerate(dataloader):
                    # print(idx)
                    # print(batch_id)
                    x = ResNet_transformer_model(batch_pydicom_data.float().to(device))
                    __, output_logits = ResNet_transformer_lstm(x)
                    batch_ending = to_categorical(batch_ending, opt.num_out)
                    loss = criterion(F.softmax(output_logits, dim=-1), torch.from_numpy(batch_ending).float().to(device))
                    loss.backward();
                    optimizer_Resnet.step();
                    optimizer_Resnet.zero_grad()
                    optimizer_lstm.step();
                    optimizer_lstm.zero_grad()

                if (epoch + 1) % 50 == 0:
                    print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));



            with torch.no_grad():

                ResNet_transformer_model.eval()
                ResNet_transformer_lstm.eval()
                flag = False
                for batch_patient_id, batch_index_id,batch_pydicom_data,batch_ending in dataloader:
                    x = ResNet_transformer_model(batch_pydicom_data.float().to(device))
                    batch_pydicom_hidden, output_logits = ResNet_transformer_lstm(x)
                    # for i in range(len(batch_ending)):
                    #     if batch_ending[i] == 1:
                    #         print(batch_patient_id[i])
                    #         print(F.softmax(output_logits[i], dim=-1))
                    if flag:
                        output = torch.cat([output, output_logits], dim=0)
                        output_label = torch.cat([output_label ,batch_ending],dim=0)
                        patient_id = torch.cat([patient_id,batch_patient_id],dim=0)
                        patients_id_index = torch.cat([patients_id_index,batch_index_id],dim=0)
                        pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
                    else:
                        output = output_logits
                        output_label = batch_ending
                        patient_id = batch_patient_id
                        patients_id_index = batch_index_id
                        pydicom_hidden = batch_pydicom_hidden
                        flag = True
                print("Train_Counter")
                print(Counter(output_label.detach().cpu().numpy().tolist()))
                pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
                pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
                pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
                pydicom_hidden_df['patient_id'] = patient_id
                df_path = './data/Integration_data/CT_hidden_train_df_{}_{}_{}.xlsx'.format(sick_name, py_type, str(fold))
                pydicom_hidden_df.to_excel(df_path)

                flag = False
                dataloader = DataLoader(X_valid, batch_size=opt.batch_size)
                for batch_patient_id, batch_index_id,batch_pydicom_data,batch_ending in dataloader:
                    x = ResNet_transformer_model(batch_pydicom_data.float().to(device))
                    batch_pydicom_hidden, output_logits = ResNet_transformer_lstm(x)
                    # for i in range(len(batch_ending)):
                    #     if batch_ending[i] == 1:
                    #         print(batch_patient_id[i])
                    #         print(F.softmax(output_logits[i], dim=-1))
                    if flag:
                        output = torch.cat([output,output_logits], dim=0)
                        output_label = torch.cat([output_label,batch_ending],dim=0)
                        patient_id = torch.cat([patient_id,batch_patient_id],dim=0)
                        patients_id_index = torch.cat([patients_id_index,batch_index_id],dim=0)
                        pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
                    else:
                        output = output_logits
                        output_label = batch_ending
                        patient_id = batch_patient_id
                        patients_id_index = batch_index_id
                        pydicom_hidden = batch_pydicom_hidden
                        flag = True
                    # print(logits)
                    # print(label_batch_valid)
                print("Valid_Counter")
                print(Counter(output_label.detach().cpu().numpy().tolist()))
                logits = F.softmax(output, dim=-1);
                output_label_ = to_categorical(output_label.detach().cpu().numpy().tolist(),2)
                for i in range(logits.shape[1]):
                    fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
                    print(roc_auc)

                pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
                pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
                pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
                pydicom_hidden_df['patient_id'] = patient_id
                df_path = './data/Integration_data/CT_hidden_valid_df_{}_{}_{}.xlsx'.format(sick_name, py_type, str(fold))
                pydicom_hidden_df.to_excel(df_path)

                torch.save(output_label_[:, 1],
                           './data/Integration_data/patients_valid_CT_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
                torch.save(logits[:, 1].detach().cpu().numpy(),
                           './data/Integration_data/logits_CT_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
                for i in range(len(output_label_)):
                    if output_label_[i][1] == 1:
                        if logits[i][0] > 0.8:
                            patient_id_index_drop = patients_id_index[i].item()
                            if len(df_CT.loc[df_CT['index_id']==patient_id_index_drop,'patient_id']) == 0:
                                continue
                            else:
                                patient_id_drop = df_CT.loc[df_CT['index_id']==patient_id_index_drop,'patient_id'].values[0]
                                print(df_CT.loc[df_CT['patient_id']==patient_id_drop,:])
                                df_CT = df_CT[df_CT['index_id']!=patients_id_index[i].item()]
                print(fold)
                print(len(df_CT.patient_id.unique()))
        if sick_name == 'is_lung':
            df_CT.to_pickle("./data/filter_CT/df_CT_all_shaped_lung_2.pkl")
        else:
            df_CT.to_pickle("./data/filter_CT/df_CT_all_shaped_{}_1.pkl".format(sick_name))







