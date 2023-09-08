import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
import cv2,re
from gensim.models import word2vec
import torch
import math, copy, time, os,jieba
import torch.utils.data as data
from collections import Counter
from tqdm import tqdm
from utils import to_categorical,caculate_auc,acu_curve
import torch.nn.functional as F
from models_building import Transformer
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils


class MyData(data.Dataset):
    def __init__(self, patients_id,patients_CF_before_operation,patients_CF_len_before_operation,patients_CF_after_operation,patients_CF_len_after_operation,patients_ending):
        self.patients_id = patients_id
        self.patients_CF_before_operation = patients_CF_before_operation
        self.patients_CF_len_before_operation = patients_CF_len_before_operation
        self.patients_CF_after_operation = patients_CF_after_operation
        self.patients_CF_len_after_operation = patients_CF_len_after_operation
        self.patients_ending = patients_ending
    def __len__(self):
        return len(self.patients_CF_before_operation)
    def __getitem__(self, item):
        return self.patients_id[item],self.patients_CF_before_operation[item], self.patients_CF_len_before_operation[item],self.patients_CF_after_operation[item], self.patients_CF_len_after_operation[item],self.patients_ending[item]



class Config(object):
    lr = 0.001
    epochs = 100
    num_out = 2
    feed_forward_hidden = 48
    d_model = 64
    d_k = 32
    d_v = 32
    batch_size = 36
    neg_ratio = 10
def collate_func(data):
    #print(type(data))
    patients_id = []
    patients_first_trip_dim = []
    patients_first_trip_len = []
    patients_pdf_dim = []
    patients_pdf_word_len = []
    patients_ending = []
    for i in data:
        patients_id.append(i[0])
        patients_first_trip_dim.append(i[1])
        patients_first_trip_len.append(i[2])
        patients_pdf_dim.append(i[3])
        patients_pdf_word_len.append(i[4])
        patients_ending.append(i[5])

    packed_first_trip_dim = rnn_utils.pad_sequence(patients_first_trip_dim, batch_first=True, padding_value=0)
    packed_patients_pdf_dim = rnn_utils.pad_sequence(patients_pdf_dim, batch_first=True, padding_value=0)

    return patients_id,packed_first_trip_dim ,patients_first_trip_len,packed_patients_pdf_dim,patients_pdf_word_len,patients_ending

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    df = pd.read_excel('./data/NLP_patient_context_concat_all.xlsx',index_col=0)
    # complication_df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx', index_col=0).drop_duplicates(subset=['patient_id'])
    complication_df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0).drop_duplicates(subset=['patient_id'])
    sick_name = 'ending'
    df = df.merge(complication_df[['patient_id']+[sick_name]],on='patient_id')
    py_type = 'bf'
    is_weight = False
    is_copy = True


    patient_context_first_trip = []
    patient_context_dis_day = []
    stopwords = [' ', '、', '：', '；']


    model = word2vec.Word2Vec.load('./data/models/patient_word2vec_all.model')
    jieba.load_userdict('./data/medical_ner_dict.txt')

    patients_id = []
    patients_clinical_diagnose_first_trip = []
    patients_clinical_diagnose_first_trip_len = []
    patients_dis_day_pdf = []
    patients_dis_day_pdf_len = []
    patients_ending = []
    patients_clinical_diagnose_first_trip_jieba = []
    patients_clinical_diagnose_day_pdf_jieba = []
    if py_type == 'bf':
        for index, row in df.iterrows():
            patients_id.append(int(row['patient_id']))
            patients_ending.append(row[sick_name])
            first_trip = row['临床诊断']
            dis_day_pdf = row['临床诊断'].strip()
            sentence_list = []
            # print(row['patient_name'])
            first_trip = first_trip.strip().replace("\n", "").replace("，", "").replace(',', "").replace("。",
                                                                                                        "").replace('“',
                                                                                                                    "").replace(
                '"', "")

            for word in jieba.lcut(first_trip):
                if word not in stopwords:
                    sentence_list.append(word)
            patients_clinical_diagnose_first_trip_jieba.append(sentence_list)
            patients_clinical_diagnose_first_trip.append(torch.tensor(model.wv[sentence_list]))
            patients_clinical_diagnose_first_trip_len.append(len(sentence_list))
            if len(sentence_list) > 1000:
                print(row['patient_id'])
                print(len(sentence_list))
                print(row['patient_name'])
                print(sentence_list)

            sentence_list = []
            dis_day_pdf = dis_day_pdf.strip().strip().replace("\n", "").replace("，", "").replace(',', "").replace("。","").replace('“', "").replace('"', "")

            for word in jieba.lcut(dis_day_pdf):
                if word not in stopwords:
                    sentence_list.append(word)
            if len(sentence_list) > 1000:
                print(row['patient_id'])
                print(len(sentence_list))
                print(row['patient_name'])
                print(sentence_list)
            patients_clinical_diagnose_day_pdf_jieba.append(dis_day_pdf)
            patients_dis_day_pdf.append(torch.tensor(model.wv[sentence_list]))
            patients_dis_day_pdf_len.append(len(sentence_list))



    else:
        for index, row in df.iterrows():
            patients_id.append(row['patient_id'])
            patients_ending.append(row[sick_name])
            first_trip =  row['临床诊断']
            if pd.isna(row['pdf']):
                dis_day_pdf = row['临床诊断'].strip()
            else:
                dis_day_pdf = row['pdf'].strip()
            sentence_list = []
            # print(row['patient_name'])
            first_trip = first_trip.strip().replace("\n","").replace("，","").replace(',',"").replace("。","").replace('“',"").replace('"',"")

            for word in jieba.lcut(first_trip):
                if word not in stopwords:
                    sentence_list.append(word)
            patients_clinical_diagnose_first_trip_jieba.append(sentence_list)
            patients_clinical_diagnose_first_trip.append(torch.tensor(model.wv[sentence_list]))
            patients_clinical_diagnose_first_trip_len.append(len(sentence_list))
            if len(sentence_list) > 1000:
                print(row['patient_id'])
                print(len(sentence_list))
                print(row['patient_name'])
                print(sentence_list)

            sentence_list = []
            dis_day_pdf = dis_day_pdf.strip().strip().replace("\n","").replace("，","").replace(',',"").replace("。","").replace('“',"").replace('"',"")

            for word in jieba.lcut(dis_day_pdf):
                if word not in stopwords:
                    sentence_list.append(word)
            if len(sentence_list) > 1000:
                print(row['patient_id'])
                print(len(sentence_list))
                print(row['patient_name'])
                print(sentence_list)
            patients_clinical_diagnose_day_pdf_jieba.append(dis_day_pdf)
            patients_dis_day_pdf.append(torch.tensor(model.wv[sentence_list]))
            patients_dis_day_pdf_len.append(len(sentence_list))




    from torch.nn.utils.rnn import pad_sequence
    # patients_clinical_diagnose_first_trip_pad = pad_sequence(patients_clinical_diagnose_first_trip, batch_first=True)
    # patients_dis_day_pdf_pad = pad_sequence(patients_dis_day_pdf, batch_first=True)
    dataset = MyData(patients_id, patients_clinical_diagnose_first_trip, patients_clinical_diagnose_first_trip_len,patients_dis_day_pdf,patients_dis_day_pdf_len,patients_ending)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patients_id = np.array(patients_id)
    opt = Config()
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    valid_is = False
    if valid_is:
        patient_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
        patient_info = patient_info.sort_values(by='入院日期').reset_index(drop=True)
        patient_info_valid_id = patient_info.loc[437:, '编号']
        X_train = []
        X_valid = []
        X_train_copy = []
        for i in dataset:
            if i[0] in patient_info_valid_id:
                X_valid.append(i)
            else:
                X_train.append(i)
                if i[-1] == 0:
                    X_train_copy.append(i)
        # X_train = X_train + X_train_copy


        transformer_model = Transformer(128, opt.d_model, max(patients_clinical_diagnose_first_trip_len),
                                        max(patients_dis_day_pdf_len), opt.feed_forward_hidden,
                                        opt.d_k, opt.d_v,
                                        tgt_vocab_dim=opt.num_out).to(device)
        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=opt.lr)
        # criterion = nn.BCELoss().to(device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1,opt.neg_ratio])).to(device)
        criterion = nn.BCELoss(weight=torch.tensor([opt.neg_ratio,1])).to(device)
        accum_steps = 8
        dataloader = DataLoader(X_train, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_func)
        for epoch in tqdm(range(opt.epochs)):
            for idx, (batch_patient_id, batch_src_pad, batch_src_len, batch_tgt_pad, batch_tgt_len,
                      batch_ending) in enumerate(dataloader):
                __, dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model(
                    batch_src_pad.to(device), batch_tgt_pad.to(device), torch.tensor(batch_tgt_len))
                batch_ending = to_categorical(batch_ending, opt.num_out)
                loss = criterion(F.softmax(dec_logits, dim=-1), torch.FloatTensor(batch_ending).to(device))
                loss = loss / accum_steps
                loss.backward();
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item() * accum_steps));
        torch.save(X_train, './data/Integration_data/NLP_X_train_{}_bf-{}.pt'.format(sick_name, 'VALID'))
        torch.save(X_valid, './data/Integration_data/NLP_X_valid_{}_bf-{}.pt'.format(sick_name, 'VALID'))
        torch.save(transformer_model,
                   './data/models/Integration_ending/Transformer_NLP_{}_bf-{}.pkl'.format(sick_name, 'VALID'))



    else:
        if py_type == 'bf':
            # for fold, (train_idx, val_idx) in enumerate(skf.split(patients_id, patients_ending)):
            #     patients_id_train, patients_id_test = patients_id[train_idx], patients_id[val_idx]
            for fold in range(0, 5):
                NLP_X_train = torch.load('./data/Integration_data/NLP_X_train_{}_bf-{}.pt'.format(sick_name, str(fold)))
                patients_id_train = []
                for i in NLP_X_train:
                    patients_id_train.append(i[0])

                X_train, X_valid = [], []
                X_train_copy = []
                for i in dataset:
                    if i[0] in patients_id_train:
                        X_train.append(i)
                        if i[-1] == 0:
                            X_train_copy.append(i)
                    else:
                        X_valid.append(i)
                # if is_copy == True:
                X_train = X_train + X_train_copy
                # torch.save(X_train, './data/Integration_data/NLP_X_train_{}_bf-{}.pt'.format(sick_name, str(fold)))
                # torch.save(X_valid, './data/Integration_data/NLP_X_valid_{}_bf-{}.pt'.format(sick_name, str(fold)))
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

                transformer_model = Transformer(128, opt.d_model, max(patients_clinical_diagnose_first_trip_len),
                                                max(patients_dis_day_pdf_len), opt.feed_forward_hidden,
                                                opt.d_k, opt.d_v,
                                                tgt_vocab_dim=opt.num_out).to(device)
                optimizer = torch.optim.Adam(transformer_model.parameters(), lr=opt.lr)
                criterion = nn.BCELoss().to(device)
                # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1,opt.neg_ratio])).to(device)
                # criterion = nn.BCELoss(weight=torch.tensor([opt.neg_ratio,1])).to(device)
                accum_steps = 8
                dataloader = DataLoader(X_train, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_func)
                for epoch in tqdm(range(opt.epochs)):
                    for idx, (batch_patient_id, batch_src_pad, batch_src_len, batch_tgt_pad, batch_tgt_len,
                              batch_ending) in enumerate(dataloader):
                        __, dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model(
                            batch_src_pad.to(device), batch_tgt_pad.to(device), torch.tensor(batch_tgt_len))
                        batch_ending = to_categorical(batch_ending, opt.num_out)
                        loss = criterion(F.softmax(dec_logits, dim=-1), torch.FloatTensor(batch_ending).to(device))
                        loss = loss / accum_steps
                        loss.backward();
                    if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                        optimizer.step()
                        optimizer.zero_grad()
                    if (epoch + 1) % 50 == 0:
                        print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item() * accum_steps));

                torch.save(transformer_model,
                           './data/models/Integration_ending/Transformer_NLP_{}_bf-{}.pkl'.format(sick_name,str(fold)))


        else:
            for fold, (train_idx, val_idx) in enumerate(skf.split(patients_id, patients_ending)):
                patients_id_train, patients_id_test = patients_id[train_idx], patients_id[val_idx]
            # for fold in range(0,1):
            #     NLP_X_train = torch.load('./data/Integration_data/NLP_X_train_{}_multi-{}.pt'.format(sick_name,str(fold)))
            #     patients_id_train = []
            #     for i in NLP_X_train:
            #         patients_id_train.append(i[0])
                X_train, X_valid = [], []
                X_train_copy = []
                X_train_id = []
                for i in dataset:
                    if i[0] in patients_id_train:
                        X_train.append(i)
                        if i[-1] == 1:
                            X_train_copy.append(i)
                    else:
                        X_valid.append(i)
                if is_copy:
                    X_train = X_train + X_train_copy
                torch.save(X_train, './data/Integration_data/NLP_X_train_{}_multi-{}.pt'.format(sick_name, str(fold)))
                torch.save(X_valid, './data/Integration_data/NLP_X_valid_{}_multi-{}.pt'.format(sick_name, str(fold)))
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                transformer_model = Transformer(128, opt.d_model,max(patients_clinical_diagnose_first_trip_len), max(patients_dis_day_pdf_len),opt.feed_forward_hidden,
                                                        opt.d_k, opt.d_v,
                                                        tgt_vocab_dim=opt.num_out).to(device)
                optimizer = torch.optim.Adam(transformer_model.parameters(), lr=opt.lr)
                # criterion = nn.BCELoss().to(device)
                # if fold !=4:
                #     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1,1])).to(device)
                # else:
                # criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([1,1])).to(device)
                criterion = nn.BCELoss().to(device)
                accum_steps = 8
                dataloader = DataLoader(X_train, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_func)
                for epoch in tqdm(range(opt.epochs)):
                    for idx, (batch_patient_id, batch_src_pad, batch_src_len,batch_tgt_pad, batch_tgt_len,batch_ending) in enumerate(dataloader):
                        __, dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model(batch_src_pad.to(device),batch_tgt_pad.to(device),torch.tensor(batch_tgt_len))
                        batch_ending = to_categorical(batch_ending, opt.num_out)
                        loss = criterion(F.softmax(dec_logits, dim=-1), torch.FloatTensor(batch_ending).to(device))
                        loss = loss / accum_steps
                        loss.backward(retain_graph=True);
                    if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                        optimizer.step();
                        optimizer.zero_grad()
                    if (epoch + 1) % 50 == 0:
                        print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()*accum_steps));

                torch.save(transformer_model,'./data/models/Integration_ending/Transformer_NLP_{}_multi-{}.pkl'.format(sick_name,str(fold)))







