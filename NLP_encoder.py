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
from models_building import Transformer_encoder
from torch.utils.data import DataLoader

class MyData(data.Dataset):
    def __init__(self, patients_id,patients_CF_before_operation,patients_CF_len_before_operation,patients_ending):
        self.patients_id = patients_id
        self.patients_CF_before_operation = patients_CF_before_operation
        self.patients_CF_len_before_operation = patients_CF_len_before_operation
        self.patients_ending = patients_ending
    def __len__(self):
        return len(self.patients_CF_before_operation)
    def __getitem__(self, item):
        return self.patients_id[item],self.patients_CF_before_operation[item], self.patients_CF_len_before_operation[item],self.patients_ending[item]
def collate_func(data):
    #print(type(data))

    patients_id = []
    patients_first_trip_dim = []
    patients_first_trip_len = []
    patients_ending = []


    for i in data:
        patients_id.append(i[0])
        patients_first_trip_dim.append(i[1])
        patients_first_trip_len.append(i[2])
        patients_ending.append(i[3])

    return patients_id,torch.stack(patients_first_trip_dim),patients_first_trip_len,patients_ending


class Config(object):
    lr = 0.0001
    epochs = 100
    num_out = 2
    feed_forward_hidden = 32
    d_model = 64
    d_k = 32
    d_v = 32
    batch_size = 16


if __name__ == '__main__':
    patient_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'姓名':'patient_name'})
    patients_context_df = pd.read_excel('./data/NLP_patient_context_concat.xlsx',index_col=0)
    patients_context_df = patients_context_df.merge(patient_info[['patient_name','出院时结局','编号']],on='patient_name').rename(columns={'编号':'patient_id'})
    jieba.load_userdict('./data/medical_ner_dict.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patients_context_df = patients_context_df[patients_context_df['出院时结局'].isin(['临床治愈','好转','死亡'])]
    patients_context_df.loc[:,"出院时结局"] = patients_context_df['出院时结局'].apply(lambda x:1 if x in ['临床治愈','好转'] else 0)
    patients_context_df_patient_id = patients_context_df.patient_id.unique()

    patients_id = []
    patients_first_trip_dim = []
    patients_first_trip_len = []
    patients_ending = []
    stopwords = [' ', '、', '：', '；']
    model = word2vec.Word2Vec.load('./data/models/patient_word2vec_encoder.model')

    for index, row in patients_context_df.iterrows():
        first_trip = row['first_trip']
        sentence_trip_list = []
        for word in jieba.lcut(first_trip.strip().replace("\n","").replace("，", "").replace(',', "").replace("。", "").replace('“', "").replace('"', "")):
            if word not in stopwords:
                sentence_trip_list.append(word)

        if len(sentence_trip_list) > 1000:
            print(row['patient_name'])
            print('sentence_trip_list')
            print(len(sentence_trip_list))
            print(sentence_trip_list)
            print(row['出院时结局'])
        else:
            patients_id.append(row['patient_id'])
            patients_ending.append(row['出院时结局'])
            patients_first_trip_dim.append(torch.tensor(model.wv[sentence_trip_list]))
            patients_first_trip_len.append(len(sentence_trip_list))

    from torch.nn.utils.rnn import pad_sequence

    patients_first_trip_dim = pad_sequence(patients_first_trip_dim, batch_first=True)
    skf = StratifiedKFold(n_splits=5)
    patients_id = np.array(patients_id)
    dataset = MyData(patients_id, patients_first_trip_dim, patients_first_trip_len, patients_ending)
    opt = Config()


    for fold, (train_idx, val_idx) in enumerate(skf.split(patients_id, patients_ending)):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        patients_id_train, patients_id_test = patients_id[train_idx], patients_id[val_idx]
        X_train, X_valid = [], []
        X_train_copy = []
        patient_train_label = []
        for i in dataset:
            if i[0] in patients_id_train:
                X_train.append(i)
                patient_train_label.append(i[-1])
                if i[-1] == 0:
                    X_train_copy.append(i)
                    # patient_train_label.append(i[-1])
            else:
                X_valid.append(i)
        X_train = X_train + X_train_copy + X_train_copy
        transformer_model = Transformer_encoder(patients_first_trip_dim[0].shape[1], opt.d_model,
                                                             max(patients_first_trip_len), opt.feed_forward_hidden,
                                                             opt.d_k, opt.d_v,
                                                             tgt_vocab_dim=opt.num_out).to(device)
        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=opt.lr)
        # criterion = nn.BCELoss().to(device)
        print(Counter(patient_train_label))
        neg_ratio = int(Counter(patient_train_label)[1]/ Counter(patient_train_label)[0]) - 4
        print(neg_ratio)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_ratio,1])).to(device)
        accum_steps = 8
        dataloader = DataLoader(X_train, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_func)
        for epoch in tqdm(range(opt.epochs)):
            for idx,(batch_patient_id,batch_src_pad,batch_src_len,batch_ending) in enumerate(dataloader):
                batch_src_len_max = max(batch_src_len)
                batch_src_pad = batch_src_pad[:, :batch_src_len_max, :]
                __,dec_logits, enc_self_attns,enc_context_attns = transformer_model(batch_src_pad.to(device))
                batch_ending = to_categorical(batch_ending,opt.num_out)
                loss = criterion(F.softmax(dec_logits,dim=-1), torch.FloatTensor(batch_ending).to(device))
                loss = loss / accum_steps
                loss.backward();
                if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                    optimizer.step();
                    optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()*8));

        torch.save(transformer_model,'./data/models/Integration_ending/Transformer_NLP_ending-{}.pkl'.format(str(fold)))

        torch.save(X_train, './data/Integration_data/NLP_X_train_ending-{}.pt'.format(str(fold)))
        torch.save(X_valid, './data/Integration_data/NLP_X_valid_ending-{}.pt'.format(str(fold)))



