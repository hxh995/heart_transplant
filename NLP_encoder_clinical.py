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
import torch.nn.utils.rnn as rnn_utils


class MyData(data.Dataset):
    def __init__(self, patients_id,patients_CF_before_operation,patients_CF_len_before_operation,patients_is_diabetes,patients_ending):
        self.patients_id = patients_id
        self.patients_CF_before_operation = patients_CF_before_operation
        self.patients_CF_len_before_operation = patients_CF_len_before_operation
        self.patients_is_diabetes = patients_is_diabetes
        self.patients_ending = patients_ending
    def __len__(self):
        return len(self.patients_CF_before_operation)
    def __getitem__(self, item):
        return self.patients_id[item],self.patients_CF_before_operation[item], self.patients_CF_len_before_operation[item],self.patients_is_diabetes[item],self.patients_ending[item]


class Config(object):
    lr = 0.0001
    epochs = 150
    num_out = 2
    feed_forward_hidden = 48
    d_model = 72
    d_k = 48
    d_v = 48
    batch_size = 512
    neg_ratio = 100
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

def collate_func_bf(data):
    #print(type(data))

    patients_id = []
    patients_first_trip_dim = []
    patients_first_trip_len = []
    patients_is_diabetes = []
    patients_ending = []


    for i in data:
        patients_id.append(i[0])
        patients_first_trip_dim.append(i[1])
        patients_first_trip_len.append(i[2])
        patients_is_diabetes.append(i[3])
        patients_ending.append(i[4])

    packed_first_trip_dim = rnn_utils.pad_sequence(patients_first_trip_dim, batch_first=True, padding_value=0)


    return patients_id,packed_first_trip_dim ,patients_first_trip_len,torch.FloatTensor(patients_is_diabetes).unsqueeze(1),patients_ending

if __name__ == '__main__':
    patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    patients_info['临床诊断'] = patients_info ['临床诊断'].apply(lambda x:x.replace('、',' ').replace('，',' ').replace('：',' ').replace(',',' ').replace('\n',' ').replace('Ⅳ','IV'))
    patients_info = patients_info.drop('临床诊断',axis=1).join(patients_info['临床诊断'].str.split(' ', expand=True).stack().reset_index(level=1, drop=True).rename('临床诊断'))
    patients_info['临床诊断'] = patients_info['临床诊断'].apply(lambda x:x.strip())
    patients_info = patients_info[patients_info['临床诊断']!='']
    # patients_info.groupby('临床诊断').size().sort_values(ascending=False).to_csv('./data/clinical_diagnose_sort.xlsx')
    patients_info = patients_info[patients_info['出院时结局'].isin(['临床治愈', '好转', '死亡'])]
    patients_info.loc[:, "出院时结局"] = patients_info['出院时结局'].apply(lambda x: 1 if x in ['临床治愈', '好转'] else 0)
    dict_status = {0:'死亡',1:'好转'}
    # for name,grouped in patients_info.groupby('出院时结局'):
    #     print(dict_status[name])
    #     print(len(grouped[grouped['临床诊断'].str.contains('肺动脉高压')]))
    #     print(len(grouped[grouped['临床诊断'].str.contains('肺动脉高压')])/len(grouped['编号'].unique()))
    patients_clinical_diagnose_jieba = []
    stopwords = [' ', '、', '：', '；', ',', ']', '】', '[']
    df = pd.read_excel('./data/NLP_patient_context_concat.xlsx', index_col=0)
    jieba.load_userdict('./data/medical_ner_dict.txt')
    patient_context_sentens = []
    patients_info = patients_info[patients_info['临床诊断'].str.len() >= 2]

    for index, row in df.iterrows():
        first_trip = row['first_trip']
        sentence_list = []
        # print(row['patient_name'])
        grouped = patients_info[patients_info['姓名']==row['patient_name']]
        first_trip = first_trip.strip() + ','.join(grouped['临床诊断'].values.tolist())
        first_trip = first_trip.strip().replace("\n", "").replace("，", "").replace(',', "").replace("。", "").replace('“',"").replace('"', "")
        first_trip = first_trip.replace('型', '性').replace('瓣膜性心脏病','').replace('20161026','').replace('Ⅳ','IV').replace('Ⅲ','III').replace('(NYHA分级)','').replace('很','').replace('？','')
        for word in jieba.lcut(first_trip):
            if word not in stopwords:
                sentence_list.append(word)

        patient_context_sentens.append(sentence_list)

    model = word2vec.Word2Vec(patient_context_sentens, sg=0, epochs=8, vector_size=128, window=5, negative=3,
                              sample=0.001, hs=1, workers=16, min_count=1)


    patients_id = []
    patients_name = []
    patients_clinical_diagnose = []
    patients_clinical_diagnose_len = []
    patients_ending = []
    patients_clinical_diagnose_jieba = []
    patients_is_diabetes = []

    for name,grouped in patients_info.groupby('编号'):
        patients_id.append(name)
        patients_name.append(grouped['姓名'].unique()[0])
        patients_ending.append(grouped['出院时结局'].unique()[0])
        clinical_diagnose = ','.join(grouped['临床诊断'].values.tolist())
        clinical_diagnose = clinical_diagnose.replace("\n", "").replace("，", "").replace(',', "").replace("。", "").replace('“', "").replace('"', "")
        clinical_diagnose = clinical_diagnose.replace('型', '性').replace('瓣膜性心脏病', '').replace('20161026', '').replace('Ⅳ','IV').replace('Ⅲ', 'III').replace('(NYHA分级)', '').replace('很','').replace('？','')
        sentence_list = []
        if '肺动脉高压' in clinical_diagnose:
            patients_is_diabetes.append(1)
        else:
            patients_is_diabetes.append(0)
        for word in jieba.lcut(clinical_diagnose):
            if word not in stopwords:
                sentence_list.append(word)
        patients_clinical_diagnose_jieba.append(sentence_list)
        patients_clinical_diagnose.append(torch.tensor(model.wv[sentence_list]))
        patients_clinical_diagnose_len.append(len(sentence_list))
    from torch.nn.utils.rnn import pad_sequence
    patients_clinical_diagnose = pad_sequence(patients_clinical_diagnose, batch_first=True)
    dataset = MyData(patients_id, patients_clinical_diagnose, patients_clinical_diagnose_len, patients_is_diabetes,patients_ending)
    jieba_df = pd.DataFrame(columns=['patient_id', 'name', 'jieba'])
    jieba_df['patient_id'] = patients_id
    jieba_df['name'] = patients_name
    jieba_df['jieba'] = patients_clinical_diagnose_jieba
    jieba_df.to_excel('./data/jieba_df.xlsx')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patients_id = np.array(patients_id)
    opt = Config()
    for fold in range(4,5):
        NLP_X_train = torch.load('./data/Integration_data/NLP_X_train_ending-{}.pt'.format(str(fold)))
        X_train = []
        X_valid = []
        X_train_copy = []
        X_train_id = []
        for i in NLP_X_train:
            X_train_id.append(i[0])
        for i in dataset:
            if i[0] in X_train_id:
                X_train.append(i)
                if i[-1] == 0:
                    X_train_copy.append(i)
            else:
                X_valid.append(i)
        X_train = X_train + X_train_copy + X_train_copy
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        transformer_model = Transformer_encoder(patients_clinical_diagnose[0].shape[1], opt.d_model,
                                                max(patients_clinical_diagnose_len), opt.feed_forward_hidden,
                                                opt.d_k, opt.d_v,
                                                tgt_vocab_dim=opt.num_out).to(device)
        print(max(patients_clinical_diagnose_len))
        optimizer = torch.optim.Adam(transformer_model.parameters(), lr=opt.lr)
        # criterion = nn.BCELoss().to(device)
        weight = [opt.neg_ratio,1]
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([opt.neg_ratio, 10])).to(device)
        # criterion = nn.BCELoss(weight=torch.tensor([opt.neg_ratio,1])).to(device)
        accum_steps = 8
        dataloader = DataLoader(X_train, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_func)
        for epoch in tqdm(range(opt.epochs)):
            for idx, (batch_patient_id, batch_src_pad, batch_src_len, batch_patients_is_diabetes,batch_ending) in enumerate(dataloader):
                batch_src_len_max = max(batch_src_len)
                batch_src_pad = batch_src_pad[:, :batch_src_len_max, :]
                __, dec_logits, enc_self_attns, enc_context_attns = transformer_model(batch_src_pad.to(device),batch_patients_is_diabetes.to(device))
                batch_ending = to_categorical(batch_ending, opt.num_out)
                loss = criterion(F.softmax(dec_logits, dim=-1), torch.FloatTensor(batch_ending).to(device))
                # loss = loss / accum_steps
                loss.backward(retain_graph=True);
                optimizer.step();
                optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()));

        torch.save(transformer_model,
                   './data/models/Integration_ending/Transformer_NLP_ending_clinical-{}.pkl'.format(str(fold)))

        torch.save(X_train, './data/Integration_data/NLP_X_train_ending_clinical-{}.pt'.format(str(fold)))
        torch.save(X_valid, './data/Integration_data/NLP_X_valid_ending_clinical-{}.pt'.format(str(fold)))


