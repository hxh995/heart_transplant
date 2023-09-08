import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import nn
from gensim.models import word2vec
import torch
import jieba
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split
from ending_predict import Encoder,Decoder,Seq2Seq
from utils import to_categorical,caculate_auc
import torch.nn.functional as F
from models_building import Transformer
from CF_CT_NLP_ending import Multimodality_dataset,collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Config(object):
    lr = 0.0001
    epochs = 100
    num_out = 2
    feed_forward_hidden = 32
    d_k = 32
    d_v = 32
    batch_size = 8


class TransformerDataset_complication(data.Dataset):
    def __init__(self, patients_id,patients_first_trip_dim, patients_first_trip_len, patients_pdf_dim,patients_pdf_word_len,patients_ending):
        self.patients_id = patients_id
        self.patient_first_trip_dim = patients_first_trip_dim
        self.patients_pdf_dim = patients_pdf_dim
        self.patients_first_trip_len = patients_first_trip_len
        self.patient_pdf_word_len = patients_pdf_word_len
        self.patients_ending = patients_ending

    def __getitem__(self,item):
        return self.patients_id[item],self.patient_first_trip_dim[item],self.patients_first_trip_len[item],self.patients_pdf_dim[item], self.patient_pdf_word_len[item],self.patients_ending[item]

    def __len__(self):
        return len(self.patients_pdf_dim)


def collate_func(data):
    #print(type(data))

    patients_id = []
    patients_first_trip_dim = []
    patients_pdf_dim = []
    patients_first_trip_len = []
    patients_pdf_word_len = []
    patients_ending = []


    for i in data:
        patients_id.append(i[0])
        patients_first_trip_dim.append(i[1])
        patients_first_trip_len.append(i[2])
        patients_pdf_dim.append(i[3])
        # print(i[7])
        patients_pdf_word_len.append(i[4])
        patients_ending.append(i[5])

    return patients_id,torch.stack(patients_first_trip_dim),patients_first_trip_len,torch.stack(patients_pdf_dim)\
        ,patients_pdf_word_len,patients_ending





if __name__ == '__main__':
    patients_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'姓名': 'patient_name','编号': 'patient_id'})
    patients_Context_df = pd.read_excel('./data/NLP_patient_context_concat.xlsx', index_col=0)
    patients_Context_df = patients_Context_df.merge(patients_info[['patient_name', 'patient_id']],on='patient_name')

    is_lung_id = patients_info[patients_info['术后并发症'].str.contains('肺')]['patient_id']
    patients_Context_df['is_lung'] = 0
    patients_Context_df.loc[patients_Context_df.patient_id.isin(is_lung_id), 'is_lung'] = 1

    is_Kidney_id = patients_info[patients_info['术后并发症'].str.contains('肾')]['patient_id']
    patients_Context_df['is_Kidney'] = 0
    patients_Context_df.loc[patients_Context_df.patient_id.isin(is_Kidney_id), 'is_Kidney'] = 1

    is_Hypertension_id = patients_info[patients_info['术后并发症'].str.contains('高血压')]['patient_id']
    patients_Context_df['is_Hypertension_id'] = 0
    patients_Context_df.loc[ patients_Context_df.patient_id.isin(is_Hypertension_id), 'is_Hypertension_id'] = 1

    is_Hyperglycemia_id = patients_info[patients_info['术后并发症'].str.contains('高血糖')]['patient_id']
    patients_Context_df['is_Hyperglycemia_id'] = 0
    patients_Context_df.loc[ patients_Context_df.patient_id.isin(is_Hyperglycemia_id), 'is_Hyperglycemia_id'] = 1

    is_na = patients_info[patients_info['术后并发症'].str.contains('无')]['patient_id']
    is_not_others = pd.concat([is_lung_id, is_Kidney_id, is_Hypertension_id, is_Hyperglycemia_id]).unique()

    patients_Context_df['Others'] = 0
    patients_Context_df.loc[~( patients_Context_df.patient_id.isin(is_na) |  patients_Context_df.patient_id.isin(is_not_others)), 'Others'] = 1
    jieba.load_userdict('./data/medical_ner_dict.txt')
    patient_context_sentens_dims = []
    stopwords = [' ', '、', '：', '；']
    model = word2vec.Word2Vec.load('./data/models/patient_word2vec.model')



    patients_id = []
    patients_first_trip_dim = []
    patients_first_trip_len = []
    patients_pdf_dim = []
    patients_pdf_word_len = []
    patients_ending = []

    for index,row in patients_Context_df.iterrows():

        first_trip = row['first_trip']
        pdf = row['pdf']
        sentence_trip_list = []
        patients_id.append(row['patient_id'])
        patients_ending.append(row[-5])

        for word in jieba.lcut(first_trip.strip().replace("\n","").replace("，", "").replace(',', "").replace("。", "").replace('“', "").replace('"', "")):
            if word not in stopwords:
                sentence_trip_list.append(word)

        patients_first_trip_dim.append(torch.tensor(model.wv[sentence_trip_list]))
        patients_first_trip_len.append(len(sentence_trip_list))
        if len(sentence_trip_list)>1000:
            print(row['patient_name'])
            print('sentence_trip_list')
            print(len(sentence_trip_list))
            print(sentence_trip_list)


        sentence_pdf_list = []
        if not pd.isnull(pdf):
            for word in jieba.lcut(pdf.strip().replace("\n","").replace("，", "").replace(',', "").replace("。", "").replace('”', "").replace('"', "")):
                if word not in stopwords:
                    sentence_pdf_list.append(word)
        if len(sentence_pdf_list)>1000 :
            print(row['patient_name'])
            print('sentence_pdf_list')
            print(len(sentence_trip_list))
            print(len(sentence_pdf_list))
            print(sentence_pdf_list)


        else:
            sentence_pdf_list.append('病例')
        patients_pdf_dim.append(torch.tensor(model.wv[sentence_pdf_list]))
        patients_pdf_word_len.append(len(sentence_pdf_list))




    from torch.nn.utils.rnn import pad_sequence
    patients_first_trip_dim = pad_sequence(patients_first_trip_dim, batch_first=True)
    patients_pdf_dim = pad_sequence(patients_pdf_dim, batch_first=True)

    patients_first_trip_dim = F.normalize(patients_first_trip_dim, p=2, dim=-1)
    patients_pdf_dim = F.normalize(patients_pdf_dim, p=2, dim=-1)

    # breakpoint()
    for fold in range(5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        opt = Config()
        transformer_model_complication = Transformer(patients_first_trip_dim[0].shape[1], 64, max(patients_first_trip_len),
                                        max(patients_pdf_word_len), opt.feed_forward_hidden, opt.d_k, opt.d_v,tgt_vocab_dim=opt.num_out).to(device)

        optimizer = torch.optim.Adam(transformer_model_complication.parameters(), lr=opt.lr)
        criterion = nn.BCELoss().to(device)
        dataset = TransformerDataset_complication(patients_id,patients_first_trip_dim, patients_first_trip_len, patients_pdf_dim, patients_pdf_word_len,patients_ending)
        CF_hidden_train_df = pd.read_excel('./data/Integration_data/CF_hidden_train_df_lung-{}.xlsx'.format(str(fold)), index_col=0)
        X_train_id = CF_hidden_train_df.patient_id.unique()

        X_train = []
        X_valid = []
        for i in dataset:
            if i[0] in X_train_id:
                X_train.append(i)
            else:
                X_valid.append(i)

        dataloader = DataLoader(X_train, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_func)

        accum_steps = 8
        transformer_model_complication.train()

        for epoch in tqdm(range(opt.epochs)):
            for idx,(batch_patient_id,batch_src_pad,batch_src_len,batch_tag_pad,batch_tag_len,batch_ending) in enumerate(dataloader):
                batch_src_len_max = max(batch_src_len)
                batch_tag_len_max = max(batch_tag_len)
                batch_src_pad = batch_src_pad[:, :batch_src_len_max, :]
                batch_tag_pad = batch_tag_pad[:, :batch_tag_len_max, :]
                __,dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model_complication(batch_src_pad.to(device),batch_tag_pad.to(device),
                                                                                              torch.tensor(batch_tag_len))
                batch_ending = to_categorical(batch_ending,opt.num_out)
                loss = criterion(F.softmax(dec_logits,dim=-1), torch.FloatTensor(batch_ending).to(device))
                loss = loss / accum_steps
                loss.backward();
                if (idx + 1) % accum_steps == 0 or (idx + 1) == len(dataloader):
                    optimizer.step();
                    optimizer.zero_grad()
            if (epoch + 1) % 50 == 0:
                print("Transformer_Epoch {} | Transformer_Loss {:.4f}".format(epoch + 1, loss.item()*8));

        torch.save(transformer_model_complication, './data/models/Integration_lung/Transformer_NLP_complication-{}.pkl'.format(str(fold)))

        torch.save(X_train, './data/Integration_data/NLP_X_train-{}.pt'.format(str(fold)))
        torch.save(X_valid, './data/Integration_data/NLP_X_valid-{}.pt'.format(str(fold)))
