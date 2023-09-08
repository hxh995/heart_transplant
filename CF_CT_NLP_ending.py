import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
import cv2,re
from gensim.models import word2vec
import torch
import math, copy, time,os,jieba
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split
from ending_predict import Encoder,Decoder,Seq2Seq
from utils import to_categorical,caculate_auc,acu_curve
import torch.nn.functional as F
from models_building import Transformer,TransformerDataset


class Config(object):
    CF_lr = 0.001
    CF_epochs = 400
    CF_nums_out = 2
    CF_enc_hidden_size = 36
    CF_dec_hidden_size = 36

    NLP_lr = 0.001
    NLP_epochs = 80
    NLP_num_out = 2
    NLP_feed_forward_hidden = 32
    NLP_d_k = 32
    NLP_d_v = 32




class Multimodality_dataset(data.Dataset):
    def __init__(self, patients_id,patients_CF_before_operation,patients_CF_after_operation,patients_CF_len_before_operation,patients_CF_len_after_operation,
                 patients_first_trip_dim, patients_first_trip_len, patients_pdf_dim,patients_pdf_word_len,patients_ending):
        self.patients_id = patients_id
        self.patients_CF_before_operation = patients_CF_before_operation
        self.patients_CF_after_operation = patients_CF_after_operation
        self.patients_CF_len_before_operation = patients_CF_len_before_operation
        self.patients_CF_len_after_operation = patients_CF_len_after_operation
        self.patient_first_trip_dim = patients_first_trip_dim
        self.patients_pdf_dim = patients_pdf_dim
        self.patients_first_trip_len = patients_first_trip_len
        self.patient_pdf_word_len = patients_pdf_word_len
        self.patients_ending = patients_ending

    def __getitem__(self,item):
        return self.patients_id[item],self.patients_CF_before_operation[item], self.patients_CF_after_operation[item], self.patients_CF_len_before_operation[item],\
               self.patients_CF_len_after_operation[item], self.patient_first_trip_dim[item],self.patients_first_trip_len[item],\
               self.patients_pdf_dim[item], self.patient_pdf_word_len[item],self.patients_ending[item]

    def __len__(self):
        return len(self.patients_pdf_dim)




def collate_fn(data):
    #print(type(data))

    patients_id = []
    patients_CF_before_operation = []
    patients_CF_after_operation = []
    patients_CF_len_before_operation = []
    patients_CF_len_after_operation = []
    patients_first_trip_dim = []
    patients_pdf_dim = []
    patients_first_trip_len = []
    patients_pdf_word_len = []
    patients_ending = []


    for i in data:
        patients_id.append(i[0])
        patients_CF_before_operation.append(i[1])
        patients_CF_after_operation.append(i[2])
        patients_CF_len_before_operation.append(i[3])
        patients_CF_len_after_operation.append(i[4])
        patients_first_trip_dim.append(i[5])
        patients_first_trip_len.append(i[6])
        patients_pdf_dim.append(i[7])
        # print(i[7])
        patients_pdf_word_len.append(i[8])
        patients_ending.append(i[9])

    packed_patients_CF_before_operation = rnn_utils.pad_sequence(patients_CF_before_operation, batch_first=True, padding_value=0)
    packed_patients_CF_after_operation = rnn_utils.pad_sequence(patients_CF_after_operation, batch_first=True, padding_value=0)

    packed_first_trip_dim = rnn_utils.pad_sequence(patients_first_trip_dim, batch_first=True, padding_value=0)
    packed_patients_pdf_dim = rnn_utils.pad_sequence(patients_pdf_dim, batch_first=True, padding_value=0)

    return patients_id,packed_patients_CF_before_operation,packed_patients_CF_after_operation,patients_CF_len_before_operation,patients_CF_len_after_operation,\
           packed_first_trip_dim,patients_first_trip_len,packed_patients_pdf_dim,patients_pdf_word_len,patients_ending




if __name__ == '__main__':
    df = pd.read_excel('./data/patient_dis_day_CF_ending_processed_pvalue_before.xlsx', index_col=0)
    df_before_operation = df[df.dis_day < 0]
    df_after_operation = df[df.dis_day >= 0]
    # df_before_operation = pd.read_excel('./data/df_before_operation_processed_ending.xlsx',index_col=0)
    df_before_operation = df_before_operation.fillna(df_before_operation.mean(numeric_only=True))
    df_after_operation = df_after_operation.fillna(df_after_operation.mean(numeric_only=True))
    df_before_operation.iloc[:, 3:-1] = df_before_operation.iloc[:, 3:-1].apply(lambda x: (x - x.mean()) / (x.std()))
    df_after_operation.iloc[:, 3:-1] = df_after_operation.iloc[:, 3:-1].apply(lambda x: (x - x.mean()) / (x.std()))

    opt = Config()

    jieba.load_userdict('./data/medical_ner_dict.txt')
    patient_context_sentens_dims = []
    stopwords = [' ', '、', '：', '；']
    model = word2vec.Word2Vec.load('./data/models/patient_word2vec.model')

    patient_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx').rename(columns={'姓名': 'patient_name'})
    patients_Context_df = pd.read_excel('./data/NLP_patient_context_concat.xlsx', index_col=0)
    patients_Context_df = patients_Context_df.merge(patient_info[['patient_name', '出院时结局','编号']], on='patient_name').rename(columns={'编号':'patient_id'})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patients_id = []
    patients_first_trip_dim = []
    patients_pdf_dim = []
    patients_first_trip_len = []
    patients_pdf_word_len = []

    patients_CF_before_operation = []
    patients_CF_len_before_operation = []
    patients_CF_after_operation = []
    patients_CF_len_after_operation = []
    patients_ending = []


    patients_Context_df = patients_Context_df[patients_Context_df['出院时结局'].isin(['临床治愈','好转','死亡'])]
    patients_Context_df.loc[:,'出院时结局'] = patients_Context_df['出院时结局'].apply(lambda x: 1 if x in ['临床治愈','好转'] else 0)
    patients_Context_df_patient_id = patients_Context_df.patient_id.unique()
    for name, grouped in df_before_operation.groupby('patient_id'):
        if name in patients_Context_df_patient_id:
            grouped.sort_values('dis_day', inplace=True)
            patients_CF_after_operation.append(torch.FloatTensor(grouped.iloc[:, 3:-1].values))
            patients_CF_len_after_operation.append(len(grouped))
            before_grouped = df_before_operation[df_before_operation.patient_id==int(name)].sort_values('dis_day')
            if len(before_grouped)==0:
                #print(name)
                ## 如果没有手术后的信息则补充0进去
                patients_CF_before_operation.append(torch.zeros([1,len(grouped.columns)-4]))
                patients_CF_len_before_operation.append(1)
            else:
                patients_CF_before_operation.append(torch.FloatTensor(before_grouped.iloc[:, 3:-1].values))
                patients_CF_len_before_operation.append(len(before_grouped))
            patients_id.append(name)
            row = patients_Context_df.loc[patients_Context_df.patient_id == name,:]
            first_trip = row['first_trip'].values[0]
            pdf = row['pdf'].values[0]
            sentence_trip_list = []
            for word in jieba.lcut(first_trip.strip().replace("\n","").replace("，", "").replace(',', "").replace("。", "").replace('“', "").replace('"', "")):
                if word not in stopwords:
                    sentence_trip_list.append(word)

            patients_first_trip_dim.append(torch.tensor(model.wv[sentence_trip_list]))
            patients_first_trip_len.append(len(sentence_trip_list))
            if len(sentence_trip_list)>1000:
                print(row['patient_name'].values)
                print('first_trip')
                print(sentence_trip_list)
            sentence_pdf_list = []
            if not pd.isnull(pdf):
                for word in jieba.lcut(pdf.strip().replace("\n","").replace("，", "").replace(',', "").replace("。", "").replace('”', "").replace('"', "")):
                    if word not in stopwords:
                        sentence_pdf_list.append(word)
            else:
                sentence_pdf_list.append('病例')
            if len(sentence_pdf_list)>1000:
                print(row['patient_name'].values)
                print('pdf_list')
                print(sentence_pdf_list)

            patients_pdf_dim.append(torch.tensor(model.wv[sentence_pdf_list]))
            patients_pdf_word_len.append(len(sentence_pdf_list))

            patients_ending.append(row['出院时结局'].values[0])

    from torch.utils.data import DataLoader
    # breakpoint()
    print(max(patients_first_trip_len))
    print(max(patients_pdf_word_len))
    patients_id = np.array(patients_id)
    patients_ending = np.array(patients_ending)
    dataset = Multimodality_dataset(patients_id,patients_CF_before_operation,patients_CF_after_operation,patients_CF_len_before_operation,
                                    patients_CF_len_after_operation,patients_first_trip_dim,patients_first_trip_len,patients_pdf_dim,patients_pdf_word_len,patients_ending)
    from collections import Counter
    print(Counter(patients_ending))
    skf = StratifiedKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients_id, patients_ending)):
        print(fold)
        # print(len(train_idx))
        # print(len(val_idx))
        patients_id_train,patients_id_test = patients_id[train_idx],patients_id[val_idx]
        X_train, X_valid = [],[]
        for i in dataset:
            if i[0] in patients_id_train:

                X_train.append(i)
            else:
                X_valid.append(i)
        # print(len(X_train))
        # print(len(X_valid))
        #
        device = "cuda" if torch.cuda.is_available() else "cpu"


        data_loader_train = DataLoader(X_train, batch_size=len(X_train), shuffle=True, collate_fn=collate_fn)
        encoder_model = Encoder(patients_CF_before_operation[0].shape[1], opt.CF_enc_hidden_size, opt.CF_dec_hidden_size).to(device)
        decoder_model = Decoder(patients_CF_after_operation[0].shape[1], opt.CF_enc_hidden_size, opt.CF_dec_hidden_size).to(device)
        Seq2Seq_model = Seq2Seq(encoder_model, decoder_model, device, opt.CF_dec_hidden_size, opt.CF_nums_out).to(device)

        patient_id_train,packed_patients_CF_before_operation_train, packed_patients_CF_after_operation_train, patients_CF_len_before_operation_train, patients_CF_len_after_operation_train,\
        packed_first_trip_dim_train, patients_first_trip_len_train, packed_patients_pdf_dim_train,  patients_pdf_word_len_train, patients_ending_train = iter(data_loader_train).next()


        optimizer = torch.optim.Adam(Seq2Seq_model.parameters(), lr=opt.CF_lr)
        criterion = nn.BCELoss().to(device)

        label_batch = to_categorical(patients_ending_train, opt.CF_nums_out)
        Seq2Seq_model.train()
        from tqdm import tqdm

        for epoch in tqdm(range(opt.CF_epochs)):
            optimizer.zero_grad()
            batch_x = packed_patients_CF_before_operation_train.to(device)
            batch_y = packed_patients_CF_after_operation_train.to(device)
            CF_hidden_train,logits_train = Seq2Seq_model(batch_x, patients_CF_len_before_operation_train,batch_y,patients_CF_len_after_operation_train)
            loss = criterion(F.softmax(logits_train, dim=-1), torch.FloatTensor(label_batch).to(device))
            loss.backward();
            optimizer.step();
            if (epoch + 1) % 100 == 0:
                print("Seq2Seq_Epoch {} | Seq2Seq_Loss {:.4f}".format(epoch + 1, loss.item()));



        with torch.no_grad():
            data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid),  collate_fn=collate_fn)
            patient_id_valid,packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid, \
            packed_first_trip_dim_valid, patients_first_trip_len_valid, packed_patients_pdf_dim_valid,  patients_pdf_word_len_valid, patients_ending_valid = iter(data_loader_valid).next()


            Seq2Seq_model.eval()
            CF_hidden_valid,logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),patients_CF_len_before_operation_valid
                                   ,packed_patients_CF_after_operation_valid.to(device),patients_CF_len_after_operation_valid)
            logits_valid = F.softmax(logits_valid, dim=-1);
            data_label_valid = to_categorical(patients_ending_valid,opt.CF_nums_out)

            for i in range(logits_valid.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(data_label_valid[:, i], logits_valid[:, i].detach().cpu().numpy());
                print(roc_auc)
        # break;


        figure_path = './data/figures/figure_test_CF_new.jpg'
        acu_curve(fpr, tpr, roc_auc, figure_path)

        torch.save(data_label_valid[:, i], './data/Integration_data/patients_ending_valid_CF_before-{}.pt'.format(fold))
        torch.save(logits_valid[:, i].detach().cpu().numpy(), './data/Integration_data/logits_CF_before-{}.pt'.format(fold))

        CF_hidden_train_array = CF_hidden_train.detach().cpu().numpy()
        CF_hidden_train_df = pd.DataFrame(CF_hidden_train_array)
        CF_hidden_train_df.columns = ['CF_hidden_{}'.format(str(i)) for i in range(CF_hidden_train_df.shape[1])]
        CF_hidden_train_df['patient_id'] = patient_id_train

        CF_hidden_valid_array = CF_hidden_valid.detach().cpu().numpy()
        CF_hidden_valid_df = pd.DataFrame(CF_hidden_valid_array)
        CF_hidden_valid_df.columns = ['CF_hidden_{}'.format(str(i)) for i in range(CF_hidden_valid_df.shape[1])]
        CF_hidden_valid_df['patient_id'] = patient_id_valid

        CF_hidden_train_df.to_excel('./data/Integration_data/CF_hidden_train_df_before-{}.xlsx'.format(str(fold)))
        CF_hidden_valid_df.to_excel('./data/Integration_data/CF_hidden_valid_df_before-{}.xlsx'.format(str(fold)))



        torch.save(Encoder,'./data/models/Integration_ending/Encoder_multi_CF_ending_before-{}.pkl'.format(str(fold)))
        torch.save(Decoder, './data/models/Integration_ending/Decoder_multi_CF_ending_before-{}.pkl'.format(str(fold)))
        torch.save(Seq2Seq_model, './data/models/Integration_ending/Seq2Seq_multi_CF_ending-{}.pkl'.format(str(fold)))

        torch.save(X_train,"./data/Integration_data/X_train_CF_NLP_before_ending-{}.pt".format(str(fold)))
        torch.save(X_valid, "./data/Integration_data/X_valid_CF_NLP_before_ending-{}.pt".format(str(fold)))
































