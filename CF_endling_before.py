import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.nn.functional as F
from utils import column_process,to_categorical,caculate_auc,acu_curve
from sklearn.model_selection import train_test_split
import torch
from ending_predict import Encoder,Decoder,Seq2Seq
from sklearn.model_selection import StratifiedKFold
class Config(object):
    CF_lr = 0.01
    epochs = 50
    CF_nums_out = 2
    CF_enc_hidden_size = 24
    CF_dec_hidden_size = 24




class net(nn.Module):
    def __init__(self,input_dims,hidden_dims,time_step,num_out,layers_num=2,batch_first=True):
        super(net, self).__init__()
        self.gru = nn.LSTM(input_size=input_dims, hidden_size=hidden_dims,num_layers=layers_num,batch_first=batch_first)
        self.linear_concat = nn.Linear(time_step*hidden_dims,num_out)
        self.linear_last = nn.Linear(hidden_dims,num_out)
    def forward(self,x):
        out,__ = self.gru(x)
        out_pad,out_len = rnn_utils.pad_packed_sequence(out,batch_first=True)
        #print(out_pad.shape)
        #if is_concat:
        #    out_pad_concat = out_pad.reshape(-1,out_pad.shape[1]*out_pad.shape[2])
        #    return self.linear_concat(out_pad_concat)
        #else:
        _inputs=[]
        #print(len(out_pad))
        out_len_index = [i-1 for i in out_len]
        #print(len(out_pad))
        for i in range(len(out_pad)):
            #print(i)
            _inputs.append(out_pad[i][out_len_index[i]].unsqueeze_(0))
            #return self.linear_last(torch.FloatTensor(np.array(out_pad_last)))
        _inputs = torch.cat(_inputs, dim=0)
        #print(_inputs.shape)
        return self.linear_last(_inputs)



class MyData(data.Dataset):
    def __init__(self, patients_id,patients_CF_before_operation,patients_CF_after_operation,patients_CF_len_before_operation,patients_CF_len_after_operation,patients_ending):
        self.patients_id = patients_id
        self.patients_CF_before_operation = patients_CF_before_operation
        self.patients_CF_after_operation = patients_CF_after_operation
        self.patients_CF_len_before_operation = patients_CF_len_before_operation
        self.patients_CF_len_after_operation = patients_CF_len_after_operation
        self.patients_ending = patients_ending
    def __len__(self):
        return len(self.patients_CF_before_operation)
    def __getitem__(self, item):
        return self.patients_id[item],self.patients_CF_before_operation[item], self.patients_CF_after_operation[item], self.patients_CF_len_before_operation[item],\
               self.patients_CF_len_after_operation[item],self.patients_ending[item]

def collate_fn(data):
    # print(data.shape)
    # lengths, idx = data.sort(key=lambda x: len(x), reverse=True)
    patients_id = []
    patients_CF_before_operation = []
    patients_CF_after_operation = []
    patients_CF_len_before_operation = []
    patients_CF_len_after_operation = []
    patients_ending = []

    for i in data:
        patients_id.append(i[0])
        patients_CF_before_operation.append(i[1])
        patients_CF_after_operation.append(i[2])
        patients_CF_len_before_operation.append(i[3])
        patients_CF_len_after_operation.append(i[4])
        # print(i[5].shape)
        patients_ending.append(i[5])


    packed_patients_CF_before_operation = rnn_utils.pad_sequence(patients_CF_before_operation, batch_first=True,
                                                                 padding_value=0)
    packed_patients_CF_after_operation = rnn_utils.pad_sequence(patients_CF_after_operation, batch_first=True,
                                                                padding_value=0)
    # patients_ending = torch.stack(patients_ending)

    return patients_id, packed_patients_CF_before_operation, packed_patients_CF_after_operation, \
           patients_CF_len_before_operation, patients_CF_len_after_operation,patients_ending


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx',index_col=0)

    model_input_dims = len(df.columns) - 8
    df_before_operation = df[df.dis_day < 0]

    df_before_operation = df_before_operation.fillna(df_before_operation.mean(numeric_only=True))

    df_before_operation.iloc[:, 3:-5] = df_before_operation.iloc[:, 3:-5].apply(lambda x: (x - x.mean()) / (x.std()))

    patients_id = []
    patients_CF_before_operation = []
    patients_CF_len_before_operation = []
    patients_CF_after_operation = []
    patients_CF_len_after_operation = []
    patients_CF_ending = []
    for name, grouped in df_before_operation.groupby('patient_id'):
        grouped.sort_values('dis_day', inplace=True)
        patients_CF_after_operation.append(torch.FloatTensor(grouped.iloc[:, 3:-5].values))
        patients_CF_len_after_operation.append(len(grouped))
        before_grouped = df_before_operation[df_before_operation.patient_id == int(name)].sort_values('dis_day')
        patients_id.append(name)
        if len(before_grouped) == 0:
            # print(name)
            ## 如果没有手术后的信息则补充0进去
            patients_CF_before_operation.append(torch.zeros([1, len(grouped.columns) - 8]))
            patients_CF_len_before_operation.append(1)
        else:
            patients_CF_before_operation.append(torch.FloatTensor(before_grouped.iloc[:, 3:-5].values))
            patients_CF_len_before_operation.append(len(before_grouped))
        patients_CF_ending.append(grouped.iloc[0,-1])
    print(len(patients_CF_len_before_operation) - len(patients_CF_after_operation))

    dataset = MyData(patients_id, patients_CF_before_operation, patients_CF_after_operation,
                                    patients_CF_len_before_operation,
                                    patients_CF_len_after_operation, patients_CF_ending)

    opt = Config()
    patients_id = np.array(patients_id)
    for fold in range(5):
        X_train_NLP = torch.load('./data/Integration_data/NLP_X_train_ending_clinical-{}.pt'.format(str(fold)))
        patients_id_train = []
        for i in X_train_NLP:
            patients_id_train.append(i[0])
        X_train,X_valid = [],[]
        for i in dataset:
            if i[0] in patients_id_train:
                X_train.append(i)
            else:
                X_valid.append(i)

        data_loader_train = DataLoader(X_train,batch_size=len(X_train),collate_fn=collate_fn)
        patient_id_train,packed_patients_CF_before_operation_train, packed_patients_CF_after_operation_train, patients_CF_len_before_operation_train, \
        patients_CF_len_after_operation_train,patients_ending_train = iter(data_loader_train).next()

        patients_ending_train = to_categorical(patients_ending_train, 2)

        encoder_model = Encoder(patients_CF_before_operation[0].shape[1], opt.CF_enc_hidden_size,
                                opt.CF_dec_hidden_size).to(device)
        decoder_model = Decoder(patients_CF_after_operation[0].shape[1], opt.CF_enc_hidden_size, opt.CF_dec_hidden_size).to(device)

        Seq2Seq_model = Seq2Seq(encoder_model, decoder_model, device, opt.CF_dec_hidden_size, opt.CF_nums_out).to(device)

        encoder_model.train()
        decoder_model.train()
        Seq2Seq_model.train()

        optimizer = torch.optim.Adam(Seq2Seq_model.parameters(), lr=opt.CF_lr)

        criterion = nn.BCELoss().to(device)

        # criterion = nn.BCEWithLogitsLoss().to(device)
        for epoch in tqdm(range(opt.epochs)):
            optimizer.zero_grad()
            batch_x = packed_patients_CF_before_operation_train.to(device)
            batch_y = packed_patients_CF_after_operation_train.to(device)
            CF_hidden_train, logits_train = Seq2Seq_model(batch_x, patients_CF_len_before_operation_train, batch_y,
                                                          patients_CF_len_after_operation_train)

            loss = criterion(F.softmax(logits_train,dim=-1),torch.FloatTensor(patients_ending_train).to(device))
            # loss = criterion(torch.sigmo(logits_train), torch.FloatTensor(patients_ending_train).float().to(device))
            loss.backward();
            optimizer.step();
            if (epoch + 1) % 100 == 0:
                print("Seq2Seq_Epoch {} | Seq2Seq_Loss {:.4f}".format(epoch + 1, loss.item()));

        with torch.no_grad():
            data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid), collate_fn=collate_fn)
            patient_id_valid,packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid\
                ,patients_ending_valid= iter(data_loader_valid).next()

            encoder_model.eval()
            decoder_model.eval()
            Seq2Seq_model.eval()
            CF_hidden_valid, logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),
                                                          patients_CF_len_before_operation_valid
                                                          , packed_patients_CF_after_operation_valid.to(device),
                                                          patients_CF_len_after_operation_valid)
            logits = F.softmax(logits_valid, dim=-1);
            # logits = torch.sigmoid(logits_valid)
            patients_ending_valid = to_categorical(patients_ending_valid,2)

            for i in range(logits_valid.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(patients_ending_valid[:, i], logits_valid[:, i].detach().cpu().numpy());
                print(roc_auc)

        torch.save(patients_ending_valid[:, i],'./data/Integration_data/CF_patients_ending_before_{}.pt'.format(str(fold)))
        torch.save(logits[:, i].detach().cpu().numpy(),'./data/Integration_data/logits_CF_patients_ending_before_{}.pt'.format(str(fold)))

        CF_hidden_train_array = CF_hidden_train.detach().cpu().numpy()
        CF_hidden_train_df = pd.DataFrame(CF_hidden_train_array)
        CF_hidden_train_df.columns = ['CF_hidden_{}'.format(str(i)) for i in range(CF_hidden_train_df.shape[1])]
        CF_hidden_train_df['patient_id'] = patient_id_train

        CF_hidden_valid_array = CF_hidden_valid.detach().cpu().numpy()
        CF_hidden_valid_df = pd.DataFrame(CF_hidden_valid_array)
        CF_hidden_valid_df.columns = ['CF_hidden_{}'.format(str(i)) for i in range(CF_hidden_valid_df.shape[1])]
        CF_hidden_valid_df['patient_id'] = patient_id_valid

        CF_hidden_train_df.to_excel('./data/Integration_data/CF_hidden_train_df_ending_before-{}.xlsx'.format(str(fold)))
        CF_hidden_valid_df.to_excel('./data/integration_data/CF_hidden_valid_df_ending_before-{}.xlsx'.format(str(fold)))

        torch.save(Encoder,'./data/models/Integration_lung/Encoder_multi_CF_ending_before-{}'.format(str(fold)))
        torch.save(Decoder, './data/models/Integration_lung/Decoder_multi_CF_ending_before-{}'.format(str(fold)))
        torch.save(Seq2Seq_model,'./data/models/Integration_lung/Seq2Seq_multi_CF_ending_before-{}'.format(str(fold)))

        torch.save(X_train,'./data/Integration_data/X_train_CF_ending_before-{}.pt'.format(str(fold)))
        torch.save(X_valid,'./data/Integration_data/X_valid_CF_ending_before-{}.pt'.format(str(fold)))





