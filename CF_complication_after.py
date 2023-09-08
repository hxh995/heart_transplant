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
    CF_lr = 0.001
    epochs = 200


    hidden_dims = 32
    time_step = 65
    num_out = 2




class net(nn.Module):
    def __init__(self,input_dims,hidden_dims,time_step,num_out,layers_num=2,batch_first=True,drop_prob = 0.2):
        super(net, self).__init__()
        self.gru = nn.LSTM(input_size=input_dims, hidden_size=hidden_dims,num_layers=layers_num,batch_first=batch_first,bidirectional=True)
        self.linear_concat = nn.Linear(time_step*hidden_dims,num_out)
        self.linear_last = nn.Sequential(
            nn.Linear(hidden_dims*2,hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims, num_out)
        )
        self.linear = nn.Linear(hidden_dims,num_out)

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
        return _inputs,self.linear_last(_inputs)



class MyData(data.Dataset):
    def __init__(self, patients_id,patients_CF_after_operation,patients_CF_len_after_operation,patients_ending):
        self.patients_id = patients_id

        self.patients_CF_after_operation = patients_CF_after_operation

        self.patients_CF_len_after_operation = patients_CF_len_after_operation
        self.patients_ending = patients_ending
    def __len__(self):
        return len(self.patients_CF_after_operation)
    def __getitem__(self, item):
        return self.patients_id[item], self.patients_CF_after_operation[item],self.patients_CF_len_after_operation[item],self.patients_ending[item]

def collate_fn(data):

    patients_id = []
    patients_CF_after_operation = []
    patients_CF_len_after_operation = []
    patients_ending = []

    for i in data:
        patients_id.append(i[0])
        patients_CF_after_operation.append(i[1])
        patients_CF_len_after_operation.append(i[2])
        # print(i[5].shape)
        patients_ending.append(i[3])

    packed_patients_CF_after_operation = rnn_utils.pad_sequence(patients_CF_after_operation, batch_first=True,
                                                                padding_value=0)
    # patients_ending = torch.stack(patients_ending)

    return patients_id, packed_patients_CF_after_operation,  patients_CF_len_after_operation,patients_ending;

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all_pvalue.xlsx',index_col=0)

    # df = df[(df.complication == 1)|(df.complication == 0)]
    # df = df[df.columns[df.isna().sum().values / len(df) < 0.5]]

    model_input_dims = len(df.columns) - 8
    df_before_operation = df[df.dis_day <= 0]
    df_after_operation = df[(df.dis_day > 0)]
    df_before_operation = df_before_operation.fillna(df_before_operation.mean(numeric_only=True))
    df_after_operation = df_after_operation.fillna(df_after_operation.mean(numeric_only=True))
    df_after_operation = pd.read_excel('./data/df_after_operation_processed.xlsx')
    df_before_operation.iloc[:, 3:-5] = df_before_operation.iloc[:, 3:-5].apply(lambda x: (x - x.mean()) / (x.std()))
    df_after_operation.iloc[:, 3:-5] = df_after_operation.iloc[:, 3:-5].apply(lambda x: (x - x.mean()) / (x.std()))

    patients_id = []
    patients_CF_before_operation = []
    patients_CF_len_before_operation = []
    patients_CF_after_operation = []
    patients_CF_len_after_operation = []
    patients_CF_ending = []

    for name, grouped in df_after_operation.groupby('patient_id'):
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
        patients_CF_ending.append(grouped.iloc[0,-5])
    print(len(patients_CF_len_before_operation) - len(patients_CF_after_operation))

    dataset = MyData(patients_id, patients_CF_after_operation,patients_CF_len_after_operation, patients_CF_ending)

    opt = Config()
    skf = StratifiedKFold(n_splits=5)
    patients_id = np.array(patients_id)
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients_id, patients_CF_ending)):
        patients_id_train, patients_id_test = patients_id[train_idx], patients_id[val_idx]
        X_train, X_valid = [], []
        for i in dataset:
            if i[0] in patients_id_train:
                X_train.append(i)
            else:
                X_valid.append(i)

        data_loader_train = DataLoader(X_train, batch_size=len(X_train), collate_fn=collate_fn)
        patient_id_train, packed_patients_CF_after_operation_train,patients_CF_len_after_operation_train, patients_ending_train = iter(data_loader_train).next()

        patients_ending_train = to_categorical(patients_ending_train, 2)

        Gru_model = net(packed_patients_CF_after_operation_train.shape[2],opt.hidden_dims,opt.time_step,opt.num_out,layers_num=2,batch_first=True).to(device)

        Gru_model.train()

        optimizer = torch.optim.Adam(Gru_model.parameters(), lr=opt.CF_lr)

        criterion = nn.BCELoss().to(device)

        packed_batch_y = rnn_utils.pack_padded_sequence(packed_patients_CF_after_operation_train, torch.FloatTensor(patients_CF_len_after_operation_train), batch_first=True,
                                                        enforce_sorted=False)
        for epoch in tqdm(range(opt.epochs)):
            optimizer.zero_grad()
            packed_batch_y = packed_batch_y.to(device)

            CF_hidden_train, logits_train = Gru_model(packed_batch_y)

            loss = criterion(F.softmax(logits_train,dim=-1),torch.FloatTensor(patients_ending_train).to(device))
            # loss = criterion(torch.sigmo(logits_train), torch.FloatTensor(patients_ending_train).float().to(device))
            loss.backward();
            optimizer.step();
            if (epoch + 1) % 100 == 0:
                print("Seq2Seq_Epoch {} | Seq2Seq_Loss {:.4f}".format(epoch + 1, loss.item()));
        with torch.no_grad():
            data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid), collate_fn=collate_fn)
            patient_id_valid, packed_patients_CF_after_operation_valid, patients_CF_len_after_operation_valid \
                , patients_ending_valid = iter(data_loader_valid).next()


            Gru_model.eval()
            packed_batch_y_valid = rnn_utils.pack_padded_sequence(packed_patients_CF_after_operation_valid,torch.FloatTensor(patients_CF_len_after_operation_valid),
                                                            batch_first=True,
                                                            enforce_sorted=False)
            CF_hidden_valid, logits_valid = Gru_model(packed_batch_y_valid.to(device))
            logits = F.softmax(logits_valid, dim=-1);
            # logits = torch.sigmoid(logits_valid)
            patients_ending_valid = to_categorical(patients_ending_valid, 2)

            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(patients_ending_valid[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)