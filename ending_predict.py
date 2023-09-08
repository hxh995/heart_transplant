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
from utils import column_process,to_categorical
from sklearn.model_selection import StratifiedKFold

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_dim,enc_hidden_size,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(enc_hidden_size*2, dec_hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,lengths):
        # sorted_len,sorted_idx = lengths.sort(0,descending=True)
        # x_sorted = x[sorted_idx.long()]
        packed_input = rnn_utils.pack_padded_sequence(x, lengths,batch_first=True,enforce_sorted=False)
        packed_out,hid = self.rnn(packed_input)
        #print(packed_out.shape)
        out,_ = rnn_utils.pad_packed_sequence(packed_out,batch_first=True)

        # # [batch_size,seq_len,2*enc_hidden_size]
        # out = out[original_idx.long()].contiguous()
        # # [2,batch_size,enc_hidden_size]
        # hid = hid[:, original_idx.long()].contiguous()

        hid = torch.cat([hid[-2],hid[-1]],dim=1)
        # hid = [batch_size,dec_hidden_size]
        hid = torch.tanh(self.fc(hid))
        return out,hid

class Attention(nn.Module):
    def __init__(self,enc_hidden_size,dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.attn = nn.Linear((enc_hidden_size*2)+dec_hidden_size,dec_hidden_size,bias=False)
        self.v  = nn.Linear(dec_hidden_size,1,bias=False)
    def forward(self,s,enc_output,mask):
        # s = [batch_size,dec_hidden_size]
        # enc_output : [batch_size,seq_len_x,2*enc_hidden_size]
        batch_size = enc_output.size(0)
        input_len = enc_output.size(1)

        # repeat decoder hidden state input_len times to change s.shape to [batch_size,src_len,dec_hid_dim]
        s = s.unsqueeze(1).repeat(1,input_len,1)

        # energy = [batch_size,input_len,dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s,enc_output),dim=2)))

        # attention = [batch_size,input_len]
        attention = self.v(energy).squeeze(2)
        # print(attention.shape)
        # print(mask.shape)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention,dim=1)
class Decoder(nn.Module):
    def __init__(self,output_y_size,enc_hidden_size,dec_hidden_size,dropout=0.2,num_out_size=2):
        super(Decoder, self).__init__()
        self.attention = Attention(enc_hidden_size,dec_hidden_size)
        self.rnn = nn.GRU((enc_hidden_size*2+output_y_size),dec_hidden_size,batch_first=True)
        self.fc_out = nn.Linear((enc_hidden_size*2)+dec_hidden_size+output_y_size,num_out_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,dec_input,s,enc_output,mask):
        # dec_input = [batch_size,1,output_size]
        # s = [batch_size,dec_hid_dim]
        # enc_output = [batch_size,src_len,enc_hid_dim*2]
        # mask = [batch_size,src_len]
        # input = [1,batch_size,output_size]
        #print(s.shape)
        a = self.attention(s,enc_output,mask)
        # a = [batch_size,1,src_len]
        a = a.unsqueeze(1)

        ## weighted = [batch_size,1,enc_hid_dim*2]
        weighted = torch.bmm(a,enc_output)
        ## rnn_input = [batch_size,1,enc_hid_dim*2+output_size]
        # print(weighted.shape)
        # print(dec_input.shape)
        rnn_input = torch.cat((weighted,dec_input),dim=2)


        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [batch_size,1,dec_hid_dim]
        # hidden = [1,batch_size,dec_hid_dim]
        output,hidden = self.rnn(rnn_input,s.unsqueeze(0))
        #print(output.shape)
        #print(hidden.shape)

        # this also means that output == hidden
        assert (output.squeeze() == hidden.squeeze()).all()

        return output.squeeze(),hidden.squeeze(), a.squeeze(1)



class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,device,dec_out,num_out):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # self.fcn_out = nn.Sequential(
        #     nn.Linear(dec_out*2,8),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(8, num_out)
        # )
        self.fcn_out = nn.Linear(dec_out*2,num_out)
        self.dec_out = dec_out
        # self.trg_len = trg_len

    def create_mask(self,src_len):
        src_len_max = max(src_len)
        mask = torch.zeros(len(src_len),src_len_max)
        for i in range(len(src_len)):
            mask[i,:src_len[i]] = 1
        return mask
    def forward(self,src,src_len,trg,trg_tru_len):
        ## src = [batch_size,src_len,input_size]
        ## src_len = [batch_size]
        ## trg = [batch_size,trg_len]

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        encoder_outputs, hidden = self.encoder(src, src_len)
        enc_hidden = hidden


        #print(hidden.shape)
        mask = self.create_mask(src_len).to(self.device)
        outputs = torch.zeros(batch_size, trg_len,self.dec_out).to(self.device)

        for t in range(0,trg_len):
            input = trg[:,t,:].unsqueeze(1)
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:,t,:] = output
        #print(outputs.shape)
        outputs_be_inputing = []
        out_len_index = [i - 1 for i in trg_tru_len]
        for i in range(len(outputs)):
            # print(i)
            outputs_be_inputing.append(outputs[i][out_len_index[i]].unsqueeze_(0))
            # return self.linear_last(torch.FloatTensor(np.array(out_pad_last)))
        outputs_be_inputing = torch.cat(outputs_be_inputing, dim=0)
        #print(enc_hidden.shape)
        #print(outputs_be_inputing.shape)
        input = torch.cat((enc_hidden,outputs_be_inputing),dim=1)
        #print(outputs_be_inputing.shape)
        return input,self.fcn_out(input)


class Config(object):
    lr = 0.01
    epochs=500
    nums_out=2
    enc_hidden_size = 16
    dec_hidden_size = 16



class Seq2seqDataset(data.Dataset):
    def __init__(self,src_tensor,src_len,target_tensor,tar_len,label):
        self.src_tensor = src_tensor
        self.src_len = src_len
        self.target_tensor = target_tensor
        self.target_len = tar_len
        self.label = label

    def __getitem__(self, item):
        return self.src_tensor[item],self.src_len[item],self.target_tensor[item],self.target_len[item],self.label[item]

    def __len__(self):
        return len(self.src_tensor)


def collate_fn(data):
    #print(type(data))
    data_len = torch.tensor([i[1] for i in data])
    label = torch.tensor([i[4] for i in data])
    sorted_len, sorted_idx = data_len.sort(0, descending=True)
    data_before = []
    data_after = []
    data_after_len = []
    for idx in sorted_idx:
        data_before.append(data[idx][0])
        data_after.append(data[idx][2])
        data_after_len.append(data[idx][3])
    label = label[sorted_idx.long()]
    packed_input = rnn_utils.pad_sequence(data_before,batch_first=True,padding_value=0)

    return packed_input,sorted_len,torch.stack(data_after),data_after_len,label



if __name__ == '__main__':

    # df = df[df.columns[df.isna().sum().values / len(df) < 0.6]]
    # df_columns_list = df.columns.to_list()
    # for index in range(3,len(df_columns_list)-1):
    #     df_column=df_columns_list[index]
    #     df[df_column] = df[df_column].apply(column_process)
    # df = df.iloc[:,3:-1].apply(lambda x:column_process(x),axis=1)
    df = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx', index_col=0)
    df_before_operation = df[df.dis_day<0]
    df_after_operation = df[df.dis_day>=0]
    df_before_operation = df_before_operation.fillna(df_before_operation.mean(numeric_only=True))
    df_after_operation = df_after_operation.fillna(df_after_operation.mean(numeric_only=True))
    patients_CF_before_operation = []
    patients_CF_len_before_operation = []
    patients_CF_after_operation = []
    patients_CF_len_after_operation = []
    patients_CF_ending = []
    opt = Config()
    for name, grouped in df_after_operation.groupby('patient_id'):
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
        patients_CF_ending.append(grouped.ending.unique()[0])
    print(len(patients_CF_len_before_operation)-len(patients_CF_after_operation))
    packed_output_after_operation = rnn_utils.pad_sequence(patients_CF_after_operation, batch_first=True, padding_value=0)
    # trg_len = max(patients_CF_len_after_operation)
    data = Seq2seqDataset(patients_CF_before_operation,patients_CF_len_before_operation,packed_output_after_operation,patients_CF_len_after_operation,patients_CF_ending)
    X_train, X_valid, y_train, y_valid = train_test_split(data, patients_CF_ending, test_size=0.2)
    X_train_copy = []
    for i in X_train:
        if i[-1] == 1:
            X_train_copy.append(i)
    for i in range(3):
        X_train = X_train + X_train_copy
    data_loader_train = DataLoader(X_train, batch_size=len(X_train), shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = Encoder(patients_CF_before_operation[0].shape[1], opt.enc_hidden_size, opt.dec_hidden_size).to(device)
    decoder_model = Decoder(patients_CF_after_operation[0].shape[1],opt.enc_hidden_size,opt.dec_hidden_size).to(device)
    Seq2Seq_model = Seq2Seq(encoder_model,decoder_model,device,opt.dec_hidden_size,opt.nums_out).to(device)
    optimizer = torch.optim.Adam(Seq2Seq_model.parameters(), lr=opt.lr)
    criterion = nn.BCELoss().to(device)
    data_before_batch, sorted_len_batch, data_after_batch, data_after_batch_len, label_batch = iter(data_loader_train).next()




    label_batch = to_categorical(label_batch, opt.nums_out)
    for epoch in tqdm(range(opt.epochs)):
        optimizer.zero_grad()
        batch_x = data_before_batch.to(device)
        batch_y = data_after_batch.to(device)
        logits = Seq2Seq_model(batch_x,sorted_len_batch,batch_y,data_after_batch_len)
        loss = criterion(F.softmax(logits, dim=-1), torch.FloatTensor(label_batch).to(device))
        loss.backward();
        optimizer.step();
        if (epoch + 1) % 100 == 0:
            print("Seq2Seq_Epoch {} | Seq2Seq_Loss {:.4f}".format(epoch + 1, loss.item()));


    with torch.no_grad():
        data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid),  collate_fn=collate_fn)
        data_before_valid, sorted_len_valid, data_after_valid, data_after_batch_len_valid, label_batch_valid = iter(data_loader_valid).next()
        Seq2Seq_model.eval()
        logits = Seq2Seq_model(data_before_valid.to(device),sorted_len_valid,data_after_valid.to(device),data_after_batch_len_valid)
        logits = F.softmax(logits, dim=-1);
        data_label_valid = to_categorical(label_batch_valid,opt.nums_out)
        for i in range(logits.shape[1]):
            fpr, tpr, roc_auc = caculate_auc(data_label_valid[:, i], logits[:, i].detach().cpu().numpy());
            print(roc_auc)




















