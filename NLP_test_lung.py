import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve
from collections import Counter
# from CF_CT_NLP_ending import collate_fn
from NLP_lung import collate_func
from utils import to_categorical
import pandas as pd



if __name__ == '__main__':
    sick_name = 'ending'
    type_data = 'bf'
    T_V = 'valid'


    for fold in range(0,5):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # transformer_model = torch.load('./data/models/Integration_lung/Transformer_NLP_complication-{}.pkl'.format(str(fold)))
        transformer_model = torch.load('./data/models/Integration_ending/Transformer_NLP_{}_{}-{}.pkl'.format(sick_name,type_data,str(fold)))
        transformer_model.eval()
        X_valid = torch.load('./data/Integration_data/NLP_X_{}_{}_{}-{}.pt'.format(T_V,sick_name,type_data,str(fold)))
        df_path = './data/Integration_data/NLP_hidden_CF_{}_{}_{}-{}.xlsx'.format(T_V,sick_name,type_data,str(fold))
        with torch.no_grad():
            transformer_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transformer_model.to(device)
            loader = DataLoader(dataset=X_valid, batch_size=9, collate_fn=collate_func)
            flag = False
            for batch_patient_id, batch_src_pad, batch_src_len,batch_tgt_pad, batch_tgt_len,batch_ending in loader:
                batch_NLP_hidden, logits, enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model(
                    batch_src_pad.to(device), batch_tgt_pad.to(device), torch.tensor(batch_tgt_len))
                # batch_ending = to_categorical(batch_ending, 2)

                # for i in range(len(batch_ending)):
                #     if batch_ending[i] == 1:
                #         print(batch_patient_id[i])
                #         print(batch_ending[i])
                #         print(F.softmax(logits[i], dim=-1))

                if flag:
                    output = torch.cat([output, logits], dim=0)
                    output_label = output_label + batch_ending
                    # output_patient_id = torch.cat([output_patient_id,batch_patient_id],dim=0)
                    output_patient_id = output_patient_id + batch_patient_id
                    NLP_hidden = torch.cat([NLP_hidden, batch_NLP_hidden], dim=0)
                else:
                    output = logits
                    output_label = batch_ending
                    output_patient_id = batch_patient_id
                    NLP_hidden = batch_NLP_hidden
                    flag = True
                # print(logits)
                # print(label_batch_valid)
            print(Counter(output_label))
            output_label = to_categorical(output_label, 2)
            logits = F.softmax(output, dim=-1);

            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)
            if T_V == 'valid':
                torch.save(output_label[:, i], './data/Integration_data/patients_valid_NLP_{}_{}-{}.pt'.format(sick_name,type_data,str(fold)))
                torch.save(logits[:, i].detach().cpu().numpy(),
                           './data/Integration_data/logits_NLP_{}_{}-{}.pt'.format(sick_name,type_data,str(fold)))

            NLP_hidden_array = NLP_hidden.detach().cpu().numpy()
            NLP_hidden_df = pd.DataFrame(NLP_hidden_array)
            NLP_hidden_df.columns = ['NLP_hidden_{}'.format(str(i)) for i in range(NLP_hidden_df.shape[1])]
            NLP_hidden_df['patient_id'] = output_patient_id
            NLP_hidden_df['label'] = output_label[:, 1]
            NLP_hidden_df.to_excel(df_path)
