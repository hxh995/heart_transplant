import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve
from collections import Counter
# from CF_CT_NLP_ending import collate_fn
from CF_CT_NLP_complication import collate_func
from utils import to_categorical
import pandas as pd




if __name__ == '__main__':

    # loader = torch.load('./data/NLP_test_loader.pkl')
    for fold in range(5):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # transformer_model = torch.load('./data/models/Integration_lung/Transformer_NLP_complication-{}.pkl'.format(str(fold)))
        transformer_model = torch.load('./data/models/Integration_lung/Transformer_NLP_complication-{}.pkl'.format(str(fold)))
        X_valid = torch.load("./data/integration_data/NLP_X_valid-{}.pt".format(str(fold)))
        df_path = './data/Integration_data/NLP_hidden_CF_valid_complication-{}.xlsx'.format(str(fold))
        with torch.no_grad():
            transformer_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transformer_model.to(device)
            loader = DataLoader(dataset=X_valid, batch_size=10,collate_fn=collate_func)
            flag = False
            for batch_patient_id,batch_src_pad_valid,batch_src_len,batch_tag_valid,batch_tag_len,label_batch_valid in loader:
                batch_src_len_max = max(batch_src_len)
                batch_tag_len_max = max(batch_tag_len)
                batch_src_pad = batch_src_pad_valid[:, :batch_src_len_max, :]
                batch_tag_pad = batch_tag_valid[:, :batch_tag_len_max, :]
                batch_NLP_hidden,logits,enc_self_attns, dec_self_attns, dec_enc_attns = transformer_model(batch_src_pad.to(device),
                                                                                         batch_tag_pad.to(device),torch.tensor(batch_tag_len))
                # label_batch_valid = torch.FloatTensor(to_categorical(label_batch_valid,2))
                if flag:
                    output = torch.cat([output,logits],dim=0)
                    output_label = output_label + label_batch_valid
                    # output_patient_id = torch.cat([output_patient_id,batch_patient_id],dim=0)
                    output_patient_id = output_patient_id + batch_patient_id
                    NLP_hidden = torch.cat([NLP_hidden,batch_NLP_hidden],dim=0)
                else:
                    output = logits
                    output_label = label_batch_valid
                    output_patient_id = batch_patient_id
                    NLP_hidden = batch_NLP_hidden
                    flag = True
                # print(logits)
                # print(label_batch_valid)
            output_label = to_categorical(output_label,2)

            logits = F.softmax(output,dim=-1);
            for i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)
            torch.save(output_label[:, i], './data/Integration_data/patients_valid_NLP_lung-{}.pt'.format(str(fold)))
            torch.save(logits[:, i].detach().cpu().numpy(), './data/Integration_data/logits_NLP_lung-{}.pt'.format(str(fold)))

            NLP_hidden_array = NLP_hidden.detach().cpu().numpy()
            NLP_hidden_df = pd.DataFrame(NLP_hidden_array)
            NLP_hidden_df.columns = ['NLP_hidden_{}'.format(str(i)) for i in range(NLP_hidden_df.shape[1])]
            NLP_hidden_df['patient_id'] = output_patient_id
            NLP_hidden_df['label'] = output_label[:,1]
            NLP_hidden_df.to_excel(df_path)




