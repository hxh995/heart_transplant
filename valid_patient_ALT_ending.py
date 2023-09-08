"""
Filename: valid_patient_ALT_ending.py
Author: yellower
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CF_complication import collate_fn
import torch.nn.functional as F
from utils import to_categorical,caculate_auc
import numpy as np
from collections import Counter
if __name__ == '__main__' :
    font1 = {
        'family': 'Arial',
        'size': 20
    }
    plt.figure(figsize=(10, 10))
    lw = 3
    plt.plot([0, 1], [0, 1], color='silver', lw=lw, linestyle='--')
    patient_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    patient_info = patient_info.sort_values(by='入院日期')
    patient_info_valid_id = patient_info.loc[437:, '编号']
    valid_str = 'VALID'
    sick_name = 'ending'
    py_type = 'bf'
    path = './data/figures/figure_Integration_{}_{}_valid.jpg'.format(sick_name, py_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        transformer_model = torch.load(
            './data/models/Integration_ending/Transformer_NLP_{}_{}-{}.pkl'.format(sick_name, py_type, valid_str))
        transformer_model.eval()
        X_valid = torch.load('./data/Integration_data/NLP_X_{}_{}_{}-{}.pt'.format('valid', sick_name, py_type, valid_str))
        # X_valid = torch.load('./data/Integration_data/NLP_X_{}_{}_{}-{}.pt'.format('train', sick_name, py_type, valid_str))
        from NLP_lung import collate_func
        loader = DataLoader(dataset=X_valid, batch_size=36, collate_fn=collate_func)
        flag = False
        for batch_patient_id, batch_src_pad, batch_src_len, batch_tgt_pad, batch_tgt_len, batch_ending in loader:
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

        print(Counter(output_label))
        output_label = to_categorical(output_label, 2)
        logits = F.softmax(output, dim=-1);
        NLP_hidden_array = NLP_hidden.detach().cpu().numpy()
        NLP_hidden_df = pd.DataFrame(NLP_hidden_array)
        NLP_hidden_df.columns = ['NLP_hidden_{}'.format(str(i)) for i in range(NLP_hidden_df.shape[1])]
        NLP_hidden_df['patient_id'] = output_patient_id
        NLP_hidden_df['label'] = output_label[:, 1]
        df_path = './data/Integration_data/NLP_hidden_CF_{}_{}_{}-{}.xlsx'.format('valid', sick_name, py_type, 'VALID')
        NLP_hidden_df.to_excel(df_path)

        for i in range(logits.shape[1]):
            fpr_NLP, tpr_NLP, roc_auc_NLP = caculate_auc(output_label[:, i], logits[:, i].detach().cpu().numpy());
        print('NLP')
        print(roc_auc_NLP)

        patients_ending_valid_CF = torch.load('./data/Integration_data/patients_ending_valid_CF_bf_{}_{}.pt'.format(sick_name, 'VALID'))
        logits_CF = torch.load('./data/Integration_data/logits_CF_bf_{}_{}.pt'.format(sick_name, 'VALID'))

        patients_ending_valid_CT = torch.load('./data/Integration_data/patients_valid_CT_{}_{}-{}.pt'.format(sick_name, py_type, 'VALID'))
        logits_CT = torch.load('./data/Integration_data/logits_CT_{}_{}-{}.pt'.format(sick_name, py_type, 'VALID'))

        patients_ending_valid_MR = torch.load('./data/Integration_data/patients_valid_MR_{}_{}-{}.pt'.format(sick_name, py_type,'VALID'))

        logits_MR = torch.load('./data/Integration_data/logits_MR_{}_{}-{}.pt'.format(sick_name, py_type,'VALID'))

        patients_ending_valid_inte = torch.load('./data/Integration_data/patients_valid_{}_{}_{}.pt'.format(sick_name,py_type,'VALID'))

        logits_inte = torch.load('./data/Integration_data/logits_{}_{}_{}.pt'.format(sick_name,py_type,'VALID'))

        fpr, tpr, roc_auc = caculate_auc(patients_ending_valid_CF, logits_CF)
        print('CF')
        print(roc_auc)

        fpr_CT, tpr_CT, roc_auc_CT = caculate_auc(patients_ending_valid_CT, logits_CT)
        fpr_MR_CT, tpr_MR_CT, roc_auc_MR_CT = caculate_auc(patients_ending_valid_MR, logits_MR)
        print('MR')
        print(roc_auc_MR_CT)
        print('CT')
        print(roc_auc_CT)


        fpr_inte, tpr_inte, roc_auc_inte = caculate_auc(patients_ending_valid_inte, logits_inte)
        print('multi')
        print(roc_auc_inte)
        plt.plot(fpr_CT, tpr_CT, color='dodgerblue', lw=lw, label='CT_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_CT)
        plt.plot(fpr, tpr, color='darkorange', lw=lw,
                 label='CF_{} (AUC = %0.3f)'.format(sick_name) % roc_auc)
        plt.plot(fpr_NLP, tpr_NLP, color='cyan', lw=lw, label='NLP_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_NLP)
        plt.plot(fpr_MR_CT, tpr_MR_CT, color='crimson', lw=lw,
                 label='MR_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_MR_CT)

        plt.plot(fpr_inte, tpr_inte, color='darkorchid', lw=lw, label='inte_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_inte)



        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        my_x_ticks = np.arange(0, 1.05, 0.5)
        my_y_ticks = np.arange(0, 1.05, 0.5)
        plt.xlabel('1 - Sp', fontsize=30)
        plt.ylabel('Sn', fontsize=30)
        plt.xticks(my_x_ticks, fontsize=30)
        plt.yticks(my_y_ticks, fontsize=30)

        plt.legend(loc='lower right', fontsize=20, prop=font1)

        plt.savefig(path)
        plt.close()





