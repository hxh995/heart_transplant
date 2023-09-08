import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import win32com.client as wc
import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import caculate_auc
from CF_CT_NLP_ending import collate_fn
from utils import to_categorical
if __name__ == '__main__' :
    path = './data/figures/figure_Integration_ending.jpg'
    lw = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_flag = 0
    for fold in range(0,5):
        ## CF
        Encoder = torch.load('./data/models/Integration_ending/Encoder_multi_CF_ending-{}.pkl'.format(str(fold)))
        Decoder = torch.load('./data/models/Integration_ending/Decoder_multi_CF_ending-{}.pkl'.format(str(fold)))
        Seq2Seq_model = torch.load('./data/models/Integration_ending/Seq2Seq_multi_CF_ending-{}.pkl'.format(str(fold)))
        X_valid = torch.load("./data/Integration_data/X_valid_CF_NLP-{}.pt".format(str(fold)))
        with torch.no_grad():
            data_loader_valid = DataLoader(X_valid, batch_size=len(X_valid), collate_fn=collate_fn)
            patient_id_valid, packed_patients_CF_before_operation_valid, packed_patients_CF_after_operation_valid, patients_CF_len_before_operation_valid, patients_CF_len_after_operation_valid, \
            packed_first_trip_dim_valid, patients_first_trip_len_valid, packed_patients_pdf_dim_valid, patients_pdf_word_len_valid, patients_ending_valid = iter(
                data_loader_valid).next()

            Seq2Seq_model.eval()
            CF_hidden_valid, logits_valid = Seq2Seq_model(packed_patients_CF_before_operation_valid.to(device),
                                                          patients_CF_len_before_operation_valid
                                                          , packed_patients_CF_after_operation_valid.to(device),
                                                          patients_CF_len_after_operation_valid)
            logits_valid = F.softmax(logits_valid, dim=-1);
            data_label_valid = to_categorical(patients_ending_valid, 2)

            for i in range(logits_valid.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(data_label_valid[:, i], logits_valid[:, i].detach().cpu().numpy());
                # print(roc_auc)



        #
        patients_ending_valid_CT = torch.load('./data/Integration_data/patients_valid_CT_ending-{}.pt'.format(str(fold)))
        logits_CT = torch.load('./data/Integration_data/logits_CT_ending-{}.pt'.format(str(fold)))
        #
        patients_ending_valid_NLP_clinical = torch.load('./data/Integration_data/patients_valid_NLP_ending_multi-{}.pt'.format(str(fold)))
        logits_NLP_clinical = torch.load('./data/Integration_data/logits_NLP_ending_multi-{}.pt'.format(str(fold)))
        #
        patients_ending_valid_MR= torch.load('./data/Integration_data/patients_ending_valid_CF_lung-{}_MR_ending.pt'.format(str(fold)))
        logits_MR = torch.load('./data/Integration_data/logits_CF_lung-{}_MR_ending.pt'.format(str(fold)))
        #
        patients_ending_valid_inte = torch.load('./data/Integration_data/patients_valid_ending_multi-{}.pt'.format(str(fold)))
        logits_inte = torch.load('./data/Integration_data/logits_ending_multi-{}.pt'.format(str(fold)))

        if fold_flag == 0 :
            data_label_valids = data_label_valid[:, i]
            logits_valids = logits_valid[:, i].detach().cpu().numpy()

            NLP_clinical_label = patients_ending_valid_NLP_clinical
            NLP_clinical_logits = logits_NLP_clinical

            CT_clinical_label = patients_ending_valid_CT
            CT_clinical_logits = logits_CT
            #
            MR_clinical_label = patients_ending_valid_MR
            MR_clinical_logits = logits_MR
            #
            inte_clinical_label = patients_ending_valid_inte
            inte_clinical_logits = logits_inte
            fold_flag = fold_flag + 1


        else:


            NLP_clinical_label = np.append(NLP_clinical_label, patients_ending_valid_NLP_clinical)
            NLP_clinical_logits = np.append(NLP_clinical_logits, logits_NLP_clinical)

            data_label_valids = np.append(data_label_valids, data_label_valid[:, i])
            logits_valids = np.append(logits_valids, logits_valid[:, i].detach().cpu().numpy())

            CT_clinical_label = np.append(CT_clinical_label, patients_ending_valid_CT)
            CT_clinical_logits = np.append(CT_clinical_logits, logits_CT)
            #
            MR_clinical_label = np.append(MR_clinical_label, patients_ending_valid_MR)
            MR_clinical_logits = np.append(MR_clinical_logits, logits_MR)
            #
            inte_clinical_label = np.append(inte_clinical_label, patients_ending_valid_inte)
            inte_clinical_logits = np.append(inte_clinical_logits, logits_inte)


    fpr,tpr,roc_auc = caculate_auc(NLP_clinical_label,NLP_clinical_logits)
    print(roc_auc)

    fpr_CF, tpr_CF, roc_auc_CF = caculate_auc(data_label_valids,logits_valids)
    print(roc_auc_CF)

    fpr_CT, tpr_CT, roc_auc_CT = caculate_auc(CT_clinical_label, CT_clinical_logits)
    print(roc_auc_CT)
    #
    fpr_MR, tpr_MR, roc_auc_MR = caculate_auc(MR_clinical_label, MR_clinical_logits)
    print(roc_auc_MR)
    #
    fpr_inte, tpr_inte, roc_auc_inte = caculate_auc(inte_clinical_label, inte_clinical_logits)
    print(roc_auc_inte)


    font1 = {
        'family' : 'Arial',
        'size' :20
    }

    plt.figure(figsize=(10,10))
    plt.plot(fpr_inte, tpr_inte, color='darkorange', lw=lw, label='inte_ending (AUC = %0.3f)' % roc_auc_inte)
    plt.plot(fpr,tpr,color = 'dodgerblue',lw=lw,label='NLP_ending (AUC = %0.3f)' % roc_auc)
    plt.plot(fpr_CF, tpr_CF, color='darkorchid', lw=lw, label='CF_ending (AUC = %0.3f)' % roc_auc_CF)

    plt.plot(fpr_CT, tpr_CT, color='cyan', lw=lw, label='CT_ending (AUC = %0.3f)' % roc_auc_CT)
    #
    plt.plot(fpr_MR, tpr_MR, color='crimson', lw=lw, label='MR_ending (AUC = %0.3f)' % roc_auc_MR)
    #


    plt.plot([0,1],[0,1],color='silver',lw=lw,linestyle='--')



    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    my_x_ticks = np.arange(0,1.05,0.5)
    my_y_ticks = np.arange(0,1.05,0.5)
    plt.xlabel('1 - Sp',fontsize=30)
    plt.ylabel('Sn',fontsize=30)
    plt.xticks(my_x_ticks,fontsize=30)
    plt.yticks(my_y_ticks,fontsize=30)

    plt.legend(loc='lower right',fontsize=20,prop=font1)

    plt.savefig(path)
    plt.close()

