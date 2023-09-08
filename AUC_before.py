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

if __name__ == '__main__' :
    path = './data/figures/figure_Integration_ending_before.jpg'
    lw = 3
    for fold in range(0,5):
        patients_ending_valid_CF = torch.load('./data/Integration_data/CF_patients_ending_before_{}.pt'.format(str(fold)))
        logits_CF = torch.load('./data/Integration_data/logits_CF_patients_ending_before_{}.pt'.format(str(fold)))
        #
        # patients_ending_valid_MR_CT = torch.load( './data/Integration_data/patients_ending_valid_MR_CT_lung-{}.pt'.format(str(fold)))
        # logits_MR_CT = torch.load('./data/Integration_data/logits_MR_CT_lung-{}.pt'.format(str(fold)))

        patients_ending_valid_CT = torch.load('./data/Integration_data/patients_valid_CT_bf_ending-{}.pt'.format(str(fold)))
        logits_CT = torch.load('./data/Integration_data/logits_CT_bf_ending-{}.pt'.format(str(fold)))

        patients_ending_valid_NLP_clinical = torch.load('./data/Integration_data/patients_valid_NLP_ending_clinical-{}.pt'.format(str(fold)))
        logits_NLP_clinical = torch.load('./data/Integration_data/logits_NLP_ending_clinical-{}.pt'.format(str(fold)))

        patients_ending_valid_MR= torch.load('./data/Integration_data/patients_valid_MR_ending_bf-{}.pt'.format(str(fold)))
        logits_MR = torch.load('./data/Integration_data/logits_MR_ending_bf-{}.pt'.format(str(fold)))

        patients_ending_valid_inte = torch.load(
            './data/Integration_data/patients_ending_valid-{}_inte_bf_ending.pt'.format(str(fold)))
        logits_inte = torch.load('./data/Integration_data/logits_ending-{}_bf.pt'.format(str(fold)))

        if fold == 0 :

            NLP_clinical_label = patients_ending_valid_NLP_clinical
            NLP_clinical_logits = logits_NLP_clinical

            CF_clinical_label = patients_ending_valid_CF
            CF_clinical_logits = logits_CF

            CT_clinical_label = patients_ending_valid_CT
            CT_clinical_logits = logits_CT

            MR_clinical_label = patients_ending_valid_MR
            MR_clinical_logits = logits_MR

            inte_clinical_label = patients_ending_valid_inte
            inte_clinical_logits = logits_inte


        else:


            NLP_clinical_label = np.append(NLP_clinical_label, patients_ending_valid_NLP_clinical)
            NLP_clinical_logits = np.append(NLP_clinical_logits, logits_NLP_clinical)

            CF_clinical_label = np.append(CF_clinical_label, patients_ending_valid_CF)
            CF_clinical_logits = np.append(CF_clinical_logits,  logits_CF)

            CT_clinical_label = np.append(CT_clinical_label, patients_ending_valid_CT)
            CT_clinical_logits = np.append(CT_clinical_logits, logits_CT)

            MR_clinical_label = np.append(MR_clinical_label, patients_ending_valid_MR)
            MR_clinical_logits = np.append(MR_clinical_logits, logits_MR)

            inte_clinical_label = np.append(inte_clinical_label, patients_ending_valid_inte)
            inte_clinical_logits = np.append(inte_clinical_logits, logits_inte)


    fpr,tpr,roc_auc = caculate_auc(NLP_clinical_label,NLP_clinical_logits)
    print(roc_auc)

    fpr_CF, tpr_CF, roc_auc_CF = caculate_auc(CF_clinical_label,CF_clinical_logits)
    print(roc_auc_CF)

    fpr_CT, tpr_CT, roc_auc_CT = caculate_auc(CT_clinical_label, CT_clinical_logits)
    print(roc_auc_CT)

    fpr_MR, tpr_MR, roc_auc_MR = caculate_auc(MR_clinical_label, MR_clinical_logits)
    print(roc_auc_MR)

    fpr_inte, tpr_inte, roc_auc_inte = caculate_auc(inte_clinical_label, inte_clinical_logits)
    print(roc_auc_inte)


    font1 = {
        'family' : 'Arial',
        'size' :20
    }

    plt.figure(figsize=(10,10))
    plt.plot(fpr_inte, tpr_inte, color='darkorange', lw=lw, label='inte_ending_before (AUC = %0.3f)' % roc_auc_inte)
    plt.plot(fpr,tpr,color = 'cyan',lw=lw,label='NLP_ending_before (AUC = %0.3f)' % roc_auc)
    plt.plot(fpr_CF, tpr_CF, color='darkorchid', lw=lw, label='CF_ending_before (AUC = %0.3f)' % roc_auc_CF)

    plt.plot(fpr_CT, tpr_CT, color='dodgerblue', lw=lw, label='CT_ending_before (AUC = %0.3f)' % roc_auc_CT)

    plt.plot(fpr_MR, tpr_MR, color='crimson', lw=lw, label='MR_ending_before (AUC = %0.3f)' % roc_auc_MR)



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

