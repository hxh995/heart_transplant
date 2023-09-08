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
    path = './data/figures/figure_Integration_lung_before.jpg'
    sick_name = 'is_lung'
    type_data = 'bf'
    lw = 3
    for fold in range(0,5):
        patients_ending_valid_CF = torch.load(
            './data/Integration_data/patients_ending_valid_CF_{}-{}_inte_bf.pt'.format(sick_name,str(fold)))
        logits_CF = torch.load('./data/Integration_data/logits_CF_{}-{}_inte_bf.pt'.format(sick_name,str(fold)))
        patients_ending_valid_NLP = torch.load('./data/Integration_data/patients_valid_NLP_{}_{}-{}.pt'.format(sick_name,type_data,str(fold)))
        logits_NLP = torch.load('./data/Integration_data/logits_NLP_{}_{}-{}.pt'.format(sick_name,type_data,str(fold)))

        patients_ending_valid_CT = torch.load('./data/Integration_data/patients_valid_CT_lung-{}_bf.pt'.format(str(fold)))
        logits_CT = torch.load('./data/Integration_data/logits_CT_lung-{}_bf.pt'.format(str(fold)))

        patients_ending_valid_MR = torch.load('./data/Integration_data/patients_valid_MR_{}_bf-{}.pt'.format(sick_name,str(fold)))
        logits_MR = torch.load('./data/Integration_data/logits_MR_{}_bf-{}.pt'.format(sick_name,str(fold)))

        patients_ending_valid_inte = torch.load('./data/Integration_data/patients_valid_{}_{}_{}.pt'.format(sick_name,type_data,str(fold)))
        logits_inte = torch.load('./data/Integration_data/logits_{}_{}_{}.pt'.format(sick_name,type_data,str(fold)))

        if fold == 0 :
            CF_label = patients_ending_valid_CF
            CF_logits = logits_CF

            NLP_label = patients_ending_valid_NLP
            NLP_logits = logits_NLP

            CT_label = patients_ending_valid_CT
            CT_logits = logits_CT

            MR_label = patients_ending_valid_MR
            MR_logits = logits_MR

            inte_label = patients_ending_valid_inte
            inte_logits = logits_inte


        else:
            CF_label = np.append(CF_label, patients_ending_valid_CF)
            CF_logits = np.append(CF_logits, logits_CF)
            NLP_label = np.append(NLP_label, patients_ending_valid_CF)
            NLP_logits = np.append(NLP_logits, logits_CF)
            CT_label = np.append(CT_label, patients_ending_valid_CT)
            CT_logits = np.append(CT_logits, logits_CT)

            MR_label = np.append(MR_label, patients_ending_valid_MR)
            MR_logits = np.append(MR_logits, logits_MR)

            inte_label = np.append(inte_label, patients_ending_valid_inte)
            inte_logits = np.append(inte_logits, logits_inte)




    fpr, tpr, roc_auc = caculate_auc(CF_label, CF_logits)
    print(roc_auc)
    fpr_NLP, tpr_NLP, roc_auc_NLP = caculate_auc(NLP_label, NLP_logits)
    print(roc_auc_NLP)
    fpr_CT, tpr_CT, roc_auc_CT = caculate_auc(CT_label, CT_logits)
    print(roc_auc_CT)
    fpr_MR, tpr_MR, roc_auc_MR = caculate_auc(MR_label, MR_logits)
    print(roc_auc_MR)

    fpr_inte, tpr_inte, roc_auc_inte = caculate_auc(inte_label, inte_logits)
    print(roc_auc_inte)


    font1 = {
        'family': 'Arial',
        'size': 20
    }

    plt.figure(figsize=(10, 10))
    plt.plot(fpr_CT, tpr_CT, color='darkorange', lw=lw, label='CT_lung_before (AUC = %0.3f)' % roc_auc_CT)
    plt.plot(fpr, tpr, color='darkorchid', lw=lw, label='CF_lung_before (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='silver', lw=lw, linestyle='--')


    plt.plot(fpr_NLP, tpr_NLP, color='cyan', lw=lw, label='NLP_lung_before (AUC = %0.3f)' % roc_auc_NLP)

    plt.plot(fpr_MR, tpr_MR, color='crimson', lw=lw, label='MR_lung_before (AUC = %0.3f)' % roc_auc_MR)
    plt.plot(fpr_inte, tpr_inte, color='dodgerblue', lw=lw, label='inte_lung_before (AUC = %0.3f)' % roc_auc_inte)


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


