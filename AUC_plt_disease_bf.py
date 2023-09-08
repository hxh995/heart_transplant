"""
Filename: AUC_plt_disease_bf.py
Author: yellower
"""
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
    sick_name = 'is_Hyperlipidemia'
    py_type = 'bf'
    figure_name = 'is_Hyperlipidemia'
    path = './data/figures/figure_Integration_{}_{}.jpg'.format(sick_name, py_type)
    lw = 3

    # flag = True
    for fold in range(0,5):
        patients_ending_valid_CF = torch.load('./data/Integration_data/patients_ending_valid_CF_{}-{}_inte_bf.pt'.format(sick_name,str(fold)))
        logits_CF = torch.load('./data/Integration_data/logits_CF_{}-{}_inte_bf.pt'.format(sick_name,str(fold)))
        # #
        patients_ending_valid_MR = torch.load('./data/Integration_data/patients_valid_MR_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
        logits_MR = torch.load('./data/Integration_data/logits_MR_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
        # #
        patients_ending_valid_CT = torch.load('./data/Integration_data/patients_valid_CT_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
        logits_CT = torch.load('./data/Integration_data/logits_CT_{}_{}-{}.pt'.format(sick_name,py_type, str(fold)))
        #

        patients_ending_valid_NLP = torch.load('./data/Integration_data/patients_valid_NLP_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
        logits_NLP = torch.load('./data/Integration_data/logits_NLP_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
        #
        patients_ending_valid_inte_lung = torch.load('./data/Integration_data/patients_valid_{}_{}_{}.pt'.format(sick_name,py_type,str(fold)))
        logits_valid_inte_lung = torch.load('./data/Integration_data/logits_{}_{}_{}.pt'.format(sick_name,py_type,str(fold)))


        if fold == 0 :
            CF_label = patients_ending_valid_CF
            CF_logits = logits_CF
            MR_label =  patients_ending_valid_MR
            MR_logits = logits_MR
            CT_label = patients_ending_valid_CT
            CT_logits = logits_CT
            # # print(flag)
            NLP_label = patients_ending_valid_NLP
            NLP_logits = logits_NLP
            # # flag = False
            #
            inte_lung_labels = patients_ending_valid_inte_lung
            inte_lung_logits = logits_valid_inte_lung
        else:
            CF_label = np.append(CF_label,patients_ending_valid_CF)
            CF_logits = np.append(CF_logits,logits_CF)
            # #
            MR_label = np.append(MR_label,patients_ending_valid_MR)
            MR_logits = np.append(MR_logits,logits_MR)
            CT_label = np.append(CT_label, patients_ending_valid_CT)
            CT_logits = np.append(CT_logits, logits_CT)

            #
            NLP_label = np.append(NLP_label, patients_ending_valid_NLP)
            NLP_logits = np.append(NLP_logits, logits_NLP)
            # #
            inte_lung_labels = np.append(inte_lung_labels, patients_ending_valid_inte_lung)
            inte_lung_logits = np.append(inte_lung_logits, logits_valid_inte_lung)

    fpr,tpr,roc_auc = caculate_auc(CF_label,CF_logits)
    print(roc_auc)
    font1 = {
        'family' : 'Arial',
        'size' :20
    }
    plt.figure(figsize=(10,10))

    plt.plot([0,1],[0,1],color='silver',lw=lw,linestyle='--')
    #
    fpr_MR, tpr_MR, roc_auc_MR = caculate_auc(MR_label, MR_logits)
    print(roc_auc_MR)
    # # #
    fpr_inte_lung, tpr_inte_lung, roc_auc_inte_lung = caculate_auc(inte_lung_labels,
                                                                   inte_lung_logits)
    print(roc_auc_inte_lung)



    plt.plot(fpr_inte_lung, tpr_inte_lung, color='darkorange', lw=lw,
             label='inte_{} (AUC = %0.3f)'.format(figure_name) % roc_auc_inte_lung)
    fpr_NLP, tpr_NLP, roc_auc_NLP = caculate_auc(NLP_label, NLP_logits)
    print(roc_auc_NLP)
    plt.plot(fpr_NLP, tpr_NLP, color='cyan', lw=lw, label='NLP_{} (AUC = %0.3f)'.format(figure_name) % roc_auc_NLP)
    plt.plot(fpr, tpr, color='darkorchid', lw=lw, label='CF_{} (AUC = %0.3f)'.format(figure_name) % roc_auc)
    # #
    fpr_CT, tpr_CT, roc_auc_CT = caculate_auc(CT_label, CT_logits)
    print(roc_auc_CT)
    plt.plot(fpr_CT, tpr_CT, color='dodgerblue', lw=lw, label='CT_{} (AUC = %0.3f)'.format(figure_name) % roc_auc_CT)
    # #
    # # #
    plt.plot(fpr_MR, tpr_MR, color='crimson', lw=lw, label='MR_{} (AUC = %0.3f)'.format(figure_name) % roc_auc_MR)

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