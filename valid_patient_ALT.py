"""
Filename: valid_patient_ALT.py
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

if __name__ == '__main__' :
    patient_info = pd.read_excel('./data/患者信息汇总 -最终版.xlsx')
    patient_info = patient_info.sort_values(by='入院日期')
    patient_info_valid_id = patient_info.loc[400:600,'编号']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sick_name = 'is_lung'
    py_type = 'bf'
    path = './data/figures/figure_Integration_{}_{}_valid.jpg'.format(sick_name, py_type)
    lw = 3
    # flag = True
    for fold in range(0,5):
        CF_valid_id = pd.read_excel('./data/Integration_data/CF_hidden_valid_df_{}-{}_multi.xlsx'.format(sick_name,str(fold)))['patient_id']
        patients_ending_valid_CF_ = torch.load('./data/Integration_data/patients_ending_valid_CF_{}-{}_multi.pt'.format(sick_name, str(fold)))
        logits_CF_ = torch.load('./data/Integration_data/logits_CF_{}-{}_multi.pt'.format(sick_name, str(fold)))
        patients_ending_valid_CF = []
        logits_CF = []
        for index in range(len(CF_valid_id)):
            CF_id = CF_valid_id[index]
            if CF_id in patient_info_valid_id:
                patients_ending_valid_CF.append(patients_ending_valid_CF_[index])
                logits_CF.append(logits_CF_[index])

        MR_valid_id = pd.read_excel(
            './data/Integration_data/MR_hidden_valid_df_{}_{}_{}.xlsx'.format(sick_name, py_type, str(fold)),
            index_col=0)['patient_id']
        patients_ending_valid_MR_CT = []
        logits_MR_CT = []

        patients_ending_valid_MR_CT_ = torch.load( './data/Integration_data/patients_valid_MR_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
        logits_MR_CT_ = torch.load('./data/Integration_data/logits_MR_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
        #
        for index in range(len(MR_valid_id)):
            MR_id = MR_valid_id[index]
            if MR_id in patient_info_valid_id:
                patients_ending_valid_MR_CT.append(patients_ending_valid_MR_CT_[index])
                logits_MR_CT.append(logits_MR_CT_[index])

        CT_valid_id = pd.read_excel(
                    './data/Integration_data/CT_hidden_valid_df_{}_{}_{}.xlsx'.format(sick_name, py_type, str(fold)),
                    index_col=0)['patient_id']
        patients_ending_valid_CT = []
        logits_CT = []
        patients_ending_valid_CT_ = torch.load(
            './data/Integration_data/patients_valid_CT_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
        logits_CT_ = torch.load('./data/Integration_data/logits_CT_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
        #
        for index in range(len(CT_valid_id)):
            CT_id = CT_valid_id[index]
            if CT_id in patient_info_valid_id:
                patients_ending_valid_CT.append(patients_ending_valid_CT_[index])
                logits_CT.append(logits_CT_[index])






        patients_ending_valid_NLP_ = torch.load(
            './data/Integration_data/patients_valid_NLP_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
        logits_NLP_ = torch.load('./data/Integration_data/logits_NLP_{}_{}-{}.pt'.format(sick_name, py_type, str(fold)))
        NLP_valid_id = pd.read_excel('./data/Integration_data/NLP_hidden_CF_valid_{}_{}-{}.xlsx'.format(sick_name, py_type,str(fold)),index_col=0)['patient_id']
        patients_ending_valid_NLP = []
        logits_NLP = []
        for index in range(len(NLP_valid_id)):
            NLP_id = NLP_valid_id[index]
            if NLP_id in patient_info_valid_id:
                patients_ending_valid_NLP.append(patients_ending_valid_NLP_[index])
                logits_NLP.append(logits_NLP_[index])


        patients_ending_valid_inte_lung_ = torch.load(
            './data/Integration_data/patients_valid_{}_{}_{}.pt'.format(sick_name, py_type, str(fold)))
        logits_valid_inte_lung_ = torch.load(
            './data/Integration_data/logits_{}_{}_{}.pt'.format(sick_name, py_type, str(fold)))
        multi_valid_id = torch.load('./data/Integration_data/patients_id_{}_{}_{}.pt'.format(sick_name,py_type,str(fold)))
        patients_ending_valid_inte_lung = []
        logits_valid_inte_lung = []

        for index in range(len(multi_valid_id)):
            multi_id = multi_valid_id[index]
            if multi_id in patient_info_valid_id:
                patients_ending_valid_inte_lung.append(patients_ending_valid_inte_lung_[index])
                logits_valid_inte_lung.append(logits_valid_inte_lung_[index])



        if fold == 0:
            CF_label = patients_ending_valid_CF
            CF_logits = logits_CF
            MR_CT_label =  patients_ending_valid_MR_CT
            MR_CT_logits = logits_MR_CT
            CT_label = patients_ending_valid_CT
            CT_logits = logits_CT
            # # print(flag)
            NLP_label = patients_ending_valid_NLP
            NLP_logits = logits_NLP
            flag = False
            inte_lung_labels = patients_ending_valid_inte_lung
            inte_lung_logits = logits_valid_inte_lung
        else:
            CF_label = np.append(CF_label, patients_ending_valid_CF)
            CF_logits = np.append(CF_logits, logits_CF)
            # #
            MR_CT_label = np.append(MR_CT_label,patients_ending_valid_MR_CT)
            MR_CT_logits = np.append(MR_CT_logits,logits_MR_CT)
            CT_label = np.append(CT_label, patients_ending_valid_CT)
            CT_logits = np.append(CT_logits, logits_CT)
            #
            NLP_label = np.append(NLP_label, patients_ending_valid_NLP)
            NLP_logits = np.append(NLP_logits, logits_NLP)
            # #
            inte_lung_labels = np.append(inte_lung_labels, patients_ending_valid_inte_lung)
            inte_lung_logits = np.append(inte_lung_logits, logits_valid_inte_lung)

    fpr, tpr, roc_auc = caculate_auc(CF_label, CF_logits)
    print(roc_auc)
    font1 = {
        'family': 'Arial',
        'size': 20
    }
    plt.figure(figsize=(10, 10))
    #
    plt.plot([0, 1], [0, 1], color='silver', lw=lw, linestyle='--')
    #
    fpr_MR_CT, tpr_MR_CT, roc_auc_MR_CT = caculate_auc(MR_CT_label, MR_CT_logits)
    print(roc_auc_MR_CT)
    #
    fpr_inte_lung, tpr_inte_lung, roc_auc_inte_lung = caculate_auc(inte_lung_labels,inte_lung_logits)
    print(roc_auc_inte_lung)

    #
    #
    plt.plot(fpr_inte_lung, tpr_inte_lung, color='darkorange', lw=lw,
             label='CF_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_inte_lung)
    fpr_NLP, tpr_NLP, roc_auc_NLP = caculate_auc(NLP_label, NLP_logits)
    print(roc_auc_NLP)
    plt.plot(fpr_NLP, tpr_NLP, color='cyan', lw=lw, label='NLP_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_NLP)
    plt.plot(fpr, tpr, color='darkorchid', lw=lw, label='inte_{} (AUC = %0.3f)'.format(sick_name) % roc_auc)
    # #
    fpr_CT, tpr_CT, roc_auc_CT = caculate_auc(CT_label, CT_logits)
    print(roc_auc_CT)
    plt.plot(fpr_CT, tpr_CT, color='dodgerblue', lw=lw, label='CT_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_CT)

    #
    plt.plot(fpr_MR_CT, tpr_MR_CT, color='crimson', lw=lw, label='MR_{} (AUC = %0.3f)'.format(sick_name) % roc_auc_MR_CT)
    #
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