import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import win32com.client as wc
import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import caculate_auc
import seaborn as sns

def get_pred_label(pred_logits,threshold):
    pred_labels = []
    for i in pred_logits:
        if i>threshold:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    return pred_labels


if __name__ == '__main__' :
    for fold in range(0, 5):
        patients_ending_valid_inte_bf_ending = torch.load(
            './data/Integration_data/patients_ending_valid_CF_lung-{}_inte_bf_ending.pt'.format(str(fold)))
        logits_inte_bf_ending = torch.load('./data/Integration_data/logits_CF_lung-{}_inte_bf_ending.pt'.format(str(fold)))

        patients_ending_valid_inte_ending = torch.load(
            './data/Integration_data/patients_ending_valid_CF_lung-{}_inte_ending.pt'.format(str(fold)))
        logits_inte_ending = torch.load('./data/Integration_data/logits_CF_lung-{}_inte_ending.pt'.format(str(fold)))

        patients_ending_valid_inte_bf_lung = torch.load(
            './data/Integration_data/patients_ending_valid_CF_lung-{}_inte_bf.pt'.format(str(fold)))
        logits_inte_bf_lung  = torch.load('./data/Integration_data/logits_CF_lung-{}_inte_bf.pt'.format(str(fold)))

        patients_ending_valid_inte_lung = torch.load(
            './data/Integration_data/patients_ending_valid_CF_lung-{}_pltinte.pt'.format(str(fold)))
        logits_valid_inte_lung = torch.load('./data/Integration_data/logits_CF_lung-{}_pltinte.pt'.format(fold))

        if fold == 0:
            inte_labels_bf_ending = patients_ending_valid_inte_bf_ending
            inte_logits_bf_ending = logits_inte_bf_ending

            inte_labels_inte_ending = patients_ending_valid_inte_ending
            inte_logits_inte_ending = logits_inte_ending

            inte_labels_inte_bf_lung = patients_ending_valid_inte_bf_lung
            inte_logits_inte_bf_lung = logits_inte_bf_lung

            inte_labels_lung = patients_ending_valid_inte_lung
            inte_logits_lung = logits_valid_inte_lung


        else:
            inte_labels_bf_ending = np.append(inte_labels_bf_ending, patients_ending_valid_inte_bf_ending)
            inte_logits_bf_ending = np.append(inte_logits_bf_ending, logits_inte_bf_ending)

            inte_labels_inte_ending =  np.append(inte_labels_inte_ending, patients_ending_valid_inte_ending)
            inte_logits_inte_ending = np.append(inte_logits_inte_ending, logits_inte_ending)

            inte_labels_inte_bf_lung = np.append(inte_labels_inte_bf_lung, patients_ending_valid_inte_bf_lung)
            inte_logits_inte_bf_lung = np.append(inte_logits_inte_bf_lung, logits_inte_bf_lung)



            inte_labels_lung = np.append(inte_labels_lung, patients_ending_valid_inte_lung)
            inte_logits_lung = np.append(inte_logits_lung, logits_valid_inte_lung)

    inte_logits_bf_ending_label = get_pred_label(inte_logits_bf_ending,0.5)
    inte_logits_inte_ending_label = get_pred_label(inte_logits_inte_ending, 0.5)
    inte_logits_inte_bf_lung_label = get_pred_label(inte_logits_inte_bf_lung, 0.5)
    inte_logits_lung_label = get_pred_label(inte_logits_lung, 0.5)

    C1_bf_ending = confusion_matrix(inte_labels_bf_ending,inte_logits_bf_ending_label,labels=[0,1])

    C1_ending = confusion_matrix(inte_labels_inte_ending,inte_logits_inte_ending_label,labels=[0,1])

    C1_bf_lung = confusion_matrix(inte_labels_inte_bf_lung, inte_logits_inte_bf_lung_label, labels=[0, 1])

    C1_lung = confusion_matrix(inte_labels_lung, inte_logits_lung_label, labels=[0, 1])

    xticks = ['Deceased', 'Cured']
    yticks = ['Deceased', 'Cured']
    f,ax = plt.subplots()

    C1_bf_ending = C1_bf_ending.astype('float') / C1_bf_ending.sum(axis=1)

    sns.heatmap(C1_bf_ending,annot=True,ax=ax,cmap="YlGnBu",xticklabels=xticks,
                yticklabels=yticks,annot_kws={"fontsize":15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig('./data/figures/confusion_matrix/Integration_bf_ending.jpg')
    plt.close()

    xticks = ['Deceased', 'Cured']
    yticks = ['Deceased', 'Cured']
    f, ax = plt.subplots()

    C1_ending = C1_ending.astype('float') / C1_ending.sum(axis=1)

    sns.heatmap(C1_ending, annot=True, ax=ax, cmap="YlGnBu", xticklabels=xticks,
                yticklabels=yticks,annot_kws={"fontsize":15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig('./data/figures/confusion_matrix/Integration_ending.jpg')
    plt.close()

    xticks = ['not lung infection', 'lung infection']
    yticks = ['not lung infection', 'lung infection']
    f, ax = plt.subplots()

    C1_bf_lung = C1_bf_lung.astype('float') /C1_bf_lung.sum(axis=1)

    sns.heatmap(C1_bf_lung, annot=True, ax=ax, cmap="YlGnBu", xticklabels=xticks,
                yticklabels=yticks,annot_kws={"fontsize":15})
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('./data/figures/confusion_matrix/Integration_lung_before.jpg')
    plt.close()

    xticks = ['not lung infection', 'lung infection']
    yticks = ['not lung infection', 'lung infection']
    f, ax = plt.subplots()

    C1_lung = C1_lung.astype('float') / C1_lung.sum(axis=1)

    sns.heatmap(C1_lung, annot=True, ax=ax, cmap="YlGnBu", xticklabels=xticks,
                yticklabels=yticks,annot_kws={"fontsize":15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig('./data/figures/confusion_matrix/Integration_lung.jpg')
    plt.close()







