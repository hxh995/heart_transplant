import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch.nn.utils.rnn as rnn_utils
import win32com.client as wc
import os

def column_process(x):
    if not pd.isnull(x) :
        if type(x) == int or type(x) == float:
            return x
        elif ',' in x:
            try:
               np.mean([float (i) for i in x.split(',')])
            except:
                x = str(x)
                if ('<' in x) or ('>' in x) or (';' in x) or ('&gt;' in x) or ('&lt;' in x):
                    return np.mean([float(i.replace('<', '').replace('>', '').replace('&gt;','').replace('&lt;','').replace(';','').replace('=','')) for i in x.split(',')])
                else:
                    print(x)
        else:
            try:
                x=float(x)
                return x
            except:
                x=str(x)
                if ('<' in x) or ('>' in x) or (';' in x) or ('&gt;' in x) or ('&lt;' in x):
                    return float(x.replace('<','').replace('>','').replace('&gt;','').replace('&lt;','').replace(';','').replace('=',''))
                else:
                    print(x)
    else:
        return np.nan

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def caculate_auc(y,prob):
    fpr, tpr, threshold = roc_curve(y, prob);
    roc_auc = auc(fpr, tpr);
    return fpr,tpr,roc_auc;

## 画ACU
def acu_curve(fpr,tpr,roc_auc,path):
    #plt.figure()
    lw = 3
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right",fontsize=20)
    plt.savefig(path)
    plt.close()


def doc_to_docx(path,doc_name):
    patients_Context_path = path
    doc_name = doc_name + '.doc'
    docx_name = doc_name + 'docx'
    word = wc.Dispatch("Word.Application")
    for root, dirs, files in os.walk(patients_Context_path):
        if files:
            if docx_name in files:
                print(root.split("\\")[-1])
                print("got it")
                continue
            else:
                root = os.path.abspath(root)
                patient_root = os.path.join(root, doc_name)
                print(root.split("\\")[-1])
                word = wc.Dispatch("Word.Application")
                doc = word.Documents.Open(patient_root.replace('\\', '/'))
                patient_root_docx = os.path.join(root, docx_name)
                doc.SaveAs(patient_root_docx, 12, False, "", True, "", False, False, False, False)
                doc.Close()

    word.Quit()



