"""
Filename: figure_5.py
Author: yellower
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
import seaborn as sns
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve,to_categorical
import pandas as pd
import torch
import scipy.stats as stats

if __name__ == '__main__':
    CF_ending = pd.read_excel('./data/patient_dis_day_CF_ending_processed.xlsx',index_col=0)
    CF_ending_before = CF_ending[CF_ending.dis_day<0]
    MR_before = pd.read_excel('./data/df_before_MR.xlsx',index_col=0)
    OCR = pd.read_table('./data/OCR.txt',sep='\t',encoding='gbk',error_bad_lines=False,header=None)
    OCR.columns = ['name','MR','score']
    OCR['patient_id'] = OCR['name'].apply(lambda x:int(x.split('术')[0]))
    OCR_before = OCR[OCR['name'].str.contains('前')]

    print(CF_ending_before.groupby('patient_id').size())
    print(OCR_before.groupby('patient_id').size())
    OCR_id = OCR_before.groupby('patient_id').size()[OCR_before.groupby('patient_id').size()>15].index.values
    CF_id = CF_ending_before.groupby('patient_id').size()[CF_ending_before.groupby('patient_id').size()>5].index.values
    for i in OCR_id:
        if i in CF_id:
            print(i)
    patient_id = 533
    print(OCR_before[(OCR_before['patient_id']==patient_id) & (OCR_before['MR']=='左房LA')])
    print(CF_ending_before.loc[CF_ending_before['patient_id']==patient_id,['dis_day','白细胞','ending']])
    # OCR['patient_id'] = OCR['']