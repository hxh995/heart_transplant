"""
Filename: R_edges.py
Author: yellower
"""
import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


if __name__ == '__main__' :
    edges = pd.read_table('./data/stats_data/pred_edges.txt',sep='\t')
    pred_label_df = pd.read_table('./data/stats_data/pred_cytoscope_bf_all_NLP.txt',sep='\t')
    for index,row in edges.iterrows():
        if len(row['to'].split('/'))==3:
            to_list = row['to'].split('/')
            if to_list[1]=='':
                node = to_list[0]
                name = to_list[2]
            else:
                node = to_list[1]
                name = to_list[2]
            color = pred_label_df.loc[(pred_label_df['name']==name)&(pred_label_df['node']==node),'cov'].values[0]
            if color >0:
                color=1
            else:
                color=-1
        else:
            color = 0
        edges.loc[index,'color'] = color
    edges.to_csv('./data/stats_data/pred_edges_color.txt',sep='\t',index=False)


