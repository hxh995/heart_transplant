import numpy as np
import pandas as pd
import torch,os,re
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from utils import to_categorical
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from torch.utils.data import Dataset
from models_building import ResNet_transformer_encoder_CT,ResNet_transformer_decoder,ResNet_transformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy.ndimage
import cv2

from torch.utils.data import DataLoader



if __name__ == '__main__':
    # df_CT = torch.load('./data/filter_CT/df_CT_all_shaped_lung.pkl')
    #
    # patients_info = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',
    #                               index_col=0).drop_duplicates(subset=['patient_id'])
    # df_CT_before = df_CT[df_CT.dis_day < 0]
    # df_CT_after = df_CT[df_CT.dis_day >= 0]
    # patients_id = []
    # patients_pydicom_bf = []
    # patients_pydicom_af = []
    # patients_disday_bf = []
    # patients_disday_af = []
    # patients_pydicom_af_bf = []
    # patients_pydicom_len_bf = []
    # patients_pydicom_len_af = []
    # patients_pydicom_len_af_bf = []
    #
    # patients_ending = []
    #
    # patient_id_before = []
    # patient_id_after = []
    #
    #
    # for name, grouped in df_CT.groupby('patient_id'):
    #     # print(name)
    #     if int(name) in patients_info['patient_id'].values:
    #         grouped_before = grouped[grouped.dis_day < 0]
    #         grouped_after = grouped[grouped.dis_day >= 0]
    #         patients_id.append(name)
    #         patients_ending.append(patients_info.loc[patients_info['patient_id'] == name, 'is_lung'].values.item())
    #         grouped_images_before = ''
    #         if len(grouped_before) != 0:
    #             patient_id_before.append(name)
    #             grouped_before = grouped_before.sort_values(by='dis_day')
    #             grouped_disday_before = []
    #             grouped_images_before = []
    #
    #             for index, row in grouped_before.iterrows():
    #                 grouped_images_before.append(np.array(row['images']))
    #                 grouped_disday_before.append(row['dis_day'])
    #
    #             patients_disday_bf.append(grouped_disday_before)
    #             patients_pydicom_bf.append(grouped_images_before)
    #             patients_pydicom_len_bf.append(len(grouped_before))
    #
    #         if len(grouped_after) != 0:
    #             patient_id_after.append(name)
    #             grouped_after = grouped_after.sort_values(by='dis_day')
    #             if len(grouped_after)>4:
    #                 grouped_after = grouped_after.iloc[:4,:]
    #
    #             grouped_disday_after = []
    #             grouped_images_after = []
    #
    #             for index, row in grouped_after.iterrows():
    #
    #                 grouped_images_after.append(np.array(row['images']))
    #                 grouped_disday_after.append(row['dis_day'])
    #             patients_disday_af.append(grouped_disday_after)
    #             patients_pydicom_af.append(grouped_images_after)
    #             patients_pydicom_len_af.append(len(grouped_after))
    #             if len(grouped_before)!=0:
    #                 patients_pydicom_af_bf.append(grouped_images_before)
    #                 patients_pydicom_len_af_bf.append(len(grouped_before))
    #             else:
    #                 print(np.zeros_like(np.array(row['images'])).shape)
    #                 patients_pydicom_af_bf.append([np.zeros_like(np.array(row['images']))])
    #                 patients_pydicom_len_af_bf.append(1)
    #
    #
    #     else:
    #         print('not')
    #         print(name)
    #
    # torch.save(patient_id_before,'./data/filter_CT/load_CT_data/patients_id_pydicom_ending_bf_all.pt')
    # torch.save(patients_pydicom_bf, "./data/filter_CT/load_CT_data/patients_pydicom_ending_bf_all.pt")
    # torch.save(patients_pydicom_len_bf, "./data/filter_CT/load_CT_data/patients_pydicom_len_ending_bf_all.pt")
    #
    # torch.save(patient_id_after,"./data/filter_CT/load_CT_data/patients_id_pydicom_ending_af_all.pt")
    # torch.save(patients_pydicom_af_bf,"./data/filter_CT/load_CT_data/patients_id_pydicom_ending_af_all_bf.pt")
    # torch.save(patients_pydicom_len_af_bf, "./data/filter_CT/load_CT_data/patients_pydicom_len_ending_af_all_bf.pt")
    # torch.save(patients_pydicom_af, "./data/filter_CT/load_CT_data/patients_pydicom_ending_af_all.pt")
    # torch.save(patients_pydicom_len_af, "./data/filter_CT/load_CT_data/patients_pydicom_len_ending_af_all.pt")

    df_CT = torch.load('./data/filter_CT/df_CT_all_shaped_lung.pkl')

    patients_info = pd.read_excel('./data/patient_dis_day_CF_complication_processed_all.xlsx',
                                  index_col=0).drop_duplicates(subset=['patient_id'])
    df_CT_before = df_CT[df_CT.dis_day < 0]
    df_CT_after = df_CT[df_CT.dis_day >= 0]
    patients_id = []
    patients_pydicom_bf = []
    patients_pydicom_af = []
    patients_disday_bf = []
    patients_disday_af = []
    patients_pydicom_af_bf = []
    patients_pydicom_len_bf = []
    patients_pydicom_len_af = []
    patients_pydicom_len_af_bf = []

    patients_ending = []

    patient_id_before = []
    patient_id_after = []

    for name, grouped in df_CT.groupby('patient_id'):
        # print(name)
        if int(name) in patients_info['patient_id'].values:
            grouped_before = grouped[grouped.dis_day < 0]
            grouped_after = grouped[grouped.dis_day >= 0]
            patients_id.append(name)
            patients_ending.append(patients_info.loc[patients_info['patient_id'] == name, 'is_lung'].values.item())
            grouped_images_before = ''

            if len(grouped_after) != 0:

                grouped_after = grouped_after.sort_values(by='dis_day')
                if len(grouped_after) > 4:
                    grouped_after = grouped_after.iloc[:4, :]
                for index, row in grouped_after.iterrows():
                    patient_id_after.append(name)
                    patients_pydicom_af.append(np.array(row['images']))
                    patients_disday_af.append(row['dis_day'])
                    patients_pydicom_len_af.append(1)
        else:
            print('not')
            print(name)

    torch.save(patient_id_after, "./data/filter_CT/load_CT_data/patients_id_pydicom_ending_af_no_lstm.pt")
    torch.save(patients_pydicom_af, "./data/filter_CT/load_CT_data/patients_pydicom_ending_af_no_lstm.pt")
    torch.save(patients_pydicom_len_af, "./data/filter_CT/load_CT_data/patients_pydicom_len_ending_af_no_lstm.pt")
