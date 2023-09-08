import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve,to_categorical
import pandas as pd
from Radiology_CT_CT import collate_func_CT_MR



if __name__ == '__main__':

    # patients_pydicom_disday = torch.load("./data/patients_pydicom_complication_dis_day.pt")
    #
    #
    # with torch.no_grad():
    #     for fold in range(5):
    #     # fold = 3
    #
    #         torch.cuda.empty_cache()
    #         torch.cuda.empty_cache()
    #         torch.cuda.empty_cache()
    #         torch.cuda.empty_cache()
    #         torch.cuda.empty_cache()
    #         df_path = './data/Integration_data/CT_hidden_valid_df_lung-{}.xlsx'.format(str(fold))
    #         X_valid = torch.load('./data/Integration_data/CT_X_valid-{}.pt'.format(str(fold)))
    #         ResNet_transformer_model_encoder = torch.load('./data/models/Integration_lung/ResNet_transformer_model_encoder_lung_CT-{}.pkl'.format(str(fold)))
    #         ResNet_model = torch.load('./data/models/Integration_lung/ResNet_model_lung_CT-{}.pkl'.format(str(fold)))
    #         ResNet_transformer_model_decoder = torch.load('./data/models/Integration_lung/ResNet_transformer_model_decoder_lung_CT-{}.pkl'.format(str(fold)))
    #
    #         ResNet_transformer_model_encoder.eval()
    #         ResNet_model.eval()
    #         ResNet_transformer_model_decoder.eval()
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         loader = DataLoader(dataset=X_valid, batch_size=33,collate_fn=collate_func_CT_MR)
    #         flag = False
    #         for batch_id, batch_age, batch_pydicom_MR, batch_MR_len,batch_pydicom_CT, batch_CT_len ,batch_ending in loader:
    #             batch_dis_day = get_keys_from_dict(patients_pydicom_disday, batch_id)
    #             batch_dis_day = [i for j in batch_dis_day for i in j]
    #
    #             batch_pydicom_MR_pad = torch.stack(batch_pydicom_MR)
    #             encoder_output, encoder_atten,encoder_logits = ResNet_transformer_model_encoder(batch_pydicom_MR_pad.to(device),
    #                                                                              batch_MR_len, batch_dis_day)
    #
    #             x = ResNet_model(torch.stack(batch_pydicom_CT).to(device), batch_CT_len)
    #             batch_pydicom_hidden, output_logits = ResNet_transformer_model_decoder(encoder_output, x, batch_CT_len, batch_dis_day, batch_age)
    #
    #             # for i in range(len(batch_ending)):
    #             #     if batch_ending[i] == 0:
    #             #         print(batch_id[i])
    #             #         print(F.softmax(output_logits[i], dim=-1))
    #             if flag:
    #                 output = torch.cat([output, output_logits], dim=0)
    #                 output_label = output_label + batch_ending
    #                 patient_id = patient_id + batch_id
    #                 pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
    #             else:
    #                 output = output_logits
    #                 output_label = batch_ending
    #                 patient_id = batch_id
    #                 pydicom_hidden = batch_pydicom_hidden
    #                 flag = True
    #             # print(logits)
    #             # print(label_batch_valid)nn.Linear(512,64),
    #             #             nn.ReLU(),
    #         logits = F.softmax(output, dim=-1);
    #         output_label_ = to_categorical(output_label,2)
    #         print(fold)
    #         print(Counter(output_label))
    #         for i in range(logits.shape[1]):
    #             fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
    #             print(roc_auc)
    #         # figure_path = './data/figures/figure_test_Radiology_complication.jpg'
    #         # acu_curve(fpr, tpr, roc_auc,figure_path)
    #         #
    #         pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
    #         pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
    #         pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
    #         pydicom_hidden_df['patient_id'] = patient_id
    #         pydicom_hidden_df.to_excel(df_path)
    #
    #         torch.save(output_label_[:, i],'./data/Integration_data/patients_ending_valid_CT_lung-{}.pt'.format(str(fold)))
    #         torch.save(logits[:, i].detach().cpu().numpy(), './data/Integration_data/logits_CT_lung-{}.pt'.format(str(fold)))

    with torch.no_grad():
        for fold in range(0,5):
            # print(fold)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            df_path = './data/Integration_data/CT_hidden_valid_df_ending_all-{}.xlsx'.format(str(fold))
            X_valid = torch.load('./data/Integration_data/CT_X_valid_ending-{}.pt'.format(str(fold)))
            ResNet_transformer_model = torch.load('./data/models/Integration_lung/ResNet_transformer_model_CT_ending-{}.pkl'.format(str(fold)))
            ResNet_transformer_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loader = DataLoader(dataset=X_valid, batch_size=24, collate_fn=collate_func_CT_MR)
            flag = False
            for batch_id, batch_pydicom_bf, batch_bf_len,batch_pydicom_CT, batch_CT_len ,batch_ending in loader:
                batch_pydicom_bf_pad = torch.FloatTensor(batch_pydicom_bf)
                batch_pydicom_af_pad = torch.FloatTensor(batch_pydicom_CT)
                batch_encoder_hidden, output_logits = ResNet_transformer_model(batch_pydicom_bf_pad.to(device),batch_pydicom_af_pad.to(device),batch_bf_len,batch_CT_len)

                # for i in range(len(batch_ending)):
                #     if batch_ending[i] == 1:
                #         print(batch_id[i])
                #         print(output_logits[i])
                if flag:
                    output = torch.cat([output, output_logits], dim=0)
                    output_label = output_label + batch_ending
                    patient_id = patient_id + batch_id
                    pydicom_hidden = torch.cat([pydicom_hidden, batch_encoder_hidden], dim=0)
                else:
                    output = output_logits
                    output_label = batch_ending
                    patient_id = batch_id
                    pydicom_hidden = batch_encoder_hidden
                    flag = True
                # print(logits)
                # print(label_batch_valid)nn.Linear(512,64),
                #             nn.ReLU(),
            logits = F.softmax(output, dim=-1);
            logits = output
            output_label_ = to_categorical(output_label, 2)
            print(fold)
            print(Counter(output_label))
            # for  i in range(logits.shape[1]):
            #     fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
            #     print(roc_auc)
            # figure_path = './data/figures/figure_test_Radiology_complication.jpg'
            # acu_curve(fpr, tpr, roc_auc,figure_path)
            #
            pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
            pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
            pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
            pydicom_hidden_df['patient_id'] = patient_id
            pydicom_hidden_df.to_excel(df_path)

            torch.save(output_label_[:, 1],
                       './data/Integration_data/patients_valid_CT_ending-{}.pt'.format(str(fold)))
            torch.save(logits[:, 1].detach().cpu().numpy(),
                       './data/Integration_data/logits_CT_ending-{}.pt'.format(str(fold)))









