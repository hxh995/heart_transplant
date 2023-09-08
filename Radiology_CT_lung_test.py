import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import caculate_auc,acu_curve,to_categorical
import pandas as pd
from Radiology_CT_CT import collate_func_CT_MR



if __name__ == '__main__':
    with torch.no_grad():
        py_type = 'multi'
        sick_name = 'is_lung'
        for fold in range(0,5):
            # print(fold)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            df_path = './data/Integration_data/CT_hidden_train_df_{}_{}_{}.xlsx'.format(sick_name,py_type,str(fold))
            X_valid = torch.load('./data/Integration_data/CT_X_valid_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
            ResNet_transformer_model = torch.load('./data/models/Integration_lung/ResNet_transformer_model_CT_{}_{}-{}.pkl'.format(sick_name,py_type,str(fold)))
            ResNet_transformer_model_lstm = torch.load('./data/models/Integration_lung/ResNet_transformer_model_CT_lstm_{}_{}-{}.pkl'.format(sick_name,py_type,str(fold)))
            ResNet_transformer_model.eval()
            ResNet_transformer_model_lstm.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loader = DataLoader(dataset=X_valid, batch_size=48)
            flag = False
            for idx, (batch_patient_id, batch_pydicom_bf, batch_bf_len, batch_pydicom_CT, batch_CT_len,
                      batch_ending) in enumerate(loader):
                batch_pydicom_af_pad = torch.from_numpy(np.array(batch_pydicom_CT)).float()
                dec_outputs = ResNet_transformer_model(batch_pydicom_af_pad.to(device))
                batch_pydicom_hidden, output_logits = ResNet_transformer_model_lstm(dec_outputs, batch_CT_len)
                # output_logits = F.softmax(output_logits, dim=-1);
                # for i in range(len(batch_ending)):
                #     if batch_ending[i] == 1:
                #         print(batch_patient_id[i])
                #         print(F.softmax(output_logits[i], dim=-1))
                if flag:
                    output = torch.cat([output, output_logits], dim=0)
                    output_label = torch.cat([output_label, batch_ending], dim=0)
                    patient_id = torch.cat([patient_id, batch_patient_id], dim=0)
                    pydicom_hidden = torch.cat([pydicom_hidden, batch_pydicom_hidden], dim=0)
                else:
                    output = output_logits
                    output_label = batch_ending
                    patient_id = batch_patient_id
                    pydicom_hidden = batch_pydicom_hidden
                    flag = True
            logits = F.softmax(output, dim=-1);
            output_label_ = to_categorical(output_label.numpy().tolist(), 2)
            print(fold)
            print(Counter(output_label.numpy().tolist()))
            for  i in range(logits.shape[1]):
                fpr, tpr, roc_auc = caculate_auc(output_label_[:, i], logits[:, i].detach().cpu().numpy());
                print(roc_auc)
            # figure_path = './data/figures/figure_test_Radiology_complication.jpg'
            # acu_curve(fpr, tpr, roc_auc,figure_path)
            #
            pydicom_hidden_array = pydicom_hidden.detach().cpu().numpy()
            pydicom_hidden_df = pd.DataFrame(pydicom_hidden_array)
            pydicom_hidden_df.columns = ['pydicom_hidden_{}'.format(str(i)) for i in range(pydicom_hidden_df.shape[1])]
            pydicom_hidden_df['patient_id'] = patient_id
            pydicom_hidden_df.to_excel(df_path)

            torch.save(output_label_[:, 1],
                       './data/Integration_data/patients_valid_CT_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))
            torch.save(logits[:, 1].detach().cpu().numpy(),
                       './data/Integration_data/logits_CT_{}_{}-{}.pt'.format(sick_name,py_type,str(fold)))









