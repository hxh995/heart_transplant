import jieba
from gensim.models import word2vec
import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel('./data/NLP_patient_context_concat_all.xlsx',index_col=0)
    jieba.load_userdict('./data/medical_ner_dict.txt')
    patient_context_sentens = []
    stopwords = [' ','、','：','；']
    for index,row in df.iterrows():
        first_trip = row['first_trip']
        pdf = row['pdf']
        dis_day_pdf = row['dis_day_pdf']
        clinical_diagnose = row['临床诊断']
        first_trip = first_trip.strip() + clinical_diagnose
        sentence_list = []
        for word in jieba.lcut(first_trip.strip().replace("\n","").replace("，","").replace(',',"").replace("。","").replace('“',"").replace('"',"")):
            if word not in stopwords:
                sentence_list.append(word)
        if not pd.isnull(pdf):
            for word in jieba.lcut(pdf.strip().replace("\n","").replace("，","").replace(',',"").replace("。","").replace('“',"").replace('"',"")):
                if word not in stopwords:
                    sentence_list.append(word)
        if not pd.isnull(dis_day_pdf):
            for word in jieba.lcut(dis_day_pdf.strip().replace("\n","").replace("，","").replace(',',"").replace("。","").replace('“',"").replace('"',"")):
                if word not in stopwords:
                    sentence_list.append(word)
        if not pd.isnull(clinical_diagnose):
            for word in jieba.lcut(clinical_diagnose.strip().replace("\n","").replace("，","").replace(',',"").replace("。","").replace('“',"").replace('"',"")):
                if word not in stopwords:
                    sentence_list.append(word)
        patient_context_sentens.append(sentence_list)

    model = word2vec.Word2Vec(patient_context_sentens,sg=0,epochs=8,vector_size=128,window=5,negative=3,sample=0.001,hs=1,workers=16,min_count=1)
    model.save('./data/models/patient_word2vec_all.model')