import pandas as pd

if __name__ == '__main__':
    df_MR_CF_before = pd.read_excel('./data/df_MR_CF_p_value_before.xlsx',index_col=0).rename(columns={'correlation':'correlation_MR','p_value':'p_value_MR'})
    df_NLP_CF_before = pd.read_excel('./data/df_NLP_CF_p_value_before.xlsx',index_col=0).rename(columns={'correlation':'correlation_NLP','p_value':'p_value_NLP'})
    df_NLP_MR_before = pd.read_excel('./data/df_MR_NLP_p_value_before.xlsx' ,index_col=0).rename(columns={'correlation':'correlation_NLP','p_value':'p_value_NLP'})


    df_MR_NLP_CF_before = df_NLP_CF_before.merge(df_MR_CF_before,on='CF')


    # df_CF_MR_NLP_before = df_MR_CF_before.merge(df_NLP_MR_before,on='MR_item')
    #
    # df_CF_MR_NLP_before[(df_CF_MR_NLP_before.p_value_NLP < 0.002) & (df_CF_MR_NLP_before.p_value_MR < 0.001)].to_excel(
    #     './data/df_CF_MR_NLP_set_before_correlation.xlsx')



