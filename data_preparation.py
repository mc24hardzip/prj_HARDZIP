import pandas as pd
import warnings

from after_kakao_preprocess import after_kakao_preprocess
from crawler_zigbang import crawl_main
from preprocess1 import preprocess1
from preprocess2 import preprocess2
from preprocess_for_clustering import preprocess_for_clustering

def get_data(): 
    warnings.filterwarnings('ignore')

    # zigbang_df = crawl_main() 
    zigbang_df = pd.read_csv('/Users/sjparky/Desktop/recodeee/zigbang.csv') # 35774, 20 

    zigbang_df_p1 = preprocess1(zigbang_df) # 35774, 23 

    # zigbang_df_emd = preprocess2(zigbang_df_p1)
    zigbang_df_emd = pd.read_csv('/Users/sjparky/Desktop/recodeee/zigbang_emd.csv') # 35774, 27 

    # local_sgg local_emd df로 주면 join 
    # emd_sgg_combine -> join 코드는 sgg emd df 주면 만들게요 
    # zb_emd_sgg_df = emd_sgg_combine(zigbang_df_emd) ######################### 

    # kakao_df = kakao_api_combine(zb_emd_sgg_df)
    kakao_df = pd.read_csv('/Users/sjparky/Desktop/recodeee/kakao.csv') # 34618, 101 

    zb_final_df = after_kakao_preprocess(kakao_df) # 34247, 87 
    
    return zb_final_df 

def get_cluster_df(): 
    zb_final_df = get_data() 
    clustering_df = preprocess_for_clustering(zb_final_df) 
    return clustering_df

# def get_regression_df():
#     zb_final_df = get_data() 
#     regression_df = preprocess_for_regression(zb_final_df)
#     return regression_df 

# def get_text_df(): 
#     zb_final_df = get_data() 
#     text_df = preprocess_for_text(zb_final_df)
#     return text_df 