import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from after_kakao_preprocess import after_kakao_preprocess
from crawler_zigbang import crawl_main
from preprocess1 import preprocess1
from preprocess2 import preprocess2
from preprocess_for_clustering import preprocess_for_clustering

def main():
    # zigbang_df = crawl_main() 
    zigbang_df = pd.read_csv('C:\\Users\\sjpar\\Desktop\\iCloudDrive\\Desktop\\recode\\zigbang.csv') 

    zigbang_df_p1 = preprocess1(zigbang_df) 

    # zigbang_df_emd = preprocess2(zigbang_df_p1)
    zigbang_df_emd = pd.read_csv('C:\\Users\\sjpar\\Desktop\\iCloudDrive\\Desktop\\recode\\zigbang_emd.csv') 

    # local_sgg local_emd df로 주면 join 
    # emd_sgg_combine -> join 코드는 sgg emd df 주면 만들게요 
    # zb_emd_sgg_df = emd_sgg_combine(zigbang_df_emd) ###### 

    # kakao_df = kakao_api_combine(zb_emd_sgg_df)
    kakao_df = pd.read_csv('C:\\Users\\sjpar\\Desktop\\iCloudDrive\\Desktop\\recode\\kakao.csv') 

    zb_final_df = after_kakao_preprocess(kakao_df) 

    # 군집 전처리 
    clustering_df = preprocess_for_clustering(zb_final_df) 

    # 회귀 전처리 
    # regression_df = preprocess_for_regression(zb_final_df)

if __name__ == "__main__":
    main()
