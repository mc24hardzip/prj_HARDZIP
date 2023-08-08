"""

이 파이썬 파일은 웹 크롤링으로 수집된 데이터를 전처리하고 군집/회귀/텍스트 분석을 
위해 데이터프레임을 준비합니다 

기능 밑 단계: 
1. 'crawler_zigbang' 모듈로부터 크롤링 함수 'geo_crawl"을 불러옵니다 
2. 'data_preprocess' 모듈로부터 전처리 함수들을 불러옵니다 
3. 'get_data' 함수를 통해 기존 데이터를 불러오고 전처리를 하고, 필요한 기능들을 
계산하고 데이터를 정제합니다 
4. 'get_cluster_df' / 'get_regression_df' 함수를 통해 데이터를 군집/회귀 분석에 
적합한 형태로 각각 전처리합니다 

"""

import pandas as pd
import warnings

from crawler_zigbang import geo_crawl
from data_preprocess import (
    process_address_and_format,
    add_geocode_and_adm_code,
    emd_sgg_combine,
    kakao_add_facility_distances,
    calculate_ratios_and_refine_features,
    preprocess_for_clustering,
    preprocess_for_regression,
)


def get_data():
    warnings.filterwarnings("ignore")

    # zigbang_df = crawl_main()
    raw_zigbang_df = pd.read_csv("./data_prep/zigbang.csv")

    address_processed_df = process_address_and_format(raw_zigbang_df)

    # geocode_adm_df = add_geocode_and_adm_code(address_processed_df)
    geocode_adm_df = pd.read_csv("./data_prep/zigbang_emd.csv")

    # emd_sgg_combined_df = emd_sgg_combine(geocode_adm_df)

    # kakao_api_df = kakao_add_facility_distances(emd_sgg_combined_df, 'API_KEY')
    kakao_api_df = pd.read_csv("./data_prep/kakao.csv")

    final_refined_df = calculate_ratios_and_refine_features(kakao_api_df)

    return final_refined_df


def get_cluster_df():
    final_refined_df = get_data()
    clustering_df = preprocess_for_clustering(final_refined_df)
    return clustering_df


def get_regression_df():
    final_refined_df = get_data()
    regression_df = preprocess_for_regression(final_refined_df)
    return regression_df


# def get_text_df():
#     final_refined_df = get_data()
#     text_df = preprocess_for_text(final_refined_df)
#     return text_df
