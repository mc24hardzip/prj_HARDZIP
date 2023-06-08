import pandas as pd
import warnings

from crawler_zigbang import geo_crawl
from data_preprocess import process_address_and_format, add_geocode_and_adm_code, emd_sgg_combine, kakao_add_facility_distances, calculate_ratios_and_refine_features, preprocess_for_clustering, preprocess_for_regression

def get_data(): 
    warnings.filterwarnings('ignore')

    # zigbang_df = crawl_main() 
    raw_zigbang_df = pd.read_csv('zigbang.csv') 

    address_processed_df = process_address_and_format(raw_zigbang_df) 

    # geocode_adm_df = add_geocode_and_adm_code(address_processed_df)
    geocode_adm_df = pd.read_csv('zigbang_emd.csv') 

    emd_sgg_combined_df = emd_sgg_combine(geocode_adm_df)

    # kakao_api_df = kakao_add_facility_distances(emd_sgg_combined_df, 'API_KEY')
    kakao_api_df = pd.read_csv('kakao.csv') 

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