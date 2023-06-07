import pandas as pd

def preprocess_for_clustering(df):
    drop_column = [
        'id', 'address1', 'address2', 'manage_cost_inc', 'images',
        'near_subways', 'options', 'description', 'title', 'add1', 'add2',
        'add3', 'x_w84', 'y_w84', 'sgg_cd', 'emd_cd_2022', 'emd_cd_2020',
        'sido_nm', 'sgg_nm', 'emd_nm','building_total', 'ppltn_total',
        'ppltn_upper_65_p', 'hhd_total', 'sgg_cd', 'sido_nm', 'sgg_nm',
        'tenure_total', 'tenure_self', 'tenure_jeonse', 'tenure_free',
        'tenure_monthly', 'app_dasedae', 'app_dandok', 'app_nonresidential',
        'app_apt', 'app_yeonlip', 'app_officetel', 'park_per_area',
        'green_per_area', 'dust_level', 'ssg_ppltn', 'convenience_per_ppltn',
        'shopping_per_ppltn', 'grocery_per_ppltn', 'restaurant_per_ppltn',
        'medical_per_ppltn','culture_per_ppltn', 'gym_per_ppltn',
        'ppltn_net_migration_rate', 'hhd_member_avg'
    ]

    df = df.drop(drop_column, axis=1)

    df = pd.get_dummies(df, columns=['service_type', 'room_direction_text',
                                     'sales_type'], drop_first=False)

    safety_range = list(range(7, 28))
    safety_df = pd.DataFrame(safety_range, columns=['safety_value'])
    percentiles = [0.10, 0.35, 0.65, 0.90, 1]
    safety_list = safety_df['safety_value'].quantile(percentiles).tolist()

    for i in range(len(df)): 
        for j in range(len(safety_list)):
            if df['safety_idx'].iloc[i] <= safety_list[j]:
                df['safety_idx'].iloc[i] = j
                break 

    return df
