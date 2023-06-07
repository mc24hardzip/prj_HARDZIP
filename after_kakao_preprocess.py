def after_kakao_preprocess(df):
    df = df[df['building_total'] != 0.0]
    df.rename(columns={'hopsital_dist': 'hospital_dist'}, inplace=True)

    df['building_dandok_p'] = (df['building_dandok'] / df['building_total']).round(3)
    df['building_apt_p'] = (df['building_apt'] / df['building_total']).round(3)
    df['building_yeonlip_p'] = (df['building_yeonlip'] / df['building_total']).round(3)
    df['building_dasedae_p'] = (df['building_dasedae'] / df['building_total']).round(3)
    df['building_nonresidential_p'] = (df['building_nonresidential'] / df['building_total']).round(3)
    df['building_others_p'] = (df['building_others'] / df['building_total']).round(3)

    df['gender_ratio'] = (df['ppltn_m'] / df['ppltn_f']).round(3)
    df['ppltn_foreign_domestic_ratio'] = (df['ppltn_foreign'] / df['ppltn_domestic']).round(3)
    df['ppltn_upper_65_p'] = (df['ppltn_upper_65'] / df['ppltn_total']).round(3)

    df['aging'] = df['ppltn_upper_65_p'].apply(lambda x: 1 if 0.07 < x < 0.14 else 0)
    df['aged'] = df['ppltn_upper_65_p'].apply(lambda x: 1 if 0.14 < x < 0.20 else 0)
    df['post_aged'] = df['ppltn_upper_65_p'].apply(lambda x: 1 if 0.20 < x <= 1 else 0)

    df['corp_to_pop'] = (df['ppltn_corp_cnt'] / df['ppltn_total']).round(3)
    df['ppltn_adult_p'] = (df['ppltn_adult'] / df['ppltn_total']).round(3)

    df['hhd_collective_p'] = (df['hhd_collective'] / df['hhd_total']).round(3)
    df['hhd_private_p'] = (df['hhd_private'] / df['hhd_total']).round(3)
    df['hhd_family_p'] = (df['hhd_family'] / df['hhd_total']).round(3)
    df['hhd_alone_p'] = (df['hhd_alone'] / df['hhd_total']).round(3)

    df['manage_cost_count'] = df['manage_cost_inc'].apply(lambda x: 0 if x == '-' else len(str(x).split(",")))
    df['near_subways_count'] = df['near_subways'].apply(lambda x: 0 if x == '-' else len(str(x).split(",")))
    df['options_count'] = df['options'].apply(lambda x: 0 if x == '-' else len(str(x).split(",")))

    df['park_per_area'] = df['park_per_area'].round(3) 
    df['green_per_area'] = df['green_per_area'].round(3)
    df['dust_level'] = df['dust_level'].round(3)
    df['convenience_per_ppltn'] = df['convenience_per_ppltn'].round(3)
    df['shopping_per_ppltn'] = df['shopping_per_ppltn'].round(3)
    df['grocery_per_ppltn'] = df['grocery_per_ppltn'].round(3)
    df['restaurant_per_ppltn'] = df['restaurant_per_ppltn'].round(3)
    df['medical_per_ppltn'] = df['medical_per_ppltn'].round(3)
    df['culture_per_ppltn'] = df['culture_per_ppltn'].round(3)
    df['gym_per_ppltn'] = df['gym_per_ppltn'].round(3)
    df['ppltn_net_migration_rate'] = df['ppltn_net_migration_rate'].round(3)
    

    # 122 columns to 87 columns 
    column_include = [
        'id',
        'service_type',
        'address1',
        'address2',
        '_floor',
        'size_m2',
        'sales_type',
        'rent',
        'deposit',
        'manage_cost',
        'manage_cost_inc',
        'manage_cost_count',
        'elevator',
        'room_direction_text',
        'images',
        'parking',
        'near_subways',
        'near_subways_count',
        'options',
        'options_count',
        'description',
        'title',
        'add1',
        'add2',
        'add3',
        'supermarket_dist',
        'convenience_store_dist',
        'school_dist',
        'subway_dist',
        'cultural_venue_dist',
        'public_institution_dist',
        'hospital_dist',
        'x_w84',
        'y_w84',
        'sgg_cd',
        'emd_cd_2022',
        'emd_cd_2020',
        'sido_nm',
        'sgg_nm',
        'emd_nm',
        'building_total',
        'building_dandok_p',
        'building_apt_p',
        'building_yeonlip_p',
        'building_dasedae_p',
        'building_nonresidential_p',
        'building_others_p',
        'ppltn_total',
        'gender_ratio',
        'ppltn_foreign_domestic_ratio',
        'ppltn_upper_65_p',
        'aging',
        'aged',
        'post_aged',
        'ppltn_dnsty',
        'corp_to_pop',
        'ppltn_adult_p',
        'hhd_member_avg',
        'hhd_total',
        'hhd_collective_p',
        'hhd_private_p',
        'hhd_family_p',
        'hhd_alone_p',
        'tenure_total',
        'tenure_self',
        'tenure_jeonse',
        'tenure_free',
        'tenure_monthly',
        'app_dasedae',
        'app_dandok',
        'app_nonresidential',
        'app_apt',
        'app_yeonlip',
        'app_officetel',
        'park_per_area',
        'green_per_area',
        'dust_level',
        'ssg_ppltn',
        'convenience_per_ppltn',
        'shopping_per_ppltn',
        'grocery_per_ppltn',
        'restaurant_per_ppltn',
        'medical_per_ppltn',
        'culture_per_ppltn',
        'gym_per_ppltn',
        'ppltn_net_migration_rate',
        'safety_idx',
    ]

    return df[column_include]
