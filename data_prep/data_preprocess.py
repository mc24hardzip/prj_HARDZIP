import re 
import requests
import pandas as pd
import concurrent.futures

def process_address_and_format(df):
    df = df.fillna(' ') 
    df[['add1', 'add2', 'add3']] = [' ', ' ', ' ']

    classify1 = ['시', '군', '구']
    classify2 = ['동', '면', '리', '읍']
    
    def process_address(row):
        address1 = row['address1']
        address2 = row['address2']
        if re.search(r'[^0-9-]', address2):
            address1 += ' ' + address2[0] 
            address2 = ' '.join(address2.split()[1:])
        
        address1_list = address1.split()
        row['add1'] = address1_list.pop(0)     
        
        addr2 = " ".join([addr for addr in address1_list if addr[-1] in classify1])
        if addr2: 
            row['add2'] = addr2.strip() 

        for address in address1_list: 
            if address[-1] in classify2 and row['add3'] == ' ':
                row['add3'] = address
                
        return row

    df = df.apply(process_address, axis=1)

    df['manage_cost_inc'] = df['manage_cost_inc'].str.replace(',', ', ', regex=True) 
    df['options'] = df['options'].str.replace(',', ', ', regex=True) 
    df['description'] = df['description'].str.split().str.join(' ') 
    df['title'] = df['title'].str.split().str.join(' ') 

    return df


def getAccessToken():
    REAL_TIME_API = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json?consumer_key={}&consumer_secret={}"
    C_KEY = "4cd7bf2024d84cfbaf8f" # 서비스 ID
    C_SECRET = "abcabcabcabcabcabcab" # 보안키
    response = requests.get(REAL_TIME_API.format(C_KEY, C_SECRET))
    data = response.json()
    return data['result']['accessToken'] # 인증 토큰

def get_adm_cd(search_address):
    access_token = getAccessToken()

    try:
        REAL_TIME_API = "https://sgisapi.kostat.go.kr/OpenAPI3/addr/geocode.json?accessToken={}&address={}".format(access_token, search_address)
        response = requests.get(REAL_TIME_API)
        data = response.json()
        adm_cd = data['result']['resultdata'][0]['adm_cd']
        x = data['result']['resultdata'][0]['x']
        y = data['result']['resultdata'][0]['y']
    except:
        adm_cd = 0
        x = 0.0
        y = 0.0
    return adm_cd, x, y

def add_geocode_and_adm_code(df):
    df['search_address'] = df['address1'] + " " + df['address2']
    df['emd_cd_2022'] = 0
    df['X'] = 0.0
    df['Y'] = 0.0

    results = df['search_address'].apply(get_adm_cd)

    df['emd_cd_2022'], df['X'], df['Y'] = zip(*results)
    return df


def load_data():
    emd = pd.read_csv('emd.csv')
    sgg = pd.read_csv('sgg.csv')
    return emd, sgg

def rename_columns(sgg):
    sgg = sgg.rename(columns={
        'sgg_cd': 'sgg_cd_2',
        'sido_nm': 'sido_nm_2',
        'sgg_nm': 'sgg_nm_2',
    })
    return sgg

def merge_dataframes(emd, sgg):
    combined = pd.merge(emd, sgg, left_on='sgg_cd', right_on='sgg_cd_2', how='left')
    combined = combined.rename(columns={'emd_cd_2022': 'emd_cd_2022_2'})
    return combined

def final_merge_and_clean(zigbang_emd, combined):
    final = pd.merge(zigbang_emd, combined, left_on='emd_cd_2022', right_on='emd_cd_2022_2', how='left')
    final = final.drop(columns=['sgg_cd_2', 'sido_nm_2', 'sgg_nm_2', 'emd_cd_2022_2'])
    return final

def emd_sgg_combine(df):
    emd, sgg = load_data()
    sgg = rename_columns(sgg)
    combined = merge_dataframes(emd, sgg)
    final = final_merge_and_clean(df, combined)
    return final


def get_distance(row, query, category_group_code, API_KEY):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    queryString = {"query": query,
                   "x": row["x_w84"],
                   "y": row["y_w84"],
                   "category_group_code": category_group_code}
    header = {'Authorization': f'KakaoAK {API_KEY}'}

    response = requests.get(url, headers=header, params=queryString)
    tokens = response.json()
    return tokens['documents'][0]['distance']


def add_distances_to_df(df, queries, API_KEY):
    for query, category_group_code, column_name in queries:
        print(f"Processing {query}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            params = [(row, query, category_group_code, API_KEY) for _, row in df[["x_w84", "y_w84"]].iterrows()]
            distances = list(executor.map(lambda p: get_distance(*p), params))
        df[column_name] = distances
    return df


def kakao_add_facility_distances(df, API_KEY):
    queries = [
        ('대형마트', 'MT1', 'supermarket_dist'),
        ('편의점', 'CS2', 'conv_store_dist'),
        ('학교', 'SC4', 'school_dist'),
        ('지하철역', 'SW8', 'subway_dist'),
        ('은행', 'BK9', 'bank_dist'),
        ('문화시설', 'CT1', 'cultural_venue_dist'),
        ('공공기관', 'PO3', 'public_institution_dist'),
        ('병원', 'HP8', 'hospital_dist')
    ]
    
    return add_distances_to_df(df, queries, API_KEY)


def calculate_ratios_and_refine_features(df):
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
    
    df['rent_adjusted'] = (df['rent'] + df['deposit'] * 0.05 / 12 + df['manage_cost']).round(3)

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
        'rent_adjusted',
    ]
    return df[column_include]


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
        'ppltn_net_migration_rate', 'hhd_member_avg', 'rent', 'deposit',
        'manage_cost'
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


def get_room_direction_score(row):
    if row["room_direction_text"] == "남향":
        return 3
    elif row["room_direction_text"] in ["남동향", "남서향"]:
        return 2
    elif row["room_direction_text"] in ["동향", "서향"]:
        return 1
    else:
        return 0

def get_floor_score(row):
    if row["_floor"] == 1:
        return 0
    elif row["_floor"] == 2:
        return 1
    else:
        return 2

def get_ev_score(row):
    if (row["_floor"] >= 3) & (row["elevator"] == 0):
        return 0
    else:
        return 1

def preprocess_for_regression(df):
    df["rent_adjusted"] = df["deposit"] * 0.05 / 12 + df["rent"] + df["manage_cost"]

    df["room_direction_score"] = df.apply(get_room_direction_score, axis=1)
    df["room_floor_score"] = df.apply(get_floor_score, axis=1)
    df["get_ev_score"] = df.apply(get_ev_score, axis=1)
    df["manage_cost_inc_num"] = df["manage_cost_inc"].str.split(",").apply(len)
    df["near_subways_num"] = df["near_subways"].str.split(",").apply(len)
    
    df.drop(["manage_cost_inc", "near_subways", "options"], axis=1, inplace=True)
    
    df["tenure_self_ratio"] = df["tenure_self"] / df["tenure_total"]
    df["tenure_jeonse_ratio"] = df["tenure_jeonse"] / df["tenure_total"]
    df["tenure_free_ratio"] = df["tenure_free"] / df["tenure_total"]
    df["tenure_monthly_ratio"] = df["tenure_monthly"] / df["tenure_total"]

    df.drop(["tenure_self", "tenure_jeonse", "tenure_free", "tenure_monthly"], axis=1, inplace=True)

    drop_columns = [
        "address1",
        "address2",
        "_floor",
        "room_direction_text",
        "images",
        "description",
        "title",
        "add1",
        "add2",
        "add3",
        "sgg_cd",
        "emd_cd_2022",
        "emd_cd_2020",
        "sido_nm",
        "sgg_nm",
        "emd_nm",
        "building_total",
        "hhd_total",
    ]
    df.drop(drop_columns, axis=1, inplace=True)

    df = df[df["manage_cost"] <= 100]
    df = df[df["rent"] <= 1000]
    df = df[df["size_m2"] <= 150]
    df = df[~((df["service_type"] == "원룸") & (df["size_m2"] > 99))]
    df = df[~((df["service_type"] == "원룸") & (df["manage_cost"] > 50))]
    
    # df = pd.get_dummies(df)
    
    df.drop(["deposit", "rent", "school_dist", "elevator", "manage_cost"], axis=1, inplace=True)
    
    return df