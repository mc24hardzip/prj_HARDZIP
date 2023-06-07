# DB 호출 -> 공통 모듈
def _get_db_connection_():
    import pymysql

    _host_ = "",
    _user_ = "",
    _passwd_ = "",
    _database_name_ = ""

    conn = pymysql.connect(
        host = _host_,
        user = _user_,
        password = _passwd_,
        database= _database_name_,
        charset= "utf8"
    )

    return conn

def get_dataframe(table):
    import pandas as pd

    conn = _get_db_connection_()
    dataframe = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()

    return dataframe


# 전체 데이터 전처리 -> 공통 모듈
# 전체 데이터 EDA

# 변수간의 상관계수 히트맵 확인
def visualizer(feature):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 5))
    sns.heatmap(feature.corr())
    return plt.show()

# ------------ 군집화를 위한 데이터 전처리 ------------

# 범주형 데이터 one-hot encoding 적용
def ohe_encoding(dataframe, columns_list):
    import pandas as pd
    dataframe = pd.get_dummies(dataframe, columns = columns_list)

    return dataframe

# 안전점수 -> 등급으로 변환
def get_safety_grade(dataframe, score_col):
    import pandas as pd

    # 최고점, 최저점 구하기
    safety_score_min = dataframe[score_col].min()
    safety_score_max = dataframe[score_col].max()

    # float -> int 형변환
    safety_score_min, safety_score_max = int(safety_score_min, safety_score_max)
    
    data = list(range(safety_score_min, safety_score_max))

    safety_df = pd.DataFrame(data, columns=['value']) # 임시 데이터프레임 생성
    safety_grade_list = [] # 안전등급을 담을 리스트 변수

    percentiles = [0.10, 0.35, 0.65, 0.90, 1] # 등급 나누는 기준

    for percent in percentiles:
        safety_grade_list.append(safety_df['value'].quantile(percent))
        
    for i in dataframe.index:
        for j in range(len(safety_grade_list)):
            if dataframe[score_col][i] <= safety_grade_list[j]:
                dataframe[score_col][i] = j
                break

    return dataframe

# 결측값, 이상치 제거
def get_drop_columns():
    pass

# 군집화 EDA 1






# ------------ 특성 선택, 추출 ------------

# PCA로 고윳값, 누적 기여율 확인
def get_PCA(feature):
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA

    feature_cnt = len(feature.columns)
    pca = PCA(n_components = feature_cnt)
    pca_feature = pca.fit_transform(feature)
    df_pca = pd.DataFrame(pca_feature, index = feature.index, columns=[f'pca{num + 1}' for num in range(feature_cnt)])

    df_pca.value_counts()

    result = pd.DataFrame({'고윳값': pca.explained_variance_,
                      '기여율': pca.explained_variance_ratio_},
                      index = np.array([f'pca{num + 1}' for num in range(feature_cnt)]))

    result['누적기여율'] = result['기여율'].cumsum()
    return print(result, end="\n\n")


# SVD
def get_SVD():
    from sklearn.decomposition import TruncatedSVD
    pass

# 불필요한 특성 제거
def delete_features():
    pass

# 군집화를 위한 데이터에 Scaler 적용
# robust, minmax, standard
def get_scaled_data(dataframe):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()

    scaled_list = [standard_scaler, minmax_scaler, robust_scaler]
    scaled_dataframes = []
    
    for idx, scaled_data in enumerate(scaled_list):
        globals()[scaled_data + f"{idx}"] = scaled_data.fit_transform(dataframe)
        scaled_dataframes.append(globals()[scaled_data + f"{idx}"])
    
    return scaled_dataframes

# 군집화 EDA 2





# ------------ 모델 학습 ------------

# DBSCAN

# eps, min_samples 최적값 => 수정 필요
def get_dbscan_parameters():
    eps = 0
    min_samples = 0

    return eps, min_samples

# 모델 적용 => 수정 필요
def get_dbscan():
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN().fit()

    return dbscan


# MeanShift

# bandwidth 최적값 구하기 => 수정필요.....
def get_best_bandwidth():
    from sklearn.cluster import estimate_bandwidth

    # 전체 데이터 중 25%만 사용해서 best_bandwidth 값을 구함
    bw = estimate_bandwidth(quantile=0.25)
    return bw

# 모델 적용 => 수정필요....................................
def get_meanshift():
    from sklearn.cluster import MeanShift


    meanshift = MeanShift()
    meanshift.fit_predict()

    return meanshift

# 군집화 EDA 3

# ------------ 모델 평가 ------------

# 실루엣 계수
def get_silhouette_score(feature, labels):
    from sklearn.metrics import silhouette_score

    silhouette_score = round(silhouette_score(feature, labels), 3)

    return silhouette_score

# elbow
def get_elbow():
    pass

# 군집화 EDA 4
