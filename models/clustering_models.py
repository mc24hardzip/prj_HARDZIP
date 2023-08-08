import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans,  MeanShift, DBSCAN , estimate_bandwidth
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.neighbors import NearestNeighbors

# service_type별로 데이터 나누기  
def service_type_sep(dataframe):
    df_ohe = dataframe 
    
    villa_df = df_ohe[df_ohe['service_type_빌라'] == 1]
    oneroom_df = df_ohe[df_ohe['service_type_원룸'] == 1]
    officetel_df = df_ohe[df_ohe['service_type_오피스텔'] == 1]

    villa_drop_df = villa_df.drop(['service_type_빌라','service_type_오피스텔','service_type_원룸'], axis=1)
    oneroom_drop_df = oneroom_df.drop(['service_type_빌라','service_type_오피스텔','service_type_원룸'], axis=1)
    officetel_drop_df = officetel_df.drop(['service_type_빌라','service_type_오피스텔','service_type_원룸'], axis=1)

    return villa_drop_df, oneroom_drop_df, officetel_drop_df

# data scaling 
def make_scaled_df(dataframe):

    df_ohe = dataframe

    standard_scaler = StandardScaler()
    standard_df = standard_scaler.fit_transform(df_ohe)

    minmax_scaler = MinMaxScaler() 
    minmax_df = minmax_scaler.fit_transform(df_ohe) 

    robust_scaler = RobustScaler()
    robust_df = robust_scaler.fit_transform(df_ohe)

    return standard_df, minmax_df, robust_df, df_ohe

# 차원 개수 정하기
def num_of_components(dataframe):
    standard_df, minmax_df, robust_df, df_ohe = make_scaled_df(dataframe)

    df_list = [df_ohe, standard_df, minmax_df, robust_df]
    df_name_list = ['Data Unscaled', 'Data Standard Scaled', 'Data Minmax Scaled', 'Data Robust Scaled']

    result_list = []

    for i in range(len(df_list)):
        print(df_name_list[i])
        pca = PCA(n_components=df_list[i].shape[1])
        pca_arr = pca.fit_transform(df_list[i])
        pca_df = pd.DataFrame(pca_arr, index=df_ohe.index, columns=[f'pca{num + 1}' for num in range(df_list[i].shape[1])])

        pca_df.value_counts()
        result = pd.DataFrame({'고윳값': pca.explained_variance_,
                               '기여율': pca.explained_variance_ratio_},
                              index=np.array([f'pca{num + 1}' for num in range(df_list[i].shape[1])]))

        result['누적기여율'] = result['기여율'].cumsum()
        result_list.append(result)

        print(result, end="\n\n")

# 각 주성분을 설명하는 변수 파악
def feature_explained_variance(dataframe, scaling_method, n_components, dim_reduction_method):

    standard_df, minmax_df, robust_df, df_ohe = make_scaled_df(dataframe)

    if scaling_method == 'standard':
        scaled_df = pd.DataFrame(standard_df, columns=df_ohe.columns)
    elif scaling_method == 'minmax':
        scaled_df = pd.DataFrame(minmax_df, columns=df_ohe.columns)
    elif scaling_method == 'robust':
        scaled_df = pd.DataFrame(robust_df, columns=df_ohe.columns)
    elif scaling_method == 'unscaled':
        scaled_df = df_ohe


    if dim_reduction_method == "PCA":
        dim_model = PCA(n_components=n_components)
    elif dim_reduction_method == "TruncatedSVD":
        dim_model = TruncatedSVD(n_components=n_components)

    df_transformed = dim_model.fit_transform(scaled_df)

    for i, comp in enumerate(df_transformed.components_):
        print(f'Component {i+1}:')  

        weight_feature_pairs = [(abs(weight), feature) for weight, feature in zip(comp, scaled_df.columns)]

        weight_feature_pairs.sort(key=lambda x: x[0], reverse=True)#[:5]

        for weight, feature in weight_feature_pairs:
            print(f'{feature}: {round(weight, 4)}')
        print()

def scree_plot(dataframe):

    standard_df, minmax_df, robust_df, df_ohe = make_scaled_df(dataframe)

    df_list = [df_ohe,  standard_df, minmax_df, robust_df]

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    for i, ax in enumerate(axs.flatten()):
        pca = PCA().fit(df_list[i])
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        ax.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, 'o-')
        ax.set_title('Scree Plot for DataFrame {}'.format(i+1))
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def elbow_method(dataframe):
    standard_df, minmax_df, robust_df, df_ohe = make_scaled_df(dataframe)

    df_list = [df_ohe, standard_df, minmax_df, robust_df]
    df_names = ['Data Unscaled', 'Data Standard Scaled', 'Data Minmax Scaled', 'Data Robust Scaled']
    n_clusters = range(1,20)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12)) 
    axes = axes.ravel() 

    for i, (data, name) in enumerate(zip(df_list, df_names)):
        inertias = []
        for n_cluster in n_clusters:
            model = KMeans(n_clusters=n_cluster, algorithm='auto')
            model.fit(data)
            inertias.append(model.inertia_)
        
        axes[i].plot(n_clusters, inertias, '-o')
        axes[i].set_xlabel('Number of clusters, k')
        axes[i].set_ylabel('Inertia')
        axes[i].set_title(f'Elbow Method For Optimal k - {name}')
        axes[i].set_xticks(n_clusters)

    plt.tight_layout()
    plt.show()

# eps 구하기위한 k-distance <- Nearest Neighbors
def k_distance_plot(dataframe):

    standard_df, minmax_df, robust_df, df_ohe = make_scaled_df(dataframe)

    df_list = [df_ohe, standard_df, minmax_df, robust_df]
    df_names = ['Data Unscaled', 'Data Standard Scaled', 'Data Minmax Scaled', 'Data Robust Scaled']

    fig, axes = plt.subplots(2, 2, figsize=(20, 12)) 
    axes = axes.ravel()
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(dataframe)
    
    distances, indices = nbrs.kneighbors(dataframe)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    for i, (data, name) in enumerate(zip(df_list, df_names)):
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        distances, indices = nbrs.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        axes[i].plot(distances)
        axes[i].set_xlabel('Data Points sorted by distance')
        axes[i].set_ylabel('Epsilon')
        axes[i].set_title(f'K-distance Graph - {name}')
    plt.tight_layout()
    plt.show()

# KMeans, DBSCAN, MeanShift 모델 피팅 
def clustering_model(dataframe, *args, **kwargs):
    df_ohe = dataframe
    
    method = kwargs.get('method')
    scaling_method = kwargs.get('scaling_method')
    dim_reduction_method = kwargs.get('dim_reduction_method')
    n_components = kwargs.get('n_components', None)
    n_clusters = kwargs.get('n_clusters', None)
    eps = kwargs.get('eps', None)
    min_samples = kwargs.get('min_samples', None)
    
    if scaling_method == 'standard':
        preprocessing_model = StandardScaler()
        data = preprocessing_model.fit_transform(df_ohe)
    elif scaling_method == 'minmax':
        preprocessing_model = MinMaxScaler()
        data = preprocessing_model.fit_transform(df_ohe)
    elif scaling_method == 'robust':
        preprocessing_model = RobustScaler()
        data = preprocessing_model.fit_transform(df_ohe)
    elif scaling_method == 'unscaled':
        data = df_ohe
    
    if dim_reduction_method == "PCA":
        dim_model = PCA(n_components=n_components)
    elif dim_reduction_method == "TruncatedSVD":
        dim_model = TruncatedSVD(n_components=n_components)
    
    dim_reduce_data = dim_model.fit_transform(data)
    
    if method == 'KMeans':
        clustering_model = KMeans(n_clusters=n_clusters, init='k-means++', algorithm='auto', max_iter=300, random_state=0)
    elif method == 'DBSCAN':
        if eps is None or min_samples is None:
            raise ValueError("Both 'eps' and 'min_samples' must be provided for DBSCAN.")
        clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'MeanShift':
        best_bandwidth = estimate_bandwidth(dim_reduce_data, quantile=0.25)
        clustering_model = MeanShift(bandwidth=best_bandwidth, n_jobs=10)
    else:
        raise ValueError("Invalid clustering method specified.")
    
    cluster_labels = clustering_model.fit_predict(dim_reduce_data)
    
    return dim_reduce_data, cluster_labels

# 모델 평가 
def cluster_evaluation(dataframe, *args, **kwargs):
    dim_reduce_data, cluster_labels = clustering_model(dataframe, *args, **kwargs)

    # Evaluation metrics
    silhouette_avg = silhouette_score(dim_reduce_data, cluster_labels)
    calinski_harabasz_avg = calinski_harabasz_score(dim_reduce_data, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(dim_reduce_data, cluster_labels)

    # Entropy
    entopy = entropy(cluster_labels)

    return dim_reduce_data, cluster_labels, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg, entopy

# KMeans를 위한 파라미터 조합 비교 함수
def comparison_of_scores_kmeans(clustering_df, param_combinations, params):
    for combination in param_combinations:
        kwargs = dict(zip(params.keys(), combination))
        method = kwargs['method']
        scaling_method = kwargs['scaling_method']
        dim_reduction_method = kwargs['dim_reduction_method']
        n_components = kwargs['n_components']
        n_clusters = kwargs['n_clusters']
        _, _, silhouette_avg, _, _, _ = cluster_evaluation(clustering_df, **kwargs)
        print(f"method: {method}, scaling_method: {scaling_method}, dim_reduction_method: {dim_reduction_method}, n_components: {n_components}, n_clusters: {n_clusters}, silhouette_avg: {silhouette_avg}")

# DBSCAN 파라미터 조합 비교 함수
def comparison_of_scores_dbscan(clustering_df, param_combinations, params):
    for combination in param_combinations:
        kwargs = dict(zip(params.keys(), combination))
        method = kwargs['method']
        scaling_method = kwargs['scaling_method']
        dim_reduction_method = kwargs['dim_reduction_method']
        n_components = kwargs['n_components']
        eps = kwargs['eps']
        min_samples = kwargs['min_samples']
        _, _, silhouette_avg, _, _, _ = cluster_evaluation(clustering_df, **kwargs)
        print(f"method: {method}, scaling_method: {scaling_method}, dim_reduction_method: {dim_reduction_method}, n_components: {n_components}, eps: {eps}, min_samples:{min_samples}, silhouette_avg: {silhouette_avg:.3f}")

# 3차원 그래프 시각화 
def cluster_visualization(dataframe, *args, **kwargs):

    method = kwargs.get('method')
    scaling_method = kwargs.get('scaling_method')
    dim_reduction_method = kwargs.get('dim_reduction_method')
    n_components = kwargs.get('n_components', None)
    n_clusters = kwargs.get('n_clusters', None)
    eps = kwargs.get('eps', None)
    min_samples = kwargs.get('min_samples', None)
    
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    dim_reduce_data, cluster_labels, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg, entopy = cluster_evaluation(dataframe, *args, **kwargs)

    ax.scatter(dim_reduce_data[:, 0], dim_reduce_data[:, 1], dim_reduce_data[:, 2], c=cluster_labels)
    
    if method == "KMeans":
        ax.set_title(f'{scaling_method}, {dim_reduction_method},n_com:{n_components}, n_clu:{n_clusters}' +
                    f'\nSilhouette Score: {silhouette_avg:.2f}' +
                    f'\nCalinski Harabasz Score: {calinski_harabasz_avg:.2f}' +
                    f'\nDavies Bouldin Score: {davies_bouldin_avg:.2f}' +
                    f'\nEntropy: {entopy:.2f}')
    elif method == "DBSCAN":
        ax.set_title(f'{scaling_method}, {dim_reduction_method}, n_com:{n_components}, eps:{eps}, min_sam:{min_samples}' +
                    f'\nSilhouette Score: {silhouette_avg:.2f}' +
                    f'\nCalinski Harabasz Score: {calinski_harabasz_avg:.2f}' +
                    f'\nDavies Bouldin Score: {davies_bouldin_avg:.2f}' +
                    f'\nEntropy: {entopy:.2f}')
    elif method == "MeanShift":
        ax.set_title(f'{scaling_method}, {dim_reduction_method}, n_com:{n_components}' +
                    f'\nSilhouette Score: {silhouette_avg:.2f}' +
                    f'\nCalinski Harabasz Score: {calinski_harabasz_avg:.2f}' +
                    f'\nDavies Bouldin Score: {davies_bouldin_avg:.2f}' +
                    f'\nEntropy: {entopy:.2f}')

    plt.tight_layout()
    plt.show()


# 군집 특징 파악
# 군집별 df 만들기
def cluster_separation(dataframe, *args, **kwargs):

    n_clusters = kwargs.get('n_clusters', None)
    
    dim_reduce_data, cluster_labels, silhouette_avg, calinski_harabasz_avg, davies_bouldin_avg, entopy = cluster_evaluation(dataframe, *args, **kwargs)
    
    preprocessed_df = dataframe
    
    preprocessed_df['clusters'] = cluster_labels

    df_clusters = []  # 군집별 데이터프레임을 저장할 리스트

    for i in range(n_clusters):
        df_cluster = preprocessed_df[preprocessed_df['clusters'] == i]
        df_clusters.append(df_cluster)

    return df_clusters

# 군집 중심점 계산 
def clusters_center(dataframe, *args, **kwargs):

    n_clusters = kwargs.get('n_clusters', None)
    
    df_clusters = cluster_separation(dataframe, *args, **kwargs)
    preprocessed_df = dataframe
    
    cluster_stats = []

    for i, df_cluster in enumerate(df_clusters):
        cluster_stat = round(df_cluster.mean(), 2)
        cluster_stats.append(cluster_stat)

    cluster_stats.append(round(preprocessed_df.mean(), 2))  # 전체 데이터프레임의 평균도 추가

    cluster_labels = ['cluster_{}'.format(i+1) for i in range(n_clusters)] + ['dataframe']

    comparison = pd.DataFrame(cluster_stats, index=cluster_labels)

    return comparison

def compare_columns(comparison_df, columns):
    df_subset = comparison_df.loc[:, columns]
    df_subset = df_subset.transpose() 
    df_subset.plot(kind='bar', figsize=(2*len(columns), 4))
    plt.title('Comparison of feature statistics across clusters and big dataframe')
    plt.ylabel('Value')
    plt.xlabel('Feature')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# 변수별 군집 중심점 시각화
def cluster_center_visualize(dataframe, *args, **kwargs):
    comparison = clusters_center(dataframe, *args, **kwargs)
    
    columns_to_compare = [
        ['hhd_collective_p', 'building_nonresidential_p', 'building_yeonlip_p', 'hhd_family_p', 'hhd_alone_p'],
        ['ppltn_adult_p', 'building_dasedae_p', 'building_dandok_p', 'hhd_private_p'],
        #['service_type_빌라', 'service_type_오피스텔', 'service_type_원룸'],
        ['aging', 'aged', 'post_aged'],
        ['room_direction_text_동향', 'room_direction_text_서향', 'room_direction_text_남향', 'room_direction_text_북향', 'room_direction_text_남동향', 'room_direction_text_남서향', 'room_direction_text_북동향', 'room_direction_text_북서향'],
        ['elevator', 'parking', 'building_apt_p'],
        ['gender_ratio', 'building_others_p'],
        ['safety_idx'],
        ['ppltn_foreign_domestic_ratio'],
        ['manage_cost_count'],
        ['corp_to_pop'],
        ['near_subways_count'],
        ['options_count'],
        ['_floor'],
        ['size_m2'],
        ['convenience_store_dist', 'school_dist', 'public_institution_dist', 'hospital_dist', 'supermarket_dist', 'subway_dist'],
        ['rent_adjusted'],
        ['ppltn_dnsty']
    ]

    for columns in columns_to_compare:
        compare_columns(comparison, columns)


# 변수별 군집 중심점 시각화(service_type)
def visualize_service_type(dataframe, *args, **kwargs):
    comparison = clusters_center(dataframe, *args, **kwargs)
    
    columns_to_compare = [
        ['service_type_빌라', 'service_type_오피스텔', 'service_type_원룸']   
    ]

    for columns in columns_to_compare:
        compare_columns(comparison, columns)
        
