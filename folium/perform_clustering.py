from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def perform_clustering(dataframe, n_components=4, n_clusters=6):
    minmax_scaled = MinMaxScaler().fit_transform(dataframe)
    
    dim = TruncatedSVD(n_components=n_components).fit_transform(minmax_scaled)
    
    model = KMeans(n_clusters=n_clusters, init='k-means++',
                   algorithm='auto', max_iter=300, random_state=0)
    model.fit(dim)
    
    dataframe['cluster'] = model.labels_
    
    return dataframe