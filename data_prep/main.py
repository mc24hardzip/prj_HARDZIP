from data_preparation import get_data, get_cluster_df, get_regression_df

def main():
    clustering_df = get_cluster_df() 
    
    print(clustering_df.shape) 
    
if __name__ == "__main__":
    main()
