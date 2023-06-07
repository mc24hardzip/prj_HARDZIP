from data_preparation import get_data, get_cluster_df, get_regression_df

def main():
    # clustering_df = get_cluster_df() 

    regression_df = get_regression_df() 

    print(regression_df.shape)

if __name__ == "__main__":
    main()
