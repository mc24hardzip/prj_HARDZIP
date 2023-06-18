def filter_by_service(df, column_name, service_type):
    return df[df[column_name] == service_type]

def drop_columns(df, columns_to_drop):
    return df.drop(columns_to_drop, axis=1)

def join_with_previous_df(df, previous_df, service_type): 
    return df.combine_first(
        filter_by_service(previous_df, 'service_type', service_type)) 