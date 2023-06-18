import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use ('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 7.5

#location_columns = ['sido_nm', 'sgg_nm', 'emd_nm', 'x_w84', 'y_w84', 'sgg_cd', 'emd_cd_2022', 'emd_cd_2020']
#cate_tobe_cnt = ['manage_cost_inc', 'near_subways', 'options']

zigbang_categorical = ['service_type', 'sales_type', 'room_direction_text']
# numeric columns
room_cost = ['manage_cost', 'rent', 'deposit', 'rent_adjusted']
room_cond = ['_floor', 'size_m2', 'manage_cost_count', 'options_count', 'elevator', 'parking']
room_env = ['near_subways_count', 'supermarket_dist', 'convenience_store_dist', 'school_dist', 'subway_dist', 'cultural_venue_dist', 'public_institution_dist', 'hospital_dist']

local_building_type = ['building_total', 'building_dandok_p', 'building_apt_p', 'building_yeonlip_p', 'building_dasedae_p', 'building_nonresidential_p', 'building_others_p']
local_ppltn = ['ppltn_total', 'gender_ratio', 'ppltn_foreign_domestic_ratio', 'ppltn_upper_65_p', 'aging', 'aged', 'post_aged', 'ppltn_dnsty', 'ppltn_adult_p', 'ssg_ppltn', 'ppltn_net_migration_rate']
local_hhd = ['hhd_member_avg', 'hhd_total', 'hhd_collective_p', 'hhd_private_p', 'hhd_family_p', 'hhd_alone_p']
local_per_ppltn = ['corp_to_pop', 'convenience_per_ppltn', 'shopping_per_ppltn', 'grocery_per_ppltn', 'restaurant_per_ppltn', 'medical_per_ppltn', 'culture_per_ppltn', 'gym_per_ppltn']
local_tenure = ['tenure_total', 'tenure_self', 'tenure_jeonse', 'tenure_free', 'tenure_monthly']
local_app = ['app_dasedae', 'app_dandok', 'app_nonresidential', 'app_apt', 'app_yeonlip', 'app_officetel']
local_comf = ['park_per_area', 'green_per_area', 'dust_level', 'safety_idx']


def cate_countplot(data, categoricals):
    n=len(categoricals)
    fig, ax = plt.subplots(ncols=(n-1), nrows=n, figsize=(n*4,n*4))
    row, col = 0, 0
    for X in categoricals:
        for hue in categoricals:
            if X==hue:
                continue
            sns.countplot(data=data, x=X , hue=hue, ax=ax[row][col])
            col+=1
        row+=1
        col=0
    plt.show()
    return


def numer_distplot(data, numericals):
    n=len(numericals) #37 4 9
    nrows=int(np.ceil(n/4))
    if nrows == 1:
        fig, ax = plt.subplots(ncols=4, figsize=(15, nrows*4))
        col = 0
        for X in numericals:
            sns.kdeplot(data=data, x=X , ax=ax[col])
            col+=1
    else:
        fig, ax = plt.subplots(ncols=4, nrows=nrows, figsize=(15, nrows*4))
        row, col = 0, 0
        for X in numericals:
            sns.kdeplot(data=data, x=X , ax=ax[row][col])
            col+=1
            if col==4:
                row+=1
                col=0
    plt.tight_layout()
    plt.show()
    return

def bar_plot(data, X, Y):
    fig, ax = plt.subplots(figsize=(15,8))
    result = data.groupby([X])[Y].aggregate(np.median).reset_index().sort_values(Y)
    sns.barplot(data = data, x = X, y=Y, order=result[X])
    s=plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.show()
    return

def correlation_plot(data, ax, row, col):
    corr_matrix = data.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    h=sns.heatmap(corr_matrix,
               annot = True,
               cmap = 'RdYlBu_r',
               vmin = -1, vmax = 1,
               cbar=False,
               mask=mask,
               fmt='.2f',
               ax=ax[row][col])
    h.set_xticklabels(h.get_xticklabels(),horizontalalignment='right', rotation=45)
    h.set_yticklabels(h.get_yticklabels(),verticalalignment='top', rotation=45)
    return

def correlation_by_group(data):
    fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(12,25))
    plt.rcParams['font.size'] = 10
    row, col = 0, 0
    for group in [room_cost, room_cond, room_env, local_building_type, local_ppltn, local_hhd, local_per_ppltn, local_tenure, local_app, local_comf]:
        correlation_plot(data[group], ax, row, col)
        col +=1
        if col==2:
            col=0
            row+=1
    plt.tight_layout()
    plt.show()
    return

def box_plot(data):
    data = data._get_numeric_data()
    n = len(data.columns)
    fig, axes = plt.subplots(n,1, figsize = (8, n*1))
    row=0
    for i in data.columns:
        sns.boxplot(data=data, x=i, ax = axes[row])
        row+=1
    plt.tight_layout()
    plt.show()
    return