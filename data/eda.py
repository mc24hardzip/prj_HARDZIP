import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

def count_plot(data, X, hue=None):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=data, x=X , hue=hue)
    plt.title(f'X:{X} hue:{hue}')
    plt.show()
    return

def bar_plot(data, X, Y):
    fig, ax = plt.subplots(figsize=(15,8))
    sns.barplot(data = data, x = X, y=Y)
    s=plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title(f'X:{X} Y:{Y}')
    plt.show()
    return

def distribution_plot(data, name):
    sns.kdeplot(data[name])
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'{name} Kernel Density Estimation')
    plt.show()
    return

def correlation_plot(data):
    size=len(data.columns)
    sns.clustermap(data.corr(), 
               annot = True,     
               cmap = 'RdYlBu_r',  
               vmin = -1, vmax = 1,
               figsize=(size,size)
              )
    plt.show()
    return

def box_plot(data):
    num_col = len(data.columns)
    fig, axes = plt.subplots(num_col,1, figsize = (8, num_col*3))
    row=0
    for i in num_col:
        sns.boxplot(data=data, x=i, ax = axes[row])
        row+=1
    plt.tight_layout()
    plt.show()
    return 

def main(data):
    count_plot(data, 'sales_type', hue='service_type')
    count_plot(data, 'service_type', hue='sales_type')
    count_plot(data, 'service_type', hue='room_direction_text')
    bar_plot(data[data['sales_type']=='전세'], '시군구', 'deposit')
    bar_plot(data[data['sales_type']=='월세'], '시군구', 'deposit')
    bar_plot(data[data['sales_type']=='월세'], '시군구', 'rent')
    distribution_plot(data, 'deposit')
    distribution_plot(data, 'rent')
    correlation_plot(data[['building_total','building_dandok','building_apt','building_yeonlip','building_dasedae','building_nonresidential','building_others','household_family_cnt', 'household_alone_cnt','population_adult_cnt']])
    correlation_plot(data[['park_cnt','park_area(㎡)','greenArea_per_person','total_green_area(㎡)','dust_level','deposit']])
    box_plot(data)
    return