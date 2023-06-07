import pymysql
import pandas as pd
import numpy as np

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
    df = df[~((df["service_type"] == "원룸") & (df["size_m2"] > 120))]
    df = df[~((df["service_type"] == "원룸") & (df["size_m2"] > 99))]
    df = df[~((df["service_type"] == "원룸") & (df["manage_cost"] > 50))]
    
    df = pd.get_dummies(df)
    
    df.drop(["deposit", "rent", "school_dist", "elevator", "manage_cost"], axis=1, inplace=True)
    
    return df