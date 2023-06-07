import pymysql
import pandas as pd

def get_room_direction_score(row):
    if row['room_direction_text'] == '남향':
        return 3
    elif row['room_direction_text'] in ['남동향', '남서향']:
        return 2
    elif row['room_direction_text'] in ['동향', '서향']:
        return 1
    else:
        return 0

def get_floor_score(row):
    if row['_floor'] == 1:
        return 0
    elif row['_floor']==2:
        return 1
    else:
        return 2

def get_ev_score(row):
    if (row['_floor'] >= 3) & (row['elevator'] == 0) :
        return 0
    else:
        return 1

def get_finaldf():
    df = pd.read_csv('./final_df.csv',encoding='cp949')
#   db= pymysql.connect(
#       host="ec2-15-152-249-56.ap-northeast-3.compute.amazonaws.com", user="mt-1", password="1111", db="zigbang_project", charset="utf8")
#   # cursor = conn.cursor()
#   df = pd.read_sql("SELECT * FROM zb_final", db, index_col='id')
#   # result = cursor.fetchall()
#   # for record in result:
#   #     print(record)
#   db.close()

  df['rent_adjusted'] = df['deposit']*0.05/12 + df['rent'] + df['manage_cost']


  # 창 반향 전환
  df['room_direction_score'] = df.apply(get_room_direction_score, axis=1)

  # 옵션 갯수, 지하철 수 숫자로 변환
  df['manage_cost_inc_num'] = df['manage_cost_inc'].str.split(',').apply(len)
  df['near_subways_num'] = df['near_subways'].str.split(',').apply(len)

  # 매물 변수 drop
  df.drop('manage_cost_inc', axis=1, inplace=True)
  df.drop('near_subways', axis=1, inplace=True)
  df.drop('options', axis=1, inplace=True)
  # 읍면동 주거유형 갯수에서 비율로
  df['tenure_self_ratio'] = df['tenure_self'] / df['tenure_total']
  df['tenure_jeonse_ratio'] = df['tenure_jeonse'] / df['tenure_total']
  df['tenure_free_ratio'] = df['tenure_free'] / df['tenure_total']
  df['tenure_monthly_ratio'] = df['tenure_monthly'] / df['tenure_total']

  # 주거유형 갯수 drop
  df.drop(['tenure_self', 'tenure_jeonse', 'tenure_free', 'tenure_monthly'], axis=1, inplace=True)

  # 층 점수화
  df['room_floor_score'] = df.apply(get_floor_score, axis=1)

  # 엘리베이터 점수화
  df['get_ev_score'] = df.apply(get_ev_score, axis=1)

  # 문자열, 코드 등 회귀변수 안쓰는 변수 drop
  drop_columns = ['address1', 'address2', '_floor' , 'room_direction_text', 'images',
          'description', 'title', 'add1', 'add2', 'add3', 'sgg_cd',
          'emd_cd_2022',
          'emd_cd_2020',
          'sido_nm',
          'sgg_nm',
          'emd_nm', 
          'building_total',
          'hhd_total'
          ]
  # 안쓰는 변수 drop 된 df 정의
  df =  df.drop(drop_columns, axis=1, inplace=False)
  # 관리비 100 이상 말이 안됨 -> 원세랑 비교
  df[df['manage_cost']>100]
  df=df.drop(df[df['manage_cost']>100].index)

  df[df['rent']>1000]
  df=df.drop(df[df['rent']>1000].index)

  df[df['size_m2']>150]
  df=df.drop(df[df['size_m2']>150].index)
  df=df.drop(df[(df['service_type']=='원룸')&(df['size_m2']>120)].index)

  # df[(df['service_type']=='원룸')&(df['size']>40)]
  df=df.drop(df[(df['service_type']=='원룸')&(df['size_m2']>99)].index)

  df=df.drop(df[(df['service_type']=='원룸')&(df['manage_cost']>50)].index)
  df = pd.get_dummies(df)

  df.drop(['deposit', 'rent','school_dist', 'elevator','manage_cost'], axis=1, inplace=True)
  return df

