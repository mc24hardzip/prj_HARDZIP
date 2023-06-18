# 모듈 가져오기
from pandas import json_normalize
import pandas as pd
import requests
import json

# SGIS 인증 토큰 무한루프 받아오기
def getAccessToken():
  REAL_TIME_API = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json?consumer_key={}&consumer_secret={}"
  C_KEY = "" # 서비스 ID
  C_SECRET = "" # 보안
  response = requests.get(REAL_TIME_API.format(C_KEY, C_SECRET))
  response.status_code
  data = response.json()
  token = data['result']['accessToken'] # 인증 토큰

  return token

# 인증 토큰 변수에 저장
token = getAccessToken()

# 읍면동 인구수 데이터 API 호출
def getPopulationDf():

  adm_cd_list = ["11", "23", "31"] # 행정동 코드
  low_search = "2" # 읍면동 레벨 설정
  dfs = []
  
  for adm_cd in adm_cd_list:
    SEARCH_POPULATION_API = f"https://sgisapi.kostat.go.kr/OpenAPI3/stats/searchpopulation.json?year=2020&accessToken={token}&age_type=23&adm_cd={adm_cd}&low_search={low_search}"
    response = requests.get(SEARCH_POPULATION_API)
    population_data = response.json()
    if adm_cd == "11":
      df_pptln_seoul= json_normalize(population_data['result'])
      dfs.append(df_pptln_seoul)
    elif adm_cd == "23":
      df_pptln_incheon= json_normalize(population_data['result'])
      dfs.append(df_pptln_incheon)
    else: 
      df_pptln_gyeongi= json_normalize(population_data['result'])
      dfs.append(df_pptln_gyeongi)

  df_adult_population = pd.concat(dfs, ignore_index=True)

  for adm_cd in adm_cd_list:
    TOTAL_POPULATION_API = f"https://sgisapi.kostat.go.kr/OpenAPI3/stats/population.json?year=2020&accessToken={token}&adm_cd={adm_cd}&low_search={low_search}"
    response = requests.get(TOTAL_POPULATION_API)
    population_data = response.json()
    if adm_cd == "11":
      df_pptln_seoul= json_normalize(population_data['result'])
      dfs.append(df_pptln_seoul)
    elif adm_cd == "23":
      df_pptln_incheon= json_normalize(population_data['result'])
      dfs.append(df_pptln_incheon)
    else: 
      df_pptln_gyeongi= json_normalize(population_data['result'])
      dfs.append(df_pptln_gyeongi)

  df_total_ppltn_population = pd.concat(dfs, ignore_index=True)
  
  for adm_cd in adm_cd_list:
    COMPANY_API = f"https://sgisapi.kostat.go.kr/OpenAPI3/stats/company.json?year=2020&accessToken={token}&adm_cd={adm_cd}&low_search={low_search}"
    response = requests.get(COMPANY_API)
    population_data = response.json()
    if adm_cd == "11":
      df_pptln_seoul= json_normalize(population_data['result'])
      dfs.append(df_pptln_seoul)
    elif adm_cd == "23":
      df_pptln_incheon= json_normalize(population_data['result'])
      dfs.append(df_pptln_incheon)
    else: 
      df_pptln_gyeongi= json_normalize(population_data['result'])
      dfs.append(df_pptln_gyeongi)

  df_adult_population = pd.concat(dfs, ignore_index=True)

  # 청장년 인구 수 (15~64)
  df_adult_population
  adult_population = df_adult_population.drop(labels=['avg_age'], axis=1).astype({'population': 'int64'})

  # 총 인구
  total_population = df_total_ppltn_population.loc[:, ['adm_cd', 'adm_nm', 'tot_ppltn', 'ppltn_dnsty', 'corp_cnt']]
  total_population.rename(columns={'adm_nm':'읍면동','adm_cd': '읍면동_2022', 'ppltn_dnsty':'population_dnsty', 'corp_cnt': 'population_corp_cnt'}, inplace=True)

  # 청장년 인구 비율, 사업체 종사자 비
  adult_ratio = (adult_population['population'] / total_population['tot_ppltn']) * 100

  # adult_population, worker 데이터 결과 병합
  df_ppltn = pd.concat(
      [
      total_population,
      adult_ratio,
      ], axis = 1
  )

  df_ppltn = df_ppltn.drop(['adm_nm'], axis = 1)
  return df_ppltn

# 시군구 시설 데이터 API 호출
def getFacility(theme_cddf):
  dfs = []

  # 테마코드 반복문
  for theme_cd in theme_cddf:
      REAL_TIME_API = "https://sgisapi.kostat.go.kr/OpenAPI3/startupbiz/sggtobcorpcount.json?accessToken={}&theme_cd={}".format(token,theme_cd)
      response = requests.get(REAL_TIME_API)
      data = response.json()
      df = json_normalize(data['result'])
      df['corp_vs_ppltn_rate'] = df['corp_vs_ppltn_rate'].astype('float')
      df1 = df[df['sido_nm']== '서울특별시']
      df2 = df[df['sido_nm']== '경기도']
      df3 = df[df['sido_nm']== '인천광역시']
      dfs.append(df1)
      dfs.append(df2)
      dfs.append(df3)

  # concatenate and 컬럼추출
  result_df = pd.concat(dfs, ignore_index=True)
  result_df= result_df[['sgg_cd', 'sido_nm', 'corp_cnt', 'sgg_nm', 'sido_cd', 'corp_vs_ppltn_rate']]
  return result_df


# 인구수 통계 데이터 df
df_ppltn = getPopulationDf()


# 행정기관(시청, 주민센터 등), 우체국, 경찰서, 소방서, 은행, 주차장
convcode = ['6001', '6002', '6003', '6004', '9002']
convdf = getFacility(convcode)

# 쇼핑시설 : 백화점, 중대형마트, 편의점, 슈퍼마켓
shoopingcode = [2003, 2011, 9001]
shooping = getFacility(shoopingcode)

# 잡화점 수	식료품점 수
grocerycode = [2004]
grocerydf = getFacility(grocerycode)

# 외식시설 : 한식, 중식, 일식, 분식, 서양식, 제과점, 패스트푸드, 치킨, 호프 및 간이주점, 카페, 기타외국식
restcode = [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011]
restdf = getFacility(restcode)

# 문화시설 수 극장, 영화관, 박물관, 사적지, 서, 동물원, 자연공원, 유원지, 테마파크
culcode = [2002, 9004, 9005]
culdf = getFacility(culcode)

# 체육시설 
gymcode = ['F001']
gymdf = getFacility(gymcode)

# 병의원 및 약국 판매업 대비 총 인구수
hospitalcode = [9003, 8007, 'J002', 'J003']
hospitaldf =getFacility(hospitalcode)

'''
데이터의 크기가 크지 않고 육안으로 확인하면서 수정해야하는 데이터의 경우 직접 처리한 후 불러옴

예시)
hyunjun_sgg=pd.read_csv('hyunjun_ssg.csv', encoding='euc-kr')
hyunjun_sgg.info()

'''

# csv 파일 데이터프레임으로 받아오기
eunbi_emd = pd.read_csv('eunbi_emd.csv', encoding='euc-kr')
jiyoon_emd = pd.read_csv('jiyoon_emd.csv', encoding='euc-kr')
hyunjun_sgg = pd.read_csv('hyunjun_sgg.csv', encoding='euc-kr')
jiyoon_sgg = pd.read_csv('jiyoon_sgg.csv', encoding='euc-kr')

# 읍면동 데이터 결합 함수
def getEmdMergeDf(emd_df, emd_csv_df):
  df_merge = pd.merge(emd_df, emd_csv_df, left_on=['시도','시군구'], right_on=['시도','시군구'], how='left')
  df_merge = df_merge.drop(['시도','시군구'], axis=1)
  return df_merge

# 시군구 데이터 결합 함수
def getSggMergeDf(sgg_df, sgg_csv_df):
  df_merge = pd.merge(sgg_df, sgg_csv_df, left_on=['시도','시군구'], right_on=['시도','시군구'], how='left')
  df_merge = df_merge.drop(['시도','시군구'], axis=1)
  return df_merge

# csv 파일로 받아온 데이터와 API로 받아온 데이터 결합
local_emd = getEmdMergeDf(eunbi_emd, jiyoon_emd)
local_sgg = getSggMergeDf(hyunjun_sgg, jiyoon_sgg)

local_emd.to_csv('local_emd.csv', encoding='euc-kr', index=False)
local_sgg.to_csv('local_sgg.csv', encoding='euc-kr', index=False)
