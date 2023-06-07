import requests

def getAccessToken():
    REAL_TIME_API = "https://sgisapi.kostat.go.kr/OpenAPI3/auth/authentication.json?consumer_key={}&consumer_secret={}"
    C_KEY = "4cd7bf2024d84cfbaf8f" # 서비스 ID
    C_SECRET = "abcabcabcabcabcabcab" # 보안키
    response = requests.get(REAL_TIME_API.format(C_KEY, C_SECRET))
    data = response.json()
    return data['result']['accessToken'] # 인증 토큰

def get_adm_cd(search_address):
    access_token = getAccessToken()

    try:
        REAL_TIME_API = "https://sgisapi.kostat.go.kr/OpenAPI3/addr/geocode.json?accessToken={}&address={}".format(access_token, search_address)
        response = requests.get(REAL_TIME_API)
        data = response.json()
        adm_cd = data['result']['resultdata'][0]['adm_cd']
        x = data['result']['resultdata'][0]['x']
        y = data['result']['resultdata'][0]['y']
    except:
        adm_cd = 0
        x = 0.0
        y = 0.0
    return adm_cd, x, y

def preprocess2(df):
    df['search_address'] = df['address1'] + " " + df['address2']
    df['emd_cd_2022'] = 0
    df['X'] = 0.0
    df['Y'] = 0.0

    results = df['search_address'].apply(get_adm_cd)

    df['emd_cd_2022'], df['X'], df['Y'] = zip(*results)
    return df
