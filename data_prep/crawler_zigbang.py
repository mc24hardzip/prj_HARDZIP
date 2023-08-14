"""

이 파이썬 파일은 zigbang 웹사이트로부터 월세와 전세 부동산 매물 정보를 크롤링하는데 
사용됩니다 

1. requests와 json 라이브러리를 사용하여 zigbang API에 요청을 보내고 응답을 
처리합니다
2. 필요한 정보를 수집하기 위해 여러 API 엔드포인트와 헤더를 사용합니다 
3. geocode로 선택한 여러 지역의 매물 정보를 크롤링 하고 데이터 필터링 후에 기본
전처리를 해서 pandas DataFrame에 저장합니다 

"""

import requests
import json
import uuid
import inspect
import re 
import pandas as pd 

uuid = str(uuid.uuid4())
item_list_api = "https://apis.zigbang.com/v2/items"
describe_list_api = item_list_api + '/list'
item_describe_api = "https://apis.zigbang.com/v3/items?item_ids=\
    {item_id}&detail=true"
item_view_url = "https://zigbang.com/home/oneroom/items/{item_id}"
referer = "https://zigbang.com/home/oneroom/subways/414/items"

headers = {
    'Host':'apis.zigbang.com',
    'Connection':'keep-alive',
    'Pragma':'no-cache',
    'Cache-Control':'no-cache',
    'Accept':'application/json, text/plain, */*',
    'Origin':'https://zigbang.com',
    'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N)\
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Mobile\
            Safari/537.36',
    'DNT':'1',
    'Sec-Fetch-Site':'same-site',
    'Sec-Fetch-Mode':'cors',
    'Referer':'https://zigbang.com/home/oneroom/subways/414/items',
    'Referer': 'https://www.zigbang.com/home/oneroom/map',
    'Accept-Encoding':'gzip, deflate, br',
    'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
}


class magic_fstring_function:
    def __init__(self, payload):
        self.payload = payload
        self.cached = None
    def __str__(self):
        if self.cached is None:
            vars = inspect.currentframe().f_back.f_globals.copy()
            vars.update(inspect.currentframe().f_back.f_locals)
            self.cached = self.payload.format(**vars)
        return self.cached


def export_items(items):
    fieldnames = [
        'id',
        'service_type',
        'address1',
        'address2',
        '_floor',
        'size',
        'size_m2',
        'sales_type',
        'rent',
        'deposit',
        'manage_cost',
        'manage_cost_inc',
        'elevator',
        'room_direction_text',
        'images',
        'parking',
        'near_subways',
        'options',
        'description',
        'title'
    ]
    filtered_item = [] 
    
    for item in items: 
        item = item['item'] 
        item_id = item['id']
        url = magic_fstring_function(item_view_url)
        item['url'] = str(url)
        new_item = {}
        for fieldname in fieldnames:
            new_item[fieldname] = item[fieldname]
        filtered_item.append(new_item)
    return filtered_item


def describe_room_list(items):
    item_ids = list(items.keys())
    max_idx = int(len(item_ids) / 30)
    if len(item_ids) % 30 != 0:
        max_idx += 1
    
    items = []
    headers['Accept'] = 'application/json'
    headers['Referer'] = 'https://www.zigbang.com/home/oneroom/map'
    
    for i in range(max_idx):
        start_idx = i * 30
        end_idx = min((i+1)*30, len(item_ids))
        sub_ids = item_ids[start_idx:end_idx-1]
        item_id = str(sub_ids).replace(' ', '').replace('\'','')
        api = magic_fstring_function(item_describe_api)
        resp = requests.get(
            url=str(api),
            headers=headers,
            timeout=5,
        )
        resp.encoding = 'utf-8'
        result = resp.json()
        resp.close()
        items += result['items']
    return items


def get_room_list(
    items,
    deposit_gteq: int = 0,
    deposit_lteq: int = 8000,
    domain: str = "zigbang",
    floor_in: str = "ground",
    geohash: str = "wydjr",
    rent_gteq: int = 0,
    sales_type_in: str = "월세",
    service_type_eq: str = "원룸",
    ):
    resp = requests.get(
        url=item_list_api,
        params={
            "deposit_gteq": deposit_gteq,
            "deposit_lteq": deposit_lteq,
            "domain": domain,
            "floor_in": floor_in,
            "geohash": geohash,
            "rent_gteq": rent_gteq,
            "sales_type_in": sales_type_in,
            "service_type_eq": service_type_eq,
        },
        headers=headers,
        timeout=5,
    )
    new_items = json.loads(resp.content.decode('utf-8'))
    resp.close()

    for section in new_items['sections']:
        for item in section['item_ids']:
            items[str(item)] = {}
    return items


def get_full_address(item):
    addr = item['address1']
    addr += ' '
    addr += item['address2'] or ''
    addr += ' '
    addr += item['_floor'] or '' 
    addr += ' ' 
    addr += item['title'] or '' 
    return addr


def remove_non_digits(s):
    return re.sub(r'\D', '', s)


def crawl(geo_loc, apt_type): 
    if apt_type == 'ws':
        apt_type_kor = '월세'
    elif apt_type == 'js':
        apt_type_kor = '전세'

    items = {} 
    items = get_room_list(items, geohash=geo_loc, sales_type_in=apt_type_kor)
    
    print(f'Crawling {geo_loc}({apt_type_kor})')
    
    describe_items = describe_room_list(items) 
    define_column = export_items(describe_items) 
    
    print(f'{geo_loc}의 {apt_type_kor} 매물 개수: {len(define_column)}')
    
    geo_dict = {} 
    
    for item in define_column: 
        full_addr = get_full_address(item) 
        item['address'] = full_addr 

        if full_addr not in geo_dict:
            geo_dict[full_addr] = []
            
        geo_dict[full_addr].append(item)
    
    item_list = [] 
    
    for apt in geo_dict.keys(): 
        item_list.append(geo_dict[apt][0]) 
        
    df = pd.DataFrame(item_list)

    processed_item_list = [] 
    outside_provinces = ['강원도', '충청남도', '충청북도']
    
    for item in item_list:
        if item['address1'].split()[0] in outside_provinces\
            or item['size'] > 100:
            continue 
        else:
            item['description'] = item['description'].replace('\n', ' ')
            item['description'] = re.sub(r"[^A-Za-z0-9가-힣 ]", " ",
                                         item['description'])
            item['description'] = item['description'].strip() 
            item['title'] = re.sub(r"[^A-Za-z0-9가-힣 ]", " ", item['title'])
            item['title'] = item['title'].strip() 
            item['manage_cost'] = item['manage_cost'].replace(
                "없음", "0").replace("만원", "")        
            item['elevator'] = item['elevator'].replace(
                "없음", "0").replace("있음", "1")
            item['parking'] = item['parking'].replace(
                "불가능", "0").replace("가능", "1")
            item['_floor'] = remove_non_digits(item['_floor'])
            processed_item_list.append(item)

    df = pd.DataFrame(processed_item_list)  
    return df
            
def geo_crawl():
    geo_list = ['wydr', 'wydn', 'wydq', 'wydw', 'wydy', 'wydj', 'wydm', 'wydt', 
            'wydh', 'wydk', 'wyds', 'wydu', 'wyd5', 'wyd7', 'wyd3', 'wydg', 
            'wyd4', 'wyd6', 'wydd', 'wy9v']
    apt_type_list = ['ws', 'js']

    result_df = pd.DataFrame()

    for geo in geo_list: 
        for apt_type in apt_type_list: 
            result = crawl(geo, apt_type)
            result_df = result_df.append(result, ignore_index=True)
    return result_df