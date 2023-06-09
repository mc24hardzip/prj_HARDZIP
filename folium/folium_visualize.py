import folium
from folium import Popup, IFrame, Circle

def adjust_cluster(row):
    if row['service_type'] == '원룸':
        return row['cluster'] + 6
    elif row['service_type'] == '오피스텔':
        return row['cluster'] + 12
    else:
        return row['cluster']

def color_select(row):
    cluster = row['cluster']
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'indigo',
              'black', 'pink', 'brown', 'purple', 'gray', 'olive', 'cyan',
              'magenta', 'lime', 'teal', 'navy']
    
    return colors[cluster]

def popup_text(row):
    pop_up_text = f"""
    <div style="
    font-size:14px;
    line-height: 1.7;
    color: #333;
    background-color: #f9f9f9;
    padding: 10px;
    border-radius: 4px;
    font-family: Arial, sans-serif;
    background-size: cover;
    ">
    <b style="color: #ff4500;">매물id:</b> {row['id']} <br>
    <b style="color: #ff4500;">주소:</b> {row['address1']} {row['address2']} <br> 
    <b style="color: #ff4500;">월세(조정된):</b> {row['rent_adjusted']} <br> 
    <b style="color: #ff4500;">크기:</b> {row['size_m2']}m2 <br> 
    <b style="color: #ff4500;">엘레베이터:</b> {row['elevator']} <br> 
    <b style="color: #ff4500;">주차:</b> {row['parking']} <br> 
    <b style="color: #ff4500;">군집:</b> {row['cluster']+1} <br>
    
    {'<b style="color: #ff4500;">외국인 비율 자취 비율이 높고 안전지수가 높은 군집 </b>' if row['cluster'] == 0 else ''} 
    {'<b style="color: #ff4500;">건물 설비가 좋지 않고 낮은 층수지만 임대료가 저렴한 군집 </b>' if row['cluster'] == 1 else ''} 
    {'<b style="color: #ff4500;">건물 설비가 좋지만 교통이 좋지 않은 군집</b>' if row['cluster'] == 2 else ''} 
    {'<b style="color: #ff4500;">노령화지수가 높고 건물 설비가 좋지 않은 임대료가 낮은 군집</b>' if row['cluster'] == 3 else ''} 
    {'<b style="color: #ff4500;">건물 설비가 좋고 주거 환경 좋지만 평수가 좁은데 비교적 집값이 높은 군집</b>' if row['cluster'] == 4 else ''} 
    {'<b style="color: #ff4500;">남녀성비가 높고 안전지수가 낮고 교통이 좋지 않지만 임대료가 낮은 군집</b>' if row['cluster'] == 5 else ''} 
    
    {'<b style="color: #ff4500;">건물 설비가 좋지만 주거 환경은 안좋고 교통이 좋지 않은 군집</b>' if row['cluster'] == 6 else ''} 
    {'<b style="color: #ff4500;">자취비율이 높고 안전지수가 높지만 작은 평수의 매물인 군집 </b>' if row['cluster'] == 7 else ''} 
    {'<b style="color: #ff4500;">교통이 좋지 않고 안전하지 않지만 좋은 건물 설비에 비해 집값이 저렴한 군집</b>' if row['cluster'] == 8 else ''} 
    {'<b style="color: #ff4500;">저층의 작은 평수의 매물이 많지만 교통이 편리하고 집값이 저렴한 군집</b>' if row['cluster'] == 9 else ''} 
    {'<b style="color: #ff4500;">인구 밀도가 높고 작은 평수의 매물이지만 편의시설이랑 가까운 군집</b>' if row['cluster'] == 10 else ''} 
    {'<b style="color: #ff4500;">안전지수가 높고 교통이 좋지만 임대료가 비싼 군집</b>' if row['cluster'] == 11 else ''} 
    
    {'<b style="color: #ff4500;">집값이 비교적 저렴하지만 교통이 좋지 않은 군집</b>' if row['cluster'] == 12 else ''} 
    {'<b style="color: #ff4500;">모든 특징이 오피스텔 군집과 비슷해서 되게 무난한 군집</b>' if row['cluster'] == 13 else ''} 
    {'<b style="color: #ff4500;">노령화지수와 안전지수가 높고 교통이 좋지만 임대료가 비싼 군집 </b>' if row['cluster'] == 14 else ''} 
    {'<b style="color: #ff4500;">다른 특징들은 평균적으로 무난하지만 안전지수가 낮은 군집 </b>' if row['cluster'] == 15 else ''} 
    {'<b style="color: #ff4500;">큰 평수의 매물이 많지만 임대료가 비싼 군집 </b>' if row['cluster'] == 16 else ''} 
    {'<b style="color: #ff4500;">자취비율이 높고 인구가 밀집되어 교통이 좋지만 임대료가 비싼 군집</b>' if row['cluster'] == 17 else ''} 
    </div>
    """
    return pop_up_text

def create_map(dataframe, num_clusters, location=[37.55, 127.08],
               zoom_start=15):
    map = folium.Map(location=location, zoom_start=zoom_start)

    for i, row in dataframe.iterrows():
        address = dataframe['address1'][i] + dataframe['address2'][i]
        iframe = IFrame(popup_text(row))
        popup = Popup(iframe, min_width=500, max_width=500)
        Circle(location = [row['y_w84'], row['x_w84']],
               popup=popup, color=color_select(row, num_clusters),
               radius=50, tooltip=address).add_to(map)
    
    return map