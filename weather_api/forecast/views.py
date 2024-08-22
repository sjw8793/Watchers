from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
import pickle
from django.http import JsonResponse
from rest_framework.views import APIView
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import io  # StringIO를 사용하기 위해 추가
import json
import logging

# 로거 설정
logger = logging.getLogger(__name__)

# CatBoost 모델 로드
with open('XGboost_best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# 서울시 그리드와 고도 및 관측소 번호 데이터 로드
grid_data = pd.read_csv('서울시그리드와고도_행정동_구_관측소번호.csv')

def parse_weather_data(data):
    lines = data.strip().split('\n')
    records = []
    for line in lines:
        if line.startswith('#'):  # 주석 라인은 건너뜁니다.
            continue
        fields = line.split(',')
        record = {
            "일시": datetime.strptime(fields[0], '%Y%m%d%H%M'),
            "관측소번호": fields[1],
            "풍향(1분)": float(fields[2]),
            "풍속(1분)": float(fields[3]),
            "풍향(10분)": float(fields[4]),
            "풍속(10분)": float(fields[5]),
            "풍향(최대)": float(fields[6]),
            "풍속(최대)": float(fields[7]),
            "기온(°C)": float(fields[8]),
            "강수량(mm)": float(fields[9]),
            "습도(%)": float(fields[14]),
            "해면기압(hPa)": float(fields[16])
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def fetch_weather_data(station_id, start_time, end_time):
    tm1 = start_time.strftime('%Y%m%d%H%M')
    tm2 = end_time.strftime('%Y%m%d%H%M')

    url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"

    params = {
        "tm1": tm1,
        "tm2": tm2,
        "stn": station_id,
        "disp": "1",
        "help": "0",
        "authKey": "x2f0WT1-Qqen9Fk9foKnhA"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        response_text = response.text
        try:
            # 받은 데이터를 파싱합니다.
            weather_data = parse_weather_data(response_text)
            return weather_data
        except Exception as e:
            logger.error("Error parsing weather data: %s", e)
            return None
    else:
        logger.error("Failed to fetch weather data: %d", response.status_code)
        return None

class WeatherPredictionAPIView(APIView):
    def post(self, request):
        try:
            gu_name = request.data.get('gu_name')
            dong_name = request.data.get('dong_name')

            # 구 이름과 동 이름에 해당하는 모든 행 가져오기
            filtered_grid = grid_data[(grid_data['구'] == gu_name) & (grid_data['ADM_NM'] == dong_name)]
            
            if filtered_grid.empty:
                return JsonResponse({'error': 'Invalid gu_name or dong_name'}, status=400)

            current_time = datetime.now()
            start_time = current_time - timedelta(hours=24)

            station_id = filtered_grid['관측소번호'].values[0]
            weather_data = fetch_weather_data(station_id, start_time, current_time)

            if weather_data is None or weather_data.empty:
                return JsonResponse({'error': 'Failed to fetch weather data'}, status=500)

            # 시간 범위별 강수량 데이터 계산
            time_ranges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            flood_data_list = []

            for _, row in filtered_grid.iterrows():
                flood_data = {}
                for hours in time_ranges:
                    time_window = weather_data[(weather_data['일시'] > current_time - timedelta(hours=hours)) & (weather_data['일시'] <= current_time)]
                    
                    if not time_window.empty:
                        total_rainfall = time_window['강수량(mm)'].sum()
                        avg_rainfall_per_10min = total_rainfall / (hours * 6)
                        hourly_rainfall = time_window['강수량(mm)'].mean()

                        flood_data[f'{hours}시간 누적 강수량(mm)'] = total_rainfall
                        flood_data[f'{hours}시간 10분 평균 강수량(mm)'] = avg_rainfall_per_10min
                        flood_data[f'{hours}시간 시간당 강수량(mm)'] = hourly_rainfall
                    else:
                        flood_data[f'{hours}시간 누적 강수량(mm)'] = np.nan
                        flood_data[f'{hours}시간 10분 평균 강수량(mm)'] = np.nan
                        flood_data[f'{hours}시간 시간당 강수량(mm)'] = np.nan

                # 각 행에 대한 입력 데이터 준비
                input_data = {
                    '침수된 지역의 평균 지형 고도': float(row['elevation']),
                    **flood_data
                }

                # XGBoost 모델의 피처 순서 가져오기
                model_features = model.get_booster().feature_names
                input_df = pd.DataFrame([input_data])[model_features]

                # 모델 예측 수행
                score = model.predict_proba(input_df)[:, 1]  # 양성 클래스(침수)의 확률을 가져옴

                # 원본 데이터에 예측 점수 추가
                result_row = row.copy()
                result_row['flood-percent'] = score[0]
                flood_data_list.append(result_row)

            # 결과를 데이터프레임으로 변환
            result_df = pd.DataFrame(flood_data_list)

            # 필요한 컬럼만 선택
            result_df = result_df[['longitude', 'latitude', 'elevation','flood-percent']]

            # JSON 형식으로 변환
            result_dict = result_df.to_dict(orient='records')

            # API 응답을 로그로 기록
            logger.info("API Response: %s", json.dumps(result_dict, indent=4, ensure_ascii=False))

            # 결과를 JSON으로 반환
            return JsonResponse(result_dict, safe=False)
        
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            return JsonResponse({'error': str(e)}, status=500)
