import numpy as np
import pandas as pd
from haversine import haversine

xy_data = {}

data_origin = pd.read_csv("subway.csv", encoding='cp949')

max_row = data_origin.shape[0]

for row_idx in range(max_row):
    row = data_origin.loc[row_idx].tolist()
    if row[1] == np.nan  or row[2] == np.nan:
        continue
    xy_data[row[0]] = [row[1], row[2]]

print(xy_data)

data = pd.read_csv("seoul_apt_test.csv", encoding='cp949')
data_refine = data.drop(["번지", "본번", "부번", "단지명", "계약년월", "도로명"], axis=1)

max_row = data_refine.shape[0]

add_data = []
for row_idx in range(max_row):
    row = data_refine.loc[row_idx].tolist()
    gu_name = row[0]

    x = row[2]
    y = row[3]
    cnt = 0

    for latlon in xy_data.values():
        lat = latlon[0]
        lon = latlon[1]
        result = haversine((x, y), (lat, lon), unit='km')
        if result <= 1:
          cnt+=1
    add_data.append(cnt)
data_refine['역 개수'] = add_data

print(data_refine)

data_refine.to_csv("station_added_seoul_apt_test.csv", mode='w')