import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt

test_df = pd.read_csv('./test.csv', parse_dates=["datetime"])
train_df = pd.read_csv('./train.csv', parse_dates=["datetime"])


#데이터 가공을 위해 train데이터와 test데이터를 합침
all_df = pd.concat((train_df, test_df),axis=0).reset_index()

#인덱스로 train data와 test data를 나누기 위해 인덱스를 나눠줌
train_index = list(range(len(train_df)))
test_index = list(range(len(train_df), len(all_df)))


def rmsle(y, y_):
    log1 = np.nan_to_num(np.log(y+1))
    log2 = np.nan_to_num(np.log(y_+1))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


#필요없을 것 같은 피처는 제외
del all_df['index']
del all_df['registered']
del all_df['casual']


#season과 weather를 one-hot encoding 하여 합쳐줌
pre_df = all_df.merge(pd.get_dummies(all_df['season'], prefix='season'), left_index=True, right_index=True)
pre_df = pre_df.merge(pd.get_dummies(all_df['weather'], prefix='weather'), left_index=True, right_index=True)


#날짜 관련 새로운 피처를 생성
pre_df['year'] = pre_df['datetime'].dt.year
pre_df['month'] = pre_df['datetime'].dt.month
pre_df['day'] = pre_df['datetime'].dt.day
pre_df['hour'] = pre_df['datetime'].dt.hour
pre_df['dayofweek'] = pre_df['datetime'].dt.dayofweek

#요일에 대해서 one-hot encoding
pre_df = pre_df.merge(pd.get_dummies(pre_df['dayofweek'], prefix='dayofweek'), left_index=True, right_index=True)


category_variable_list = ['season','holiday','workingday','weather','dayofweek',
                         'year','month','hour']

#category 타입으로 바꾸어줍니다
for var_name in category_variable_list:
    pre_df[var_name] = pre_df[var_name].astype("category")

train_df = pre_df.iloc[train_index]
test_df = pre_df.iloc[test_index]

# feature들과 count사이의 상관관계를 분석합니다.

fig, ax = plt.subplots()

train_dffig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_size_inches(12, 5)

axes[0][0].bar(train_df['year'], train_df['count'])
axes[0][0].set(xlabel='count', ylabel='year')

axes[0][1].bar(train_df['weather'], train_df['count'])
axes[0][2].bar(train_df['workingday'], train_df['count'])
axes[1][0].bar(train_df['holiday'], train_df['count'])
axes[1][1].bar(train_df['dayofweek'], train_df['count'])
axes[1][2].bar(train_df['month'], train_df['count'])
axes[2][0].bar(train_df['day'], train_df['count'])
axes[2][1].bar(train_df['hour'], train_df['count'])
plt.show()


continuous_variable_list = ['temp','humidity','windspeed','atemp']
season_list = ['season_1','season_2','season_3','season_4']
weather_list = ['weather_1','weather_2','weather_3','weather_4']
dayofweek_list = ['dayofweek_0','dayofweek_1','dayofweek_2','dayofweek_3','dayofweek_4','dayofweek_5','dayofweek_6']

all_variable_list = continuous_variable_list + category_variable_list
all_variable_list.append(season_list)
all_variable_list.append(weather_list)
all_variable_list.append(dayofweek_list)

number_of_variables = len(all_variable_list)

variable_combinations = []

#총 15개의 피쳐중에서 13,14,15개 를 뽑아 만드는 조합을 계산합니다
for L in range(13, number_of_variables + 1):
    for subset in itertools.combinations(all_variable_list, L):
        temp = []
        for variable in subset:
            if isinstance(variable,list):
                for value in variable:
                    temp.append(value)

            else:
                temp.append(variable)
        variable_combinations.append(temp)

del pre_df['count']

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import datetime

kf = KFold(n_splits=10)

y = train_df['count'].values
final_output = []
models = []

ts = datetime.datetime.now()

for i, combination in enumerate(variable_combinations):
    lr = LinearRegression(n_jobs=8)
    ridge = Ridge()
    lasso = Lasso()

    lr_result = []
    ridge_result = []
    lasso_result = []

    target_df = pre_df[combination]

    ALL = target_df.values
    std = StandardScaler()
    std.fit(ALL)
    ALL_scaled = std.transform(ALL)
    X = ALL_scaled[train_index]

    for train_data_index, test_data_index in kf.split(X):
        X_train = X[train_data_index]
        X_test = X[test_data_index]
        y_train = y[train_data_index]
        y_test = y[test_data_index]

        lr.fit(X_train,y_train)
        result = rmsle(y_test,lr.predict(X_test))
        lr_result.append(result)

        ridge.fit(X_train, y_train)
        result = rmsle(y_test, ridge.predict(X_test))
        ridge_result.append(result)

        lasso.fit(X_train, y_train)
        result = rmsle(y_test, lasso.predict(X_test))
        lasso_result.append(result)

    final_output.append([i, np.mean(lr_result), np.mean(ridge_result), np.mean(lasso_result)])
    models.append([lr, ridge, lasso])

    if i % 100 == 0:
        #100개마다 얼마나 걸렸는지 계산하기
        tf = datetime.datetime.now()
        te = tf - ts
        print(i, te)
        ts = datetime.datetime.now()

labels = ['combination', 'lr', 'ridge', 'lasso']

from pandas import DataFrame

result_df = DataFrame(final_output, columns=labels)
result_df['lasso'].sort_values().head()

target_df = pre_df[variable_combinations[8]]
ALL = target_df.values
std = StandardScaler()
std.fit(ALL)
ALL_scaled = std.transform(ALL)
X_submission_test = ALL_scaled[test_index]

final_result = models[8][2].predict(X_submission_test)
final_result[final_result < 0] = 0

data = {"datetime": pre_df.iloc[test_index]["datetime"], "count": final_result}
df_submission = DataFrame(data, columns=["datetime","count"])
df_submission.set_index("datetime").to_csv("submission_lasso_data.csv")









