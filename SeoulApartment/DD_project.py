import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

train_data_origin = pd.read_csv("station_added_seoul_apt.csv", encoding='cp949')
test_data_origin = pd.read_csv("station_added_seoul_apt_test.csv", encoding='utf8')

train_data_refine = train_data_origin.drop(["위도", "경도"], axis=1)
test_data_refine = test_data_origin.drop(["위도", "경도"], axis=1)

data_with_gu_onehot = pd.get_dummies(train_data_refine, columns=['지역구'])
data_with_gu_onehot = data_with_gu_onehot.drop(["동"], axis=1)

train_data_with_dong_onehot = pd.get_dummies(train_data_refine, columns=['지역구'])
train_data_with_dong_onehot = train_data_with_dong_onehot.drop(["동"], axis=1)

test_data_with_dong_onehot = pd.get_dummies(test_data_refine, columns=['지역구'])
test_data_with_dong_onehot = test_data_with_dong_onehot.drop(["동"], axis=1)

train_set = train_data_with_dong_onehot
test_set = test_data_with_dong_onehot

# from sklearn.model_selection import train_test_split
#
# train_set, test_set = train_test_split(train_data_with_dong_onehot, test_size=0.2, random_state=42)

print(train_set)
"""
    data - onehot split
"""
# gu_data = data_with_gu_onehot.iloc[:, 0:7]
# dong_data = data_with_dong_onehot.iloc[:, 0:7]
#
# gu_onehot = data_with_gu_onehot.iloc[:, 7:]
# dong_onehot = data_with_dong_onehot.iloc[:, 7:]


"""
    scaler
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# pipeline = Pipeline([
#     ('std_scaler', StandardScaler()),
#     ])
# train_data = pipeline.fit_transform(data_with_gu)

y_train = train_set['거래금액(만원)']
x_train = train_set.drop(['거래금액(만원)'], axis=1)

y_test = test_set['거래금액(만원)']
x_test = test_set.drop(['거래금액(만원)'], axis=1)

x_train = x_train.values.tolist()
y_train = y_train.astype(float).values.tolist()

x_test = x_test.values.tolist()
y_test = y_test.astype(float).values.tolist()

"""
    linearRegression
"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

predictions = lin_reg.predict(x_test)
print(predictions.tolist())
print(y_test)

lin_mae = mean_absolute_error(y_test, predictions)
lin_mse = mean_squared_error(y_test, predictions)
lin_rmse = np.sqrt(lin_mse)

print("lin_score : ", lin_reg.score(x_test,y_test))
print("lin_mae : ", lin_mae)
print("lin_mse : ", lin_mse)
print("lin_rmse : ", lin_rmse)
print("")

"""
        Lasso
"""

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x_train, y_train)

predictions = lasso.predict(x_test)

print(predictions.tolist())
print(y_test)

lasso_mae = mean_absolute_error(y_test, predictions)
lasso_mse = mean_squared_error(y_test, predictions)
lasso_rmse = np.sqrt(lasso_mse)

print("lasso_score : ", lasso.score(x_test,y_test))
print("lasso_mae ", lasso_mae)
print("lasso_mse ", lasso_mse)
print("lasso_rmse ", lasso_rmse)
print("")

"""
        Ridge
"""
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, normalize=False)
ridge.fit(x_train, y_train)

predictions = ridge.predict(x_test)
print(predictions.tolist())
print(y_test)

ridge_mae = mean_absolute_error(y_test, predictions)
ridge_mse = mean_squared_error(y_test, predictions)
ridge_rmse = np.sqrt(ridge_mse)

print("ridge_score : ", ridge.score(x_test,y_test))
print("ridge_mae ", ridge_mae)
print("ridge_mse ", ridge_mse)
print("ridge_rmse ", ridge_rmse)
print("")

"""
    decision tree 
"""

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

predictions = tree_reg.predict(x_test)

print(predictions.tolist())
print(y_test)

tree_mae = mean_absolute_error(y_test, predictions)
tree_mse = mean_squared_error(y_test, predictions)
tree_rmse = np.sqrt(tree_mse)

print("tree_score : ", tree_reg.score(x_test, y_test))
print("tree_mae : ", tree_mae)
print("tree_mse : ", tree_mse)
print("tree_rmse : ", tree_rmse)
print("")

"""
    random forest
"""

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=1, n_jobs=1)

forest.fit(x_train, y_train)
predictions = forest.predict(x_test)

print(predictions.tolist())
print(y_test)

forest_mae = mean_absolute_error(y_test, predictions)
forest_mse = mean_squared_error(y_test, predictions)
forest_rmse = np.sqrt(forest_mse)

print("forest_score : ", forest.score(x_test, y_test))
print("forest_mae : ", forest_mae)
print("forest_mse : ", forest_mse)
print("forest_rmse : ", forest_rmse)
print("")


"""
    print scatter matrix
"""
# attr = list(data_with_dong.corrwith(data_with_dong['price']).sort_values(ascending=False).head(4).index)
# scatter_matrix(data_with_dong[attr], figsize=(12,8))
# plt.show()
# print(data_with_dong.corrwith(data_with_dong['price']).sort_values(ascending=False).head(30))
# print(data_with_gu.corrwith(data_with_gu['price']).sort_values(ascending=False).head(30))


