from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import feature_pp as fp


# get dataframe
def get_xy(df):
    X = df.drop(['rent_adjusted'],axis=1) 
    y = df['rent_adjusted']
    return X, y

# 데이터 분할
def data_split(X, y):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df['service_type'], random_state = 42)
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    return X_train, X_test, y_train, y_test

# poly_transform
def poly_data(X_train, X_test):
    # 2차 다항식으로 변환
    poly = PolynomialFeatures(degree=2,include_bias=False)
    X_poly_2_train = poly.fit_transform(X_train)
    X_poly_2_test = poly.fit_transform(X_test)
    return X_poly_2_train, X_poly_2_test


# ML 모델 with best parameter 
def get_model():
    lr = LinearRegression()
    lr_ridge = Ridge(alpha=0.001)
    lr_lasso = Lasso(alpha=0.1)
    elastic = ElasticNet(alpha = 0.0000155, l1_ratio = 0.005, max_iter = 500)
    rf = RandomForestRegressor(n_estimators =621, min_samples_leaf =1, min_sampels_split =5)
    gb = GradientBoostingRegressor(n_estimators=954, learning_rate=0.09, subsample=0.8)
    xgb = xgboost.XGBRegressor(n_jobs=-1, n_estimators=2000, colsample_bytree=0.75,
                                learning_rate = 0.1, max_depth=6,
                                subsample=0.75, gamma=10)
    lgbm = LGBMRegressor(n_estimators=448, learning_rate=0.1,
                        max_depth=15, min_child_samples=40,
                        num_leaves=23)
    return lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm


# fit 
def model_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train,y_pred_train)


    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test,y_pred_test)


    train_rmse = np.sqrt(abs(mse_train))
    test_rmse = np.sqrt(abs(mse_test))
    
    return train_rmse, test_rmse, r2_train, r2_test


# feature_importance 뽑기
def feature_importance(model,X_train):
    feat_impt = model.feature_importances_

    graph_data=pd.DataFrame()

    graph_data['feature']=X_train.columns.values
    graph_data['importance']=feat_impt
    graph_data_top=graph_data.nlargest(10,'importance')

    g=sns.barplot(y='feature',x='importance',data=graph_data_top,orient='h')
    g.set_ylabel('Features',fontsize=12)
    g.set_xlabel('Relative Importance')
    g.set_title(type(model).__name__ )
    g.tick_params(labelsize=8)
    plt.tight_layout()
    
    return print(graph_data.sort_values(by='importance', ascending=False).iloc[:11])









# df로 test, train결과 넣기도 하면 좋을듯

#               data : }
# df = pd.DataFrame()
# print(df)





################## 불러올 것 ###########################################
df = fp.get_finaldf()
X, y = get_xy(df)
X_train, X_test, y_train, y_test = data_split(X,y)
X_poly_2_train, X_poly_2_test = poly_data(X_train, X_test)
lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm = get_model()

# 1) original data 결과
# 1-1) Linear Regression
lr_train_rmse, lr_test_rmse, lr_r2_train, lr_r2_test = model_eval(lr, X_train, X_test, y_train, y_test)

# 1-2) ridge
ridge_train_rmse, ridge_test_rmse, ridge_r2_train, ridge_r2_test = model_eval(lr_ridge, X_train, X_test, y_train, y_test)

# 1-3) lasso
lasso_train_rmse, lasso_test_rmse, lasso_r2_train, lasso_r2_test = model_eval(lr_lasso, X_train, X_test, y_train, y_test)

# 1-4) elastic
elastic_train_rmse, elastic_test_rmse, elastic_r2_train, elastic_r2_test = model_eval(elastic, X_train, X_test, y_train, y_test)

# 1-5) rf
rf_train_rmse, rf_test_rmse, rf_r2_train, rf_r2_test = model_eval(rf, X_train, X_test, y_train, y_test)

# 1-6) gb
gb_train_rmse, gb_test_rmse, gb_r2_train, gb_r2_test = model_eval(gb, X_train, X_test, y_train, y_test)

# 1-7) xgb
xgb_train_rmse, xgb_test_rmse, xgb_r2_train, xgb_r2_test = model_eval(xgb, X_train, X_test, y_train, y_test)

# 1-8) lgmb
lgb_train_rmse, lgb_test_rmse, lgb_r2_train, lgb_r2_test = model_eval(lgbm, X_train, X_test, y_train, y_test)


# 2) poly data 결과
# 2-1) Linear Regression

lr_poly_train_rmse, lr_poly_test_rmse, lr_poly_r2_train, lr_poly_r2_test = model_eval(lr, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-2) ridge
ridge_poly_train_rmse, ridge_poly_test_rmse, ridge_poly_r2_train, ridge_poly_r2_test = model_eval(lr_ridge, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-3) lasso
lasso_poly_train_rmse, lasso_poly_test_rmse, lasso_poly_r2_train, lasso_poly_r2_test = model_eval(lr_lasso, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-4) elastic
elastic_poly_train_rmse, elastic_poly_test_rmse, elastic_poly_r2_train, elastic_poly_r2_test = model_eval(elastic, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-5) rf
rf_poly_train_rmse, rf_poly_test_rmse, rf_poly_r2_train, rf_poly_r2_test = model_eval(rf, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-6) gb
gb_poly_train_rmse, gb_poly_test_rmse, gb_poly_r2_train, gb_poly_r2_test = model_eval(gb, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-7) xgb
xgb_poly_train_rmse, xgb_poly_test_rmse, xgb_poly_r2_train, xgb_poly_r2_test = model_eval(xgb, X_poly_2_train, X_poly_2_test, y_train, y_test)

# 1-8) lgmb
lgb_poly_train_rmse, lgb_poly_test_rmse, lgb_poly_r2_train, lgb_poly_r2_test = model_eval(lgbm, X_poly_2_train, X_poly_2_test, y_train, y_test)