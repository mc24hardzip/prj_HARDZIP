import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost
from lightgbm import LGBMRegressor

# import feature_pp as fp

df = pd.read_csv("regression_df.csv")


# get dataframe
def get_xy(df):
    X = df.drop(["rent_adjusted"], axis=1)
    y = df["rent_adjusted"]
    return X, y


# 데이터 분할
def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=df["service_type"], random_state=42
    )
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    return X_train, X_test, y_train, y_test


# poly_transform
def poly_data(X_train, X_test):
    # 2차 다항식으로 변환
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_2_train = poly.fit_transform(X_train)
    X_poly_2_test = poly.fit_transform(X_test)
    return X_poly_2_train, X_poly_2_test


# 데이터 스케일링
def scaled_data(scaling_method, X_train, X_test):
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaling method")

    # 훈련 데이터에 스케일링 적용
    X_train_scaled = scaler.fit_transform(X_train)

    # 테스트 데이터에 스케일링 적용
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ML 모델 with best parameter
def raw_model():
    lr = LinearRegression()
    lr_ridge = Ridge()
    lr_lasso = Lasso()
    elastic = ElasticNet()
    rf = RandomForestRegressor(random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    xgb = xgboost.XGBRegressor(random_state=42)
    lgbm = LGBMRegressor(random_state=42)
    return lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm


# ML 모델 with best parameter
def get_bestparam_model():
    lr = LinearRegression()
    lr_ridge = Ridge(alpha=0.001)
    lr_lasso = Lasso(alpha=0.1)
    elastic = ElasticNet(alpha=0.0000155, l1_ratio=0.005, max_iter=500)
    rf = RandomForestRegressor(
        n_estimators=621, min_samples_leaf=1, min_samples_split=5, random_state=42
    )
    gb = GradientBoostingRegressor(
        n_estimators=954, learning_rate=0.09, subsample=0.8, random_state=42
    )
    xgb = xgboost.XGBRegressor(
        n_jobs=-1,
        n_estimators=2000,
        colsample_bytree=0.75,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.75,
        gamma=10,
        random_state=42,
    )
    lgbm = LGBMRegressor(
        n_estimators=448,
        learning_rate=0.1,
        max_depth=15,
        min_child_samples=40,
        num_leaves=23,
        random_state=42,
    )
    return lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm


# eval
def model_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    train_rmse = np.sqrt(abs(mse_train))
    test_rmse = np.sqrt(abs(mse_test))

    return model, train_rmse, r2_train, test_rmse, r2_test


# 모델결과 dataframe으로 변환하는 함수
def making_df(models, model_names, X_train, X_test, y_train, y_test):
    results = []
    for model, name in zip(models, model_names):
        model, train_rmse, r2_train, test_rmse, r2_test = model_eval(
            model, X_train, X_test, y_train, y_test
        )
        results.append((name, train_rmse, r2_train, test_rmse, r2_test))
    # 결과 출력
    results_df = pd.DataFrame(
        results, columns=["Model", "Train RMSE", "Train R^2", "Test RMSE", "Test R^2"]
    )
    return results_df


# 선형 모델 결과를 dataframe으로 받아오기 (input: 데이터 종류(original, poly))
def get_linear_result_df(input, scaling_method=None, wanted_model=None):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = data_split(X, y)

    if scaling_method == None:
        if input == "original":
            pass
        elif input == "poly":
            X_train, X_test = poly_data(X_train, X_test)

    elif scaling_method != None:
        # 스케일링 적용
        scaling_method = scaling_method  # 원하는 스케일링 방법 선택
        X_train, X_test = scaled_data(scaling_method, X_train, X_test)

    # 사용할 모델 가져오기
    lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm = wanted_model()

    # 모델 평가
    models = [lr, lr_ridge, lr_lasso, elastic]
    model_names = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet",
    ]

    results_df = making_df(models, model_names, X_train, X_test, y_train, y_test)
    print(results_df)
    return results_df


#  트리 모델 결과를 dataframe으로 받아오기 (input: 데이터 종류(original, poly))
def get_treemodel_result_df(input, scaling_method=None, wanted_model=None):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = data_split(X, y)

    if input == "original":
        pass
    elif input == "poly":
        X_train, X_test = poly_data(X_train, X_test)
    elif input == "scaled":
        # 스케일링 적용
        scaling_method = scaling_method  # 원하는 스케일링 방법 선택
        X_train, X_test = scaled_data(scaling_method, X_train, X_test)

    # 사용할 모델 가져오기
    lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm = wanted_model()

    # 모델 평가
    models = [rf, gb, xgb, lgbm]
    model_names = [
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
    ]

    results_df = making_df(models, model_names, X_train, X_test, y_train, y_test)
    print(results_df)

    return results_df


# Top 10 feature_importance 뽑기
def feature_importance(model, X_train):  # fit이 끝난 모델이 필요
    plt.rcParams["font.family"] = "Malgun Gothic"  # 한글깨짐 방지

    feat_impt = model.feature_importances_

    graph_data = pd.DataFrame()

    graph_data["feature"] = X_train.columns.values
    graph_data["importance"] = feat_impt
    graph_data_top = graph_data.nlargest(10, "importance")

    g = sns.barplot(y="feature", x="importance", data=graph_data_top, orient="h")
    g.set_ylabel("Features", fontsize=12)
    g.set_xlabel("Relative Importance")
    g.set_title(type(model).__name__)
    g.tick_params(labelsize=8)
    plt.tight_layout()

    # print(graph_data.sort_values(by='importance', ascending=False).iloc[:11])
    return


# 최종 모델 (service_type 별로 트리모델 돌리기) 결과 df로 반환
def get_model_each_servcie(df, wanted_model=None):
    X = df.drop(["rent_adjusted"], axis=1)
    X = pd.get_dummies(X)
    y = df["rent_adjusted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 사용할 모델 가져오기
    lr, lr_ridge, lr_lasso, elastic, rf, gb, xgb, lgbm = wanted_model()

    # 모델 평가
    model = xgb
    model_name = "XGBoost"

    model, train_rmse, r2_train, test_rmse, r2_test = model_eval(
        model, X_train, X_test, y_train, y_test
    )
    results_data = [[model_name, train_rmse, r2_train, test_rmse, r2_test]]
    results_df = pd.DataFrame(
        data=results_data,
        columns=["Model", "Train RMSE", "Train R^2", "Test RMSE", "Test R^2"],
    )
    print(results_df)
    feature_importance(model, X_train)

    return results_df


def gradient_loss_function(learning_rate, n_estimators, subsample):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = data_split(X, y)

    # Define the parameters for the GradientBoostingRegressor
    params = {
        "loss": "absolute_error",
        "learning_rate": learning_rate,
        "n_estimators": int(n_estimators),
        "subsample": subsample,
    }

    # Train the GradientBoostingRegressor model with the specified parameters
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Return the negative MSE as the loss to be minimized
    return -mse


def random_forest_loss_function(
    n_estimators, max_depth, min_samples_split, min_samples_leaf
):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = data_split(X, y)

    # Define the parameters for the RandomForestRegressor
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
    }

    # Train the RandomForestRegressor model with the specified parameters
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Return the negative MSE as the loss to be minimized
    return -mse


def xgboost_loss_function(learning_rate, max_depth, subsample, colsample_bytree):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = data_split(X, y)

    # Define the parameters for the XGBRegressor
    params = {
        # "tree_method": "gpu_hist",
        "learning_rate": learning_rate,
        "max_depth": int(max_depth),
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
    }

    # Train the XGBRegressor model with the specified parameters
    model = xgboost.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Return the negative MSE as the loss to be minimized
    return -mse


def lightgbm_loss_function(
    learning_rate, num_leaves, max_depth, n_estimators, min_child_samples
):
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = data_split(X, y)

    # Define the parameters for the LGBMRegressor
    params = {
        # "tree_method": "gpu_hist",
        "objective": "regression",
        "metric": "mse",
        "learning_rate": learning_rate,
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "n_estimators": int(n_estimators),
        "min_child_samples": int(min_child_samples),
    }

    # Train the LGBMRegressor model with the specified parameters
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Return the negative MSE as the loss to be minimized
    return -mse


from bayes_opt import BayesianOptimization


def optimize_models(
    gradient_pbounds, random_forest_pbounds, xgboost_pbounds, lightgbm_pbounds
):
    # Create the Bayesian optimizers for each model
    gradient_optimizer = BayesianOptimization(
        f=gradient_loss_function, pbounds=gradient_pbounds, random_state=42
    )
    random_forest_optimizer = BayesianOptimization(
        f=random_forest_loss_function, pbounds=random_forest_pbounds, random_state=42
    )
    xgboost_optimizer = BayesianOptimization(
        f=xgboost_loss_function, pbounds=xgboost_pbounds, random_state=42
    )
    lightgbm_optimizer = BayesianOptimization(
        f=lightgbm_loss_function, pbounds=lightgbm_pbounds, random_state=42
    )

    # Run the optimization for each model
    gradient_optimizer.maximize(init_points=5, n_iter=12)
    random_forest_optimizer.maximize(init_points=5, n_iter=12)
    xgboost_optimizer.maximize(init_points=5, n_iter=12)
    lightgbm_optimizer.maximize(init_points=5, n_iter=12)

    # Get the best parameters and the best loss achieved for each model
    best_params = {
        "gradient": {
            "params": gradient_optimizer.max["params"],
            "loss": -gradient_optimizer.max["target"],
        },
        "random_forest": {
            "params": random_forest_optimizer.max["params"],
            "loss": -random_forest_optimizer.max["target"],
        },
        "xgboost": {
            "params": xgboost_optimizer.max["params"],
            "loss": -xgboost_optimizer.max["target"],
        },
        "lightgbm": {
            "params": lightgbm_optimizer.max["params"],
            "loss": -lightgbm_optimizer.max["target"],
        },
    }

    return best_params


# def grid_search_model(model, param_grid, X_train, X_test, y_train, y_test):
#     grid_search = GridSearchCV(
#         estimator=model, param_grid=param_grid, scoring="neg_mean_absolute_error"
#     )
#     grid_search.fit(X_train, y_train)

#     best_params = grid_search.best_params_
#     best_score = -grid_search.best_score_

#     # Evaluate the best model on the test set
#     best_model = model.set_params(**best_params)
#     best_model.fit(X_train, y_train)
#     test_score = -best_model.score(X_test, y_test)

#     return best_params, best_score, test_score


# def get_best_params():
#     X, y = get_xy(df)
#     X_train, X_test, y_train, y_test = data_split(X, y)

#     models = {
#         "Linear Regression": LinearRegression(),
#         "Lasso": Lasso(),
#         "Ridge": Ridge(),
#         "ElasticNet": ElasticNet(),
#     }

#     param_grids = {
#         "Linear Regression": {"normalize": [True, False]},
#         "Lasso": {"alpha": [0.1, 0.5, 1.0]},
#         "Ridge": {"alpha": [0.1, 0.5, 1.0]},
#         "ElasticNet": {"alpha": [0.1, 0.5, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
#     }

#     best_params = {}
#     best_scores = {}
#     test_scores = {}

#     for model_name, model in models.items():
#         param_grid = param_grids[model_name]
#         (
#             best_params[model_name],
#             best_scores[model_name],
#             test_scores[model_name],
#         ) = grid_search_model(model, param_grid, X_train, X_test, y_train, y_test)

#     return best_params, best_scores, test_scores
