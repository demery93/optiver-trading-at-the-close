import numpy as np
import pandas as pd
import lightgbm as lgb

cutoff = 390
df = pd.read_feather("input/train_processed.f")

ir = pd.read_csv("ir.csv")
#features = ir[ir.ir>-0.2].feature.values.tolist() + ['seconds_in_bucket','stock_id']
#-0.2 -> 5.8733
#-0.25 -> 5.87202
#-0.3 -> 5.87248


y = df['target']
dates = df.date_id

dropcols = ['target','date_id','time_id']
df.drop(dropcols, axis=1, inplace=True)
#df = df[features]
x_train = df[dates <= cutoff].reset_index(drop=True)
x_val = df[dates > cutoff].reset_index(drop=True)
y_train = y[dates <= cutoff]
y_val = y[dates > cutoff]

lgb_params = {
        "objective": "mae",
        "n_estimators": 5500,
        "num_leaves": 256,
        "learning_rate": 0.00877,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
        "max_depth": 12,
        "min_child_samples": 15,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "min_split_gain": 0.2,
        "min_child_weight": 0.001,
        "subsample": 0.9,
        "subsample_freq": 5,
        "colsample_bytree": 0.9,
        "n_jobs": 4,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    x_train,
    y_train,
    eval_set=[(x_val, y_val)],
    callbacks=[
        lgb.callback.early_stopping(stopping_rounds=100),
        lgb.callback.log_evaluation(period=100),
    ],
)

def center_target_at_zero(df, col):
    df['weight'] = df.groupby("time_id")['weight'].transform(lambda x: x / x.sum())  # Normalize weight to sum to 1
    df['weighted_prediction'] = df[col] * df['weight']
    df['weighted_sum_prediction'] = df.groupby("time_id")['weighted_prediction'].transform(
        lambda x: x.sum())  # Sum predictions in time_id
    df[col] = df[col].values - df['weighted_sum_prediction']

    return df[col]

weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]

stock_weights = {}
for i in range(200):
    stock_weights[i] = weights[i]

val_pred = lgb_model.predict(x_val)

val_df = pd.read_csv("input/train.csv", usecols=['time_id','stock_id','ask_size','bid_size','target'])
val_df = val_df[val_df.target.notnull()].reset_index(drop=True)
val_df = val_df[dates > cutoff].reset_index(drop=True)
val_df['weight'] = val_df['stock_id'].map(stock_weights)
val_df['prediction'] = val_pred
val_df['centered_prediction'] = center_target_at_zero(val_df.copy(), 'prediction')
print(f"Model: {np.mean(np.abs(val_df['target'] - val_df['prediction']))}")
print(f"Model with centered target: {np.mean(np.abs(val_df['target'] - val_df['centered_prediction']))}")
#Model: 5.871045387661461
#Model with centered target: 5.867787327767533
import plotly.express as px
feat_imp = pd.Series(lgb_model.feature_importances_, index=x_train.columns).sort_values()
print('Columns with poor contribution', feat_imp[feat_imp<10].index)
fig = px.bar(x=feat_imp, y=feat_imp.index, orientation='h')
fig.show()

feat_imp.sort_values()[-40:]

