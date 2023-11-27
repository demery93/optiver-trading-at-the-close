import pandas as pd
import lightgbm as lgb

import os
import joblib

cutoff = 390
df = pd.read_feather("input/train_processed.f")

y = df['target']
dates = df.date_id

dropcols = ['target','date_id','time_id']
df.drop(dropcols, axis=1, inplace=True)

lgb_params = {
        "objective": "mae",
        "n_estimators": 2500,
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
    df,
    y,
)

model_save_path = 'trained_models'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_filename = os.path.join(model_save_path, 'lgbm.pkl')

# save model
joblib.dump(lgb_model, model_filename)
print(f"Full model saved to {model_filename}")