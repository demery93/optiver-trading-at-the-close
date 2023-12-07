import os
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from typing import Dict, List, Optional, Tuple
from utils import timer
import joblib

def plot_importance(cvbooster, figsize=(10, 10)):
    raw_importances = cvbooster.feature_importance(importance_type='gain')
    feature_name = cvbooster.boosters[0].feature_name()
    importance_df = pd.DataFrame(data=raw_importances,
                                 columns=feature_name)
    # order by average importance across folds
    sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    # plot top-n
    PLOT_TOP_N = 50
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()

class EnsembleModel:
    def __init__(self, models: List[lgb.Booster], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights

        features = list(self.models[0].feature_name())

        for m in self.models[1:]:
            assert features == list(m.feature_name())

    def predict(self, x):
        predicted = np.zeros((len(x), len(self.models)))

        for i, m in enumerate(self.models):
            w = self.weights[i] if self.weights is not None else 1
            predicted[:, i] = w * m.predict(x)

        ttl = np.sum(self.weights) if self.weights is not None else len(self.models)
        return np.sum(predicted, axis=1) / ttl

    def feature_name(self) -> List[str]:
        return self.models[0].feature_name()


df = pd.read_feather("input/train_processed.f")

y = df['target']
dates = df.date_id

dropcols = ['time_id','date_id','stock_id','target']
df.drop(dropcols, axis=1, inplace=True)
gc.collect()
print(df.shape)

lr = 0.01
params = {
        "objective": "mae",
        "num_leaves": 256,
        "learning_rate": lr,
        "device": "gpu",
        "verbosity": -1,
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

with timer('make folds'):
    folds_border = [481 - 45*4, 481 - 45*3, 481 - 45*2, 481 - 45*1]

    folds = []
    for i, border in enumerate(folds_border):
        idx_train = np.where(dates < border)[0]
        idx_valid = np.where((border <= dates) & (dates < border + 45))[0]
        folds.append((idx_train, idx_valid))

        print(f"folds{i}: train={len(idx_train)}, valid={len(idx_valid)}")


ds = lgb.Dataset(df, y)
with timer('lgb.cv'):
    ret = lgb.cv(params, ds, num_boost_round=5500, folds=folds,
                 stratified=False, return_cvbooster=True,
                 callbacks=[lgb.callback.early_stopping(stopping_rounds=int(40*0.1/lr)), lgb.callback.log_evaluation(period=100)]
                 )

    print(f"# overall MAE: {ret['l1-mean'][-1]}")


best_iteration = len(ret['l1-mean'])
for i in range(len(folds)):
    y_pred = ret['cvbooster'].boosters[i].predict(df.iloc[folds[i][1]], num_iteration=best_iteration)
    y_true = y.iloc[folds[i][1]]
    print(f"# fold{i} MAE: {np.mean(np.abs((y_true - y_pred)))}")

    if i == len(folds) - 1:
        np.save('pred_gbdt.npy', y_pred)

plot_importance(ret['cvbooster'], figsize=(10, 20))

GBDT_NUM_MODELS = 5

# Save the model to a file
model_save_path = 'trained_models'

boosters = []
with timer('retraining'):
    for i in range(GBDT_NUM_MODELS):
        params['seed'] = i
        model_filename = os.path.join(model_save_path, f'lgbm_{i + 1}.pkl')
        lgb_model = lgb.train(params, ds, num_boost_round=int(1.1 * best_iteration))

        # save model
        joblib.dump(lgb_model, model_filename)





booster = EnsembleModel(boosters)
del ret
del ds