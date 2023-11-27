import numpy as np
import pandas as pd

import gc
import warnings

from numba import njit, prange
from itertools import combinations
from warnings import simplefilter
from utils import reduce_mem_usage, timer
from tqdm import tqdm


warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df = pd.read_csv("input/train.csv")
df = df[df.target.notnull()].reset_index(drop=True)
df = reduce_mem_usage(df)
date_id = df['date_id']

# Copied from public kernels
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]

        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val

            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


# Copied from public kernels
def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features

global_stock_id_feats = {
    "median_size": df.groupby("stock_id")["bid_size"].median() + df.groupby("stock_id")[
        "ask_size"].median(),
    "median_bid_size": df.groupby("stock_id")['bid_size'].median(),
    "median_ask_size": df.groupby("stock_id")['ask_size'].median(),
    "median_imbalance_size": df.groupby("stock_id")['imbalance_size'].median(),
    "median_matched_size": df.groupby("stock_id")['matched_size'].median(),
    "std_size": df.groupby("stock_id")["bid_size"].std() + df.groupby("stock_id")["ask_size"].std(),
    "ptp_size": df.groupby("stock_id")["bid_size"].max() - df.groupby("stock_id")["bid_size"].min(),
    "median_price": df.groupby("stock_id")["bid_price"].median() + df.groupby("stock_id")[
        "ask_price"].median(),
    "std_price": df.groupby("stock_id")["bid_price"].std() + df.groupby("stock_id")["ask_price"].std(),
    "ptp_price": df.groupby("stock_id")["bid_price"].max() - df.groupby("stock_id")["ask_price"].min(),
}


def imbalance_features(df, verbose=True):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap", "mid_price"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size", "mid_size"]
    first_features = ['matched_size', 'imbalance_size', 'imbalance_buy_sell_flag', 'ask_size', 'bid_size']
    rank_features = ['imbalance_buy_sell_flag', 'wap', 'imbalance_buy_sell_flag_cumsum', 'wap_mid_price_imb',
                     'volume_global_ratio', 'imbalance_global_ratio', 'matched_global_ratio', 'bid_size_global_ratio',
                     'ask_size_global_ratio', 'market_imbalance_buy_sell_flag']

    with timer("Created ask/bid features", verbose=verbose):
        df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
        df["mid_size"] = df.eval("(ask_size + bid_size)/2")
        df["price_spread"] = df.eval("ask_price - bid_price")
        df["spread_percentage"] = df.eval("(ask_price - bid_price)/mid_price")
        df["volume"] = df.eval("ask_size + bid_size")
        df["imbalance"] = df.eval('imbalance_size * imbalance_buy_sell_flag')
        df["market_imbalance_buy_sell_flag"] = (df['bid_size'] > df['ask_size']).astype(int)
        df['volume_global_ratio'] = df.eval("volume/global_median_size")
        df['bid_size_global_ratio'] = df.eval("bid_size/global_median_bid_size")
        df['ask_size_global_ratio'] = df.eval("ask_size/global_median_ask_size")
        df['imbalance_global_ratio'] = df.eval("imbalance_size/global_median_imbalance_size")
        df['matched_global_ratio'] = df.eval("matched_size/global_median_matched_size")
        df['price_pressure'] = df.eval("imbalance_size * (ask_price - bid_price)")
        df['depth_pressure'] = df.eval("(ask_size - bid_size) * (far_price - near_price)")

    with timer("Created price combination features", verbose=verbose):
        for c in combinations(prices, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    with timer("Created size combination features", verbose=verbose):
        for c in combinations(sizes, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    with timer("Created wap and volatility features", verbose=verbose):
        df['log_wap'] = np.log(df['wap'])
        df['log_return'] = df.groupby(['stock_id'])['log_wap'].diff()
        df.loc[df.seconds_in_bucket < 10, "log_return"] = np.nan

    with timer("Created normalized features within time id", verbose=verbose):
        df['norm_wap'] = df.groupby("time_id")['wap'].transform(lambda x: x - x.mean())
        df['norm_imbalance_buy_sell_flag'] = df.groupby("time_id")['imbalance_buy_sell_flag'].transform(
            lambda x: x - x.mean())

    with timer("Created differenced features", verbose=verbose):
        df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
        df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff() / df['wap']
        df['rsi'] = (df.groupby(['stock_id'])['wap'].diff() > 0).astype(int)
        df['auction_direction_alignment'] = (
                df.groupby(['stock_id'])['wap'].diff() * df['imbalance_buy_sell_flag'] > 0).astype(int)
        df['market_direction_alignment'] = (
                df.groupby(['stock_id'])['wap'].diff() * df['market_imbalance_buy_sell_flag'] > 0).astype(int)
        df.loc[df.seconds_in_bucket < 10, "spread_intensity"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "imbalance_momentum"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "auction_direction_alignment"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "market_direction_alignment"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "rsi"] = np.nan

    # Calculate triplet imbalance features using the Numba-optimized function
    with timer("Created triplet features", verbose=verbose):
        for c in [['ask_price', 'bid_price', 'wap', 'reference_price', 'mid_price'], sizes]:
            triplet_feature = calculate_triplet_imbalance_numba(c, df)
            df[triplet_feature.columns] = triplet_feature.values

    # Calculate various statistical aggregation features
    with timer("Created statistical agg features", verbose=verbose):
        for func in ["mean", "std", "skew", "kurt", "min", "max"]:
            df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
            df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)

    # Additional Features
    rank_df = df.groupby('time_id')[['wap', 'imbalance_buy_sell_flag']].rank(pct=True)
    rank_df.columns = [f'{c}_rank' for c in rank_df.columns]
    df = pd.concat([df, rank_df], axis=1)

    return df.replace([np.inf, -np.inf], 0)


def time_features(df, verbose):
    with timer("Created time based features", verbose=verbose):
        df["seconds"] = df["seconds_in_bucket"] % 60
        df["minute"] = df["seconds_in_bucket"] // 60

    return df


def embed_stock(df, verbose=True):
    with timer("Embedding global stock info", verbose=verbose):
        for key, value in global_stock_id_feats.items():
            df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df


def generate_all_features(df, verbose=True):
    cols = [c for c in df.columns if c not in ["row_id"]]
    df = df[cols]

    df = embed_stock(df, verbose)
    df = imbalance_features(df, verbose)
    df = time_features(df, verbose)

    gc.collect()

    # Select and return the generated features
    feature_name = [i for i in df.columns if
                    i not in ["row_id", "target", "time_id", "date_id", "index_return", "stock_return", "target_norm",
                              "currently_scored"]]

    return df[feature_name]

time_id = df.time_id.values
date_id = df.date_id.values
time_index = time_id % 55
stock_id = df.stock_id.values
y = df['target']

df = generate_all_features(df)
df['target'] = y

ntime = 55
ndate = len(np.unique(date_id))
nstock = 200
nfeatures = df.shape[1]

X = np.zeros((ndate, ntime, nstock, nfeatures))
feature_dict = {}
for i, c in enumerate(df.columns.tolist()):
    feature_dict[c] = i

X[date_id, time_index, stock_id, :] = df.values
def generate_features(X, date, time, stock, feature_dict):
    res = pd.DataFrame(columns=list(feature_dict.keys()))
    res[list(feature_dict.keys())] = X[date, time, :, :]
    first_features = ['matched_size', 'imbalance_size', 'imbalance_buy_sell_flag', 'ask_size', 'bid_size']
    rank_features = ['imbalance_buy_sell_flag', 'wap', 'wap_mid_price_imb',
                     'volume_global_ratio', 'imbalance_global_ratio', 'matched_global_ratio', 'bid_size_global_ratio',
                     'ask_size_global_ratio', 'market_imbalance_buy_sell_flag']

    for f in first_features:
        c_first = X[date, 0, :, feature_dict[f]]
        c_curr = X[date, time, :, feature_dict[f]]
        first_ratio = c_curr / c_first
        res[f"{f}_first"] = c_first
        res[f"{f}_first_ratio"] = first_ratio

    ## Cumulative Features
    cumulative_features = ['imbalance_buy_sell_flag', 'rsi']
    for feat in cumulative_features:
        cumsum = np.sum(np.nan_to_num(X[date, :time+1, :, feature_dict[feat]]), axis=0)
        cummean = cumsum / X[date, time, :, feature_dict['seconds_in_bucket']]
        res[f'{feat}_cumsum'] = cumsum
        res[f'{feat}_cummean'] = cummean

    for f in rank_features:
        c_curr = X[date, time, :, feature_dict[f]]
        res[f"{f}_rank"] = pd.Series(c_curr).rank(pct=True).values

    ## Additional Rank Features
    res[f"imbalance_buy_sell_flag_cumsum_rank"] = res['imbalance_buy_sell_flag_cumsum'].rank(pct=True).values

    ## Percent Change Features
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'wap', 'ask_price', 'bid_price', 'ask_size','bid_size']:
        for window in [1, 2, 3, 6, 10]:
            pct_change = (X[date, time, :, feature_dict[col]] / X[date, time - window, :, feature_dict[col]] - 1)
            res[f"{col}_ret_{window}"] = pct_change
            if (time - window < 0):
                res[f"{col}_ret_{window}"] = np.nan

    for col in ['wap', 'imbalance_buy_sell_flag', 'imbalance_size', 'matched_size', 'norm_wap']:
        for window in [3, 6, 10]:
            mean = np.mean(X[date, time - window + 1:time + 1, :, feature_dict[col]], axis=0)
            std = np.std(X[date, time - window + 1:time + 1, :, feature_dict[col]], axis=0)
            res[f"{col}_rolling_mean_{window}"] = mean
            res[f"{col}_rolling_std_{window}"] = std
            if (time - window + 1 < 0):
                res[f"{col}_rolling_mean_{window}"] = np.nan
                res[f"{col}_rolling_std_{window}"] = np.nan

    for col in ['auction_direction_alignment', 'rsi']:
        for window in [3, 6, 10]:
            mean = np.mean(X[date, time - window + 1:time + 1, :, feature_dict[col]], axis=0)
            res[f"{col}_rolling_mean_{window}"] = mean
            if (time - window + 1 < 0):
                res[f"{col}_rolling_mean_{window}"] = np.nan

    for col in ['wap_mid_price_imb', 'reference_price_wap_imb', 'norm_wap', 'imbalance_buy_sell_flag', 'wap_rank',
                'imbalance_buy_sell_flag_rank']:
        for window in [1, 2, 3, 4, 5, 6]:
            lag = X[date, time - window, :, feature_dict[col]]
            res[f"{col}_shift_{window}"] = lag
            if (time - window < 0):
                res[f"{col}_shift_{window}"] = np.nan

    shift_features = ['imbalance_size', 'imbalance_buy_sell_flag', 'wap_rank', 'imbalance_buy_sell_flag_rank',
                      'reference_price_wap_imb', 'target']

    for shift_idx in [1, 2]:
        for f in shift_features:
            shift = X[date - shift_idx, time, :, feature_dict[f]].copy()
            res[f"shifted_{shift_idx}_{f}"] = shift
            if (date - shift_idx < 0):
                res[f"shifted_{shift_idx}_{f}"] = np.nan

    # Handling edge case cumsum feature
    for shift_idx in [1,2]:
        cumsum = np.sum(np.nan_to_num(X[date-shift_idx, :time+1, :, feature_dict[feat]]), axis=0)
        res[f"shifted_{shift_idx}_imbalance_buy_sell_flag_cumsum"] = cumsum
        if (date - shift_idx < 0):
            res[f"shifted_{shift_idx}_imbalance_buy_sell_flag_cumsum"] = np.nan

    res = res.iloc[stock].reset_index(drop=True)
    res['stock_id'] = stock

    return res.replace([np.inf, -np.inf], 0)


df['time_id'] = time_id
df['date_id'] = date_id
df_l = []
for (t), frame in tqdm(df.groupby("time_id")):
    stock_idx = frame.stock_id.values
    date = frame.date_id.values[0]
    time_idx = t % 55
    feat = generate_features(X, date, time_idx, stock_idx, feature_dict)
    df_l.append(feat)

df = pd.concat(df_l, axis=0, ignore_index=True)
df['time_id'] = time_id
df['date_id'] = date_id
df.to_feather("input/train_processed.f")

