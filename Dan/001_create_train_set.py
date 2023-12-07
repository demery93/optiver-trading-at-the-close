import numpy as np
import pandas as pd

import gc
import warnings

from numba import njit, prange
from itertools import combinations
from warnings import simplefilter
from utils import reduce_mem_usage, timer
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df = pd.read_csv("input/train.csv")
features = ['seconds_in_bucket', 'imbalance_size', 'far_price', 'near_price', 'bid_price', 'ask_size', 'global_median_size', 'global_median_bid_size', 'global_median_ask_size', 'global_median_imbalance_size', 'global_median_matched_size', 'global_mean_imbalance_buy_sell_flag', 'global_std_size', 'global_ptp_size', 'global_median_price', 'global_std_price', 'global_std_wap', 'global_ptp_price', 'imbalance', 'imbalance_global_ratio', 'price_pressure', 'depth_pressure', 'reference_price_far_price_imb', 'reference_price_near_price_imb', 'reference_price_ask_price_imb', 'reference_price_bid_price_imb', 'reference_price_wap_imb', 'reference_price_mid_price_imb', 'far_price_near_price_imb', 'far_price_mid_price_imb', 'near_price_ask_price_imb', 'near_price_bid_price_imb', 'near_price_wap_imb', 'near_price_mid_price_imb', 'wap_mid_price_imb', 'matched_size_bid_size_imb', 'matched_size_ask_size_imb', 'matched_size_imbalance_size_imb', 'matched_size_mid_size_imb', 'bid_size_ask_size_imb', 'bid_size_imbalance_size_imb', 'ask_size_imbalance_size_imb', 'imbalance_size_mid_size_imb', 'imbalance_momentum', 'log_return', 'norm_wap', 'norm_imbalance_buy_sell_flag', 'norm_log_return', 'ask_price_bid_price_wap_imb2', 'ask_price_wap_reference_price_imb2', 'ask_price_reference_price_mid_price_imb2', 'bid_price_wap_reference_price_imb2', 'bid_price_reference_price_mid_price_imb2', 'matched_size_bid_size_ask_size_imb2', 'matched_size_bid_size_imbalance_size_imb2', 'matched_size_bid_size_mid_size_imb2', 'matched_size_ask_size_imbalance_size_imb2', 'matched_size_ask_size_mid_size_imb2', 'matched_size_imbalance_size_mid_size_imb2', 'bid_size_ask_size_imbalance_size_imb2', 'all_sizes_mean', 'all_prices_std', 'all_sizes_std', 'all_prices_skew', 'all_sizes_skew', 'all_prices_kurt', 'all_sizes_kurt', 'all_sizes_max', 'wap_rank', 'imbalance_buy_sell_flag_rank', 'seconds', 'minute', 'matched_size_first', 'matched_size_first_ratio', 'imbalance_size_first', 'imbalance_size_first_ratio', 'ask_size_first', 'ask_size_first_ratio', 'bid_size_first', 'imbalance_buy_sell_flag_cumsum', 'imbalance_buy_sell_flag_cummean', 'rsi_cumsum', 'rsi_cummean', 'wap_mid_price_imb_rank', 'imbalance_global_ratio_rank', 'matched_global_ratio_rank', 'ask_size_global_ratio_rank', 'market_imbalance_buy_sell_flag_rank', 'imbalance_buy_sell_flag_cumsum_rank', 'matched_size_ret_1', 'matched_size_ret_2', 'matched_size_ret_10', 'imbalance_size_ret_1', 'imbalance_size_ret_2', 'imbalance_size_ret_3', 'imbalance_size_ret_6', 'imbalance_size_ret_10', 'reference_price_ret_1', 'reference_price_ret_2', 'reference_price_ret_3', 'reference_price_ret_10', 'wap_ret_1', 'ask_price_ret_1', 'bid_price_ret_1', 'bid_price_ret_10', 'ask_size_ret_1', 'bid_size_ret_1', 'bid_size_ret_2', 'bid_size_ret_3', 'wap_rolling_std_3', 'wap_rolling_std_10', 'imbalance_buy_sell_flag_rolling_mean_3', 'imbalance_buy_sell_flag_rolling_std_3', 'imbalance_buy_sell_flag_rolling_mean_6', 'imbalance_buy_sell_flag_rolling_mean_10', 'imbalance_buy_sell_flag_rolling_std_10', 'imbalance_size_rolling_mean_3', 'imbalance_size_rolling_mean_10', 'matched_size_rolling_std_10', 'norm_wap_rolling_mean_6', 'norm_wap_rolling_mean_10', 'rsi_rolling_mean_3', 'rsi_rolling_mean_10', 'matched_size_ema', 'imbalance_size_ema', 'reference_price_wap_imb_shift_3', 'reference_price_wap_imb_shift_4', 'reference_price_wap_imb_shift_5', 'reference_price_wap_imb_shift_6', 'norm_wap_shift_1', 'norm_wap_shift_3', 'norm_wap_shift_5', 'imbalance_buy_sell_flag_shift_1', 'imbalance_buy_sell_flag_shift_2', 'imbalance_buy_sell_flag_shift_3', 'wap_rank_shift_1', 'wap_rank_shift_3', 'wap_rank_shift_4', 'wap_rank_shift_5', 'imbalance_buy_sell_flag_rank_shift_1', 'imbalance_buy_sell_flag_rank_shift_2', 'imbalance_buy_sell_flag_rank_shift_4', 'imbalance_buy_sell_flag_rank_shift_6', 'norm_log_return_shift_1', 'norm_log_return_shift_2', 'norm_log_return_shift_3', 'norm_log_return_shift_4', 'norm_log_return_shift_5', 'norm_log_return_shift_6', 'shifted_1_imbalance_buy_sell_flag', 'shifted_1_imbalance_buy_sell_flag_rank', 'shifted_1_target', 'shifted_2_imbalance_buy_sell_flag_rank', 'shifted_2_reference_price_wap_imb', 'shifted_2_target', 'shifted_1_imbalance_buy_sell_flag_cumsum', 'shifted_1_rsi_cumsum', 'shifted_2_rsi_cumsum', 'shifted_1_imbalance_cumsum']

def clean_data(df):
    df = df[df.target.notnull()].reset_index(drop=True)
    return df

df = clean_data(df)
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
    "median_size": df.groupby("stock_id")["bid_size"].median() + df.groupby("stock_id")["ask_size"].median(),
    "median_bid_size": df.groupby("stock_id")['bid_size'].median(),
    "median_ask_size": df.groupby("stock_id")['ask_size'].median(),
    "median_imbalance_size": df.groupby("stock_id")['imbalance_size'].median(),
    "median_matched_size": df.groupby("stock_id")['matched_size'].median(),
    "mean_imbalance_buy_sell_flag": df.groupby("stock_id")['imbalance_buy_sell_flag'].mean(),
    "std_size": df.groupby("stock_id")["bid_size"].std() + df.groupby("stock_id")["ask_size"].std(),
    "ptp_size": df.groupby("stock_id")["bid_size"].max() - df.groupby("stock_id")["bid_size"].min(),
    "median_price": df.groupby("stock_id")["bid_price"].median() + df.groupby("stock_id")["ask_price"].median(),
    "std_price": df.groupby("stock_id")["bid_price"].std() + df.groupby("stock_id")["ask_price"].std(),
    "std_wap": df.groupby("stock_id")["wap"].std(),
    "ptp_price": df.groupby("stock_id")["bid_price"].max() - df.groupby("stock_id")["ask_price"].min(),
}

def imbalance_features(df, verbose=True):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap", "mid_price"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size", "mid_size"]
    norm_features = ['wap', 'imbalance_buy_sell_flag', 'log_return', 'wap_mid_price_imb']

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

    # Calculate various statistical aggregation features
    # Can drop all but skew and kurt
    with timer("Created statistical price agg features", verbose=verbose):
        for func in ["std", "skew", "kurt"]:
            df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)

    with timer("Created statistical agg features", verbose=verbose):
        for func in ["mean", "std", "skew", "kurt", "max"]:
            df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)

    with timer("Created price combination features", verbose=verbose):
        for c in combinations(prices, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    with timer("Created size combination features", verbose=verbose):
        for c in combinations(sizes, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    with timer("Created wap", verbose=verbose):
        df['log_wap'] = np.log(df['wap'])

    with timer("Created differenced features", verbose=verbose):
        df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
        df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff() / df['wap']
        df['rsi'] = (df.groupby(['stock_id'])['wap'].diff() > 0).astype(int)
        df['auction_direction_alignment'] = (df.groupby(['stock_id'])['wap'].diff() * df['imbalance_buy_sell_flag'] > 0).astype(int)
        df['market_direction_alignment'] = (df.groupby(['stock_id'])['wap'].diff() * df['market_imbalance_buy_sell_flag'] > 0).astype(int)
        df['log_return'] = df.groupby(['stock_id'])['log_wap'].diff()

        df.loc[df.seconds_in_bucket < 10, "imbalance_momentum"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "spread_intensity"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "rsi"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "auction_direction_alignment"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "market_direction_alignment"] = np.nan
        df.loc[df.seconds_in_bucket < 10, "log_return"] = np.nan

    with timer("Created normalized features within time id", verbose=verbose):
        for c in norm_features:
            df[f'norm_{c}'] = df.groupby("time_id")[c].transform(lambda x: x - x.mean())

    # Calculate triplet imbalance features using the Numba-optimized function
    with timer("Created triplet features", verbose=verbose):
        for c in [['ask_price', 'bid_price', 'wap', 'reference_price', 'mid_price'], sizes]:
            triplet_feature = calculate_triplet_imbalance_numba(c, df)
            df[triplet_feature.columns] = triplet_feature.values

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

X = np.zeros((ndate, ntime, nstock, nfeatures)) * np.nan
feature_dict = {}
for i, c in enumerate(df.columns.tolist()):
    feature_dict[c] = i

#stock_mapping = {166: 'KHC', 121: 'KDP', 105: 'MDLZ', 151: 'CSCO', 170: 'HOLX', 0: 'MNST', 65: 'EXC', 109: 'CSX', 123: 'GILD', 198: 'CMCSA', 131: 'INCY', 21: 'SSNC', 148: 'XEL', 38: 'ATVI', 30: 'LNT', 63: 'LKQ', 24: 'AKAM', 130: 'SBUX', 120: 'FOXA', 195: 'AEP', 53: 'EBAY', 81: 'SGEN', 47: 'TXRH', 154: 'DBX', 160: 'PEP', 55: 'FAST', 37: 'ADP', 90: 'FFIV', 186: 'CTSH', 187: 'EA', 76: 'HAS', 117: 'AGNC', 134: 'VTRS', 3: 'HON', 165: 'HST', 97: 'NBIX', 145: 'CG', 25: 'EXPD', 68: 'PAYX', 52: 'CINF', 112: 'LSTR', 181: 'JBHT', 28: 'FANG', 43: 'JKHY', 12: 'AMGN', 149: 'VRSK', 144: 'PCAR', 192: 'PTC', 153: 'HTZ', 175: 'GOOGL', 189: 'CSGP', 116: 'CDW', 35: 'ROST', 46: 'TECH', 164: 'CPRT', 44: 'CME', 146: 'MIDD', 125: 'UTHR', 171: 'TROW', 73: 'GEN', 196: 'XRAY', 9: 'PFG', 199: 'BKR', 193: 'PYPL', 106: 'TRMB', 122: 'AAL', 4: 'MAR', 176: 'FTNT', 2: 'AXON', 167: 'CHRW', 91: 'DOCU', 128: 'WDC', 152: 'SBAC', 155: 'CHK', 26: 'HBAN', 132: 'TSCO', 119: 'MASI', 27: 'QRVO', 84: 'GOOGL', 23: 'SWKS', 110: 'TMUS', 182: 'UAL', 157: 'ADSK', 168: 'AMZN', 147: 'APA', 64: 'MKTX', 190: 'DXCM', 19: 'ALNY', 1: 'WING', 49: 'FITB', 194: 'PENN', 140: 'TXN', 133: 'ISRG', 177: 'SWAV', 32: 'NTAP', 22: 'ON', 77: 'VRTX', 104: 'WBA', 107: 'PODD', 59: 'Z', 72: 'ADI', 158: 'APLS', 169: 'MSFT', 94: 'PCTY', 66: 'LBRDK', 126: 'MU', 139: 'EXPE', 159: 'STLD', 137: 'TTWO', 78: 'HOOD', 114: 'LPLA', 141: 'AMAT', 15: 'ABNB', 60: 'CRWD', 183: 'MCHP', 10: 'NDAQ', 135: 'DKNG', 197: 'SPLK', 99: 'PARA', 56: 'ETSY', 13: 'TER', 62: 'RGEN', 80: 'TXG', 67: 'MRNA', 178: 'ZM', 39: 'CTAS', 173: 'FIVE', 184: 'DDOG', 162: 'ENPH', 16: 'ZBRA', 89: 'ENTG', 45: 'MSFT', 124: 'ASO', 42: 'SAIA', 115: 'ILMN', 50: 'MTCH', 98: 'JBLU', 103: 'ZS', 40: 'CZR', 108: 'SEDG', 179: 'META', 6: 'POOL', 100: 'MQ', 48: 'WDAY', 150: 'PANW', 74: 'ALGN', 113: 'LULU', 163: 'COST', 111: 'SPWR', 36: 'DLTR', 85: 'CAR', 79: 'WBD', 83: 'INTC', 75: 'CDNS', 57: 'IDXX', 180: 'GH', 93: 'ZION', 87: 'LSCC', 51: 'ROKU', 33: 'CROX', 58: 'ROP', 7: 'LRCX', 172: 'APP', 61: 'LYFT', 185: 'ODFL', 102: 'TEAM', 188: 'RUN', 17: 'KLAC', 88: 'NFLX', 95: 'AMD', 14: 'ADBE', 54: 'SNPS', 18: 'ZI', 129: 'CFLT', 136: 'LITE', 191: 'TSLA', 20: 'ULTA', 161: 'PTON', 5: 'OKTA', 71: 'EQIX', 34: 'REGN', 142: 'AVGO', 92: 'MSTR', 156: 'LCID', 41: 'NVDA', 69: 'SOFI', 138: 'SMCI', 174: 'AFRM', 11: 'COIN', 70: 'BYND', 96: 'MRVL', 118: 'FCNCA', 29: 'ORLY', 143: 'TLRY', 86: 'ONEW', 82: 'OPEN', 127: 'MDB', 101: 'FCNCA', 8: 'BKNG', 31: 'NVCR'}
#sector_id = {0: 3, 1: 2, 2: 7, 3: 7, 4: 2, 5: 9, 6: 7, 7: 9, 8: 2, 9: 5, 10: 5, 11: 5, 12: 6, 13: 9, 14: 9, 15: 2, 16: 9, 17: 9, 18: 9, 19: 6, 20: 2, 21: 9, 22: 9, 23: 9, 24: 9, 25: 7, 26: 5, 27: 9, 28: 4, 29: 2, 30: 10, 31: 6, 32: 9, 33: 2, 34: 6, 35: 2, 36: 3, 37: 7, 38: 1, 39: 7, 40: 2, 41: 9, 42: 7, 43: 9, 44: 5, 45: 9, 46: 6, 47: 2, 48: 9, 49: 5, 50: 1, 51: 1, 52: 5, 53: 2, 54: 9, 55: 7, 56: 2, 57: 6, 58: 9, 59: 1, 60: 9, 61: 9, 62: 6, 63: 2, 64: 5, 65: 10, 66: 1, 67: 6, 68: 7, 69: 3, 70: 8, 71: 9, 72: 9, 73: 6, 74: 9, 75: 2, 76: 6, 77: 6, 78: 6, 79: 8, 80: 9, 81: 1, 82: 7, 83: 2, 84: 9, 85: 1, 86: 9, 87: 9, 88: 9, 89: 9, 90: 5, 91: 9, 92: 9, 93: 9, 94: 6, 95: 7, 96: 1, 97: 9, 98: 5, 99: 9, 100: 6, 101: 3, 102: 9, 103: 6, 104: 9, 105: 7, 106: 1, 107: 9, 108: 7, 109: 2, 110: 5, 111: 6, 112: 9, 113: 8, 114: 5, 115: 6, 116: 1, 117: 3, 118: 7, 119: 6, 120: 2, 121: 6, 122: 9, 123: 9, 124: 9, 125: 9, 126: 2, 127: 6, 128: 2, 129: 6, 130: 6, 131: 9, 132: 1, 133: 9, 134: 2, 135: 9, 136: 9, 137: 9, 138: 6, 139: 7, 140: 5, 141: 7, 142: 4, 143: 10, 144: 7, 145: 9, 146: 8, 147: 9, 148: 4, 149: 9, 150: 6, 151: 0, 152: 3, 153: 2, 154: 9, 155: 3, 156: 7, 157: 8, 158: 3, 159: 7, 160: 2, 161: 9, 162: 6, 163: 5, 164: 9, 165: 2, 166: 9, 167: 1, 168: 9, 169: 6, 170: 9, 171: 1, 172: 6, 173: 7, 174: 7, 175: 9, 176: 9, 177: 7, 178: 9, 179: 1, 180: 9, 181: 8, 182: 6, 183: 2, 184: 9, 185: 5, 186: 2, 187: 10, 188: 6, 189: 9, 190: 1, 191: 9, 192: 5, 193: 2, 194: 9, 195: 7, 196: 4, 197: 1, 198: 2, 199: 9}
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
        try:
            first_ratio = c_curr / c_first
        except:
            first_ratio = np.nan
        if(f"{f}_first" in features):
            res[f"{f}_first"] = c_first
        if (f"{f}_first_ratio" in features):
            res[f"{f}_first_ratio"] = first_ratio

    ## Cumulative Features
    cumulative_features = ['imbalance_buy_sell_flag','rsi']
    for f in cumulative_features:
        cumsum = np.sum(np.nan_to_num(X[date, :time + 1, :, feature_dict[f]]), axis=0)
        try:
            cummean = cumsum / X[date, time, :, feature_dict['seconds_in_bucket']]
        except:
            cummean = np.nan
        if(f'{f}_cumsum' in features):
            res[f'{f}_cumsum'] = cumsum
        if(f'{f}_cummean' in features):
            res[f'{f}_cummean'] = cummean

    for f in rank_features:
        if(f"{f}_rank" in features):
            c_curr = X[date, time, :, feature_dict[f]]
            res[f"{f}_rank"] = pd.Series(c_curr).rank(pct=True).values

    ## Additional Rank Features
    res[f"imbalance_buy_sell_flag_cumsum_rank"] = res['imbalance_buy_sell_flag_cumsum'].rank(pct=True).values

    ## Percent Change Features
    for f in ['matched_size', 'imbalance_size', 'reference_price', 'wap', 'ask_price', 'bid_price', 'ask_size','bid_size']:
        for window in [1, 2, 3, 6, 10]:
            if(f"{f}_ret_{window}" in features):
                try:
                    pct_change = (X[date, time, :, feature_dict[f]] / X[date, time - window, :, feature_dict[f]] - 1)
                except:
                    pct_change = np.nan
                res[f"{f}_ret_{window}"] = pct_change
                if (time - window < 0):
                    res[f"{f}_ret_{window}"] = np.nan

    ## Rolling Means
    for f in ['wap', 'imbalance_buy_sell_flag', 'imbalance_size', 'matched_size', 'norm_wap','auction_direction_alignment','rsi']:
        if(f"{f}_rolling_mean_{window}" in features):
            for window in [3, 6, 10]:
                mean = np.mean(X[date, time - window + 1:time + 1, :, feature_dict[f]], axis=0)
                res[f"{f}_rolling_mean_{window}"] = mean
                if (time - window + 1 < 0):
                    res[f"{f}_rolling_mean_{window}"] = np.nan

    ## Rolling Standard Deviations
    for f in ['wap', 'imbalance_buy_sell_flag', 'imbalance_size', 'matched_size', 'norm_wap']:
        if(f"{f}_rolling_std_{window}" in features):
            for window in [3, 6, 10]:
                std = np.std(X[date, time - window + 1:time + 1, :, feature_dict[f]], axis=0)
                res[f"{f}_rolling_std_{window}"] = std
                if (time - window + 1 < 0):
                    res[f"{f}_rolling_std_{window}"] = np.nan

    ## EMA
    alpha = 0.285
    beta = 1 - alpha
    for f in ['matched_size', 'wap', 'imbalance_size','norm_wap','reference_price']:
        if(f"{f}_ema" in features):
            ema = X[date, time, :, feature_dict[f]]*alpha + \
                  X[date, time-1, :, feature_dict[f]]*alpha*beta + \
                  X[date, time-2, :, feature_dict[f]]*alpha*beta**2 + \
                  X[date, time-3, :, feature_dict[f]]*alpha*beta**3 + \
                  X[date, time-4, :, feature_dict[f]]*alpha*beta**4 + \
                  X[date, time-5, :, feature_dict[f]]*alpha*beta**5 + \
                  X[date, time-6, :, feature_dict[f]]*alpha*beta**6
            res[f"{f}_ema"] = ema
            if (time < 6):
                res[f"{f}_ema"] = np.nan



    for f in ['wap_mid_price_imb', 'reference_price_wap_imb', 'norm_wap', 'imbalance_buy_sell_flag', 'wap_rank','imbalance_buy_sell_flag_rank','norm_log_return']:
        for window in [1, 2, 3, 4, 5, 6]:
            if(f"{f}_shift_{window}" in features):
                lag = X[date, time - window, :, feature_dict[f]]
                res[f"{f}_shift_{window}"] = lag
                if (time - window < 0):
                    res[f"{f}_shift_{window}"] = np.nan

    shift_features = ['imbalance_size', 'imbalance_buy_sell_flag', 'wap_rank', 'imbalance_buy_sell_flag_rank','reference_price_wap_imb', 'target']
    for shift_idx in [1, 2]:
        for f in shift_features:
            if(f"shifted_{shift_idx}_{f}" in features):
                shift = X[date - shift_idx, time, :, feature_dict[f]].copy()
                res[f"shifted_{shift_idx}_{f}"] = shift
                if (date - shift_idx < 0):
                    res[f"shifted_{shift_idx}_{f}"] = np.nan

    # Handling edge case cumsum features
    for f in ['imbalance_buy_sell_flag','rsi','imbalance']:
        for shift_idx in [1, 2]:
            if(f"shifted_{shift_idx}_{f}_cumsum" in features):
                cumsum = np.sum(np.nan_to_num(X[date - shift_idx, :time + 1, :, feature_dict[f]]), axis=0)
                res[f"shifted_{shift_idx}_{f}_cumsum"] = cumsum
                if (date - shift_idx < 0):
                    res[f"shifted_{shift_idx}_{f}_cumsum"] = np.nan

    res = res.iloc[stock].reset_index(drop=True)
    res['stock_id'] = stock
    res = res[features]

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
df['stock_id'] = stock_id
df['target'] = y
df.to_feather("input/train_processed.f")