import os, time
from utils import *

COLS = ['sma', 'ema', 'wma', 'bbands_high', 'bbands_low', 'per_b_high', 'per_b_low', 'trima', 'rsi', 'willr', 'atr', 'trange', 'plus_di', 'minus_di', 'dx', 'adx', 'roc', '_ema_fixed_12', '_ema_fixed_26', 'macd', '_macd_self_ema',  'macd_histogram', 'cci', 'aroon_osc', 'adl', '_adl_ema_3', '_adl_ema_10', 'chaikin_osc', '_mf_vol', '_mf_mult', 'chaikin_mf', 'obv', 'stoch_per_k', 'stoch_per_d', '_avg_up', '_avg_dn', '_plus_dm_sum', '_minus_dm_sum', '_tr_sum', 'ichi_dist_day', 'ichi_dist_week', 'ichi_dist_month', 'label', 'weight', 'return']

def z(x):
    return np.zeros(x)

def rename_files(prices_path, after_sym):
    for txt in os.listdir(prices_path):
        if after_sym not in txt:
            print(f'Rewriting failed after_sym arg {after_sym} was not found in {txt}.  No files overwritten.')
            return None

    for txt in os.listdir(prices_path):
        new_name = txt.replace(after_sym, '').upper() + '.csv'
        os.rename(f'{prices_path}/{txt}', f'{prices_path}/{new_name}')

    print(f'Successfully renamed files in {prices_path}')
    
def sym_filename(prices_path, sym):
    return f'{prices_path}/{sym}.csv'

def load_overwrite_price(prices_path, sym, col_names=None):
    '''overwrite col names, remove file if no columns'''
    filename = sym_filename(prices_path, sym)
    cols_needed = ['date','open','high','low','close','volume']
    
    try:
        df = pd.read_csv(filename)
    except:
        os.remove(filename)
        return None, None

    idx = 'Unnamed: 0'
    if idx in df.columns:
        df=df.drop(idx, axis=1)
    
    if col_names is None:
        if subset(cols_needed, df.columns):
            return df, False # don't need to write b/c in correct format
        else:
            df.rename(str.lower, axis=1, inplace=True)
    else:
        df.rename(columns=col_names, inplace=True)
    
    df_to_csv(df, filename, overwrite=True, quiet=True)
    return df, True

def load_overwrite_prices(prices_path, sym_subset=[], col_names=None):
    sym_subset = update_sym_subset(prices_path, sym_subset)
    dfs = [None]*len(sym_subset)
    deleted = []
    overwritten = []
    print(f'\nLoading and reformatting prices based on column mappings in {prices_path}')
    progbar = tf.keras.utils.Progbar(len(sym_subset))
    for i, sym in enumerate(sym_subset):
        dfs[i], overwrite = load_overwrite_price(prices_path, sym, col_names=col_names)
        if dfs[i] is None:
            deleted.append(sym)
        if overwrite:
            overwritten.append(sym)
        progbar.add(1)

    print(f'Deleted the following syms which were not parsable by pandas: {deleted}')
    print(f'Reformatted columns and overwrote the following syms: {overwritten}')
    return [elt for elt in filter(lambda elt2: elt2 is not None, dfs)]


def update_prices(new_prices_path, old_prices_path, sym_subset=[], col_mapping=None):
    '''Update prices in {old_prices_path} with prices in {new_prices_path}.  If a price exists in sym_subset of new and not in old, add it'''
    sym_subset = update_sym_subset(new_prices_path, sym_subset)
    dfs = load_overwrite_prices(new_prices_path, sym_subset=sym_subset, col_names=col_mapping)

    print(f'\nUpdating prices, overwriting prices in {old_prices_path} with prices in {new_prices_path}')
    progbar = tf.keras.utils.Progbar(len(sym_subset))
    for i, (sym, df2) in enumerate(zip(sym_subset, dfs)):
        if os.path.exists(sym_filename(old_prices_path, sym)):
            df = load_df(old_prices_path, sym)
            max_df_date = df['date'].max()
            df2 = df2.loc[df2['date'] > max_df_date]
            dfs[i] = pd.merge(df, df2, how='outer')
        else:
            dfs[i] = df2
        
        # clean up nans
        df = dfs[i]
        n = len(df)
        for j in range(1, n):
            for col in df.columns:
                if df[col].dtype != 'object' and np.isnan(df.iloc[j][col]):
                    df.loc[j, col] = df.loc[j-1, col]

        df_to_csv(df, sym_filename(old_prices_path, sym), overwrite=True, quiet=True)
        progbar.add(1)

    return dfs

def periods_from_path(indicators_path):
    return [int(elt) for elt in indicators_path.split('/')[-2].split('_')[1:]]

def get_idx_of_date(df, date):
    return int(df.loc[df['date'] == date].index.to_numpy()[0])

def update_indicators(prices_path, indicators_path, periods, sym_subset=[], first_write=False):
    ''' If price in prices_path (which has presumably been updated with update_prices) is not in indicators_path, create it
    Note: DO NOT PASS NEW_PRICES_PATH - in the case that you only get the most recent prices to update_prices, this will fail as it expects all prices
    First write: delete and recreate folder from scrath.  Else update.
    '''
    sym_subset = update_sym_subset(prices_path, sym_subset)
    
    if os.path.exists(indicators_path) and first_write:
        print(f'Overwriting dir: {indicators_path}')
        shutil.rmtree(indicators_path)
    
    if not os.path.exists(indicators_path):
        pathlib.Path(indicators_path).mkdir(parents=True, exist_ok=True)

    print('\nUpdating indicators...')
    progbar = tf.keras.utils.Progbar(len(sym_subset))
    for sym in sym_subset:
        # if indicators file does not exist, create it
        if not os.path.exists(sym_filename(indicators_path, sym)):
            df = sym_to_indicators(sym, prices_path, periods)
            df_to_csv(df, sym_filename(indicators_path, sym), quiet=True)

        # else update it
        else:
            df_price = load_df(prices_path, sym)
            n = len(df_price)

            df_ind = load_df(indicators_path, sym)
            
            # but only if there are updates to be made
            df = df_ind
            offset = get_idx_of_date(df_price, df_price['date'].max()) - get_idx_of_date(df_price, df_ind['date'].max())
            if offset != 0:
                price, highs, lows, opens, volume = get_f32_cols(df_price, ['close', 'high', 'low', 'open', 'volume'])
                df_temp = df_price.loc[n-offset:].reset_index(drop=True)
                for period in periods:
                    cols = [f'{elt}_{period}' for elt in COLS]
                    indicators_in = df_ind.filter(regex=f'_{period}$').to_numpy()
                    indicators_in = np.append(indicators_in, z((offset, len(cols))), axis=0)
                    arr = get_indicators(price, period, len(cols), highs, lows, opens, volume, offset=offset, indicators_in=indicators_in)
                    df_temp = pd.concat([df_temp, pd.DataFrame(arr, columns=cols)], axis=1)
                df = pd.concat([df_ind, df_temp], axis=0).reset_index(drop=True)
                df_to_csv(df, sym_filename(indicators_path, sym), overwrite=True, quiet=True)
        
        progbar.add(1)


def get_preds_ichi(price, ichi_type):
    '''
    above cloud: buy
    below: sell
    else: .5
    '''
    periods = {'day': 1, 'week': 7, 'twoweek': 14, 'month': 30}
    per = periods[ichi_type]

    n = len(price)

    # 0: conv, 1: base, 2: lead_a, 3: lead_b, 4: lag, 5: ichi_dist
    arr = z((n, 6))

    def avg(a, b):
        return (a + b) / 2

    def phl(i, k, price):
        ''' Determine avg of high and low for k periods before this'''
        if i >= k*per:
            arr = ar(price[i-k*per:i])[range(0,k*per,per)]
            return avg(max(arr), min(arr))
        else:
            return 0
        
    for i in range(0, n):
        price_i = price[i]
        arr[i, 0] = phl(i,9,price)
        arr[i, 1] = phl(i,26,price)
        top = max(arr[i, 2], arr[i, 3])
        bot = min(arr[i, 2], arr[i, 3])

        if (n-i)/per > 26:
            arr[i+26*per, 2] = avg(arr[i, 0], arr[i, 1])
            arr[i+26*per, 3] = phl(i,52,price)
        if i/per >= 26:
            arr[i-26*per, 4] = price_i
              
        if price_i > top:
            ichi_dist = price_i - top
        elif price_i < bot:
            ichi_dist = price_i - bot
        else:
            ichi_dist = 0

        arr[i, 5] = ichi_dist
    
    return arr[:, 5]


def add(arr, i, j, elt):
    if np.isnan(elt):
        arr[i,j] = -1
    else:
        arr[i,j] = elt

def get_indicators(price, period, num_cols, highs, lows, opens, volume, offset=0, indicators_in=None):
    n = len(price)
    arr = z((n, num_cols)) if indicators_in is None else indicators_in

    # ichi
    arr[:, 39] = get_preds_ichi(price, 'day')
    arr[:, 40] = get_preds_ichi(price, 'week')
    arr[:, 41] = get_preds_ichi(price, 'month')

    # trima
    if period % 2 == 0: # even
        raw_weights = np.hstack(ar([np.arange(1, math.floor(period/2)+1), np.arange(math.floor(period/2), 0, -1)]))
    else:
        raw_weights = np.hstack(ar([np.arange(1, math.floor(period/2)+1), ar([math.ceil(period/2)]), np.arange(math.floor(period/2), 0, -1)]))
    trima_weights = raw_weights / raw_weights.sum()

    def trange(k):
        return max(highs[k], price[k-1]) - min(lows[k], price[k-1])
    
    def ema(p, ema_prev, per):
        ''' p: price; per: period'''
        return (p - ema_prev)*(2/(per + 1)) + ema_prev

    def calc_plus_minus_dm(i):
        plus_dm = max(highs[i] - highs[i-1], 0)
        minus_dm = max(lows[i-1] - lows[i], 0)
        if plus_dm == minus_dm:
            plus_dm = 0
            minus_dm = 0
        elif plus_dm > minus_dm:
            minus_dm = 0
        else:
            plus_dm = 0

        return plus_dm, minus_dm

    if indicators_in is None:
        # no valid indicator value
        for i in range(min(period+1, n)):
            for j in range(num_cols):
                arr[i,j] = -1

        obv = 0
        for j in range(1, min(n,period)):
            if price[j-1] < price[j]:
                obv += volume[j]
            else:
                obv -= volume[j]
        
        start_idx = period

    else:
        start_idx = n-offset
    
    for i in range(start_idx, n):
        if np.isnan(price[i]):
            price[i] = price[i-1]
            highs[i] = highs[i-1]
            lows[i] = lows[i-1]
            opens[i] = opens[i-1]
            volume[i] = volume[i-1]

        if i == n-1: # no possible predictions on last datapt- add -1 tag so these are filtered out
            price_diff = -1
            ret = -2
        else:
            # label (buy=1, sell = 0), weight, return
            price_diff = price[i+1] - price[i]
            ret = price_diff / price[i]
        
        add(arr, i, 42, int(price_diff > 0))
        add(arr, i, 43, abs(ret))
        add(arr, i, 44, ret + 1)
        
        # get the last <period> prices. e.g. if period = 5, get the last 5 prices
        period_prices = price[i-period+1:i+1]
        std = np.std(period_prices)

        # sma
        add(arr, i, 0, period_prices.mean())
        
        # ema
        add(arr, i, 1, ema(price[i], arr[i-1, 1], period))

        # wma
        add(arr, i, 2, sum([price[i-j] * (period-j) for j in range(period)]) / (period * (period+1) / 2))

        # bbands high, low
        add(arr, i, 3, arr[i, 0] + 2*std)
        add(arr, i, 4, arr[i, 0] - 2*std)

        # per_b high, low - measure of how far away price is from low/high bband, divided by std
        if std == 0:
            add(arr, i, 5, 0)
            add(arr, i, 6, 0)
        else:
            add(arr, i, 5, (arr[i, 3] -  price[i])/ std)
            add(arr, i, 6, (price[i] - arr[i,4]) / std)
        
        # trima
        add(arr, i, 7, np.multiply(period_prices, trima_weights).sum())

        # rsi
        if i == period:
            avg_up = ar([max(0, price[i-j] - price[i-j-1]) for j in range(0, period)]).mean()
            avg_dn = ar([abs(min(0, price[i-j] - price[i-j-1])) for j in range(0, period)]).mean()

        else:
            avg_up = (arr[i-1, 34] * (period-1) + max(0, price[i] - price[i-1])) / period
            avg_dn = (arr[i-1, 35] * (period-1) + abs(min(0, price[i] - price[i-1]))) / period

        if avg_dn == 0:
            rsi = 100

        else:
            rs = avg_up / avg_dn
            rsi = 100 - (100/(1+rs))

        add(arr, i, 8, rsi)
        add(arr, i, 34, avg_up)
        add(arr, i, 35, avg_dn)

        # willr
        add(arr, i, 9, 100 * (period_prices.max() - price[i]) / (period_prices.max() - period_prices.min()))

        # plus_di, minus_di
        if i == period:
            plus_dm_sum = 0
            minus_dm_sum = 0
            adx = 0
            for j in range(1, period):
                plus_dm, minus_dm = calc_plus_minus_dm(i)
                plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm
                minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm

            tranges = ar([trange(k) for k in range(1, period+1)])
            atr = tranges.mean()
            tr_sum = tranges.sum()
        else:
            plus_dm_sum = arr[i-1, 36]
            minus_dm_sum = arr[i-1, 37]
            atr = arr[i-1, 10]
            tr_sum = arr[i-1, 38]
            adx = arr[i-1, 15]

        true_range = trange(i)
        atr = (atr * (period - 1) + true_range) / period
        tr_sum = tr_sum - (tr_sum / period) + true_range

        plus_dm, minus_dm = calc_plus_minus_dm(i)
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm
        
        plus_di = 100 * plus_dm_sum / tr_sum
        minus_di = 100 * minus_dm_sum / tr_sum
        
        if plus_di + minus_di == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        add(arr, i, 10, atr)
        add(arr, i, 11, true_range)
        add(arr, i, 12, plus_di)
        add(arr, i, 13, minus_di)
        add(arr, i, 14, dx)

        if i < 2*period and i > period: # base case
            adx = arr[period:i, 14].mean()
        else:
            adx = (adx * (period - 1) + dx) / period
        
        add(arr, i, 15, adx)
        add(arr, i, 36, plus_dm_sum)
        add(arr, i, 37, minus_dm_sum)
        add(arr, i, 38, tr_sum)

        # roc
        add(arr, i, 16, 100 * (price[i] - price[i-period])/(price[i-period]))

        # macd
        MACD_FAST_PERIOD = 12
        MACD_SLOW_PERIOD = 26
        MACD_SELF_EMA_PERIOD = 9
        fast = ema(price[i], arr[i-1, 17], MACD_FAST_PERIOD)
        add(arr, i, 17, fast)

        slow = ema(price[i], arr[i-1, 18], MACD_SLOW_PERIOD)
        add(arr, i, 18, slow)

        macd = fast-slow
        add(arr, i, 19, macd)
        
        macd_self_ema = ema(macd, arr[i-1, 20], MACD_SELF_EMA_PERIOD)
        add(arr, i, 20, macd_self_ema)

        histogram_val = macd - macd_self_ema
        add(arr, i, 21, histogram_val)

        # cci
        def tp(k):
            return ar([highs[k], lows[k], price[k]]).mean()
        
        tps = ar([tp(j) for j in range(i-4, i+1)])
        tps_mean = tps.mean()
        mean_dev = ar([np.abs(tps[k] - tps_mean) for k in range(len(tps))]).mean()
        add(arr, i, 22, (tps[-1] - tps_mean)/(.015 * mean_dev))

        # aroon oscillator
        aroon_prices = price[i-period:i+1][::-1]

        num_days_since_max = np.argmax(aroon_prices)
        num_days_since_min = np.argmin(aroon_prices)

        bullish = 100 * (period - num_days_since_max ) / period
        bearish = 100 * (period - num_days_since_min ) / period
        add(arr, i, 23, bullish-bearish)
        

        # chaikin AD
        if (highs[i] - lows[i]) == 0:
            denom = .1
        else:
            denom = highs[i] - lows[i]
        
        mf_mult = ((price[i] - lows[i]) - (highs[i] - price[i])) / (denom)
        mf_vol = mf_mult * volume[i]

        CHAIKIN_OSC_FAST = 3
        CHAIKIN_OSC_SLOW = 10

        adl = arr[i-1, 24] + mf_vol
        add(arr, i, 24, adl)
        add(arr, i, 25, ema(adl, arr[i-1, 25], CHAIKIN_OSC_FAST))
        add(arr, i, 26, ema(adl, arr[i-1, 26], CHAIKIN_OSC_SLOW))
        add(arr, i, 27, arr[i, 25] - arr[i, 26])
        add(arr, i, 28, mf_vol)
        add(arr, i, 29, mf_mult)
        add(arr, i, 30, arr[i-period+1:i+1, 28].sum() / volume[i-period+1:i+1].sum()) # chaikin mf

        # obv
        if price[i-1] < price[i]:
            obv = arr[i-1, 31] + volume[i]
        else:
            obv = arr[i-1, 31] - volume[i]

        add(arr, i, 31, obv)

        # stochastics
        STOCH_PER_D_CONST = 3

        lowest_low = lows[i-period+1:i+1].min()
        highest_high = highs[i-period+1:i+1].max()
        per_k = 100 * (price[i] - lowest_low) / (highest_high - lowest_low)
        add(arr, i, 32, per_k)
        per_d = arr[i-STOCH_PER_D_CONST+1:i+1, 32].mean() 
        add(arr, i, 33, per_d)


    if indicators_in is None:
        return arr
    else:
        return arr[n-offset:]

def sym_to_indicators(sym, prices_path, periods):
    '''Get df with all indicators from prices csv'''
    df, _ = load_overwrite_price(prices_path, sym)

    for period in periods:
        cols = [f'{elt}_{period}' for elt in COLS]
        price, highs, lows, opens, volume = get_f32_cols(df, ['close', 'high', 'low', 'open', 'volume'])        
        arr = get_indicators(price, period, len(cols), highs, lows, opens, volume)
        df2 = pd.DataFrame(arr, columns=cols)
        df = pd.concat([df, df2], axis=1)

    return df

def get_indicators_path(indicators_path, periods):
    s = '_'.join([str(elt) for elt in periods])
    return f'{indicators_path}/indicators_{s}/'

def update(new_prices_path, old_prices_path, indicators_path, periods, sym_subset=[], col_mapping=None, download_prices=None):
    ''' Update, return all indicators'''
    remove_icon_files(old_prices_path)
    sym_subset = update_sym_subset(old_prices_path, sym_subset)
    if download_prices is not None:
        download_prices(new_prices_path, old_prices_path, sym_subset=sym_subset)
    update_prices(new_prices_path, old_prices_path, sym_subset=sym_subset, col_mapping=col_mapping)
    update_indicators(old_prices_path, indicators_path, periods, sym_subset=sym_subset, first_write=False)
    dfs = load_indicators(indicators_path, periods, sym_subset=sym_subset)
    for df in dfs:
        assert len(check_for_nans(df)) == 0, 'There is a nan in one of your dataframes.  Something went wrong with the update.'
    return dfs