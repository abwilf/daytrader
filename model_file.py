import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce
from utils import *

def arrs_equal(alpha, beta):
    return np.all(f32(alpha) == f32(beta))

def split_df(df, train_idx, test_idx):
    train_df = df.copy().loc[train_idx, :].reset_index()
    test_df = df.copy().loc[test_idx, :].reset_index()
    return train_df, test_df

def z(x):
    return np.zeros(x)

def normalize_df(df, d=None):
    '''
    Normalize each column of the df (-mean / std) and return the (mean, std) tuple for each column to be normalized by in testing.  
    d is the map {col_name: (mean, std)} to normalize with if passed in for testing or predicting
    '''
    temp_d = {}
    
    for col in df.columns:
        if d is None:
            if col in ['label', 'return', 'weight']:
                m = 0
                s = 1
            else:
                m = df.loc[:, col].mean()
                s = df.loc[:, col].std()
            temp_d[col] = (m, s)
        else:
            m, s = d[col]
        
        df.loc[:, col] = (df.loc[:, col] - m) / s

    return temp_d

def normalize_dfs(dfs, d):
    for i, _ in enumerate(dfs):
        normalize_df(dfs[i], d)
    
def get_np_arrs(df, regex):
    x = f32(df.filter(regex=regex))
    y = f32(df['label'])
    w = f32(df['weight'])
    return x, y, w

def train(model, df_train, df_test, regex, checkpts_path, num_epochs=51, plot=True, batch_size=32, k=3, threshold=.5):
    if os.path.exists(checkpts_path):
        shutil.rmtree(checkpts_path)
    pathlib.Path(checkpts_path).mkdir(parents=True)
    
    x_train, y_train, w_train = get_np_arrs(df_train, regex)
    x_test, y_test, w_test = get_np_arrs(pd.concat(df_test, axis=0, sort=True), regex)
    
    def perf(model_in):
        return model_perf(df_test, model_in, regex, k=k, just_buy=False, threshold=threshold)

    train_model(model, x_train, y_train, w_train, x_test, y_test, w_test, perf, batch_size, plot, num_epochs, checkpts_path)

def loss(model, x, y, weights, training):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    y_ = model(x, training=training)
    return loss_obj(y, y_, sample_weight=weights)

def grad(model, inputs, targets, weights, training):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, weights, training)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_helper(model, x, y, w, acc_results, loss_results, optimizer, batch_size, training=True):
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_avg = tf.keras.metrics.Mean()

    if np.isnan(x).max() > False:
        df = pd.DataFrame(x)
        for col in df.columns:
            if np.isnan(df[col]).max() > False:
                print('ERROR: column contains nan', col)
                exit()

    n = x.shape[0]
    idxs = np.split(np.arange(n), [elt for elt in range(0, n, batch_size)][1:])
    for idx in idxs:
        x_batch = x[idx]
        y_batch = y[idx]
        w_batch = w[idx]
        loss, grads = grad(model, x_batch, y_batch, w_batch, training)
        if training: 
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        accuracy(y_batch, model(x_batch, training=training), sample_weight=w_batch)
        loss_avg(loss)

    acc_results.append(accuracy.result())
    loss_results.append(loss_avg.result())

def train_model(model, x_train, y_train, w_train, x_test, y_test, w_test, perf, batch_size, plot, num_epochs, checkpts_path):
    '''
    perf: function mapping from (model) to performance over testing window - actually runs algorithm and sees what percentage it would have made
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    train_acc_results = []
    train_loss_results = []
    
    test_acc_results = []
    test_loss_results = []
    
    perf_results = []
    date_a = time.time()
    print(f'Saving epoch checkpoints to {checkpts_path}')
    print('Epoch\tTrain Loss\tTrain Accuracy\tTest Loss\tTest Accuracy\tPerf Result\tETA (mins)')
    for epoch in range(num_epochs+1):
        train_helper(model, x_train, y_train, w_train, train_acc_results, train_loss_results, optimizer, batch_size, training=True)
        train_helper(model, x_test, y_test, w_test, test_acc_results, test_loss_results, optimizer, batch_size, training=False)
        
        # set eta
        date_b = time.time()
        eta = (date_b - date_a) * (num_epochs - epoch + 1) / 60
        date_a = date_b

        perf_results.append(perf(model))

        if epoch % 5 == 0:
            print("{}/{}\t{:.3f}\t\t{:.3%}\t\t{:.3f}\t\t{:.3%}\t\t{:.2f}\t\t{:.2f}".format(epoch, num_epochs, train_loss_results[epoch], train_acc_results[epoch], test_loss_results[epoch], test_acc_results[epoch], perf_results[epoch], eta if epoch > 0 else 0.0))
    
        filename = f'{checkpts_path}/{epoch}.h5'
        model.save(filename)

    if plot:
        fig, axes = plt.subplots(3, sharex=True, figsize=(12, 8))
        fig.suptitle('Training / Testing Metrics')

        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].plot(train_loss_results, color='blue', label='Train')
        axes[0].plot(test_loss_results, color='red', label='Test')
        axes[0].legend()
        
        axes[1].set_ylabel('Accuracy', fontsize=14)
        axes[1].plot(train_acc_results, color='blue', label='Train')
        axes[1].plot(test_acc_results, color='red', label='Test')
        axes[1].legend()

        axes[2].set_ylabel('Performance', fontsize=14)
        axes[2].set_xlabel('Epoch', fontsize=14)
        axes[2].plot(perf_results, color='blue')
        plt.show()

def all_dates(dfs, asc=True):
    all = reduce(np.union1d, [df.index for df in dfs])
    return ar(sorted(all, reverse=not asc))

def index_date(dfs):
    for i, df in enumerate(dfs):
        if 'date' in df.columns:
            dfs[i] = df.set_index('date')

def split_dfs(dfs, start_date=None, ratio=.8, concat=True, quiet=True):
    '''
    ratio: ratio of train-test to split into
    returns train = one big df with rows concatenated, test = list of dfs to be evaluated using row by row & higher level strategy
    concat: if training RNN, might want these separate (concat=False)
    split based on date: find window starting at start_date - random if not initialized - ending at end_date that splits training data up into 80-20 split
    '''
    index_date(dfs)
    dates = all_dates(dfs, asc=True)
    if start_date == None:
        start_date = np.random.choice(dates[:int(len(dates)*ratio)])
    
    start_date_idx = np.searchsorted(dates, start_date)

    tot = sum([len(df) for df in dfs])
    target = (1-ratio) * tot
    
    for end_date_idx in range(start_date_idx+1, len(dates)):
        tot_captured = sum([len(df[start_date:dates[end_date_idx]]) for df in dfs])
        if tot_captured >= target:
            break
    
    if tot_captured < target:
        latest_start = dates[int(len(dates)*ratio)]
        print(f'ERROR: {start_date} was not early enough to ensure that {ratio} of the datapoints were assigned to training.  Please input a date before roughly {latest_start} (may be slightly later depending on the density of the dataframe points) or leave the field blank and it will be chosen randomly.')
        return None, None

    end_date = dates[end_date_idx]
    if not quiet: 
        print(f'Backtesting from {start_date} to {end_date}')
    
    train = [df.drop(df.loc[start_date:end_date].index) for df in dfs]
    test = [df.loc[start_date:end_date] for df in dfs]
    
    # elif style == 'stock':
    #     indeces = np.arange(len(dfs), dtype=int)
    #     np.random.shuffle(indeces)
    #     idx = int(ratio*len(dfs))
    #     train_indeces = indeces[:idx].astype(int)
    #     test_indeces = indeces[idx:].astype(int)
    #     train = [df for i, df in enumerate(dfs) if i in train_indeces]
    #     test = [df for i, df in enumerate(dfs) if i in test_indeces]
    #     if not quiet:
    #         print(f'Training on stocks with indeces: {train_indeces}\nTesting on stocks with indeces: {test_indeces}')

    if concat:
        train = pd.concat(train, axis=0, sort=True)
    return train, test

def right_join(a, b):
    return ar([elt for elt in b if elt not in a])

def predictions(mod, dates, dfs, regex, k, threshold, just_buy=False):
    '''
    Get predictions model will generate on dfs at given dates, regex to filter columns, top k confident stocks (with confidence), eliminating buy predictions below threshold
    e.g. This array indicates that you should allocate 70% of your money to stock 0 at date t0, 30% to stock 2.  At date t1, you should allocate 20% to stock 0, 80% to stock 2.
    [
    [.7, 0, .3],
    [.2, 0, .8]
    ]
    if just_buy, invest equally in all stocks - no top k members
    '''
    n = len(dates)
    d = len(dfs)

    k = min(k,d) # can't choose top 3 stocks if there are only 2 stocks to choose from
    arr = z((n, d))
    dates_dict = {elt: i for i, elt in enumerate(dates)}

    # calc confidences - only where the df is valid at that date
    for i, df in enumerate(dfs):
        if just_buy:
            preds = np.ones((len(df),2))
        else:
            x = f32(df.filter(regex=regex))
            preds = mod(x)

        df_dates = dfs[i].index
        df_dates_idxs = np.zeros(len(df_dates), dtype=int)
        for j, date in enumerate(df_dates):
            df_dates_idxs[j] = dates_dict[date]

        arr[df_dates_idxs, i] = preds[:, 1]

    # zero out all non k-top members, turn into prob distribution of confidences    
    for i in range(n):
        if not just_buy:
            for j in range(d):
                if arr[i, j] < threshold or np.isnan(arr[i,j]):
                    arr[i,j] = 0
            k_top_idxs = np.argpartition(arr[i,:], -k)[-k:]

            # zero out other idxs
            rj = right_join(k_top_idxs, range(d))
            if len(rj) > 0:
                arr[i, rj] = 0

        s = sum(arr[i, :])
        arr[i, :] = arr[i, :] / (s if s != 0 else 1)
    
    # calculated weighted return for each date
    weighted_return = z(n)
    for i in range(n):
        weighted_return[i] = sum([0 if arr[i,j]==0 else dfs[j].loc[dates[i], 'return']*arr[i,j] for j in range(d)])
    
    return arr, np.product(weighted_return[weighted_return != 0]) # don't include where weighted_return==0 (algorithm did not choose any stocks)

def get_date(date_str):
    date_str = date_str.split('-')
    return date(int(date_str[0]), int(date_str[1]), int(date_str[2]))

def annual_return(tot_return, num_years, rounded=True):
    num = ((tot_return ** (1/num_years)) - 1) * 100
    return np.round(num, decimals=2) if rounded else num

sp_df = None
def sp_perf(df_test, prices_path, rounded=True):
    '''Calculates return from just buying and holding the s&p500 over the period in df_test'''
    dates, num_years = get_dates_yrs(df_test)
    global sp_df

    if sp_df is None:
        print('Loading GSPC dataframe')
        sp_df = load_df(prices_path, 'GSPC')
    

    sp_dates = sp_df['date'].to_numpy()
    min_date = sp_dates[np.searchsorted(sp_dates, dates[0])]
    max_date = sp_dates[np.searchsorted(sp_dates, dates[-1])]

    price_0 = sp_df.loc[sp_df['date'] == min_date]['close'].to_numpy()
    assert len(price_0) == 1, f'S&P prices csv has 0 or multiple dates for {min_date}'
    price_n = sp_df.loc[sp_df['date'] == max_date]['close'].to_numpy()
    assert len(price_n) == 1, f'S&P prices csv has 0 or multiple dates for {max_date}'

    price_0 = price_0[0]
    price_n = price_n[0]
    ret = (price_n-price_0)/price_0 + 1
    
    return annual_return(ret, num_years, rounded=rounded)

def get_dates_yrs(df_test):
    dates = all_dates(df_test, asc=True)
    first_date = get_date(dates[0])
    last_date = get_date(dates[-1])
    num_years = float((last_date - first_date).days) / 365
    return dates, num_years

def model_perf(df_test, mod, regex, k=3, threshold=.5, just_buy=False, minus_sp=False, prices_path=None, rounded=True):
    ''' 
    just_buy: calcualtes the return from allocating your money evenly across the stocks in df_test, holding over the duration

    Get avg annual return of strategy involving model, k, and threshold.  
        E.g if you ended up with 1.8x your initial over 3 years, 1 * x**3 = 1.8, x = 1.216 -> 21.6 percent annual return.  
    '''
    dates, num_years = get_dates_yrs(df_test)
    
    _, tot_return = predictions(mod, dates, df_test, regex, k, threshold, just_buy=just_buy)
    to_ret = annual_return(tot_return, num_years, rounded=False)
    
    if minus_sp:
        num = to_ret - sp_perf(df_test, prices_path, rounded=False)
        return np.round(num, decimals=2) if rounded else num
    else:
        return to_ret

def apply_scalars(bullish_scalar, dfs):
    '''
    bullish_scalar: weight mistakes of selling when you should have bought (penalty for not being bullish enough) (FN).  If 0 < weight < 1, weight mistake of buying when you should have sold (FP) more
    '''
    def apply_scalar_helper(df): # to get around settingwithcopy warning
        if len(df) > 0:
            df.loc[:, 'weight'] = df.apply(lambda row: row['weight'] if row['label'] < .5 else bullish_scalar*row['weight'], axis=1)

    if type(dfs) is not list:
        apply_scalar_helper(dfs)
    
    else:
        for i, _ in enumerate(dfs):
            apply_scalar_helper(dfs[i])

def save_model(m, models_path, name=None):
    pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)
    if name == None:
        name = len(os.listdir(models_path)) + 1
    filename = f'{models_path}/{name}.h5'
    m.save(filename)
    print(f'Saved model to {filename}')
    return filename

def load_model(path, name):
    model_path = path if name is None else f'{path}/{name}.h5'
    print(f'Loading model from {model_path}')
    return tf.keras.models.load_model(model_path, compile=False)

def get_today_preds(model, indicators_path, periods, sym_subset, norm_params, regex, k, threshold, bullish_scalar, quiet=False):
    dfs_today = load_indicators(indicators_path, periods, sym_subset=sym_subset, predict_today=True)
    index_date(dfs_today)
    apply_scalars(bullish_scalar, dfs_today)
    normalize_dfs(dfs_today, norm_params)
    dates = [dfs_today[0].index[0]]
    preds, _ = predictions(model, dates, dfs_today, regex, k, threshold)
    to_ret = list(zip(sym_subset, preds[0]))
    if not quiet:
        dashes = 27*'-'
        print(f'\n\nPredictions for {dates[0]}\n{dashes}\nStock\tProportion of Funds')
        [print(f'{elt[0]}\t{np.round(elt[1], decimals=3)}') for elt in to_ret]
    return to_ret