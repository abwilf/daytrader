import os, time, math, requests, shutil, glob, threading, pathlib
import numpy as np
import pandas as pd
from datetime import datetime, date
import tensorflow as tf

pd.options.mode.chained_assignment = None

def remove_icon_files(this_dir):
    [os.remove(x) for x in glob.iglob(f'{this_dir}/**/' + 'Icon*', recursive=True)]

def f32(col):
    return col.to_numpy(dtype='float32')

def get_f32_cols(df, cols):
    return [f32(df[col]) for col in cols]

def normalize(arr):
    arr = ar(arr)
    m = np.mean(arr)
    s = np.std(arr)
    arr = (arr-m) / s
    return arr

def hi(colab=False):
    if colab:
        from google.colab import output
        output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/0d/Es_Argentina.ogg").play()')
    else:
        os.system('say "all done dog"')

def ar(x):
    return np.array(x)

def subset(arr1, arr2):
    ''' arr1 is subset of arr2 '''
    return min([elt in arr2 for elt in arr1]) > 0

def df_to_csv(df, filename, overwrite=False, quiet=False):
    s = df.to_csv()
    if os.path.exists(filename):
        if overwrite:
            os.remove(filename)
        else:
            if not quiet:
                print(f'File already existed.  Did not overwrite {filename}.  Please use overwrite flag if you want to overwrite this file.')
            return
    with open(filename, 'w') as f:
        f.write(s)
    if not quiet: print(f'Wrote df to {filename}')

def timestamp_to_iso(timestamp):
    return datetime.fromtimestamp(timestamp).date().isoformat()

def iso_to_timestamp(iso):
    return time.mktime(datetime.strptime(iso, "%Y-%m-%d").timetuple())

def testing_df():
    return pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], columns=['a', 'b', 'c'])

def check_for_nans(df):
    # return columns that have nans in them
    cols = []
    for column in df.columns:
        if column != 'date' and max(np.isnan(df[column].to_numpy())) > .5:
            cols.append(column)
    return cols


def read_csv(filename):
    return pd.read_csv(filename).drop('Unnamed: 0', axis=1)


def update_sym_subset(prices_path, sym_subset):
    if len(sym_subset) == 0:
        sym_subset = csv_names_to_syms(prices_path)
    return sym_subset

def csv_names_to_syms(path):
    names = filter(lambda elt: 'Icon' not in elt, os.listdir(path))
    return [elt.replace('.csv', '').upper() for elt in names]

def load_df(path, sym):
    return load_dfs(path, [sym])[0]

def load_dfs(path, syms, quiet=True):
    if not quiet: 
        print(f'Loading syms from path: \n{path}\n{syms}')
    dfs = [None]*len(syms)
    for i, sym in enumerate(syms):
        dfs[i] = read_csv(f'{path}{sym}.csv').drop('openint', axis=1, errors='ignore')
    return dfs

def drop_nonperiod(df, regex):
    ''' Drop duplicated columns not dependent on period - e.g. trange & obv'''
    all_cols = df.filter(regex=regex).columns
    first_col = all_cols[0]
    col_name = '_'.join(first_col.split('_')[:-1])
    df[col_name] = df[first_col]
    return df.drop(columns=all_cols)

def load_indicators(indicators_path, periods, sym_subset=[], predict_today=False):
    '''Load indicator dfs from periods. This function expects a directory indicators_<periods> to exist in csvs/, and will print an error else'''
    if not os.path.exists(indicators_path):
        print('Path to indicators folder does not exist.  use update_indicators to create this path if you wish to use it')
        return None
    
    sym_subset = update_sym_subset(indicators_path, sym_subset)
    dfs = load_dfs(indicators_path, sym_subset)

    print('\nLoading indicator dataframes...')
    progbar = tf.keras.utils.Progbar(len(dfs))
    for i, df in enumerate(dfs):
        dfs[i] = df.drop(columns=df.filter(regex='^_')) # all columns important only to internal execution - e.g. _adl_ema_10
        for regex in ['^trange', '^obv', '^adl', '^ichi_dist_day', '^ichi_dist_week', '^ichi_dist_month', '^label', '^weight', '^return']:
            dfs[i] = drop_nonperiod(dfs[i], regex) 
    
        # drop all rows where invalid predictions
        if not predict_today:
            for j in range(len(df)):
                if -1 in df.loc[j].to_numpy():
                    dfs[i].drop(j, inplace=True)
                
        progbar.add(1)
    
    if predict_today:
        return [df.tail(1) for df in dfs]
    else:
        return dfs