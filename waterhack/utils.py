import re
import pandas as pd


def find_col(df, regex_str):
    """Return column names that satify regex d condition

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to folter columns
    regex_str : str
        regex condition

    Returns
    -------
    list
        list of column names that satisfy condition
    """
    r = re.compile(regex_str)
    mylist = df.columns.to_list()
    return list(filter(r.findall, mylist))


def series_to_supervised(data_in, lagged_cols, n_in=1, n_out=1, dropnan=True):
    """Add lags in the past and future. Used to prepare time series as a supervised ML
    problem

    Parameters
    -----------
    data_in: pd.DataFRame
        pandas DataFrame with input data
    lagged_cols: lisr of str
        list of names of columns that are lagged. The rest of the columns remain the same.
    n_in: int, list of int
        Number of lags to look in the past. Could be either an integer or a list of them.
    n_out: in, list of int
        Number of lags in the future

    Returns
    --------
    agg_all: pd.DataFrame
        Dataset containing lagged columns
    names_in: list of str
         Names of columns lagged in the past
    names_out: list of str
         Names of columns lagged in the future to be forecast.
    """

    data = pd.DataFrame(data_in[lagged_cols].copy())  # columns that will be lagged
    df_notlag = pd.DataFrame(
        data_in.drop(lagged_cols, axis=1)
    )  # Here keep unlagged cols

    # print("Columns that will NOT be lagged: {}".format(df_notlag.columns))
    # print("Columns that will be lagged: {}".format(data.columns))

    # Number of variables to lag
    n_vars = data.shape[1]

    cols, names_in, names_out = list(), list(), list()
    # input sequence (t-n, ... t-1)
    if isinstance(n_in, int):
        for i in range(n_in, 0, -1):
            cols.append(data.shift(i))
            names_in += [("%s(t-%d)" % (list(data)[j], i)) for j in range(n_vars)]

    else:
        # USer-define set of lags
        for i in n_in:
            cols.append(data.shift(i))
            names_in += [("%s(t-%d)" % (list(data)[j], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names_out += [("%s(t)" % (list(data)[j])) for j in range(n_vars)]
        else:
            names_out += [("%s(t+%d)" % (list(data)[j], i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names_in + names_out

    # drop rows with NaN values. (Inevitably happens when lagging)
    if dropnan:
        agg.dropna(inplace=True)

    # Merge with unlagged columns
    agg_all = pd.merge(agg, df_notlag, right_index=True, left_index=True)

    # DONE
    return agg_all, names_in, names_out