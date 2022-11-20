import pandas as pd, re, numpy as np
# from google.cloud import storage


def read_src_file_as_df(dir_name, src_filename, verbose=True):
    """Reads the `src_filename` as a pandas df if exists on GCS storage.

    Args:
        dir_name (str): Source Directory name
        src_filename (str): Target file name.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame
    """
    abs_srcfilename = dir_name + src_filename
    if verbose:  print(f'Trying to read {abs_srcfilename}')
    
    try:
        df = pd.read_csv(abs_srcfilename) if '.csv' in abs_srcfilename else pd.read_json(abs_srcfilename, lines=True)
    except Exception as e:
        print(f'Are you sure {abs_srcfilename} exists? {e}')
    if verbose:  print(df.info())
    
    return df



def levenshtein_distance(s1, s2):
    """Computes the Levenshtein distance between 2 strings: Minimum no. of character substitutions/additions/removals.

    Args:
        s1 (str): String 1
        s2 (str): String 2

    Returns:
        float: Distance between 2 strings
    """
    r = len(s1)+1
    c = len(s2)+1
    distance = np.zeros((r,c),dtype = int)

    for i in range(1, r):
        for k in range(1,c):
            distance[i][0] = i
            distance[0][k] = k
            
    for col in range(1, c):
        for row in range(1, r):
            if s1[row-1]==s2[col-1]:
                cost = 0
            else:
                cost = 2
            distance[row][col] = min(
                distance[row-1][col] + 1,         # Cost of deletions
                distance[row][col-1] + 1,         # Cost of insertions
                distance[row-1][col-1] + cost     # Cost of substitutions
            )
    return distance[row][col] / (len(s1)+len(s2))



def add_time_dependent_features(df, col, include_time_fields=False):
    """For the input datetime column, engineer new features. Also drops out the original column after creating these new features.

    Args:
        df (pd.DataFrame): Input dataframe containing the `col` datetime-format
        col (str): Column name within df.
        include_time_fields (bool, optional): Flag to indicate if we want to produce new hour & minute features. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with new engineered features, but original column dropped out.
    """
    df[col+'Year'] = df[col].dt.year
    df[col+'Month'] = df[col].dt.month
    df[col+'Day'] = df[col].dt.day
    df[col+'DayOfWeek'] = df[col].dt.weekday
    df[col+'Date'] = df[col].dt.date
    if include_time_fields:
        df[col+'Hour'] = df[col].dt.hour
        df[col+'Min'] = df[col].dt.minute
    df = df.drop(col, axis=1)
    return df




def drop_unary_columns(df, verbose=False):
    """Drop the features with just one value. Won't add any value to the model.

    Args:
        df (pd.DataFrame): Input Dataframe of features.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with dropped out columns that contained only <=1 value.
    """
    nvalue_counts_df = df.nunique()
    columns_to_drop = nvalue_counts_df.index[nvalue_counts_df.values<=1]
    if verbose:  print(f'Columns that will be dropped: {columns_to_drop.values}')
    return df.drop(labels=columns_to_drop, axis=1)



def impute_transactionType(main_df, col='transactionType', replacement='UNK', verbose=True):
    """Imputes the cells with just whitespaces, with a replacement string.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        col (str): Column name that we want to impute. Defaults to 'transactionType'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with imputed `transactionType`.
    """
    main_df[col] = main_df[col].replace(r'^\s*$', replacement, regex=True)
    if verbose:  print(main_df[['transactionAmount','transactionType']].groupby(by=['transactionType'], as_index=False).agg({'transactionAmount':['count','nunique','min','max']}))
    return main_df




def convert_boolean_to_int(main_df, verbose=True):
    """Converts all boolean type features into integers.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame: Final Dataframe containing integers instead of booleans.
    """
    boolean_columns = main_df.select_dtypes(include='bool').columns
    for c in boolean_columns:
        main_df[c] = main_df[c].replace(True, 1).replace(False, 0)
    
    if verbose:  print(main_df[['cardPresent','expirationDateKeyInMatch','isFraud']].value_counts())
    return main_df





def drop_irrelevant_columns(main_df, verbose=True):
    """Drop the non-numeric columns since we've standardized all our features by now.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe containing ready-to-use feature columns.
    """
    for c in main_df.columns:
        mode = main_df[c].mode().values[0]
        main_df[c] = main_df[c].fillna(value=mode)
    if verbose:  print('Imputed NaN values with mode of each column.')
    cols_to_drop = list(main_df.select_dtypes(exclude=['int64','float64','int32']).columns)
    if verbose:  print(f'Dropping {cols_to_drop}')
    return main_df.drop(cols_to_drop, axis=1)