import pandas as pd, matplotlib.pyplot as plt, re, os, numpy as np, joblib, fs_gcsfs
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from google.cloud import storage


def read_src_file_as_df(dir_name, src_filename, verbose=True):
    """Reads the `src_filename` as a pandas df if already exists, otherwise attempts to unzip it first.

    Args:
        dir_name (str): Source Directory name
        src_filename (str): Target file name.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame
    """
    abs_srcfilename = dir_name + src_filename
    if verbose:  print(f'Trying to read {abs_srcfilename}')

    # fs = gcsfs.GCSFileSystem(project='I535-Final-Project')
    # with fs.open(abs_srcfilename) as f:
    #     # df = pd.read_csv(f)
    #     lines = f.readlines()
    #     print(lines)
    #     df = pd.read_json(f, lines=True)
    
    df = pd.read_json(abs_srcfilename, lines=True)
    if verbose:  print(df.info())
    
    return df



def write_data_to_cloud_storage(df, bucket_name='my-bucket-name', tgt_filename='abs-file-path', verbose=True):
    """Write dataframe to remote storage on GCP.

    Args:
        df (pd.DataFrame): Input dataframe.
        bucket_name (str, optional): Google-Cloud-Storage bucket name. Defaults to 'my-bucket-name'.
        tgt_filename (str, optional): Absolute file path for target location. Defaults to 'abs-file-path'.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to False.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    if verbose:  print(f'Uploading df={df.shape} into {bucket_name}/{tgt_filename}')
    bucket.blob(tgt_filename).upload_from_string(df.to_csv(), 'text/csv')





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
        pd.DataFrame: Dataframe with dropped out columns that contained only 1 value.
    """
    nvalue_counts_df = df.nunique()
    columns_to_drop = nvalue_counts_df.index[nvalue_counts_df.values==1]
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


# Applies the lag function to input dataframe's `colname` within each `customerId` partition.
get_shifted_column = lambda df, colname : df.groupby(by=['customerId'], as_index=False)[[colname]].shift(-1)



def get_reversals_report(main_df):
    """Generates a report of reversal transactions that occured within the input dataset.
    First sorts by ['customerId','merchantName','transactionAmount','transactionDtDate'].
    Now apply the lag function to ['transactionAmount','transactionDtDate','merchantName','transactionType','isFraud'] to get the next transaction that this customer did.
    Computes the duration between transactions for each customer using these newly created columns.
    If the first transaction was a 'PURCHASE' and the next was 'REVERSAL', with same merchant-name and transaction-amount, only then report out these as Reversal-transactions.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.

    Returns:
        pd.DataFrame: Final Report-Dataframe containing my interpretation of Reversal-transactions.
    """
    REVERSAL_COLS_TO_VIEW = ['customerId','transactionDtDate','transactionAmount','merchantName','transactionType','cardPresent','isFraud']
    COLNAMES_T1_VS_T2 = {
        'transactionAmount' : 'nextTransAmt',
        'transactionDtDate' : 'nextTransDate',
        'merchantName' : 'nextMerchantName',
        'transactionType' : 'nextTransType',
        'isFraud' : 'nextIsFraud'
    }
    reversals_df = main_df[REVERSAL_COLS_TO_VIEW].sort_values(by=['customerId','merchantName','transactionAmount','transactionDtDate'])
    for col, newcol in COLNAMES_T1_VS_T2.items():
        reversals_df[newcol] = get_shifted_column(reversals_df, col)
    reversals_df['durationBetweenTrans'] = reversals_df[COLNAMES_T1_VS_T2['transactionDtDate']] - reversals_df['transactionDtDate']
    reversals_df['durationBetweenTrans'] = (reversals_df['durationBetweenTrans'] / np.timedelta64(1, 'D')).convert_dtypes('int')
    report_reversals_df = reversals_df[
        (reversals_df['merchantName']==reversals_df[COLNAMES_T1_VS_T2['merchantName']])
        &
        (reversals_df['transactionType']=='PURCHASE') & (reversals_df[COLNAMES_T1_VS_T2['transactionType']]=='REVERSAL')
        &
        (reversals_df['transactionAmount']==reversals_df[COLNAMES_T1_VS_T2['transactionAmount']])
    ]
    return report_reversals_df





def get_multiswipe_transactions(main_df):
    """Assuming a Multiswipe transaction has no `Reversal` and card was presented at POS, I'll assume such a transaction occurs more than once on the same day.
    We'll just groupby the number of transactions by a customer (identified by merchantName, date), and look at the number of transactionAmounts occured.
    If transactions occured at the same place on the same day for the same customer, and same amount, but multiple times, then it should show up on our report.
    Easier to think in SQL?
    ```sql
    SELECT ACC, MERCHANT WHERE [ACC,MERCHANT] TRANSACTION_TYPE!="REVERSAL" AND SAME TRANSACTIONAMOUNT ON SAME DAY
    ```

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.

    Returns:
        pd.DataFrame: Final Report-Dataframe containing my interpretation of Multiswipe-transactions.
    """
    transactions_without_reversals = main_df[(main_df['transactionType']!='REVERSAL') & main_df['cardPresent']==True]
    transactions_without_reversals = transactions_without_reversals.groupby(
        by=['customerId','merchantName','transactionDtDate'], as_index=False).agg(
            { 'transactionAmount': ['nunique', 'count'] }
        )
    
    transactions_without_reversals.columns = ['customerId','merchantName','transactionDate','transAmtNunique','transAmtCnt']
    multiswipes_df = transactions_without_reversals[
        (transactions_without_reversals['transAmtNunique']==1) & 
        (transactions_without_reversals['transAmtCnt']>1)
        ].sort_values(by=['customerId','merchantName','transactionDate'])
    
    return multiswipes_df




def display_class_imbalance(main_df, label_col='isFraud'):
    """Just logs the amount of fraudulent cases within our dataset of transactions.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        label_col (str, optional): Target-column for classification task. Defaults to 'isFraud'.
    """
    count_frauds = main_df[label_col].value_counts()
    print(f'The dataset contains {round(count_frauds[True]/count_frauds[False]*100,2)}% of fraudulent cases.')





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




# Cleans the merchant-name string
get_clean_merchant_name = lambda name: re.sub(r' #.*', '', name).replace('.com', '')
	
	
def encode_categorical_cols(main_df, model_path=None, verbose=True):
    """Encodes the categorical columns using Ordinal encoding.
    Applies custom ordinal encoder for `merchantName` since it has >30 basenames, even after cleaning the strings.
    Using ordinal encoder rather than one-hot-encoding since even a 5-valued categorical feature requires 55GB!

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.

    Returns:
        pd.DataFrame, dict: Final dataframe with ordinal encodings for categorical features, and the encoding-transformers used for each column.
    """
    categorical_cols = set(main_df.select_dtypes(include='object').columns)
    categorical_cols -= set(['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange', 'currentExpDate'])

    abs_model_path = f'{model_path}/categorical_cols_encoders.pickle'
    if model_path and os.path.exists(abs_model_path):
        if verbose:  print(f'Using the MinMaxScaler() objects available in the pickle file at: {abs_model_path}')
        categorical_cols_encoders = joblib.load(abs_model_path)
    else:
        categorical_cols_encoders = {c : None for c in categorical_cols}
    

    nvalues_categorical_df = main_df[categorical_cols].nunique()
    cols_to_encode = list(nvalues_categorical_df.index[nvalues_categorical_df.values<=20])
    for c in cols_to_encode[::-1]:
        if verbose:  print(f'Ordinal-encoding {c} instead of OHE...')
        if not categorical_cols_encoders[c]:
            if verbose:  print(f'Not using one-hot-encoding for "{c}" since even a 5-valued categorical feature requires 55GB!')
            categorical_cols_encoders[c] = OrdinalEncoder().fit(main_df[c].values.reshape(-1,1))
        main_df[c] = categorical_cols_encoders[c].transform(main_df[c].values.reshape(-1,1))


    # Cleaning the 'Merchant name' column first to reduce 2,400 categories into 200 categories
    c = 'merchantName'
    if verbose:  print(f'Ordinal-encoding {c} instead of OHE...')
    main_df[c] = main_df[c].apply(get_clean_merchant_name)
    if not categorical_cols_encoders[c]:
        if verbose:  print(f'Not using one-hot-encoding for "{c}" since even a 5-valued categorical feature requires 55GB!')
        categorical_cols_encoders[c] = OrdinalEncoder().fit(main_df[c].values.reshape(-1,1))
    main_df[c] = categorical_cols_encoders[c].transform(main_df[c].values.reshape(-1,1))

    if model_path and not os.path.exists(abs_model_path):
        if verbose:  print(f'Caching the scaling-transformation objects at: {abs_model_path}...')
        joblib.dump(categorical_cols_encoders, abs_model_path)

    return main_df, categorical_cols_encoders




def drop_irrelevant_columns(main_df, verbose=True):
    """Drop the non-numeric columns since we've standardized all our features by now.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe containing ready-to-use feature columns.
    """
    cols_to_drop = list(main_df.select_dtypes(exclude=['int64','float64','int32']).columns)
    if verbose:  print(f'Dropping {cols_to_drop}')
    return main_df.drop(cols_to_drop, axis=1)





def scaledown_numerical_cols(main_df, numerical_cols=['creditLimit', 'availableMoney'], model_path=None, verbose=True):
    """Use MinMaxScaler for each of the input specified `numerical_columns`.
    If `model_path` specified and pickle file of the scalers already exists, then recreate the scaler transformation objects.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        numerical_cols (list): List of numerical columns that we need to scale into standard range.
        model_path (str): Example path = '../models/numerical_col_scalers.pickle'. Defaults to None to avoid caching the models.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame, dict: Final dataframe with ordinal encodings for categorical features, and the encoding-transformers used for each column.
    """
    abs_model_path = f'{model_path}/numerical_col_scalers.pickle'
    
    if model_path and os.path.exists(abs_model_path):
        if verbose:  print(f'Using the MinMaxScaler() objects available in the pickle file at: {abs_model_path}')
        numerical_col_scalers = joblib.load(abs_model_path)
    else:
        numerical_col_scalers = {c : None for c in numerical_cols}

    for c in numerical_cols:
        if verbose:  print(f'Scaling down column `{c}`...')
        if not numerical_col_scalers[c]:
            if verbose:  print(f'Training a MinMaxScaler() for column `{c}`')
            numerical_col_scalers[c] = MinMaxScaler().fit(main_df[[c]])
        main_df[c] = numerical_col_scalers[c].transform(main_df[[c]])

    if model_path and not os.path.exists(abs_model_path):
        if verbose:  print(f'Caching the scaling-transformation objects at: {abs_model_path}...')
        joblib.dump(numerical_col_scalers, abs_model_path)
    return main_df, numerical_col_scalers






def print_neat_metrics(expected, preds, model_name='Random-Forest', show_confusion_matrix=True):
    """Print all classification metrics in a pretty format for analysis.

    Args:
        expected (np.array): Expected output.
        preds (np.array): Predictions.
        model_name (str, optional): Model Description. Defaults to 'Random-Forest'.
        show_confusion_matrix (bool, optional): Flag to indicate confusion-matrix logging in verbose mode. Defaults to True.
    """
    print(f'Precision = {round(100*precision_score(expected, preds),2)}%.')
    print(f'Recall = {round(100*recall_score(expected, preds),2)}%.')
    print(f'Accuracy = {round(100*accuracy_score(expected, preds),2)}%.')
    print(f'F1 score = {round(100*f1_score(expected, preds),2)}%.')
    
    if show_confusion_matrix:
        tn, fp, fn, tp = confusion_matrix(expected, preds).ravel()
        print(f'Confusion matrix for {model_name} model = \n\tTP={tp}  FN={fn}\n\tFP={fp}  TN={tn}\n')








def train_random_forest_classifier(train_features, train_labels, param_grid, model_path=None):
    """Trains a Random-Forest classifier using GridSearch with K-Fold Cross-validation of 10 folds.
    If `model_path!=None`, then stores the model as a pickle file at the specified path.

    Args:
        train_features (np.array): Numpy array of numerical train-features.
        train_labels (np.array): Numpy array of numerical train-labels.
        param_grid (dict): Parameter-grid to be used for the GridSearchCV module.
        model_path (str): Example path = '../models/random_forest_classifier.pickle'. Defaults to None to avoid caching the model.

    Returns:
        GridSearchCV: Output of the GridSearchCV module.
    """
    strat_kfold_cv = StratifiedKFold(n_splits=10, shuffle=True)
    random_forest = RandomForestClassifier()
    cv_output = GridSearchCV(random_forest, param_grid=param_grid, cv=strat_kfold_cv, verbose=1).fit(train_features, train_labels)
    if model_path:
        joblib.dump(cv_output, f'{model_path}/random_forest_classifier.pickle')
    return cv_output




def predict_random_forest_classifier(test_features, model_path='models'):
    """Reads the previously-cached model at the specified path, recreates it, and predicts on the test_features.

    Args:
        test_features (np.array): Numpy array of numerical test-features.
        model_path (str, optional): Path to read for retrieving the cached model. Defaults to 'models'.
    """
    new_random_forest = joblib.load(f'{model_path}/random_forest_classifier.pickle')
    preds = new_random_forest.predict(test_features)
    return preds




def get_random_forest_feature_importances(gridsearch_model, feature_cols):
    """Creates a dataframe of features-importances from the GridSearchCV module.

    Args:
        gridsearch_model (GridSearchCV): Output of the GridSearchCV module.
        feature_cols (list): Column names of the features dataframe.

    Returns:
        pd.DataFrame: Output dataframe containing sorted feature-importances.
    """
    importances = pd.DataFrame(gridsearch_model.best_estimator_.feature_importances_)
    importances.index = feature_cols
    importances = importances.sort_values(by=0)
    return importances