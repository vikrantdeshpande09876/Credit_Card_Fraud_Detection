import pandas as pd, re, numpy as np, joblib, gcsfs
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from google.cloud import storage




def cache_pickle_object_to_storage(model, project_name=None, tgt_abs_path='foldername/modelname.pickle', use_gcs=True, verbose=True):
    """Opens a connection with Google-Cloud-Storage and writes the Python model-object as a pickle-file, if `project_name!=''` and `use_gcs=True`.
    Otherwise attempts to write to local directory.

    Args:
        model (Any): Can be a dictionary/sklearn-model/Tensorflow-model/etc.
        project_name (str, optional): Google Cloud Project-name. Defaults to None.
        tgt_abs_path (str, optional): Absolute path of the destination pickle file. Defaults to 'foldername/modelname.pickle'.
        use_gcs (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.
    """
    if project_name and use_gcs:
        try:
            fs = gcsfs.GCSFileSystem(project=project_name)
            if verbose:  print(f'Connected with GCSFS: Trying to write the input model to {tgt_abs_path}')
            with fs.open(tgt_abs_path, 'wb') as file:
                joblib.dump(model, file)
            if verbose:  print(f'Successfully cached input model to {tgt_abs_path}')
        except Exception as e:
            print(f'Something went wrong while trying to cache input model to {tgt_abs_path}.\n{e}')
    else:
        try:
            with open(tgt_abs_path, 'wb') as file:
                joblib.dump(model, file)
            if verbose:  print(f'Successfully cached input model to {tgt_abs_path}')
        except Exception as e:
            print(f'Something went wrong while trying to cache input model to {tgt_abs_path}.\n{e}')






def retrieve_cached_model(project_name=None, tgt_abs_path='foldername/modelname.pickle', use_gcs=True, verbose=True):
    """Opens a connection with Google-Cloud-Storage and retrieves the pickle-object if it exists, and if `project_name!=''` and `use_gcs=True`.
    Otherwise attempts to read from local directory.

    Args:
        project_name (str, optional): Google Cloud Project-name. Defaults to None.
        tgt_abs_path (str, optional): Absolutie path of the source pickle file. Defaults to 'foldername/modelname.pickle'.
        use_gcs (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        Any: Can be a dictionary/sklearn-model/Tensorflow-model/etc. depending on the remote pickle-object.
    """
    newmodel = None
    if project_name and use_gcs:
        try:
            fs = gcsfs.GCSFileSystem(project=project_name)
            if verbose:  print(f'Connected with GCSFS: Trying to fetch model at {tgt_abs_path}')
            with fs.open(tgt_abs_path, 'rb') as f:
                newmodel = joblib.load(f)
                print(newmodel)
            if verbose:  print(f'Successfully retrieved cached model at {tgt_abs_path}')
        except Exception as e:
            print(f'Something went wrong. Are you sure the model-pickle file exists at {tgt_abs_path}?: {e}')
    else:
        try:
            with open(tgt_abs_path, 'rb') as f:
                newmodel = joblib.load(f)
            if verbose:  print(f'Successfully retrieved cached model at {tgt_abs_path}: {newmodel}')
        except Exception as e:
            print(f'Something went wrong. Are you sure the model-pickle file exists at {tgt_abs_path}?: {e}')
    return newmodel






# Cleans the merchant-name string
get_clean_merchant_name = lambda name: re.sub(r' #.*', '', name).replace('.com', '')
	
	
def encode_categorical_cols(main_df, categorical_cols=['transactionType','merchantCategoryCode','merchantCountryCode','merchantName','acqCountry'], 
                            model_path=None, project_name=None, use_gcs=True, verbose=True):
    """Encodes the categorical columns using Ordinal encoding.
    Applies custom ordinal encoder for `merchantName` since it has >30 basenames, even after cleaning the strings.
    Using ordinal encoder rather than one-hot-encoding since even a 5-valued categorical feature requires 55GB!

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        categorical_cols (list): List of numerical columns that we need to scale into standard range.
        model_path (str): Example path = '../models/'. Defaults to None to avoid caching the models.
        project_name (str, optional): Google Cloud Project-name. Defaults to None.
        use_gcs (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.


    Returns:
        pd.DataFrame, dict: Final dataframe with ordinal encodings for categorical features, and the encoding-transformers used for each column.
    """
    pickle_filename = 'categorical_cols_encoders.pickle'
    abs_model_path = model_path + pickle_filename
    categorical_cols_encoders = None
    
    if model_path:
        categorical_cols_encoders = retrieve_cached_model(tgt_abs_path=abs_model_path, project_name=project_name, use_gcs=use_gcs, verbose=verbose)
        
    if not categorical_cols_encoders:
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

    if model_path:
        if verbose:  print(f'Caching the encoding-transformation objects at: {abs_model_path}...')
        cache_pickle_object_to_storage(categorical_cols_encoders, tgt_abs_path=abs_model_path, project_name=project_name, use_gcs=use_gcs, verbose=verbose)
    return main_df, categorical_cols_encoders






def scaledown_numerical_cols(main_df, numerical_cols=['creditLimit', 'availableMoney'], model_path=None, project_name=None, use_gcs=True, verbose=True):
    """Use MinMaxScaler for each of the input specified `numerical_columns`.
    If `model_path` specified and pickle file of the scalers already exists, then recreate the scaler transformation objects.

    Args:
        main_df (pd.DataFrame): Input Dataframe of features.
        numerical_cols (list): List of numerical columns that we need to scale into standard range.
        model_path (str): Example path = '../models/'. Defaults to None to avoid caching the models.
        project_name (str, optional): Google Cloud Project-name. Defaults to None.
        use_gcs (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame, dict: Final dataframe with ordinal encodings for categorical features, and the encoding-transformers used for each column.
    """
    abs_model_path = f'{model_path}/numerical_col_scalers.pickle'

    pickle_filename = 'numerical_col_scalers.pickle'
    abs_model_path = model_path + pickle_filename
    numerical_col_scalers = None

    if model_path:
        numerical_col_scalers = retrieve_cached_model(tgt_abs_path=abs_model_path, project_name=project_name, use_gcs=use_gcs, verbose=verbose)
        
    if not numerical_col_scalers:
        numerical_col_scalers = {c : None for c in numerical_cols}
    

    for c in numerical_cols:
        if verbose:  print(f'Scaling down column `{c}`...')
        if not numerical_col_scalers[c]:
            if verbose:  print(f'Training a MinMaxScaler() for column `{c}`')
            numerical_col_scalers[c] = MinMaxScaler().fit(main_df[[c]])
        main_df[c] = numerical_col_scalers[c].transform(main_df[[c]])

    if model_path:
        if verbose:  print(f'Caching the scaling-transformation objects at: {abs_model_path}...')
        cache_pickle_object_to_storage(numerical_col_scalers, tgt_abs_path=abs_model_path, project_name=project_name, use_gcs=use_gcs, verbose=verbose)
    return main_df, numerical_col_scalers







def train_random_forest_classifier(train_features, train_labels, param_grid, model_path=None, project_name=None, use_gcs=True, verbose=True):
    """Trains a Random-Forest classifier using GridSearch with K-Fold Cross-validation of 10 folds.
    If `model_path!=None`, then stores the model as a pickle file at the specified path.

    Args:
        train_features (np.array): Numpy array of numerical train-features.
        train_labels (np.array): Numpy array of numerical train-labels.
        param_grid (dict): Parameter-grid to be used for the GridSearchCV module.
        model_path (str): Example path = '../models/random_forest_classifier.pickle'. Defaults to None to avoid caching the model.
        project_name (str, optional): Google Cloud Project-name. Defaults to None.
        use_gcs (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        GridSearchCV: Output of the GridSearchCV module.
    """
    model_picklefile = 'random_forest_classifier.pickle'
    abs_model_path = model_path + model_picklefile
    strat_kfold_cv = StratifiedKFold(n_splits=10, shuffle=True)
    random_forest = RandomForestClassifier()
    cv_output = GridSearchCV(random_forest, param_grid=param_grid, cv=strat_kfold_cv, verbose=1).fit(train_features, train_labels)
    if model_path:
        cache_pickle_object_to_storage(cv_output, tgt_abs_path=abs_model_path, project_name=project_name, use_gcs=use_gcs, verbose=verbose)
    return cv_output




def predict_random_forest_classifier(test_features, model_path='models/', project_name=None, use_gcs=True, verbose=True):
    """Reads the previously-cached model at the specified path, recreates it, and predicts on the test_features.

    Args:
        test_features (np.array): Numpy array of numerical test-features.
        model_path (str, optional): Path to read for retrieving the cached model. Defaults to 'models/'.
        project_name (str, optional): Google Cloud Project-name. Defaults to None.
        use_gcs (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
        verbose (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.

    Returns:
        pd.DataFrame, model: Predictions dataframe, and the random-forest model retrieved from cache.
    """
    model_picklefile = 'random_forest_classifier.pickle'
    abs_model_path = model_path + model_picklefile
    new_random_forest = retrieve_cached_model(tgt_abs_path=abs_model_path, project_name=project_name, use_gcs=use_gcs, verbose=verbose)
    preds = new_random_forest.predict(test_features)
    return preds, new_random_forest