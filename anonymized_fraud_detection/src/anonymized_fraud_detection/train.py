import pandas as pd

from sklearn.model_selection import train_test_split
from anonymized_fraud_detection.utilities.util_functions import read_src_file_as_df, levenshtein_distance, add_time_dependent_features, drop_unary_columns, impute_transactionType
from anonymized_fraud_detection.utilities.util_functions import get_reversals_report, get_multiswipe_transactions, display_class_imbalance, convert_boolean_to_int
from anonymized_fraud_detection.utilities.util_functions import encode_categorical_cols, drop_irrelevant_columns, scaledown_numerical_cols, print_neat_metrics
from anonymized_fraud_detection.utilities.util_functions import train_random_forest_classifier


def run_train_pipeline(SRC_DIR_NAME, SRC_FILE_NAME, MODEL_PATH, PARAM_GRID, TGT_DIR_NAME, VERBOSE=True):
    """Uses the following series of functions:
    ```python
        1. read_src_file_as_df()- The Airflow task reads the “Train/Test_transactions.csv” file from GCS, and converts it into a dataframe.
        2. add_time_dependent_features()- For the input datetime columns, engineer new date and time features. Also drop out the original column.
        3. levenshtein_distance()- Rather than just binary non-equality, engineer a score of mismatch for enteredCVV vs cardCVV: a single digit wrong might be just a minor blunder.
        4. drop_unary_columns()- Filter out features with <=1 unique values.
        5. get_reversals_report()- Create a report of reversal transactions.
        6. get_multiswipe_transactions()- Create a report of multiswipe transactions.
        7. convert_boolean_to_int()- Convert Boolean-columns in the data to integer encodings.
        8. encode_categorical_cols()- Apply Ordinal-Encoding to each of categorical-columns and keep track of the transformations, ie- store the sklearn models as pickle objects on GCS for use during prediction.
        9. scaledown_numerical_cols()- Apply MinMax-Scaling to each of numerical-columns and keep track of the transformations, ie- store the sklearn models as pickle objects on GCS for use during prediction.
        10. drop_irrelevant_columns()- Filters out the non-numerical columns from data.
        11. train_random_forest_classifier()- Apply grid-search-cv for a random-forest-classifier and store it on GCS if model-path provided.
    ```


    Args:
        SRC_DIR_NAME (str): Source directory name
        SRC_FILE_NAME (str): Source file name name
        MODEL_PATH (str): Directory/GCS-bucketname where models should be stored
        TGT_DIR_NAME (str): Directory/GCS-bucketname where reports should be stored
        VERBOSE (bool, optional): Flag to indicate logging in verbose mode. Defaults to True.
    """
    
    df = read_src_file_as_df(dir_name=SRC_DIR_NAME, src_filename=SRC_FILE_NAME, verbose=VERBOSE)
    print(f'Successfully read the remote source-file {SRC_FILE_NAME} and created a dataframe of shape df={df.shape}')

    # Engineer date-time features from the input datetime column and drop the original column
    df['transactionDt'] = pd.to_datetime(df['transactionDateTime'])
    df = add_time_dependent_features(df, 'transactionDt', True)

    df['accountOpenDt'] = pd.to_datetime(df['accountOpenDate'])
    df = add_time_dependent_features(df, 'accountOpenDt')

    df['lastAddressChangeDt'] = pd.to_datetime(df['dateOfLastAddressChange'])
    df = add_time_dependent_features(df, 'lastAddressChangeDt')


    # Rather than just binary non-equality, check score of mismatch: a single digit wrong might be just a minor blunder
    df = df.astype({'cardCVV':'str', 'enteredCVV':'str'})
    df['cvvMismatchScore'] = df.apply(lambda x : levenshtein_distance(x['cardCVV'], x['enteredCVV']), axis=1)


    # Some basic cleansing/preprocessing logic
    main_df = df.copy(deep=True)
    main_df = drop_unary_columns(main_df, verbose=VERBOSE)
    main_df = impute_transactionType(main_df, verbose=VERBOSE)

    # Fetch the report of reversal-transactions
    report_reversals_df = get_reversals_report(main_df)
    report_reversals_df.to_csv(f'{TGT_DIR_NAME}Overall_Report_Reversals.csv', index=False)

    # Fetch the report of multiswipe-transactions
    multiswipes_df = get_multiswipe_transactions(main_df)
    multiswipes_df.to_csv(f'{TGT_DIR_NAME}Overall_Report_Multiswipes.csv', index=False)


    # Log the class-imbalance within our current dataset
    display_class_imbalance(main_df)

    # Convert the boolean columns to integer-type
    main_df = convert_boolean_to_int(main_df, verbose=VERBOSE)

    # Apply Ordinal-Encoding to each of categorical-column and keep track of the transformations
    main_df, categorical_cols_encoders = encode_categorical_cols(main_df, model_path=MODEL_PATH, verbose=VERBOSE)

    # Apply Scaling to each of numerical-column and keep track of the transformations
    main_df, numerical_col_scalers = scaledown_numerical_cols(main_df, model_path=MODEL_PATH, verbose=VERBOSE)

    # Drop off the irrelevant non-numerical columns now
    main_df = drop_irrelevant_columns(main_df)


    # Create the train-test split of features and labels
    feature_cols = set(main_df.columns) - set(['isFraud', 'cardCVV', 'enteredCVV', 'cardLast4Digits', 'accountNumber', 'customerId'])
    labels = main_df['isFraud']
    features = main_df[feature_cols]
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)

    # Apply grid-search-cv for a random-forest-classifier and cache the model
    model = train_random_forest_classifier(x_train, y_train, param_grid=PARAM_GRID, model_path=MODEL_PATH)

    # Log out the metrics for current model
    predictions = model.predict(x_test)
    print_neat_metrics(expected=y_test, preds=predictions)

    # Create the final set of validation-set-predictions
    validations_df = x_test.copy()
    validations_df['predictions'] = predictions
    validations_df['expected'] = y_train.values
    
    TGT_FILENAME = f'{TGT_DIR_NAME}Random_Forest_Validation_Set_predictions.csv'
    validations_df.to_csv(TGT_FILENAME, index=False)
    print(f'Successfully wrote the final predictions file {TGT_FILENAME}')