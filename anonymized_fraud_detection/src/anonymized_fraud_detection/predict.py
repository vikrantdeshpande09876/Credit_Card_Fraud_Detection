import pandas as pd

from anonymized_fraud_detection.utilities.util_functions import read_src_file_as_df, levenshtein_distance, add_time_dependent_features, drop_unary_columns
from anonymized_fraud_detection.utilities.util_functions import impute_transactionType, convert_boolean_to_int, drop_irrelevant_columns
from anonymized_fraud_detection.utilities.analytical_reporting_functions import get_reversals_report, get_multiswipe_transactions, display_class_imbalance
from anonymized_fraud_detection.utilities.models import encode_categorical_cols, scaledown_numerical_cols, predict_random_forest_classifier


def run_prediction_pipeline(SRC_DIR_NAME, SRC_FILE_NAME, MODEL_PATH, TGT_DIR_NAME, PROJECT_NAME=None, USE_GCS=True, VERBOSE=True):
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
        11. predict_random_forest_classifier()- Reads the previously-cached model at the specified path, recreates it, and predicts on the test_features.
    ```


    Args:
        SRC_DIR_NAME (str): Source directory name
        SRC_FILE_NAME (str): Source file name name
        MODEL_PATH (str): Directory/GCS-bucketname where models should be stored
        TGT_DIR_NAME (str): Directory/GCS-bucketname where reports should be stored
        PROJECT_NAME (str, optional): Google Cloud Project-name. Defaults to None.
        USE_GCS (bool, optional): Flag to configure access to GCP cloud storage and BigQuery. Defaults to True.
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
    report_reversals_df.to_csv(f'{TGT_DIR_NAME}Test_Report_Reversals.csv', index=False)

    # Fetch the report of multiswipe-transactions
    multiswipes_df = get_multiswipe_transactions(main_df)
    multiswipes_df.to_csv(f'{TGT_DIR_NAME}Test_Report_Multiswipes.csv', index=False)


    # Log the class-imbalance within our current dataset
    display_class_imbalance(main_df)

    # Convert the boolean columns to integer-type
    main_df = convert_boolean_to_int(main_df, verbose=VERBOSE)

    # Apply Ordinal-Encoding to each of categorical-column and keep track of the transformations
    categorical_cols = ['transactionType','merchantCategoryCode','merchantCountryCode','merchantName','acqCountry']
    main_df, _ = encode_categorical_cols(main_df, categorical_cols=categorical_cols, model_path=MODEL_PATH, project_name=PROJECT_NAME, use_gcs=USE_GCS, verbose=VERBOSE)

    # Apply Scaling to each of numerical-column and keep track of the transformations
    numerical_cols = ['creditLimit', 'availableMoney']
    main_df, _ = scaledown_numerical_cols(main_df, numerical_cols=numerical_cols, model_path=MODEL_PATH, project_name=PROJECT_NAME, use_gcs=USE_GCS, verbose=VERBOSE)

    # Drop off the irrelevant non-numerical columns now
    main_df = drop_irrelevant_columns(main_df)


    # Create the test features
    feature_cols = set(main_df.columns) - set(['isFraud', 'cardCVV', 'enteredCVV', 'cardLast4Digits', 'accountNumber', 'customerId'])
    test_features = main_df[feature_cols]

    # Create the final set of test-predictions
    preds, model = predict_random_forest_classifier(test_features, model_path=MODEL_PATH, project_name=PROJECT_NAME, use_gcs=USE_GCS, verbose=VERBOSE)
    test_features['predictions'] = preds
    if 'isFraud' in main_df.columns:
        test_features['expected'] = main_df['isFraud'].values

    TGT_FILENAME = f'{TGT_DIR_NAME}Random_Forest_Test_Set_predictions.csv'
    test_features.to_csv(TGT_FILENAME, index=False)
    print(f'Successfully wrote the final predictions file {TGT_FILENAME}')