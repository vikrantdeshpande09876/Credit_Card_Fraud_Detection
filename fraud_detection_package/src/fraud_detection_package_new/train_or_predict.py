import pandas as pd

from sklearn.model_selection import train_test_split
from fraud_detection_package_new.util_functions import read_src_file_as_df, levenshtein_distance, add_time_dependent_features, drop_unary_columns, impute_transactionType
from fraud_detection_package_new.util_functions import get_reversals_report, get_multiswipe_transactions, display_class_imbalance, convert_boolean_to_int
from fraud_detection_package_new.util_functions import encode_categorical_cols, drop_irrelevant_columns, scaledown_numerical_cols, print_neat_metrics
from fraud_detection_package_new.util_functions import train_random_forest_classifier, predict_random_forest_classifier


def run_train_pipeline(SRC_DIR_NAME, TGT_FILE_NAME, MODEL_PATH, PARAM_GRID, TGT_DIR_NAME):
    
    df = read_src_file_as_df(dir_name=SRC_DIR_NAME, src_filename=TGT_FILE_NAME, verbose=True)
    print(f'Successfully read the remote text file and created a dataframe of shape df={df.shape}')

    # Engineer date-time features from the input datetime column and drop the original column
    df['transactionDt'] = pd.to_datetime(df['transactionDateTime'])
    df = add_time_dependent_features(df, 'transactionDt', True)

    df['accountOpenDt'] = pd.to_datetime(df['accountOpenDate'])
    df = add_time_dependent_features(df, 'accountOpenDt')

    df['lastAddressChangeDt'] = pd.to_datetime(df['dateOfLastAddressChange'])
    df = add_time_dependent_features(df, 'lastAddressChangeDt')


    # Rather than just binary non-equality, check score of mismatch: a single digit wrong might be just a minor blunder
    df = df.astype({'cardCVV':'str', 'enteredCVV':'str'})
    df['cvvMismatchScore'] = df.apply(lambda x : levenshtein_distance(x['enteredCVV'], x['enteredCVV']), axis=1)


    # Some basic cleansing/preprocessing logic
    main_df = df.copy(deep=True)
    main_df = drop_unary_columns(main_df, verbose=True)
    main_df = impute_transactionType(main_df, verbose=True)

    # Fetch the report of reversal-transactions
    report_reversals_df = get_reversals_report(main_df)
    report_reversals_df.to_csv(f'{TGT_DIR_NAME}/Report_Reversals.csv', index=False)

    # Fetch the report of multiswipe-transactions
    multiswipes_df = get_multiswipe_transactions(main_df)
    multiswipes_df.to_csv(f'{TGT_DIR_NAME}/Report_Multiswipes.csv', index=False)


    # Log the class-imbalance within our current dataset
    display_class_imbalance(main_df)

    # Convert the boolean columns to integer-type
    main_df = convert_boolean_to_int(main_df, verbose=True)

    # Apply Ordinal-Encoding to each of cat-column and keep track of the transformations
    main_df, categorical_cols_encoders = encode_categorical_cols(main_df, model_path=MODEL_PATH, verbose=True)

    # Apply Scaling to each of cat-column and keep track of the transformations
    main_df, numerical_col_scalers = scaledown_numerical_cols(main_df, model_path=MODEL_PATH, verbose=True)

    # Drop off the irrelevant non-numerical columns now
    main_df = drop_irrelevant_columns(main_df)


    # Create the train-test split of features and labels
    feature_cols = set(main_df.columns) - set(['isFraud', 'cardCVV', 'enteredCVV', 'cardLast4Digits', 'accountNumber', 'customerId'])
    labels = main_df['isFraud']
    features = main_df[feature_cols]
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, stratify=labels)

    print(f'x_train={x_train.info()}')
    print(f'x_test={x_test.info()}')

    print(f'y_train={y_train.info()}')
    print(f'y_test={y_test.info()}')

    # Apply grid-search-cv for a random-forest-classifier and cache the model
    model = train_random_forest_classifier(x_train, y_train, param_grid=PARAM_GRID, model_path=MODEL_PATH)

    # Log out the metrics for current model
    print_neat_metrics(expected=y_test, preds=model.predict(x_test))







def run_prediction_pipeline(SRC_DIR_NAME, TGT_FILE_NAME, MODEL_PATH, TGT_DIR_NAME):
    df = read_src_file_as_df(dir_name=SRC_DIR_NAME, src_filename=TGT_FILE_NAME, verbose=True)
    print(f'Successfully read the remote text file and created a dataframe of shape df={df.shape}')

    # Engineer date-time features from the input datetime column and drop the original column
    df['transactionDt'] = pd.to_datetime(df['transactionDateTime'])
    df = add_time_dependent_features(df, 'transactionDt', True)

    df['accountOpenDt'] = pd.to_datetime(df['accountOpenDate'])
    df = add_time_dependent_features(df, 'accountOpenDt')

    df['lastAddressChangeDt'] = pd.to_datetime(df['dateOfLastAddressChange'])
    df = add_time_dependent_features(df, 'lastAddressChangeDt')


    # Rather than just binary non-equality, check score of mismatch: a single digit wrong might be just a minor blunder
    df = df.astype({'cardCVV':'str', 'enteredCVV':'str'})
    df['cvvMismatchScore'] = df.apply(lambda x : levenshtein_distance(x['enteredCVV'], x['enteredCVV']), axis=1)


    # Some basic cleansing/preprocessing logic
    main_df = df.copy(deep=True)
    main_df = drop_unary_columns(main_df, verbose=True)
    main_df = impute_transactionType(main_df, verbose=True)

    # Fetch the report of reversal-transactions
    report_reversals_df = get_reversals_report(main_df)
    report_reversals_df.to_csv(f'{TGT_DIR_NAME}/Test_Report_Reversals.csv', index=False)

    # Fetch the report of multiswipe-transactions
    multiswipes_df = get_multiswipe_transactions(main_df)
    multiswipes_df.to_csv(f'{TGT_DIR_NAME}/Test_Report_Multiswipes.csv', index=False)


    # Log the class-imbalance within our current dataset
    display_class_imbalance(main_df)

    # Convert the boolean columns to integer-type
    main_df = convert_boolean_to_int(main_df, verbose=True)

    # Apply Ordinal-Encoding to each of cat-column and keep track of the transformations
    main_df, categorical_cols_encoders = encode_categorical_cols(main_df, model_path=MODEL_PATH, verbose=True)

    # Apply Scaling to each of cat-column and keep track of the transformations
    main_df, numerical_col_scalers = scaledown_numerical_cols(main_df, model_path=MODEL_PATH, verbose=True)

    # Drop off the irrelevant non-numerical columns now
    main_df = drop_irrelevant_columns(main_df)


    # Create the test features
    feature_cols = set(main_df.columns) - set(['isFraud', 'cardCVV', 'enteredCVV', 'cardLast4Digits', 'accountNumber', 'customerId'])
    test_features = main_df[feature_cols]

    # Create the final set of test-predictions
    test_features['predictions'] = predict_random_forest_classifier(test_features, model_path=MODEL_PATH)
    if 'isFraud' in main_df.columns:
        test_features['expected'] = main_df['isFraud']
    test_features.to_csv(f'{TGT_DIR_NAME}/Random_Forest_predictions.csv', index=False)