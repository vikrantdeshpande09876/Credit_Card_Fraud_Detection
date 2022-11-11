import pandas as pd

from sklearn.model_selection import train_test_split
from anonymized_fraud_detection.utilities.util_functions import read_src_file_as_df, levenshtein_distance, add_time_dependent_features, drop_unary_columns, impute_transactionType
from anonymized_fraud_detection.utilities.util_functions import get_reversals_report, get_multiswipe_transactions, display_class_imbalance, convert_boolean_to_int
from anonymized_fraud_detection.utilities.util_functions import encode_categorical_cols, drop_irrelevant_columns, scaledown_numerical_cols, print_neat_metrics
from anonymized_fraud_detection.utilities.util_functions import train_random_forest_classifier


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
    df['cvvMismatchScore'] = df.apply(lambda x : levenshtein_distance(x['cardCVV'], x['enteredCVV']), axis=1)


    # Some basic cleansing/preprocessing logic
    main_df = df.copy(deep=True)
    main_df = drop_unary_columns(main_df, verbose=True)
    main_df = impute_transactionType(main_df, verbose=True)

    # Fetch the report of reversal-transactions
    report_reversals_df = get_reversals_report(main_df)
    report_reversals_df.to_csv(f'{TGT_DIR_NAME}Overall_Report_Reversals.csv', index=False)

    # Fetch the report of multiswipe-transactions
    multiswipes_df = get_multiswipe_transactions(main_df)
    multiswipes_df.to_csv(f'{TGT_DIR_NAME}Overall_Report_Multiswipes.csv', index=False)


    # Log the class-imbalance within our current dataset
    display_class_imbalance(main_df)

    # Convert the boolean columns to integer-type
    main_df = convert_boolean_to_int(main_df, verbose=True)

    # Apply Ordinal-Encoding to each of categorical-column and keep track of the transformations
    main_df, categorical_cols_encoders = encode_categorical_cols(main_df, model_path=MODEL_PATH, verbose=True)

    # Apply Scaling to each of numerical-column and keep track of the transformations
    main_df, numerical_col_scalers = scaledown_numerical_cols(main_df, model_path=MODEL_PATH, verbose=True)

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
    validations_df['expected'] = y_train
    validations_df.to_csv(f'{TGT_DIR_NAME}Random_Forest_Validation_Set_predictions.csv', index=False)