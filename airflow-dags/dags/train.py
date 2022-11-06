import pandas as pd

from sklearn.model_selection import train_test_split
from utils.util_functions import read_zip_file_as_df, levenshtein_distance, add_time_dependent_features, drop_unary_columns, impute_transactionType
from utils.util_functions import get_reversals_report, get_multiswipe_transactions, display_class_imbalance, convert_boolean_to_int
from utils.util_functions import encode_categorical_cols, drop_irrelevant_columns, scaledown_numerical_cols, print_neat_metrics
from utils.util_functions import train_random_forest_classifier

from config_params.config import SRC_DIR_NAME, TGT_FILE_NAME, SRC_ZIP_FILE, MODEL_PATH, PARAM_GRID, TGT_DIR_NAME


def run_train_pipeline():
    df = read_zip_file_as_df(dir_name=SRC_DIR_NAME, zipfile_name=SRC_ZIP_FILE, tgt_filename=TGT_FILE_NAME)


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

    # Apply grid-search-cv for a random-forest-classifier and cache the model
    model = train_random_forest_classifier(x_train, y_train, param_grid=PARAM_GRID, model_path=MODEL_PATH)

    # Log out the metrics for current model
    print_neat_metrics(expected=y_test, preds=model.predict(x_test))