import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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






def get_random_forest_feature_importances(model, feature_cols):
    """Creates a dataframe of features-importances from the GridSearchCV module.

    Args:
        model (GridSearchCV): Output of the GridSearchCV module.
        feature_cols (list): Column names of the features dataframe.

    Returns:
        pd.DataFrame: Output dataframe containing sorted feature-importances.
    """
    importances = pd.DataFrame(model.best_estimator_.feature_importances_)
    importances.index = feature_cols
    importances = importances.sort_values(by=0)
    return importances