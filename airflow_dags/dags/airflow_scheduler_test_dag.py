from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonVirtualenvOperator




dag = DAG(dag_id='model_prediction_dag', default_args={'owner':'vikrant', 'retries':0, 'start_date':days_ago(1)}, schedule_interval=None, )


def run_predict_function():
        from fraud_detection_package_new.train_or_predict import run_prediction_pipeline
        print(f'Attempting to create a virtual-env')
    
        SRC_GCS_BUCKETNAME = 'gs://i535-credit-card-fraud-transactions-dir'
        SRC_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/Data/'
        TGT_FILE_NAME = 'Test_transactions.txt'
        TGT_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/outputs/'
        MODEL_PATH = f'{SRC_GCS_BUCKETNAME}/models/'

        print(SRC_DIR_NAME, TGT_FILE_NAME, MODEL_PATH, TGT_DIR_NAME)
        run_prediction_pipeline(SRC_DIR_NAME, TGT_FILE_NAME, MODEL_PATH, TGT_DIR_NAME)



def run_data_ingestion_function():
        import pandas as pd
        from datetime import datetime
        print(f'Attempting to create a virtual-env')

        SRC_GCS_BUCKETNAME = 'gs://i535-credit-card-fraud-transactions-dir'
        PROJECT_ID = 'i535-final-project-367821'
        TGT_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/outputs/'
        BIGQUERY_DATASET_NAME = 'ReportsDataset'

        REPORTS = [
            'Test_Report_Reversals',
            'Test_Report_Multiswipes',
            'Random_Forest_Test_Set_predictions'
        ]

        for report in REPORTS:
            try:
                df = pd.read_csv(f'{TGT_DIR_NAME}{report}.csv')
                current_ts = datetime.now().strftime(format='%Y-%m-%d %H:%M:%S')
                df['src_date'] = current_ts
                print(f'Pushing {TGT_DIR_NAME}{report}.csv into Google Big Table dataset.')
                df.to_gbq(
                    destination_table=f'{BIGQUERY_DATASET_NAME}.{report}', 
                    project_id=PROJECT_ID, 
                    if_exists='append', 
                    progress_bar=False
                )
                print(f'Successfully wrote {TGT_DIR_NAME}{report}.csv into Google Big Table dataset.')
            except Exception as e:
                print(f'Something went wrong while reading/writing the {report}-dataframe to BigQuery.. {e}')
                print(f'Are you sure {TGT_DIR_NAME}{report}.csv exists?')
                continue

            



with dag:

    prediction_pipeline_task = PythonVirtualenvOperator(   
        task_id = 'prediction_pipeline_task',
        python_callable = run_predict_function,
        requirements = [
            'scikit-learn==1.0.2', 
            'google-cloud-storage', 
            'pandas==1.4.2',
            # 'fs-gcsfs',
            'pandas-gbq',
            'gcsfs',
            'fsspec',
            'matplotlib==3.5.1', 
            'fraud-detection-package-new==0.1.1'
            ],
        system_site_packages = False,
    )

    data_ingestion_task = PythonVirtualenvOperator(
        task_id = 'data_ingestion_task',
        python_callable = run_data_ingestion_function,
        requirements = [
            'google-cloud-storage', 
            'pandas==1.4.2',
            'pandas-gbq',
            'gcsfs'
            ],
        system_site_packages = False,
    )

prediction_pipeline_task >> data_ingestion_task