from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonVirtualenvOperator




dag = DAG(dag_id='model_prediction_dag', default_args={'owner':'vikrant', 'retries':0, 'start_date':days_ago(1)}, schedule_interval=None, )


def run_predict_function():
        from anonymized_fraud_detection.predict import run_prediction_pipeline
        print(f'Attempting to create a virtual-env')
    
        USE_GCS = True
        PROJECT_NAME = 'I535-Final-Project' if USE_GCS else None
        SRC_GCS_BUCKETNAME = 'gs://i535-course-project-bucket' if USE_GCS else ''

        SRC_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/'
        SRC_FILE_NAME = 'Test_transactions.csv'
        TGT_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/outputs/'
        MODEL_PATH = f'{SRC_GCS_BUCKETNAME}/models/'

        print(SRC_DIR_NAME, SRC_FILE_NAME, MODEL_PATH, TGT_DIR_NAME, PROJECT_NAME, USE_GCS)
        run_prediction_pipeline(SRC_DIR_NAME, SRC_FILE_NAME, MODEL_PATH, TGT_DIR_NAME, PROJECT_NAME=PROJECT_NAME, USE_GCS=USE_GCS)



def run_data_ingestion_function():
        import pandas as pd
        from datetime import datetime
        print(f'Attempting to create a virtual-env')

        SRC_GCS_BUCKETNAME = 'gs://i535-course-project-bucket'
        PROJECT_ID = 'i535-final-project-367821'
        TGT_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/outputs/'
        BIGQUERY_DATASET_NAME = 'ReportsDataset'

        REPORTS = [
            'Test_Report_Reversals',
            'Test_Report_Multiswipes',
            'Random_Forest_Test_Set_predictions'
        ]
        CURRENT_TS = datetime.now().strftime(format='%Y-%m-%d %H:%M:%S')

        for report in REPORTS:
            try:
                df = pd.read_csv(f'{TGT_DIR_NAME}{report}.csv')
                df['src_date'] = CURRENT_TS
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

def run_file_archival_task():
    import gcsfs
    from datetime import datetime

    SRC_GCS_BUCKETNAME = 'gs://i535-course-project-bucket'
    PROJECT_NAME = 'i535-Final-Project'
    SRC_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/'
    TGT_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/archive/'
    CURRENT_TS = datetime.now().strftime(format='%Y-%m-%d %H:%M:%S')
    CURRENT_TS = CURRENT_TS.replace('-','').replace(' ','').replace(':','')

    fs = gcsfs.GCSFileSystem(project=PROJECT_NAME)
    
    FILE_NAMES = ['Test_transactions.csv']
    FILE_FORMATS = ['csv']
    for fname, format in zip(FILE_NAMES, FILE_FORMATS):
        abs_src_filename = f'{SRC_DIR_NAME}{fname}.{format}'
        abs_tgt_filename = f'{TGT_DIR_NAME}{fname}_{CURRENT_TS}.{format}'

        try:
            fs.mv(abs_src_filename, abs_tgt_filename)
        except Exception as e:
            print(f'Something went wrong while archiving the {abs_src_filename}.. {e}')
            print(f'Are you sure {abs_src_filename} exists?')
            continue


            



with dag:

    prediction_pipeline_task = PythonVirtualenvOperator(   
        task_id = 'prediction_pipeline_task',
        python_callable = run_predict_function,
        requirements = [
            'scikit-learn==1.0.2', 
            'google-cloud-storage', 
            'pandas==1.4.2',
            'pandas-gbq',
            'gcsfs',
            'fsspec',
            'matplotlib==3.5.1', 
            'anonymized-fraud-detection==0.1.3'
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

    file_archival_task = PythonVirtualenvOperator(
        task_id = 'file_archival_task',
        python_callable = run_file_archival_task,
        requirements = [
            'google-cloud-storage', 
            'gcsfs'
            ],
        system_site_packages = False,
    )

prediction_pipeline_task >> data_ingestion_task >> file_archival_task