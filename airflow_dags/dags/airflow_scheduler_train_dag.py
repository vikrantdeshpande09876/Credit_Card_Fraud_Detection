from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator




def dummy_function():
    import sys, os
    PATHS = sys.path
    print(f'PATHS={PATHS}')
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))



dag = DAG(dag_id='model_training_dag', default_args={'owner':'vikrant', 'retries':0, 'start_date':days_ago(1)}, schedule_interval=None, )

with dag:
    def run_train_function():
        from fraud_detection_package_new.train_or_predict import run_train_pipeline
        print(f'Attempts to create a virtual-env')
    
        SRC_GCS_BUCKETNAME = 'gs://i535-credit-card-fraud-transactions-dir'
        SRC_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/Data/'
        # SRC_ZIP_FILE = 'transactions.zip'
        TGT_FILE_NAME = 'Train_transactions.csv'
        TGT_DIR_NAME = f'{SRC_GCS_BUCKETNAME}/outputs/'
        MODEL_PATH = f'{SRC_GCS_BUCKETNAME}/models/'

        CLASS_WEIGHTS = { 0: 2, 1: 98 }
        PARAM_GRID = {
            'n_estimators' : [400],
            'max_depth' : [None],
            'random_state' : [1],
            'min_samples_split' : [2],
            'n_jobs' : [-1],
            'class_weight' : [CLASS_WEIGHTS]
        }

        print(SRC_DIR_NAME, TGT_FILE_NAME, MODEL_PATH, PARAM_GRID, TGT_DIR_NAME)
        run_train_pipeline(SRC_DIR_NAME, TGT_FILE_NAME, MODEL_PATH, PARAM_GRID, TGT_DIR_NAME)
    
    def run_data_ingestion_function():
        print('I am going to push your reports into the Google Big Table dataset.')
    
    train_pipeline_task = PythonVirtualenvOperator(   
        task_id = 'train_pipeline_task',
        python_callable = run_train_function,
        requirements = [
            'scikit-learn==1.0.2', 
            'google-cloud-storage', 
            'pandas==1.4.2',
            'fs-gcsfs',
            'gcsfs',
            'fsspec',
            'matplotlib==3.5.1', 
            'fraud-detection-package-new==0.0.8'
            ],
        system_site_packages = False,
    )

    data_ingestion_task = PythonOperator(
        task_id = 'data_ingestion_task',
        python_callable = run_data_ingestion_function
    )

train_pipeline_task >> data_ingestion_task