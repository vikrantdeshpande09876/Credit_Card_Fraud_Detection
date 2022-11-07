from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
# from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator






dag = DAG(dag_id='model_training_dag', default_args={'owner':'vikrant', 'retries':0, 'start_date':days_ago(1)}, schedule_interval=None, )

with dag:
    def run_data_ingestion_function():
        print('I am going to push your reports into the Google Big Table dataset.')
    
    def run_train_function():
        import sys, os
        PATHS = sys.path
        print(f'PATHS={PATHS}')
        for root, dirs, files in os.walk(".", topdown=False):
            for name in files:
                print(os.path.join(root, name))
            for name in dirs:
                print(os.path.join(root, name))
        from train import run_train_pipeline
        print(f'Might need to create a virtual-env')
        run_train_pipeline()
    
    train_pipeline_task = PythonVirtualenvOperator(   
        task_id = 'train_pipeline_task',
        python_callable = run_train_function,
        requirements = ['scikit-learn==1.0.2', 'google.cloud', 'pandas==1.4.2'],
        system_site_packages = False,
    )
    data_ingestion_task = PythonOperator(
        task_id = 'data_ingestion_task',
        python_callable = run_data_ingestion_function
    )

train_pipeline_task >> data_ingestion_task