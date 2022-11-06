from airflow.models import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.python import ExternalPythonOperator

from train import run_train_pipeline

def run_this_func():
    print('I am coming first')

# @task.virtualenv(task_id='virtualenv_python', system_site_packages=False, requirements=[
#     "scikit-learn",
#     "pandas"
#     ])
# def run_train_function_in_venv():
#     print(f'Trying to create a virtual-env')
#     run_train_pipeline()


args = {
    'owner': 'vikrant',
    'start_date': days_ago(1)
}

dag = DAG(dag_id='model_training_dag', default_args=args, schedule_interval=None)

with dag:
    initial_task = PythonOperator(
        task_id='run_this_first',
        python_callable = run_this_func
    )
    train_pipeline_task = PythonOperator(
        task_id='train_pipeline_task',
        python_callable = run_train_pipeline,
    )

initial_task >> train_pipeline_task