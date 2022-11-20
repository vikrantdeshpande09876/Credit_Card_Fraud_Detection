# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trigger a DAG in a Cloud Composer 2 environment in response to an event,
using Cloud Functions.
"""

from typing import Any

import composer2_airflow_rest_api

def trigger_dag_gcf(data, context=None):
    """
    Trigger a DAG and pass event data.

    Args:
      data: A dictionary containing the data for the event. Its format depends
      on the event.
      context: The context object for the event.

    For more information about the arguments, see:
    https://cloud.google.com/functions/docs/writing/background#function_parameters
    """

    # TODO(developer): replace with your values
    # Replace web_server_url with the Airflow web server address. To obtain this
    # URL, run the following command for your environment:
    # gcloud composer environments describe example-environment \
    #  --location=your-composer-region \
    #  --format="value(config.airflowUri)"
    web_server_url = (
        # "https://a91bb61eed044a3897cb3d36643ec15e-dot-us-central1.composer.googleusercontent.com"
        "https://lc671f9a53c03e59cp-tp.appspot.com"
    )
    # Replace with the ID of the DAG that you want to run.
    dag_id = 'model_training_dag'
    # dag_id = 'model_prediction_dag'

    composer2_airflow_rest_api.trigger_dag(web_server_url, dag_id, data)