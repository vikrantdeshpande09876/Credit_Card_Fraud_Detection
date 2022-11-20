# Credit Card Fraud Detection

## Google Cloud Functions (Serverless Function-as-a-Service)

This folder contains the source code of the Cloud Function trigger that monitors the creation of an object in a specific bucket.

Use this template to create 2 functions:

`gcs-trainfile-watcher-trigger-function` --> to trigger the `model_training_dag`

`gcs-testfile-watcher-trigger-function` --> to trigger the `model_prediction_dag`

Configure an `EventArc` trigger with the following config:
1. <b>Event Provider</b>: `Cloud Storage`
2. <b>Event</b>: `storage.objects.create`
3. <b>Resource and Path Pattern</b>:
    
    a. `projects/_/buckets/i535-course-project-bucket/objects/Train_transactions.csv`
    
    b. `projects/_/buckets/i535-course-project-bucket/objects/Test_transactions.csv`


Set the entrypoint to be the `trigger_dag()` function in the `main.py` file.