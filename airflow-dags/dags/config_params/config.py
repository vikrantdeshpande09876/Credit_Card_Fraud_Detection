SRC_DIR_NAME = 'Data/'
SRC_ZIP_FILE = 'transactions.zip'
TGT_FILE_NAME = 'transactions.txt'

TGT_DIR_NAME = 'outputs'

CLASS_WEIGHTS = { 0: 2, 1: 98 }

PARAM_GRID = {
    'n_estimators' : [400],
    'max_depth' : [None],
    'random_state' : [1],
    'min_samples_split' : [2],
    'n_jobs' : [-1],
    'class_weight' : [CLASS_WEIGHTS]
}


MODEL_PATH = 'models/'