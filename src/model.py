import os
from sklearn.externals import joblib
from SETTINGS import DATA_DIR


def load_model(model_type):

    if model_type == 'svr':
        model_path = os.path.join(DATA_DIR, 'best_ver_regressionSVR(20).pkl')
        return joblib.load(model_path)

    raise NotImplementedError(
        'Loading model "%s" has not been implemented.' % model_type
    )
