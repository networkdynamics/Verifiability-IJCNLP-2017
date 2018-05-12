import os
import scipy
import itertools as it
import numpy as np
from sklearn.svm import SVR
from sklearn.externals import joblib
import features as f
import t4k
from SETTINGS import DATA_DIR


NUM_FOLDS = 10

def get_RMSE(Y_pred, Y):
    return np.sqrt(np.mean((Y_pred - Y)**2))


def get_quintile_accuracy(Y_pred, Y):
    quintiles_pred = get_quintile(Y_pred)
    quintiles = get_quintile(Y)
    accuracy = np.mean(np.equal(quintiles_pred, quintiles))
    return accuracy
    

@np.vectorize
def get_quintile(val):
    i = 0
    for threshold in f.TRAIN_QUINTILES:
        if val < threshold:
            break
        else:
            i += 1
    return i


SVX_RUNS = (
    {'kernel':'rbf', 'C':C, 'gamma':gamma}
    for C, gamma in it.product(np.logspace(-5, 4, 10), repeat=2)
)
SMALL_RUNS = [
    {'kernel':'rbf', 'C':100, 'gamma':0.01},
    {'kernel':'rbf', 'C':100, 'gamma':0.001}
]


EXCERPTED_ATTRIBUTIONS = [
    'wsj_0956_Attribution_relation_level.xml_set_13',
    'wsj_0020_Attribution_relation_level.xml_set_1',
    'wsj_0037_Attribution_relation_level.xml_set_12',
    'wsj_2113.pdtb_85',
    'wsj_2250_PDTB_annotation_level.xml_set_2'
]
def show_regressions_for_excerpt_table():

    # Load the training data, and the best SVR model
    X, Y, attr_ids = f.load_train_features(with_ids=True)
    model = load_best_SVR_model()

    # Figure out the indices corresponding to the excerpted attributions, so we
    # can extract the features and target values for those attributions.
    attr_ids = list(attr_ids)
    excerpted_attr_idxs = [
        attr_ids.index(attr_id) for attr_id in EXCERPTED_ATTRIBUTIONS
    ]

    # Extract the features and target values for the excerpted attributions.
    excerpted_attr_X = [X[idx] for idx in excerpted_attr_idxs]
    excerpted_attr_Y = [Y[idx] for idx in excerpted_attr_idxs]

    # Get predicted verifiability for excerpted attributions.
    excerpted_attr_Y_pred = model.predict(excerpted_attr_X)

    # Return the predicted and actual verifiabilities, zipped.
    return zip(excerpted_attr_Y_pred, excerpted_attr_Y)


def optimize_model(
    test_name='test',
    data=f.load_train_features(),
    model_class=SVR,
    runs=SVX_RUNS,
    num_folds=NUM_FOLDS
):

    X, Y = data

    # Find the best performance (minimized error) accross runs
    minimizer = t4k.Min()
    performances = {}
    for kwargs in runs:
        get_model = lambda: model_class(**kwargs)
        RMSE, quint_accuracy, rho, p = crossval(get_model, X, Y, num_folds)
        minimizer.add((RMSE, quint_accuracy, rho, p), (kwargs))

    (RMSE, quint_accuracy, rho, p), (kwargs) =  minimizer.get()

    return {
        'kwargs':kwargs, 'RMSE':RMSE, 'quintile_accuracy':quint_accuracy,
        'rho':rho, 'p':p
    }


def test_model(model, X, Y):
    Y_pred = model.predict(X)

    RMSE = get_RMSE(Y_pred, Y)
    quint_accuracy = get_quintile_accuracy(Y_pred, Y)
    spearmanr = scipy.stats.spearmanr(Y_pred, Y)
    rho = spearmanr.correlation
    p = spearmanr.pvalue

    return RMSE, quint_accuracy, rho, p


def load_best_SVR_model():
    model_path = os.path.join(DATA_DIR, 'best_ver_regressionSVR(20).pkl')
    model = joblib.load(model_path)

    # We have to wrap this model and convert its predictions from percentage
    # to fractional
    class WrappedModel(object):
        def __init__(self, model):
            self.model = model
        def predict(self, X):
            return self.model.predict(X) / 100.

    model = WrappedModel(model)

    return model


def test_best_SVR_model():
    X, Y = f.load_test_features()
    model = load_best_SVR_model()
    return test_model(model, X, Y)


def crossval(get_model, X, Y, num_folds):

    RMSEs = []
    quintile_accuracies = []
    rhos = []
    ps = []

    for fold in range(num_folds):

        print 'running fold %d' % fold
    
        # Get the test-train splits for this fold
        X_train, X_test = t4k.get_fold(X, num_folds, fold)
        Y_train, Y_test = t4k.get_fold(Y, num_folds, fold)

        # Get an untrained model, and train it on this fold
        clf = get_model()
        clf.fit(X_train, Y_train)

        # Test the model, and accumulate the results for this fold
        RSME, quint_accuracy, rho, p = test_model(clf, X_test, Y_test)
        RMSEs.append(RMSE)
        quintile_accuracies.append(quint_accuracy)
        rhos.append(rho)
        ps.append(p)

    return (
        np.mean(RMSEs), np.mean(quintile_accuracies), np.mean(rhos), np.mean(ps)
    )


 
