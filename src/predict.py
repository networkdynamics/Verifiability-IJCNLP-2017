import os
import model as m 
import features as f
from SETTINGS import PARC_FEATURES_PATH, DATA_DIR


def predict_parc():

    # Load the features.
    features, attribution_ids = f.load_features(PARC_FEATURES_PATH)

    # Load the model.
    model = m.load_model('svr')

    # Make predictions.  Convert percentage to decimal.
    predictions = model.predict(features) / 100.
    results = zip(attribution_ids, predictions)

    # sort on predicted value.
    results.sort(key=lambda x: x[1])

    predictions_path = os.path.join(
        DATA_DIR, 'parc-verifiability', 'predictions.tsv')
    open(predictions_path, 'w').write(
        '\n'.join([ '%s\t%f' % (attr_id, score) for attr_id, score in results])
    )
    return results

