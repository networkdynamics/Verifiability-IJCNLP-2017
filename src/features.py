import t4k
import os
from parc_reader.new_reader import ParcCorenlpReader as P
from sklearn.externals import joblib
from SETTINGS import DATA_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH
import old.features_testing as vectors
import old.features as feats
import pickle
import numpy as np
import csv
import parc_dataset as pd
import pandas
import iterable_queue as iq
import multiprocessing

NUM_PROCESSES = 12

train_dict = pickle.load(open(os.path.join(
    DATA_DIR, 'verifiabilityNumFeatures_len_5_liwc_train_20comp'), 'rb'))

(
    sourceVocab, cueVocab, weaselWordsVocab, typeEntitiesVocab, 
    sourceLemmaVocab, amodVocab, detsVocab, cueLiwcVocab, sourceLiwcVocab, 
    contentLiwcVocab, typeEntitiesContentVocab, headers 
) = vectors.getVocabs(train_dict)


TRAIN_QUINTILES = (
    0.63745912172200003,
    0.75505605902399997,
    0.84547732519600005,
    0.91542111606999998
)

BASELINE_FEATURES = [
    'c_1_typeQuote', 's_1_typeEntities_PERSON', 's_1_typeEntities_ORGANIZATION',
]

def calc_quintiles(values):
    values.sort()
    step_size = len(values) / 5.
    return [values[int(q*step_size)] for q in range(1,5)]


def get_training_quintiles():
    X, Y, attr_ids = load_train_features()
    return calc_quintiles(Y)


def load_test_features(
    baseline=False, keep_only=None, with_headers=False, with_ids=False
):
    return load_features_with_targets(
        TEST_DATA_PATH, baseline, keep_only, with_headers, with_ids)


def load_train_features(
    baseline=False, keep_only=None, with_headers=False, with_ids=False
):
    return load_features_with_targets(
        TRAIN_DATA_PATH, baseline, keep_only, with_headers, with_ids)


def load_features(path):
    dataframe = pickle.load(open(path, 'rb'))
    return get_feature_vectors(dataframe)


def load_features_with_targets(
    path, baseline, keep_only=None, with_headers=False, with_ids=False
):
    dataframe = pickle.load(open(path, 'rb'))
    return get_feature_vectors_with_targets(
        dataframe, baseline, keep_only, with_headers, with_ids)


def get_feature_vectors_with_targets(
    dataframe,
    baseline=False,
    keep_only=None,
    with_headers=False,
    with_ids=False
):
    
    # Alter the dataframe a bit: rename one of the columns, and drop a bunch
    # of others.
    dataframe = dataframe.rename(
        index=str, columns={"s_8_contentLength": "c_2_contentLength"}
    )
    dataframe = drop_unused_columns(dataframe)

    # Get all of the headers
    headers = list(dataframe.columns.values)

    # Get just the x-values (feature values).  This means we don't want the 
    # first two columns, which contain the attribution_id and the target
    # verifiability.
    feature_names = keep_only or headers[2:]
    if baseline:
        feature_names = BASELINE_FEATURES
    X = dataframe[feature_names].astype(np.float).as_matrix()

    # Ge the verifiability and attribution ids for each attribution
    Y = dataframe[headers[1]].astype(np.float).as_matrix()
    attr_ids = dataframe[headers[0]].as_matrix()

    results = [X,Y]
    if with_ids:
        results.append(attr_ids)
    if with_headers:
        results.append(feature_names)
    return results


def drop_unused_columns(dataframe):
    """
    Drops various unused columns from the dataframe.
    """
    headers = list(dataframe.columns.values)
    feature_names = headers[2:]

    sourcecolumns = list(filter(
        lambda s: s.startswith('s_8') and s != 's_8_sourceLength',
        feature_names
    ))
    cuecolumns = list(filter(lambda s: s.startswith('q_2'), feature_names))
    dropColumns = list(filter(
    lambda s: s.startswith('c_3') or s.startswith('s_10'), feature_names))

    dataframe.drop(sourcecolumns, axis=1, inplace=True)
    dataframe.drop(dropColumns, axis=1, inplace=True)

    return dataframe


def get_feature_vectors(dataframe):
    headers = list(dataframe.columns.values)

    metaHeaders = headers[0:1]
    X_headers = headers[1:]
    meta = metaData.as_matrix().reshape(-1)

    sourcecolumns = list(filter(
        lambda s: s.startswith('s_8') and s != 's_8_sourceLength', 
        X_headers
    ))
    cuecolumns = list(filter(lambda s: s.startswith('q_2'), X_headers))
    dropColumns = list(filter(
        lambda s: s.startswith('c_3') or s.startswith('s_10'),
        X_headers
    ))

    dataframe.drop(sourcecolumns, axis=1, inplace=True)
    dataframe.drop(dropColumns, axis=1, inplace=True)

    headers = list(dataframe.columns.values)


    metaData = dataframe[metaHeaders]
    X_Values = dataframe[X_headers].astype(np.float)

    feats = X_Values.as_matrix()
    
    # Features is a 2-D array with each row being the feature vector for a 
    # single attribution.  meta is a 1-D array of attribution ids, whose order
    # matches the rows of feats.
    return feats, meta


def extract_features_from_parc(
    num_processes=NUM_PROCESSES, limit=pd.MAX_ARTICLE_NUM, out_path=None
):

    # Resolve the path to the results file and open it for writing.
    if out_path is None:
        out_dir = os.path.join(DATA_DIR, 'parc-verifiability')
        t4k.ensure_exists(out_dir)
        out_path = os.path.join(out_dir, 'features.np')
    out_file = open(out_path, 'w')

    # Make a queueu so that workers can send results back
    results_queue = iq.IterableQueue()

    # Start a bunch of workers
    for proc_num in range(num_processes):
        p = multiprocessing.Process(
            target=extract_features_from_parc_worker,
            args=(
                results_queue.get_producer(), proc_num, num_processes, limit
            )
        )
        p.start()

    # Get an endpoint to collect the work, then close the queue since we won't
    # make any more endpoints
    results_consumer = results_queue.get_consumer()
    results_queue.close()

    # Collect all the incoming work from the workers
    all_result_vectors = []
    for result_vectors in results_consumer:
        all_result_vectors.extend(result_vectors)
    
    # Turn all the results vectors into a single pandas dataframe, and save it
    use_headers = headers[:1] + headers[2:]
    data_frame = pandas.DataFrame(all_result_vectors, columns=use_headers)
    pickle.dump(data_frame, out_file)


def extract_features_from_parc_worker(
    results_producer, this_bin, num_bins, limit=pd.MAX_ARTICLE_NUM
):

    # We'll accumulate the feature vectors from many files in this list.
    result_vectors = []

    # Iterate over articles in the dataset (only process certain ones though).
    for article_num in range(limit):

        # Only process articles assigned to this bin.
        if not t4k.inbin(str(article_num), num_bins, this_bin):
            continue

        # Check if it's a valid article num, and get the paths to the files
        # representing the article.
        print 'processing %d' % article_num
        paths = pd.get_article_paths(article_num, confirm=True)
        if paths is None:
            print '\tskipping %d' % article_num
            continue

        # extract and accumulate the features
        result_vectors.extend(extract_features(*paths))

    results_producer.put(result_vectors)
    results_producer.close()



def extract_features_from_parc_file(article_num):
    parc_features_dir = os.path.join(DATA_DIR, 'parc-verifiability', 'features')
    t4k.ensure_exists(parc_features_dir)
    corenlp_path, parc_path, raw_path = pd.get_article_paths(article_num)
    out_vector_path = os.path.join(parc_features_dir, 
        pd.get_article_features_path(article_num))
    return extract_features(corenlp_path, parc_path, raw_path, out_vector_path)


def extract_and_write_features(
    corenlp_path,
    parc_path,
    raw_path,
    out_vector_path
):

    # Open files for writing results
    out_vector_file = open(out_vector_path, 'wb')

    # Extract the features
    rowsVectors = extract_features(corenlp_path, parc_path, raw_path)

    # Write the features to disc
    # Exclude header 1 (probably because it's the label column?)
    # Write the feature vectors to disk
    use_headers = headers[:1] + headers[2:]
    dataframe = pandas.DataFrame(rowsVectors, columns=use_headers)
    pickle.dump(dataframe, out_vector_file)


def extract_features(corenlp_path, parc_path, raw_path):
    """
    Predict verifiability of attributions in a single article, based on its raw
    text, corenlp annotations, and the attribution annotations in the parc
    format.  Write feature vectors and regression scores.
    """

    # Read input files
    corenlp_xml = open(corenlp_path).read()
    parc_xml = open(parc_path).read()
    raw_text = open(raw_path).read()

    # Parse the input files into an annotated article object
    article = P(corenlp_xml, parc_xml, raw_text)
    attrs = article.attributions

    # Extract features for each attribution
    rowsVectors = []
    for attr in attrs:
        thisAttribution = attrs[attr]
        featsDict = feats.featureExtract(thisAttribution, article)
        vector = vectors.vectorize_ind(
            featsDict, False, sourceVocab, cueVocab, weaselWordsVocab,
            typeEntitiesVocab, sourceLemmaVocab, amodVocab, detsVocab, 
            cueLiwcVocab, sourceLiwcVocab, contentLiwcVocab, 
            typeEntitiesContentVocab
        )
        rowsVectors.append(vector)

    return rowsVectors



def read_data(subset='all'):
    """
    Read and return the data (features and verifiability scores) for
    attributions in PARC that were scored using a crowdflower task.
    """

    # Read in the desired data
    if subset == 'all' or subset == 'train':
        X_train, y_train, attr_ids_train, headers = read_data_file(
            TRAIN_DATA_PATH)
    if subset == 'all' or subset == 'test':
        X_test, y_test, attr_ids_test, headers = read_data_file(TEST_DATA_PATH)

    # Return the desired data.  If "all" is desired, we need to concatenate the
    # test and train portions.
    if subset == 'train':
        return X_train, y_train, attr_ids_train, headers
    elif subset == 'test':
        return X_test, y_test, attr_ids_test, headers
    elif subset == 'all':
        return (
            np.concatenate([X_train, X_test]), 
            np.concatenate([y_train, y_test]),
            np.concatenate([attr_ids_train, attr_ids_test]),
            headers
        )
    else:
        raise ValueError('Unexpected subset: %s' % subset)


def read_data_file(datafile):

    dataframe = pickle.load(open(datafile, 'rb'))
    headers = list(dataframe.columns.values)

    metaHeaders = headers[0:2]
    X_headers = headers[2:]

    sourcecolumns = list(filter(
    lambda s: s.startswith('s_8') and s != 's_8_sourceLength', X_headers))
    cuecolumns = list(filter(lambda s: s.startswith('q_2'), X_headers))
    dropColumns = list(filter(
    lambda s: s.startswith('c_3') or s.startswith('s_10'), X_headers))

    dataframe.drop(sourcecolumns, axis=1, inplace=True)
    dataframe.drop(dropColumns, axis=1, inplace=True)

    headers = list(dataframe.columns.values)

    X_headers = headers[2:]

    metaData = dataframe[metaHeaders]
    X_Values = dataframe[X_headers].astype(np.float)

    metaData = metaData.as_matrix()
    X_Values = X_Values.as_matrix()

    test_meta = metaData[:, 0]
    y_test = metaData[:, 1].astype(np.float) * 100
    X_test = X_Values
    
    return X_test, y_test, test_meta, X_headers



