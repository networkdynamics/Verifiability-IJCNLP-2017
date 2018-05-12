from os import path

PROJ_DIR = '/Users/enewel3/projects/verifiability'
DATA_DIR = '/Users/enewel3/projects/verifiability/data'

LIWC_PATH = '/home/ndg/dataset/liwc/data/LIWC2007_English100131_Dictionary.txt'

TEST_DATA_PATH = path.join(
    DATA_DIR, 'verifiabilityNumFeatures_len_5_liwc_test_20comp')
TRAIN_DATA_PATH = path.join(
    DATA_DIR, 'verifiabilityNumFeatures_len_5_liwc_train_20comp')

RAW_DIR = '/home/ndg/dataset/ptb2-corenlp/masked_raw'
RAW_TRAIN_DIR = path.join(RAW_DIR, 'train')
RAW_TEST_DIR = path.join(RAW_DIR, 'test')
RAW_DEV_DIR = path.join(RAW_DIR, 'dev')

CORENLP_DIR = '/home/ndg/dataset/ptb2-corenlp/CoreNLP'
CORENLP_TRAIN_DIR = path.join(CORENLP_DIR, 'train')
CORENLP_TEST_DIR = path.join(CORENLP_DIR, 'test')
CORENLP_DEV_DIR = path.join(CORENLP_DIR, 'dev')

PARC_DIR = '/home/ndg/dataset/parc3'
PARC_TRAIN_DIR = path.join(PARC_DIR, 'train')
PARC_DEV_DIR = path.join(PARC_DIR, 'dev')
PARC_TEST_DIR = path.join(PARC_DIR, 'test')

PARC_VERIFIABILITY_DIR = path.join(DATA_DIR, 'parc-verifiability')
PARC_FEATURES_PATH = path.join(PARC_VERIFIABILITY_DIR, 'features.np')
