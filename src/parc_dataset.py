import t4k
import re
import os
import json
import random
import SETTINGS

MAX_ARTICLE_NUM = 2499

ARTICLE_NUM_MATCHER = re.compile('wsj_(\d\d\d\d)')
def get_article_num(article_fname):
    return int(ARTICLE_NUM_MATCHER.search(article_fname).group(1))


def get_parc_fname(article_num):
    return 'wsj_%s' % str(article_num).zfill(4)


def get_raw_path(article_num):
    fname = get_parc_fname(article_num)

    # Determine the subdir by getting the hundreds from the article_num
    prefix_digits = article_num / 100
    if prefix_digits < 23:
        raw_dir = SETTINGS.RAW_TRAIN_DIR
    elif prefix_digits < 24:
        raw_dir = SETTINGS.RAW_TEST_DIR
    elif prefix_digits < 25:
        raw_dir = SETTINGS.RAW_DEV_DIR
    else:
        raise ValueError(
            "Parc data has no articles with ids in the range of %d00's."
            % prefix_digits
        )
    subsubdir = str(prefix_digits).zfill(2)

    return os.path.join(raw_dir, fname)


def get_corenlp_path(article_num):
    fname = get_parc_fname(article_num)

    # Determine the subdir by getting the hundreds from the article_num
    prefix_digits = article_num / 100
    if prefix_digits < 23:
        corenlp_dir = SETTINGS.CORENLP_TRAIN_DIR
    elif prefix_digits < 24:
        corenlp_dir = SETTINGS.CORENLP_TEST_DIR
    elif prefix_digits < 25:
        corenlp_dir = SETTINGS.CORENLP_DEV_DIR
    else:
        raise ValueError(
            "Parc data has no articles with ids in the range of %d00's."
            % prefix_digits
        )
    subsubdir = str(prefix_digits).zfill(2)

    return os.path.join(corenlp_dir, fname + '.xml')


def get_parc_path(article_num):

    # Get the actual filename based on the article number
    fname = get_parc_fname(article_num)

    # Determine the subdir by getting the hundreds from the article_num
    prefix_digits = article_num / 100
    if prefix_digits < 23:
        parc_dir = SETTINGS.PARC_TRAIN_DIR
    elif prefix_digits < 24:
        parc_dir = SETTINGS.PARC_TEST_DIR
    elif prefix_digits < 25:
        parc_dir = SETTINGS.PARC_DEV_DIR
    else:
        raise ValueError(
            "Parc data has no articles with ids in the range of %d00's."
            % prefix_digits
        )
    subsubdir = str(prefix_digits).zfill(2)

    return os.path.join(parc_dir, subsubdir, fname + '.xml')


def get_article_features_path(file_num):
    return 'wsj_%s.np' % str(file_num).zfill(4)


def get_article_paths(article_num, confirm=False):
    paths = (
        get_corenlp_path(article_num),
        get_parc_path(article_num),
        get_raw_path(article_num)
    )

    # Optionally confirm that these paths actually exist.  This can help filter
    # out the occaisional gaps in the article numbering
    if confirm:
        if not all(os.path.exists(path) for path in paths):
            return None

    return paths


def iter_article_paths():
    for article_num in range(MAX_ARTICLE_NUM):
        paths = get_article_paths(article_num)
        if paths is not None:
            yield paths

