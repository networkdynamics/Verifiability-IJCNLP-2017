import numpy as np
import t4k
import iterable_queue as iq
import multiprocessing
import json
import os
from SETTINGS import DATA_DIR, PARC_VERIFIABILITY_DIR
import parc_reader as pr
from collections import defaultdict, Counter

try:
    import matplotlib.pyplot as plt 
    import matplotlib
except RuntimeError:
    print 'NO MATPLOTLIB RIGHT NOW!'


NUM_PROCESSES = 12
ATTR_INFO_PATH = os.path.join(PARC_VERIFIABILITY_DIR, 'all-attrs.json')
ATTR_BY_ARTICLE_PATH = os.path.join(
    PARC_VERIFIABILITY_DIR, 'attrs-by-article.json')

# Location in article relative to overall text
# Location in article relative to other attributions
# Distribution of attribution verifiabilities within article


def read_attr_info():
    attr_info = json.loads(open(ATTR_INFO_PATH).read())
    attr_info_by_article = json.loads(open(ATTR_BY_ARTICLE_PATH).read())
    return attr_info, attr_info_by_article


def check_first_last():
    attr_info, attr_by_article = read_attr_info()
    verif_first_three = []
    verif_last_three = []
    verif_middle = []
    for attributions in attr_by_article.values():
        if len(attributions) < 6:
            continue

        verif_first_three.extend([a[-1] for a in attributions[:1]])
        verif_last_three.extend([a[-1] for a in attributions[-1:]])
        verif_middle.extend([a[-1] for a in attributions[3:-3]])

    print np.mean(verif_first_three), np.std(verif_first_three)
    print np.mean(verif_last_three), np.std(verif_last_three)
    print np.mean(verif_middle), np.std(verif_middle)


def check_first_three_sentences():
    attr_info, attr_by_article = read_attr_info()
    first_attributions = []
    last_attributions = []
    for attributions in attr_by_article.values():
        if len(attributions) < 6:
            continue

        if attributions[0][3] < 2:
            first_attributions.append(attributions[0][-1])

        if (attributions[-1][4] - attributions[-1][3]) < 2:
            last_attributions.append(attributions[-1][-1])

    all_attributions = [attr[-1] for attr in attr_info]

    print np.mean(first_attributions), np.std(first_attributions)
    print np.mean(last_attributions), np.std(last_attributions)
    print np.mean(all_attributions), np.std(all_attributions)


def check_attrs_by_rank(limit=5):
    attr_info, attr_by_article = read_attr_info()
    attr_verifs_by_rank = defaultdict(list)
    for attributions in attr_by_article.values():
        if len(attributions) < 6:
            continue

        for rank, attr in enumerate(attributions):
            if limit and rank >= limit:
                break
            attr_verifs_by_rank[rank].append(attr[-1])

    attr_verifs_rank_means_err = [0] * limit
    for rank in attr_verifs_by_rank:

        if limit and rank >= limit:
            break

        num_attrs = len(attr_verifs_by_rank[rank])
        mean = np.mean(attr_verifs_by_rank[rank])
        std = (
            np.std(attr_verifs_by_rank[rank]) 
            / np.sqrt(len(attr_verifs_by_rank[rank]))
        )
        lower_CI = mean - 1.96 * std
        upper_CI = mean + 1.96 * std

        try:
            attr_verifs_rank_means_err[rank] = (mean, 1.96 * std)
        except IndexError:
            print rank
        print '%d\t%.4f\t(%.4f - %.4f)' % (rank, mean, lower_CI, upper_CI)
        #print '%d\t(%d)\t%f\t%f' % (rank, num_attrs, mean, std)
    
    return attr_verifs_rank_means_err


def compare_first_three_last_three(limit=5):

    attr_info, attr_by_article = read_attr_info()
    first_three_attrs = []
    last_three_attrs = []

    for attributions in attr_by_article.values():
        if len(attributions) < 6:
            continue

        for rank, attr in enumerate(attributions):

            rev_rank =  attr[3] - attr[4]

            if rank < 3:
                first_three_attrs.append(attr[-1])
            elif abs(rev_rank) <= 3:
                last_three_attrs.append(attr[-1])

    first_three_verif = np.mean(first_three_attrs)
    first_three_std = (
        np.std(first_three_attrs) 
        / np.sqrt(len(first_three_attrs))
    )

    last_three_verif = np.mean(last_three_attrs)
    last_three_std = (
        np.std(last_three_attrs) 
        / np.sqrt(len(last_three_attrs))
    )

    fig, ax = plt.subplots(figsize=(6,2))
    ax.bar(
        [0,1], 
        [first_three_verif, last_three_verif],
        yerr=[first_three_std, last_three_std]
    )
    plt.show()






def plot_verif_by_rank():
    attrs_by_rank = check_attrs_by_rank()
    attrs_by_rev_rank = check_attrs_by_reverse_rank()
    fig, ax = plt.subplots(figsize=(6,2))

    X1 = range(len(attrs_by_rank))
    Y1 = [a[0] for a in attrs_by_rank]
    Y1_err = [a[1] for a in attrs_by_rank]

    X2 = range(len(attrs_by_rev_rank))
    Y2 = [a[0] for a in attrs_by_rev_rank]
    Y2_err = [a[1] for a in attrs_by_rev_rank]

    ax.errorbar(X1, Y1, yerr=Y1_err, fmt='-o')
    ax.errorbar(X2, Y2, yerr=Y2_err, fmt='-o')
    plt.show()



def check_attrs_by_reverse_rank(limit=5):

    attr_info, attr_by_article = read_attr_info()
    attr_verifs_by_rank = defaultdict(list)
    for attributions in attr_by_article.values():
        if len(attributions) < 6:
            continue

        for rank, attr in enumerate(attributions):
            rev_rank =  attr[3] - attr[4]
            if limit and abs(rev_rank) > limit:
                continue
            attr_verifs_by_rank[rev_rank].append(attr[-1])

    attr_verifs_rank_means_err = [0] * limit
    for rank in sorted(attr_verifs_by_rank):

        if limit and abs(rank) > limit:
            continue

        num_attrs = len(attr_verifs_by_rank[rank])
        mean = np.mean(attr_verifs_by_rank[rank])

        std = (
            np.std(attr_verifs_by_rank[rank]) 
            / np.sqrt(len(attr_verifs_by_rank[rank]))
        )
        lower_CI = mean - 1.96 * std
        upper_CI = mean + 1.96 * std

        try:
            attr_verifs_rank_means_err[rank] = (mean, 1.96 * std)
        except IndexError:
            print rank

        print '%d\t%.4f\t(%.4f - %.4f)' % (rank, mean, lower_CI, upper_CI)
        #print '%d\t(%d)\t%f\t%f' % (rank, num_attrs, mean, std)

    return attr_verifs_rank_means_err


def check_attrs_correlation():

    attr_info, attr_by_article = read_attr_info()
    attr_stds = []
    for attributions in attr_by_article.values():
        if len(attributions) < 6:
            continue

        attr_stds.append(np.std([attr[-1] for attr in attributions])) 

    print np.mean(attr_stds)
    print np.std([attr[-1] for attr in attr_info])




def read_attribution_scores():
    scores_path = os.path.join(PARC_VERIFIABILITY_DIR, 'predictions.tsv')
    rows = [l.strip().split('\t') for l in open(scores_path).readlines()]
    attribution_scores = dict([(row[0], float(row[1])) for row in rows])
    return attribution_scores
    

def extract_attribution_distribution(num_processes=NUM_PROCESSES, limit=None):

    # Work out paths and open files for writing results.
    attr_info_file = open(ATTR_INFO_PATH, 'w')
    attr_info_by_article_file = open(ATTR_BY_ARTICLE_PATH, 'w')

    # Use a queue to receive results from worker processes.
    results_queue = iq.IterableQueue()

    # Start a bunch of workers.
    for proc_num in range(num_processes):
        process = multiprocessing.Process(
            target=extract_attr_dist_worker,
            args=(results_queue.get_producer(), proc_num, num_processes, limit)
        )
        process.start()

    # Get a consumer endpoint for the results queue.  No more endponts are
    # needed, so close the queue.
    results_consumer = results_queue.get_consumer()
    results_queue.close()

    # Aggregate all the results in two ways: organized on a per-article basis,
    # and as a set of "loose" rows (i.e. all attribution info together).
    all_attributions = []
    attributions_by_article = {}
    for result in results_consumer:
        for fname, attribution_info in result:
            all_attributions.extend(attribution_info)
            attributions_by_article[fname] = attribution_info

    # Now write all the info to file
    attr_info_by_article_file.write(
        json.dumps(attributions_by_article, indent=2))
    attr_info_file.write(json.dumps(all_attributions, indent=2))


def extract_attr_dist_worker(results_producer, this_bin, num_bins, limit=None):
    
    scores = read_attribution_scores()
    all_attribution_info = []

    for article_num in range(pr.parc_dataset.MAX_ARTICLE_NUM):

        # Optionally stop early.
        if limit is not None:
            if article_num > limit:
                break

        # This worker should only process articles assigned to its bin
        if not t4k.inbin(str(article_num), num_bins, this_bin):
            continue

        print 'processing %d' % article_num

        # Load the article.  Skip if it doesn't exist.
        try:
            article = pr.parc_dataset.load_article(article_num)
        except (ValueError, IOError):
            continue

        attr_info = get_article_attr_info(article, article_num, scores)
        all_attribution_info.append((article_num, attr_info))

    results_producer.put(all_attribution_info)
    results_producer.close()


def get_article_attr_info(article, article_num, scores):
    """
    Extract position information and merge it with score information
    for each attribution in this article
    """

    num_sentences = len(article.sentences)

    # Pull out the location of each attribution (corresponding to the 
    # sentence in which the attribution starts, and sort by location
    attributions = sorted(
        (min(attr.get_sentence_ids()), attr['id'], attr)
        for attr in article.attributions.values()
    )

    # Pull out other information associated to the attributions and articles
    # in which they occur
    attr_info = [
        (article_num, attr_id, rank, sentence, num_sentences, scores[attr_id])
        for rank, (sentence, attr_id, attr) in enumerate(attributions)
    ]

    return attr_info

    


