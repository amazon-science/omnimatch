import os
import re
import itertools

import pandas as pd
import numpy as np
import wordninja
from typing import List, Dict
import pickle
from collections import Counter
from math import log
import string
import math
import nltk
from scipy.stats import skew, kurtosis
from collections import OrderedDict
from tqdm import tqdm
from random import shuffle, sample, choice
from math import floor
import argparse
import configparser


# -------------------------- Data preparation routines for classifier.py --------------------------

def prepare_data(join_features: dict, non_join_features: dict, training_ratio: int, percentage: float):
    """
    Splits data into training and test ones, based on the given match and non-match column pairs.
    :param join_features: The features associated with match pairs
    :param non_join_features: The features associated with non-match pairs
    :param training_ratio: The ratio of negative to positive examples
    :param percentage: The percentage of training data to be used
    :return: Training data + labels and test data + labels
    """

    match_pairs = list(join_features.keys())
    non_match_pairs = list(non_join_features.keys())

    len_train = floor(percentage * len(match_pairs))
    len_train_negative = len_train * training_ratio

    shuffle(match_pairs)
    shuffle(non_match_pairs)

    X_train = []
    y_train = []

    for i in range(len_train):
        X_train.append(join_features[match_pairs[i]])
        y_train.append(1)

    for i in range(len_train_negative):
        X_train.append(non_join_features[non_match_pairs[i]])
        y_train.append(0)

    X_test = []
    y_test = []

    for i in range(len_train, len(match_pairs)):
        X_test.append(join_features[match_pairs[i]])
        y_test.append(1)

    for i in range(len_train_negative, len(non_match_pairs)):
        X_test.append(non_join_features[non_match_pairs[i]])
        y_test.append(0)

    return np.array(X_train, dtype=object), np.array(y_train, dtype=np.dtype(float)), np.array(X_test, dtype=object), \
           np.array(y_test, np.dtype(float))


def filter_pairs(datasets: list, join_pairs: list, non_join_pairs: list, benchmark: str):
    """
    Filters out join, non-join pairs included in a set of datasets
    :param datasets: A list of dataset names
    :param join_pairs: Column join pairs of the form ((dataset1, column1), (dataset2, column2))
    :param non_join_pairs: Column non join pairs of the form ((dataset1, column1), (dataset2, column2))
    :param benchmark: The benchmark from which the datasets come
    :return: Join/Non-join pairs included in datasets
    """
    dataset_joins = []
    dataset_non_joins = []

    for pair in join_pairs:
        dataset1, _ = pair[0]
        dataset2, _ = pair[1]

        if dataset1 in datasets and dataset2 in datasets:

            if fabricated_to_source(dataset1, benchmark) == fabricated_to_source(dataset2, benchmark):
                dataset_joins.append(pair)

    for pair in non_join_pairs:
        dataset1, _ = pair[0]
        dataset2, _ = pair[1]

        if dataset1 in datasets and dataset2 in datasets:
            
            if fabricated_to_source(dataset1, benchmark) == fabricated_to_source(dataset2, benchmark):
                dataset_non_joins.append(pair)
         
    return dataset_joins, dataset_non_joins


def prepare_data_datasets(join_features: dict, non_join_features: dict, training_ratio: int, no_datasets_sampled: int,
                          sources_fabricated: dict, samples_path: str, sample_file: str, file_sampling: str = 'random'):
    """
    Splits data into training and test ones, based on joins/non-joins among a subset of datasets.
    :param join_features: The features associated with match pairs
    :param non_join_features: The features associated with non-match pairs
    :param training_ratio: The ratio of negative to positive examples
    :param no_datasets_sampled: The number of datasets to be sampled per source (or x sources if at_random=True)
    :param sources_fabricated: A dictionary storing the corresponding fabricated datasets per source table
    :param sample_file: If specified, it stores the name of the file containing the subset of datasets
    :param file_sampling: Specifies whether we sample per source or based on all fabricated datasets
    :return: Training data + labels and test data + labels, also groups of pairs to compute ranking metrics
    """
    match_pairs = list(join_features.keys())
    non_match_pairs = list(non_join_features.keys())

    sample_datasets = []

    if sample_file:
        with open(sample_file, 'r') as s_file:
            all_lines = s_file.readlines()
        for line in all_lines:
            sample_datasets.append(line.strip())
    else:
        if file_sampling == 'random':
            all_datasets = []
            for k, v in sources_fabricated.items():
                all_datasets.extend(v)
            sample_datasets = sample(all_datasets, no_datasets_sampled)
        elif file_sampling == 'one_domain':
            sources = list(sources_fabricated.keys())
            random_source = choice(sources)
            sample_datasets.extend(sources_fabricated[random_source])
        else:
            for source, fabricated in sources_fabricated.items():
                sample_datasets.extend(sample(fabricated, no_datasets_sampled))

    random_positives, random_negatives = filter_pairs(sample_datasets, match_pairs, non_match_pairs, file_sampling)

    len_train = len(random_positives)
    len_train_negative = len_train * training_ratio if training_ratio > 0 else len(random_negatives)

    print('Number of positive examples: {}'.format(len_train))
    print('Number of negative examples: {}'.format(len(random_negatives)))
    print('Number of sampled negative examples: {}'.format(len_train_negative))

    X_train = []
    y_train = []

    for i in range(len_train):
        X_train.append(join_features[random_positives[i]])
        y_train.append(1)

    for i in range(len_train_negative):
        X_train.append(non_join_features[random_negatives[i]])
        y_train.append(0)

    X_test = []
    y_test = []

    test_match_pairs = list(set(match_pairs).difference(set(random_positives)))
    test_non_match_pairs = list(set(non_match_pairs).difference(set(random_negatives)))

    pairs_grouped = []
    for i in range(len(test_match_pairs)):
        X_test.append(join_features[test_match_pairs[i]])
        y_test.append(1)

        group = []
        column1, column2 = test_match_pairs[i]

        for pair in test_non_match_pairs:
            if column1 in pair:
                group.append(non_join_features[pair])

        shuffle(group)
        pairs_grouped.append([join_features[test_match_pairs[i]]] + group)

        group = []

        for pair in test_non_match_pairs:
            if column2 in pair:
                group.append(non_join_features[pair])

        shuffle(group)
        pairs_grouped.append([join_features[test_match_pairs[i]]] + group)

    for i in range(len(test_non_match_pairs)):
        X_test.append(non_join_features[test_non_match_pairs[i]])
        y_test.append(0)

    return np.array(X_train, dtype=object), np.array(y_train, dtype=np.dtype(float)), np.array(X_test, dtype=object), \
        np.array(y_test, np.dtype(float)), pairs_grouped

def prepare_data_datasets_self(train_join_features: dict, train_non_join_features: dict, test_join_features: dict, 
                               test_non_join_features: dict, sample_file: str, benchmark: str):
    """
    Splits data into training and test ones, based on joins/non-joins among a subset of datasets.
    :param join_features: The features associated with match pairs
    :param non_join_features: The features associated with non-match pairs
    :param training_ratio: The ratio of negative to positive examples
    :param no_datasets_sampled: The number of datasets to be sampled per source (or x sources if at_random=True)
    :param sources_fabricated: A dictionary storing the corresponding fabricated datasets per source table
    :param sample_file: If specified, it stores the name of the file containing the subset of datasets
    :param benchmark: The benchmark from which the datasets come
    :return: Training data + labels and test data + labels, also groups of pairs to compute ranking metrics
    """
    train_match_pairs = list(train_join_features.keys())
    train_non_match_pairs = list(train_non_join_features.keys())

    sample_datasets = []

    
    with open(sample_file, 'r') as s_file:
        all_lines = s_file.readlines()
        for line in all_lines:
            sample_datasets.append(line.strip())
  

    random_positives, random_negatives = filter_pairs(sample_datasets, train_match_pairs, train_non_match_pairs, benchmark)

    len_train = len(random_positives)
    len_train_negative = len(random_negatives)

    print('Number of positive examples: {}'.format(len_train))
    print('Number of negative examples: {}'.format(len(random_negatives)))

    X_train = []
    y_train = []

    for i in range(len_train):
        X_train.append(train_join_features[random_positives[i]])
        y_train.append(1)

    for i in range(len_train_negative):
        X_train.append(train_non_join_features[random_negatives[i]])
        y_train.append(0)

    X_test = []
    y_test = []

    test_match_pairs = list(test_join_features.keys())
    test_non_match_pairs = list(test_non_join_features.keys())

    pairs_grouped = []
    for i in range(len(test_match_pairs)):
        X_test.append(test_join_features[test_match_pairs[i]])
        y_test.append(1)

        group = []
        column1, column2 = test_match_pairs[i]

        for pair in test_non_match_pairs:
            if column1 in pair:
                group.append(test_non_join_features[pair])

        shuffle(group)
        pairs_grouped.append([test_join_features[test_match_pairs[i]]] + group)

        group = []

        for pair in test_non_match_pairs:
            if column2 in pair:
                group.append(test_non_join_features[pair])

        shuffle(group)
        pairs_grouped.append([test_join_features[test_match_pairs[i]]] + group)

    for i in range(len(test_non_match_pairs)):
        X_test.append(test_non_join_features[test_non_match_pairs[i]])
        y_test.append(0)

    return np.array(X_train, dtype=object), np.array(y_train, dtype=np.dtype(float)), np.array(X_test, dtype=object), \
        np.array(y_test, np.dtype(float)), pairs_grouped


# -------------------------- Data preparation routines for classifier.py [END] --------------------------


def extract_filename(path: str) -> str:
    """
    Extracts the filename from a full path
    :param path: fullpath of file
    :return: filename
    """
    return os.path.basename(path)


def camel_case_split(string_value: str) -> List[str]:
    """
    Returns a list of tokens based on CamelCase
    :param string_value: Input string
    :return: List of tokens identified based on CamelCase (in lowercase)
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', string_value)
    return [m.group(0).lower() for m in matches]


def tokenizer(string_value: str, split_words: bool = False) -> List[str]:
    """
    Returns a list of tokens based on CamelCase and _, / and space delimiters
    :param split_words: True if we want to find real world tokens in a string, e.g. 'applebanana' -> apple , banana
    :param string_value: Input string
    :return: List of tokens identified (in lowercase)
    """

    initial_tokens = list(filter(None, re.split('_| |/|(\d+)', string_value)))

    tokens = [camel_case_split(t) for t in initial_tokens]

    if split_words:
        tokens_flattened = list(itertools.chain(*tokens))

        tokens = [wordninja.split(t) for t in tokens_flattened]

    return list(itertools.chain(*tokens))


def get_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns columns of a pandas DataFrame as a list
    :param df: The input DataFrame
    :return: List of all column headers in the DataFrame
    """

    return df.columns.tolist()


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Simple function that computes the embedding cosine similarity of two vectors

    :return: Cosine similarity of embedding(word1) and embedding(word2)
    """

    cosine_sim = (np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2))

    return float(cosine_sim)


def csvs_to_dataframes(dir_path: str) -> Dict[str, pd.DataFrame]:
    """
    Reads all .csvs files under a given path and returns a list of their corresponding dataframes
    :param dir_path: Fullpath under which we crawl .csvs
    :return: A dict of all dataframes created from the .csvs found under the fullpath
    """

    dataframe_dict = dict()

    for filename in os.listdir(dir_path):

        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(dir_path, filename))

            columns = df.columns.tolist()

            for col in columns:
                df.rename(columns={col: col.rstrip()}, inplace=True)
            dataframe_dict[filename] = df

    return dataframe_dict

def fabricated_to_source(fabricated_name: str, benchmark: str) -> str:
    """
    Simple function that returns the name of the source on which the fabricated dataset is based. It uses info on the
    naming conventions we use for the fabricated files, i.e. sourcename_[clean|noisy]_id.csv
    :param fabricated_name: The filename of the fabricated dataset
    :param benchmark: The benchmark that the filename comes from
    :return: The name of the source -> sourcename
    """

    if benchmark in ['city_government', 'culture_recreation']:
        return re.split('_clean_|_noisy_', fabricated_name)[0]
    else:
        return re.split('_[0-9]+.csv', fabricated_name)[0]

def load_embeddings(embedding_file: str) -> Dict[str, np.array]:
    """
    Loads embeddings from the specified file
    :param embedding_file: The file containing the embedding vectors
    """

    embeddings = dict()
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)

            embeddings[word] = embedding

    return embeddings


def value_tokenizer(value: str) -> List[str]:
    """
    Tokenizes string values based on non-alphanumerical characters.
    :param value: A string value
    :return: Tokens when splitting on non-alphanumerical characters.
    """

    value_tokens = list(filter(lambda x: len(x) > 0, re.split('[^0-9a-zA-Z]', value)))

    return value_tokens


def unsupervised_matcher(positive_features, negative_features, threshold: float = 0.75):
    """
    A simple matcher that for each pair looks at the maximum value of a feature (with a range from 0 to 1). If the value
    of the max feature is above a threshold then the column pair is regarded as a join.

    :param positive_features: A dictionary storing features of all join pairs
    :param negative_features: A dictionary storing features of all non-join pairs
    :param threshold: The value of the threshold used to judge whether a pair is a join or not
    :return: Precision, Recall and F1-scores
    """

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for pair, features in positive_features.items():

        max_feat = max(features)

        if max_feat > threshold:
            tp += 1
        else:
            fn += 1

    for pair, features in negative_features.items():

        max_feat = max(features)

        if max_feat > threshold:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    f1_score = 0 if not precision or not recall else 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def return_features_labels(match_features, non_match_features):
    features = []
    labels = []

    for pair, features in match_features.items():
        features.append(features)
        labels.append(1)

    for pair, features in non_match_features.items():
        features.append(features)
        labels.append(0)

    return features, labels


def mask_alnums(string_value: str):
    """
    Masks all alphanumerical values in a string
    :param string_value: The string value to be masked
    :return: The masked value (* is used for the mask)
    """

    character_list = []

    for ch in string_value:
        if ch.isalnum():
            character_list.append('*')
        else:
            character_list.append(ch)

    return ''.join(character_list)


def updated_values(dict1, dict2):
    """
    Updates a dictionary by extending existing list values with values stored in another dictionary
    :param dict1: Dictionary storing key -> list
    :param dict2: Dictionary storing key -> list
    :return: Dictionary storing for each key the list values specified by both dictionaries
    """

    for key, value in dict2.items():
        if key in dict1:
            dict1[key].extend(value)
        else:
            dict1[(key[1], key[0])].extend(value)

    return dict1


def get_features(features, features_path, pair_type: str):
    """
    Reads pickle files containing computed features and returns a dictionary storing for each pair all specified features
    :param features: A list of features to be included
    :param features_path: The path where we have stored the pickle files
    :param pair_type: The type of pairs -> pos for join pairs, neg for non-join pairs
    :return: A dictionary storing for each pair the set of features specified
    """

    feat_dict = dict()

    for feature in features:
        with open('{0}/{1}_{2}.pickle'.format(features_path, feature, pair_type), 'rb') as input_file:
            feats = pickle.load(input_file)
            if not feat_dict:
                feat_dict = {pair: [] for pair in feats}
            if feature == 'jaccard_containment':              
                for pair, jaccard_features in feats.items():
                    if not jaccard_features:
                        feats[pair] = [0, 0]
                    else:
                        #feats[pair] = [max(jaccard_features[1:])]
                        #feats[pair] = [jaccard_features[0]]
                        feats[pair] = [jaccard_features[0], max(jaccard_features[1:])]
            else:
                for pair, feature in feats.items():
                    if not feature:
                        feats[pair] = [0]
            feat_dict = updated_values(feat_dict, feats)

    return feat_dict


def jensen_shanon_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Computes Jensen-Shannon (JS) similarity (1 - divergence) of two lists of tokens crawled from two DataFrame columns
    :param tokens_a: Tokens of first column
    :param tokens_b: Tokens of second column
    :return: 1 - JS divergence of the two lists of tokens
    """

    p_a = {token: token_count / len(tokens_a) for token, token_count in Counter(tokens_a).items()}
    p_b = {token: token_count / len(tokens_b) for token, token_count in Counter(tokens_b).items()}

    # initialize sums for Kullback-Leibler divergences
    kl_a = 0
    kl_b = 0

    for token, freq in p_a.items():
        freq_b = 0
        if token in p_b:
            freq_b = p_b[token]
        prod = freq * log(freq / (0.5 * (freq + freq_b)))
        kl_a += prod

    for token, freq in p_b.items():
        freq_a = 0
        if token in p_a:
            freq_a = p_a[token]
        prod = freq * log(freq / (0.5 * (freq + freq_a)))
        kl_b += prod

    js = 0.5 * (kl_a + kl_b)

    return 1 - js


# ----------------------------------- Following two routines as taken from Sherlock repository -----------------------------------

def extract_bag_of_characters_features(data):
    characters_to_check = (
            ['[' + c + ']' for c in string.printable if c not in ('\n', '\\', '\v', '\r', '\t', '^')]
            + ['[\\\\]', '[\^]']
    )

    f = OrderedDict()

    data_no_null = data.dropna()
    all_value_features = OrderedDict()

    for c in characters_to_check:
        all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    return f


def extract_bag_of_words_features(data, n_val):
    f = OrderedDict()
    data = data.dropna()

    # Entropy of column
    freq_dist = nltk.FreqDist(data)
    probs = [freq_dist.freq(l) for l in freq_dist]
    f['col_entropy'] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = data.nunique()
    f['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(data.str.contains('[0-9]', regex=True))
    text_cells = np.sum(data.str.contains('[a-z]|[A-Z]', regex=True))
    f['frac_numcells'] = num_cells / n_val
    f['frac_textcells'] = text_cells / n_val

    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    f['avg_num_cells'] = np.mean(data.str.count(num_reg))
    f['std_num_cells'] = np.std(data.str.count(num_reg))

    # Average + std number of textual tokens in cells
    text_reg = '[a-z]|[A-Z]'
    f['avg_text_cells'] = np.mean(data.str.count(text_reg))
    f['std_text_cells'] = np.std(data.str.count(text_reg))

    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    f['avg_spec_cells'] = np.mean(data.str.count(spec_reg))
    f['std_spec_cells'] = np.std(data.str.count(spec_reg))

    # Average number of words in each cell
    space_reg = '[" "]'
    f['avg_word_cells'] = np.mean(data.str.count(space_reg) + 1)
    f['std_word_cells'] = np.std(data.str.count(space_reg) + 1)

    all_value_features = OrderedDict()

    data_no_null = data.dropna()

    f['n_values'] = n_val

    all_value_features['length'] = data_no_null.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    n_none = data.size - data_no_null.size - len([e for e in data if e == ''])
    f['none-agg-has'] = n_none > 0
    f['none-agg-percent'] = n_none / len(data)
    f['none-agg-num'] = n_none
    f['none-agg-all'] = (n_none == len(data))

    return f

def best_f1(precisions, recalls):
    
    best = max([2*precisions[i]*recalls[i]/(precisions[i] + recalls[i]) for i in range(len(precisions)) if precisions[i] + recalls[i] > 0])
    
    return best

def get_individual_features(data: pd.DataFrame) -> pd.DataFrame:
    """
        Code for profiling tabular datasets and based on the featurizer of Sherlock.
        Input:
            data: A pandas DataFrame with each row a list of string values
        Output:
            a dataframe where each row represents a column and columns represent the features
            computed for the corresponding column.
    """

    # Transform data so that each column becomes a row with its corresponding values as a list

    data = data.T
    list_values = data.values.tolist()
    data = pd.DataFrame(data={'values': list_values})

    data_columns = data['values']

    features_list = []

    for column in data_columns:
        column = pd.Series(column).astype(str)

        f = OrderedDict(list(extract_bag_of_characters_features(column).items()) + list(
            extract_bag_of_words_features(column, len(column)).items()))

        features_list.append(f)

    return pd.DataFrame(features_list).reset_index(drop=True) * 1


def columns_to_features(dataframes):
    col_features = dict()

    for file, dataframe in tqdm(dataframes.items()):

        col_features_pd = get_individual_features(dataframe)

        cols = dataframe.columns.tolist()

        feature_list = col_features_pd.values.tolist()

        for i in range(len(cols)):
            col_features[(file, cols[i])] = feature_list[i]

    return col_features


def _get_negative_neighbors(negative_pairs, columns_to_ids):
    """
    Function that creates a dict of negative neighbor column ids for each column
    :param negative_pairs: The set of non-join pairs based on which we create the dict
    :param columns_to_ids: A dict storing column to id correspondence
    :return: A dictionary of column_id -> [non_join_column_1, non_join_column2 ...]
    """

    negative_neighbors = OrderedDict()

    for pair in negative_pairs:

        column1, column2 = pair

        if isinstance(column1, int):
            cid1 = column1
            cid2 = column2
        else:
            cid1 = columns_to_ids[column1]
            cid2 = columns_to_ids[column2]

        if cid1 in negative_neighbors:
            negative_neighbors[cid1].append(cid2)
        else:
            negative_neighbors[cid1] = [cid2]

        if column2 in negative_neighbors:
            negative_neighbors[cid2].append(cid1)
        else:
            negative_neighbors[cid2] = [cid1]

    return negative_neighbors


def parse_configuration():
    """
    A simple function for parsing a configuration file
    :return: The parameters of the configuration. Features is a list os it's returned as a separate variable
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-cf", "--configuration_file", type=str, help='specify path to configuration file')
    args = parser.parse_args()

    config = configparser.ConfigParser()

    config.read(args.configuration_file)

    features = []
    for feature_item in config.items('FEATURES'):
        feature, included = feature_item
        if included == 'True':
            features.append(feature)

    return config, features