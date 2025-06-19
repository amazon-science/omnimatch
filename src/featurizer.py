import numpy as np
from typing import List, Tuple
import pandas as pd
from tools import csvs_to_dataframes, get_columns, tokenizer,fabricated_to_source_filename, \
    cosine_similarity, drop_row_nums, value_tokenizer, jensen_shanon_similarity, \
    get_individual_features, parse_configuration, fabricated_to_source_self
import itertools
from tqdm import tqdm
from collections import Counter
import fasttext
import time
import pickle
from multiprocess_tools import run_multithread_jaccard_similarity
import time


class Featurizer:

    def __init__(self, path: str):
        self.dataframes = csvs_to_dataframes(path)
        self.token_itf = None
        self.tokens = None
        self.infrequent_tokens = None
        self.frequent_tokens = None
        self.value_tokens = None
        self.column_features = None

    def featurize_columns(self):
        """
        Computes statistics of columns to be used as initial features in the GNN models.
        """

        col_features = dict()

        for file, dataframe in tqdm(self.dataframes.items()):

            col_features_pd = get_individual_features(dataframe)

            cols = dataframe.columns.tolist()

            feature_list = col_features_pd.values.tolist()

            for i in range(len(cols)):
                col_features[(file, cols[i])] = feature_list[i]

        self.column_features = col_features

    def tokenize_columns(self, split_words: bool = False):
        """
        Returns the token lists of all column names of DataFrames
        :param split_words: If True then the tokenizer will try to split strings to words
        """

        dataframe_column_tokens = dict()

        for df_name, df in self.dataframes.items():

            columns = get_columns(df)
            dataframe_column_tokens[df_name] = {col: None for col in columns}

            for col in columns:
                tokens = tokenizer(col, split_words)

                dataframe_column_tokens[df_name][col] = tokens

        self.tokens = dataframe_column_tokens

    def frequent_token_sets(self):
        """
        Computes frequent, infrequent and complete token sets of all columns
        """

        dataframe_column_iftokens = dict()
        dataframe_column_ftokens = dict()
        dataframe_column_tokens = dict()

        for df_name, df in self.dataframes.items():

            columns = get_columns(df)

            dataframe_column_iftokens[df_name] = {col: set() for col in columns}
            dataframe_column_ftokens[df_name] = {col: set() for col in columns}
            dataframe_column_tokens[df_name] = {col: [] for col in columns}

            for col in columns:

                all_tokens = []

                values = list(
                    df[col].dropna().map(str).map(lambda x: x.lower()).map(lambda x: x[:-2] if x.endswith('.0') else x))
                value_tokens = dict()

                for value in values:
                    tokens = value_tokenizer(str(value))
                    all_tokens.extend(tokens)
                    value_tokens[value] = tokens

                value_frequencies = Counter(all_tokens)

                for value in values:
                    if len(value_tokens[value]) > 0:
                        dataframe_column_iftokens[df_name][col].add(
                            min(value_tokens[value], key=lambda v: value_frequencies[v]))
                        dataframe_column_ftokens[df_name][col].add(
                            max(value_tokens[value], key=lambda v: value_frequencies[v]))
                        dataframe_column_tokens[df_name][col].extend(value_tokens[value])

        self.infrequent_tokens = dataframe_column_iftokens
        self.frequent_tokens = dataframe_column_ftokens
        self.value_tokens = dataframe_column_tokens

    @staticmethod
    def compute_value_based_embedding(tokens: List[str], embeddings):
        """
        Computes embedding of a list of tokens as the mean of their embeddings
        :param tokens: A list of string tokens
        :param embeddings: The embedding fasttext model
        :return: The embedding of the list of tokens
        """

        all_embeddings = []

        for token in tokens:
            all_embeddings.append(embeddings.get_word_vector(token))

        if len(all_embeddings) == 0:
            return np.empty(0)

        return np.nanmean(np.array(all_embeddings), axis=0)

    def compute_value_embedding_similarity(self, column_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]]):
        """
        Computes value-based similarity among column pairs and stores them in a dictionary
        :param column_pairs: The list of all possible column pairs
        :return: A dict storing for each pair of columns their value-based embedding similarity
        """
        column_value_embedding_sim = dict()

        word_embeddings = fasttext.load_model('/usr/local/matching/embedding_files/cc.en.50.bin')

        for pair1, pair2 in tqdm(column_pairs):
            df1, col_name1 = pair1
            df2, col_name2 = pair2

            tokens1 = self.frequent_tokens[df1][col_name1]
            tokens2 = self.frequent_tokens[df2][col_name2]

            embedding1 = self.compute_value_based_embedding(tokens1, word_embeddings)
            embedding2 = self.compute_value_based_embedding(tokens2, word_embeddings)

            if not embedding1.size or not embedding2.size:
                column_value_embedding_sim[(pair1, pair2)] = 0.0
            else:
                column_value_embedding_sim[(pair1, pair2)] = cosine_similarity(embedding1, embedding2)

        return column_value_embedding_sim

   
    def compute_jaccard_infrequent_tokens(self, column_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]]):
        """
        Returns column similarity based on jaccard similarity of their less frequent token sets
        :param column_pairs: Contains all column pairs for which the similarity will be computed
        :return: Jaccard similarity of less frequent token sets for each column pair
        """

        column_jaccard_infrequent_similarity = dict()

        for pair1, pair2 in tqdm(column_pairs):
            df1, col_name1 = pair1
            df2, col_name2 = pair2

            tokens1 = self.infrequent_tokens[df1][col_name1]
            tokens2 = self.infrequent_tokens[df2][col_name2]

            if len(tokens1.union(tokens2)) == 0:
                column_jaccard_infrequent_similarity[(pair1, pair2)] = 0
            else:

                jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

                column_jaccard_infrequent_similarity[(pair1, pair2)] = jaccard

        return column_jaccard_infrequent_similarity

    def compute_value_distribution_similarity(self, column_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]]):
        """
        Computes distribution similarity among column pairs based on Jensen-Shanon divergence
        :param column_pairs: List containing all possible column pairs
        :return: A dictionary storing for each pair the distribution similarity score
        """

        column_dist_similarity = dict()

        for pair1, pair2 in tqdm(column_pairs):
            df1_name, col_name1 = pair1
            df2_name, col_name2 = pair2

            js_similarity = jensen_shanon_similarity(self.tokens[df1_name][col_name1], self.tokens[df2_name][col_name2])

            column_dist_similarity[(pair1, pair2)] = js_similarity

        return column_dist_similarity

    def compute_jaccard_containment(self, column_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                                    row_limit: int = 0, in_parallel: bool = False):
        """
        Returns column similarity based on jaccard and containment similarity among all columns in column_pairs
        :param column_pairs: Contains all column pairs for which jaccard/containment similarity will be computed
        :param row_limit: Determines whether jaccard/containment similarity is approximated based on top-row_limit rows
        :param in_parallel: Determines whether the Jaccard/Containment scores will be computed in parallel
        :return: Jaccard/containment similarity for each column pair
        """

        column_jaccard_similarity = dict()
        column_containment = dict()

        if in_parallel:

            similarities = run_multithread_jaccard_similarity(self.dataframes, column_pairs, 2)

            for pair, sims in similarities.items():
                column_jaccard_similarity[pair] = sims[0]
                column_containment[pair] = sims[1:]
        else:

            for pair1, pair2 in tqdm(column_pairs):

                df1, col_name1 = pair1
                df2, col_name2 = pair2

                col1 = self.dataframes[df1][col_name1].dropna().map(str).map(lambda x: x.lower()).map(
                    lambda x: x[:-2] if x.endswith('.0') else x)
                col2 = self.dataframes[df2][col_name2].dropna().map(str).map(lambda x: x.lower()).map(
                    lambda x: x[:-2] if x.endswith('.0') else x)

                p1 = p2 = p1p2 = 1
                if row_limit:
                    p1 = row_limit / len(col1)
                    p2 = row_limit / len(col2)
                    p1p2 = p1 * p2

                    col1 = col1[:row_limit]
                    col2 = col2[:row_limit]

                set1 = {element for element in set(col1) if pd.notna(element)}
                set2 = {element for element in set(col2) if pd.notna(element)}

                inter = set1.intersection(set2)
                diff1 = set1.difference(set2)
                diff2 = set2.difference(set1)

                sig1 = len(inter) == 0 and len(diff2) == 0
                sig2 = len(inter) == 0 and len(diff1) == 0


                if sig1 and sig2:
                    jaccard_sim = 0
                else:
                    jaccard_sim = (len(inter) / p1p2) / (len(diff1) / p1 + len(diff2) / p2 + len(inter) / p1p2)

                if sig1:
                    containment1 = 0
                else:
                    containment1 = (len(inter) / p1p2) / (len(diff2) / p2 + len(inter) / p1p2)
                
                if sig2:
                    containment2 = 0
                else:
                    containment2 = (len(inter) / p1p2) / (len(diff1) / p1 + len(inter) / p1p2)

                column_jaccard_similarity[(pair1, pair2)] = jaccard_sim
                column_containment[(pair1, pair2)] = (containment1, containment2)

        return column_jaccard_similarity, column_containment

    def _pair_featurization(self, column_pairs: list, embedding_file: str, included_features: list):
        """
        Function that based on the features to be included for training a column edge classification model, calls the
        corresponding functions to compute them and returns for each column its corresponding feature set
        :param column_pairs: A list containing all column pairs for which the features will be computed
        :param embedding_file: The file containing the pre-trained glove embeddings model
        :param included_features: A list that contains the features to be computed and used for training
        :return: A dictionary storing for each column pair its corresponding feature set
        """
        features = {pair: [] for pair in column_pairs}

        if 'jaccard_containment' in included_features:
            print('--------> Computing jaccard/containment scores <--------')

            jaccard_similarities, containment_similarities = self.compute_jaccard_containment(column_pairs)
        else:
            jaccard_similarities = {pair: None for pair in column_pairs}
            containment_similarities = {pair: [None] for pair in column_pairs}

        if 'jaccard_frequent' in included_features:
            print('--------> Computing jaccard frequent scores <--------')

            jf_similarities = self.compute_jaccard_infrequent_tokens(column_pairs)
        else:
            jf_similarities = {pair: None for pair in column_pairs}

        if 'value_embeddings' in included_features:
            print('-------> Computing value-based embedding similarities <--------')

            ve_similarities = self.compute_value_embedding_similarity(column_pairs)
        else:
            ve_similarities = {pair: None for pair in column_pairs}

        if 'value_distribution' in included_features:
            print('-------> Computing value-based distribution similarities <--------')

            dist_similarities = self.compute_value_distribution_similarity(column_pairs)
        else:
            dist_similarities = {pair: None for pair in column_pairs}

        for pair in column_pairs:
            features[pair].append(jaccard_similarities[pair])

            features[pair].extend([score for score in containment_similarities[pair]])

            features[pair].append(jf_similarities[pair])

            features[pair].append(ve_similarities[pair])

            features[pair].append(dist_similarities[pair])

            features[pair] = [feat for feat in features[pair] if isinstance(feat, float)]

        return features

    def featurize_column_pairs(self, embedding_file: str, match_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                               non_match_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]], feature_mask: List[str]):
        """
        Returns feature vectors for every column pair included in the given list.
        :param embedding_file: The embedding file based on which we compute column name representations
        :param match_pairs: A list of all matching column pairs for which we want to compute features
        :param non_match_pairs: A list of all non-matching column pairs for which we want to compute features
        :param feature_mask: A list containing all features that should be computed and used
        :return: A mapping from each match/non-match pair to a feature vector
        """

        print('--------> Computing uniqueness scores <--------')
        uniqueness_scores = self.compute_header_uniqueness()

        print('****POSITIVE PAIRS****')

        join_features = self._pair_featurization(match_pairs, embedding_file, feature_mask)
        print('****NEGATIVE PAIRS****')

        non_join_features = self._pair_featurization(non_match_pairs, embedding_file, feature_mask)

        return join_features, non_join_features


if __name__ == '__main__':
    config, features = parse_configuration()

    print('Read Datasets')
    featurizer = Featurizer(config['PATHS']['dataset_path'])
    column_features = config['OTHER'].getboolean('compute_column_features')
    features_path = config['PATHS']['features_path']

    if column_features:


        featurizer.featurize_columns()

        filename_features = '{}/individual_features.pickle'.format(features_path)

        with open(filename_features, 'wb') as f:
            pickle.dump(featurizer.column_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    with open(config['PATHS']['join_pairs_file'], 'rb') as join_file:
        matches = pickle.load(join_file)

    with open(config['PATHS']['non_join_pairs_file'], 'rb') as non_join_file:
        non_matches = pickle.load(non_join_file)

    print('NUMBER OF POSITIVE PAIRS = {}'.format(len(matches)))
    print('NUMBER OF NEGATIVE PAIRS = {}'.format(len(non_matches)))

    print('############ TOKENIZATION OF COLUMN NAMES ############')
    start = time.time()
    featurizer.tokenize_columns(split_words=True)
    end = time.time()
    print('############ FINISHED IN {} seconds'.format(end - start))

    if not {'jaccard_frequent', 'value_embeddings', 'value_distribution'}.isdisjoint(set(features)):
        print('############ TOKENIZATION OF COLUMN VALUES ############')
        start = time.time()
        featurizer.frequent_token_sets()
        end = time.time()
        print('############ FINISHED IN {} seconds'.format(end - start))

    print('############ COMPUTING FEATURES FOR POSITIVE/NEGATIVE PAIRS ############')
    start = time.time()
    positive_features, negative_features = featurizer.featurize_column_pairs(config['PATHS']['embeddings_file'], matches,
                                                                             non_matches, features)
    end = time.time()
    print('############ FINISHED IN {} seconds'.format(end - start))

    feature = features[0]

    filename_pos = '{0}/{1}_pos.pickle'.format(features_path, feature)

    with open(filename_pos, 'wb') as output_file:
        pickle.dump(positive_features, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    filename_neg = '{0}/{1}_neg.pickle'.format(features_path, feature)

    with open(filename_neg, 'wb') as output_file:
        pickle.dump(negative_features, output_file, protocol=pickle.HIGHEST_PROTOCOL)