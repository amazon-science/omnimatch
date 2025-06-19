import os
from rf_classifier import Classifier
import numpy as np
from  tools import get_features, prepare_data_datasets_self, parse_configuration, best_f1
import pickle
from sklearn.metrics import precision_recall_curve

def train_and_compute_metrics(train: np.ndarray, test: np.ndarray, train_labels: np.ndarray, test_labels: np.ndarray,
                              groups, num_est: int, results_path: str, write_results: bool=False):
    """
    Trains a classifier and computes evaluation metrics based on the predictions of the model
    :param train: The numpy array containing the features of the training data
    :param test: The numpy array containing the features of the test data
    :param train_labels: The numpy array containing the labels of the training data
    :param test_labels: The numpy array containing the labels of the test data
    :param groups: A list containing sublists of data features. Each group of data features contains a feature set of
    a join pair in the beginning while the rest are features of non-join pairs for one of the columns in the join pair.
    We use groups to compute ranking scores.
    :param num_est: Number of decision trees used in the Random Forest Classifier
    :return: A dictionary holding all evaluation metrics
    """
    model = Classifier(no_est=num_est)

    model.train_classifier(train, train_labels)

    pras = model.compute_metrics(test, test_labels)

    scores = model.classifier.predict_proba(test)[:, 1]

    if write_results:
        with open('{}/rf_scores.pickle'.format(results_path), 'wb') as f:
                    pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('{}/rf_labels.pickle'.format(results_path), 'wb') as f:
                pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    ranks = []
    hits_at_1_scores = []
    hits_at_3_scores = []
    hits_at_10_scores = []
    for group in groups:
        group_array = np.array(group, dtype=object)

        probabilities = model.classifier.predict_proba(group_array)[:, 1]

        indices = np.argsort(-1 * probabilities, kind='stable')

        rank = np.where(indices == 0)[0][0] + 1

        ranks.append(1 / rank)
        hits_at_1_scores.append(int(rank == 1))
        hits_at_3_scores.append(int(rank <= 3))
        hits_at_10_scores.append(int(rank <= 10))

    mean_reciprocal_rank = sum(ranks) / len(ranks)
    hits_at_1 = sum(hits_at_1_scores) / len(hits_at_1_scores)
    hits_at_3 = sum(hits_at_3_scores) / len(hits_at_3_scores)
    hits_at_10 = sum(hits_at_10_scores) / len(hits_at_10_scores)

    p, r, _ = precision_recall_curve(test_labels, scores)
    best_f1_score = best_f1(p, r)

    results_dict = {'PR-AUC': pras, 'MRR': mean_reciprocal_rank, 'Hits@1': hits_at_1, 'Hits@3': hits_at_3, 'Hits@10': hits_at_10, 'Best F1': best_f1_score}


    return results_dict


if __name__ == "__main__":

    config, features = parse_configuration()

    train_features_path = config['PATHS']['train_features_path']
    test_features_path = config['PATHS']['test_features_path']

    iterations = config['CLASSIFIER'].getint('num_iterations')
    benchmark = config['CLASSIFIER'].get('benchmark')

    train_positive_features = get_features(features, train_features_path, pair_type='pos')
    train_negative_features = get_features(features, train_features_path, pair_type='neg')

    test_positive_features = get_features(features, test_features_path, pair_type='pos')
    test_negative_features = get_features(features, test_features_path, pair_type='neg')

    results_total = {'PR-AUC': [], 'MRR': [], 'Hits@1': [], 'Hits@3': [], 'Hits@10': [], 'Best F1': []}
    # Train the Random Forest Classifier as many times as specified by iterations
    for i in range(iterations):

        print('Iteration {}'.format(i + 1))

        X_train, y_train, X_test, y_test, pair_groups = prepare_data_datasets_self(train_positive_features, train_negative_features,
                                                                                   test_positive_features, test_negative_features,
                                                                                   sample_file=config['PATHS']['sampled_datasets'],
                                                                                   benchmark=benchmark)
        

        results = train_and_compute_metrics(X_train, X_test, y_train, y_test, pair_groups,
                                            config['CLASSIFIER'].getint('num_estimators'),
                                            config['PATHS']['results_path'])

        for metric, value in results.items():
            results_total[metric].append(value)

    for metric, values in results_total.items():
        mean = sum(values) / iterations
        std = sum([((x - mean) ** 2) for x in values]) / iterations

        print('Mean {0} is {1} with a std of {2}'.format(metric, mean, std))