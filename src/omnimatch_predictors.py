import pickle
from collections import OrderedDict
import dgl
import torch
import torch.nn.functional as F
from omnimatch_models import train_model_triplet_loss, train_mlp_model, train_model
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
from tools import get_features, _get_negative_neighbors, parse_configuration, filter_pairs, fabricated_to_source, best_f1
from random import sample
import os


def get_column_ids(features_file: str):
    """
    Simple function that assigns an id to each column of a set of datasets and returns column to ids, ids to columns and
    ids to features dictionaries
    :param features_file: File containing for each column its corresponding feature set
    :return 3 dictionaries: column -> id, id -> column, id -> features
    """
    with open(features_file, 'rb') as feat_file:
        node_features = pickle.load(feat_file)
    
    column_to_id = dict()
    id_to_column = dict()
    id_to_features = []
    c_id = 0

    for column, features in node_features.items():
        column_to_id[(column[0], column[1].rstrip())] = c_id
        id_to_column[c_id] = (column[0], column[1].rstrip())
        id_to_features.append(features)
        c_id += 1

    return column_to_id, id_to_column, id_to_features


def create_edge_dict_threshold(join_features: dict, non_join_features: dict, column_ids: dict, num_of_features: int, threshold: float):
    """
    Based on column join/non-join features, this function adds a feature edge among two columns (non-directed)
    if the feature is above a specified threshold
    :param join_features: Dict holding for each column join pair its corresponding feature set
    :param non_join_features: Dict holding for each column non-join pair its corresponding feature set
    :param column_ids: A dictionary with column to id correspondences
    :param num_of_features: The total number of features computed per column pair
    :param threshold: The threshold which determines whether a feature edge will be created for a column pair
    :return: A dictionary storing for each feature the corresponding edges that will be kept
    """

    join_features.update(non_join_features)

    edges_per_feature = OrderedDict([(i, []) for i in range(num_of_features)])

    for pair, features in join_features.items():

        for i, feature in enumerate(features):
            if feature >= threshold:
                column1, column2 = pair
                edges_per_feature[i].append((column_ids[column1], column_ids[column2]))

    return edges_per_feature


def create_edge_list_topk(join_features: dict, non_join_features: dict, column_ids: dict, num_of_features: int, k: int):
    """
    Based on column join/non-join features, this function adds a feature edge (directed) among two columns if it's in the
    top-k ones of one of the two columns (based on the feature values)
    :param join_features: Dict holding for each column join pair its corresponding feature set
    :param non_join_features: Dict holding for each column non-join pair its corresponding feature set
    :param column_ids: A dictionary with column to id correspondences
    :param num_of_features: The total number of features computed per column pair
    :param k: Determines how many of the top edges per feature and per node will be regarded
    :return: A dictionary storing for each feature the corresponding edges that will be kept + A list holding the corresponding
             types of the edges (type goes from 0 to num_features -1)
    """

    join_features.update(non_join_features)

    neighbors_dict = dict()

    for pair, features in join_features.items():

        c1, c2 = pair

        if c1 in neighbors_dict:
            neighbors_dict[c1].append((c2, features))
        else:
            neighbors_dict[c1] = [(c2, features)]

        if c2 in neighbors_dict:
            neighbors_dict[c2].append((c1, features))
        else:
            neighbors_dict[c2] = [(c1, features)]

    edge_types = []
    edge_list = []


    for node, neighbors in neighbors_dict.items():

        node_id = column_ids[node]
        for i in range(num_of_features):
            neighbors.sort(key=lambda x: x[1][i], reverse=True)

            for j in range(min(k, len(neighbors))):
                if neighbors[j][1][i] > -1:
                    edge_list.append((column_ids[neighbors[j][0]], node_id))
                    edge_types.append(i)


    return edge_list, edge_types


def edge_lists_types(edges_per_feature: dict):
    """
    Receives a dict storing for each feature the corresponding directed edges to be added to the graph. For each edge,
    it also adds the opposite one (since we want to create a non-directed graph) and computes a list storing all corresponding
    edge types (in the right sequence).
    :param edges_per_feature: A dict storing for each feature the corresponding directed edges to be added to a graph
    :return A list of all edges to be added to the graph (initial ones and opposites) + a list of all corresponding feature edge types
    """
    edge_types = []
    edges = []

    for edge_type, edges in edges_per_feature.items():
        edge_types.extend([edge_type] * 2 * len(edges))

        for e in edges:
            edges.append(e)
            edges.append((e[1], e[0]))

    return edges, edge_types


def get_positive_negative_nodes(join_pairs: list, non_join_pairs: list, column_ids: dict, no_datasets_sampled: int, no_sources_sampled: int, sources_fabricated: dict,
                                sample_file: str, samples_path: str, benchmark: str):
    """

    :param join_pairs: A list containing column join pairs
    :param non_join_pairs: A list containing column non-join pairs
    :param column_ids: A dict storing for each column its corresponding id
    :param no_datasets_sampled: It controls the number of fabricated pairs per source (e.g. if it's 2 then we get two fabricated pairs, etc.) [use even number when ]
    :param no_sources_sampled: It controls the number of source datasets from which we fabricate pairs
    :param sources_fabricated: A dictionary storing source_file_name -> [fabricated_name1, fabricated_name2, ...]
    :param sample_file: The fullpath of a file storing the dataset names to be included for training
    :return: A list containing all join pairs included in the sampled datasets for training, a dictionary storing
             for each column the corresponding columns with which it doesn't join, and two lists containing the
             join/non-join pairs for testing.
    """
    dataset_joins, dataset_non_joins = _get_positive_negative_pairs(join_pairs, non_join_pairs, sources_fabricated,
                                                                    no_datasets_sampled, no_sources_sampled, sample_file, samples_path, benchmark)
    
    print("Positive pairs: {0} / Negative pairs: {1}".format(len(dataset_joins), len(dataset_non_joins)))

    non_join_neighbors = _get_negative_neighbors(dataset_non_joins, column_ids)


    return dataset_joins, non_join_neighbors


def _get_positive_negative_pairs(join_pairs, non_join_pairs, sources_fabricated, sample_datasets, sample_sources, sample_file, samples_path, benchmark):
    """
    Creates or reads a sample of datasets and returns all joins/non-joins in them
    :param join_pairs: A list containing all column join pairs
    :param non_join_pairs: A list containing all column non-join pairs
    :param sources_fabricated: A dictionary storing source_filename -> [fabricated_name1, fabricated_name2, ...]
    :param sample_pos:
    :param sample_file: If a sample file exists then this contains the full path of the file
    :param file_sampling: The file sampling technique: random randomly selects a subset of datasets,
                          one_domain samples a number of datasets from a randomly chosen source, and
                          non_random samples a number of datasets per source
    :return: All join/non-join pairs included in the sampled subset of datasets
    """
    sampled_datasets = []

    if sample_file.endswith('.txt'): # if there exists a .txt file specifying the sampled datasets
        with open(sample_file, 'r') as s_file:
            all_lines = s_file.readlines()
        for line in all_lines:
            sampled_datasets.append(line.strip())
    else:
        sources = list(sources_fabricated.keys())
        print(f'Length of source: {len(sources)}')
        random_sources = sample(sources, sample_sources)
        for source in random_sources:
            for i in range(sample_datasets):
                if benchmark in ['city_government', 'culture_recreation']:
                    sampled_datasets.append('{0}_noisy_{1}.csv'.format(source, str(i)))
                else:
                    sampled_datasets.append('{0}_{1}.csv'.format(source, str(i)))

      

        filename_datasets = 'samples_' + str(sample_sources) + '_' + str(sample_datasets) + '.txt'

        with open('{}/{}'.format(samples_path,filename_datasets), "w") as sd_file:
            for dataset in sampled_datasets:
                sd_file.write(dataset + '\n')

    dataset_joins, dataset_non_joins = filter_pairs(sampled_datasets, join_pairs, non_join_pairs, benchmark)

    return dataset_joins, dataset_non_joins

def create_join_non_join_lists(join_pairs: list, non_join_pairs: list, no_datasets_sampled: int, no_sources_sampled: int, sources_fabricated: dict,
                                sample_file: str, samples_path: str, benchmark: str):
    """
    Function for constructing join/non-join lists for training/testing
    :param join_pairs: The list of column join pairs
    :param non_join_pairs: The list of column non-join pairs
    :param column_ids: A dictionary storing for each column its corresponding id
    :param no_files_sampled: A parameter controlling the number of datasets to be sampled for training
    :param sample_file: The fullpath of the file containing the subset of datasets to be used for training
    :param sources_fabricated: Dictionary storing source_filename -> [fabricated_name1, fabricated_name2, ...]
    :param file_sampling: The file sampling technique: random randomly selects a subset of datasets,
                          one_domain samples a number of datasets from a randomly chosen source, and
                          non_random samples a number of datasets per source
    :return:
    """
    dataset_joins, dataset_non_joins = _get_positive_negative_pairs(join_pairs, non_join_pairs, sources_fabricated,
                                                                    no_datasets_sampled, no_sources_sampled, sample_file, samples_path, benchmark)

    return dataset_joins, dataset_non_joins

def compute_metrics(join_test_pairs: list, non_join_test_pairs: list, column_ids: dict,
                    column_embeddings, model, model_loss: str):
    """
    Computes metrics (MRR, Hits@k, Best F1)  based on the trained gnn model
    """

    join_scores = []
    non_join_scores = []

    labels = []

    negative_neighbors_test = _get_negative_neighbors(non_join_test_pairs, column_ids)

    ranks = []
    hits_at_1_scores = []
    hits_at_3_scores = []
    hits_at_10_scores = []

    rev = False

    if model_loss in ['rgcn_cross_entropy']:
        rev = True

 
    for pair in join_test_pairs:
        emb_node_1 = column_embeddings[pair[0]]
        emb_node_2 = column_embeddings[pair[1]]

        labels.append(1)

        pos_score = compute_score(emb_node_1, emb_node_2, model_loss, model)
        join_scores.append(pos_score)

        scores = [(pair[1], pos_score)]

        for nn in negative_neighbors_test[pair[0]]:
            neg_emb = column_embeddings[nn]
            score = compute_score(emb_node_1, neg_emb, model_loss, model)
            scores.append((nn, score))

        scores.sort(key=lambda x: x[1], reverse=rev)

        rank = scores.index((pair[1], pos_score)) + 1

        hits_at_1_scores.append(int(rank == 1))
        hits_at_3_scores.append(int(rank <= 3))
        hits_at_10_scores.append(int(rank <= 10))
        ranks.append(1 / rank)

        scores = [(pair[0], pos_score)]
        for nn in negative_neighbors_test[pair[1]]:
            neg_emb = column_embeddings[nn]
            score = compute_score(emb_node_2, neg_emb, model_loss, model)
            scores.append((nn, score))

        scores.sort(key=lambda x: x[1], reverse=rev)

        hits_at_1_scores.append(int(rank == 1))
        hits_at_3_scores.append(int(rank <= 3))
        hits_at_10_scores.append(int(rank <= 10))

        rank = scores.index((pair[0], pos_score)) + 1

        ranks.append(1 / rank)


            
    for pair in non_join_test_pairs:
        emb_node_1 = column_embeddings[pair[0]]
        emb_node_2 = column_embeddings[pair[1]]

        labels.append(0)

        neg_score = compute_score(emb_node_1, emb_node_2, model_loss, model)
        non_join_scores.append(neg_score)

    mean_reciprocal_rank = sum(ranks) / len(ranks)
    hits_at_1 = sum(hits_at_1_scores) / len(hits_at_1_scores)
    hits_at_3 = sum(hits_at_3_scores) / len(hits_at_3_scores)
    hits_at_10 = sum(hits_at_10_scores) / len(hits_at_10_scores)

    labels = np.concatenate([np.ones(len(join_scores)), np.zeros(len(non_join_scores))])

    if model_loss in ['rgcn_cross_entropy', 'mlp_cross_entropy']:
        scores = np.concatenate([np.array(join_scores), np.array(non_join_scores)])
    else:
        scores = np.concatenate([np.array([1 -p for p in join_scores]), np.array([1 - p for p in non_join_scores])])

    p, r, _ = precision_recall_curve(labels, scores)
    best_f1_score = best_f1(p, r)

    print('Length of ranks is {}'.format(len(ranks)))

    evaluation_metrics = {'MRR': mean_reciprocal_rank, 'Hits@1': hits_at_1, 'Hits@3': hits_at_3, 'Hits@10': hits_at_10, 'Best F1': best_f1_score}

    return evaluation_metrics, join_scores, non_join_scores


def compute_score(emb_1, emb_2, model_loss, model):
    """
    Computes score between a pair of column embeddings based on the loss function and model used
    """

    if model_loss == 'rgcn_margin':
        score = ((emb_1 - emb_2) ** 2).sum(axis=0).sqrt().item()
    elif model_loss == 'rgcn_cross_entropy':
        hh = torch.cat([emb_1, emb_2])
        score = torch.sigmoid(model.predictor.W2(F.relu(model.predictor.W1(hh)))).detach().item()
    else:
        hh = torch.cat([emb_1, emb_2])
        score = torch.sigmoid(model.W2(F.relu(model.W1(hh)))).detach().item()

    return score


if __name__ == '__main__':

    config, features = parse_configuration()

    num_features = len(features) + int('jaccard_containment' in features) + int('uniqueness' in features)

    benchmark = config['PARAMETERS'].get('benchmark')
    print('Number of features is {}'.format(num_features))
    print("++++++++TRAINING PHASE++++++++")
    join_train_features = get_features(features, config['PATHS']['train_features_path'], pair_type='pos')
    non_join_train_features = get_features(features, config['PATHS']['train_features_path'], pair_type='neg')

    join_train_pairs = list(join_train_features.keys())
    non_join_train_pairs = list(non_join_train_features.keys())

    all_cols = set()
    for pair in join_train_pairs:
        c1, c2 = pair
        all_cols.add(c1)
        all_cols.add(c2)
    for pair in non_join_train_pairs:
        c1, c2 = pair
        all_cols.add(c1)
        all_cols.add(c2)


    columns_to_ids, ids_to_columns, ids_to_features = get_column_ids(config['PATHS']['train_node_features'])

    if config['PARAMETERS']['graph_construction'] == 'thresholded':
        edge_dict = create_edge_dict_threshold(join_train_features, non_join_train_features, columns_to_ids, num_features,
                                               config['PARAMETERS'].getfloat('threshold'))

        edge_list, edge_types = edge_lists_types(edge_dict)
    else:
        edge_list, edge_types = create_edge_list_topk(join_train_features, non_join_train_features, columns_to_ids,
                                                      num_features, config['PARAMETERS'].getint('k'))
    
    et = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    ft = torch.tensor(ids_to_features, dtype=torch.float)

    print('Build the graph')
    graph = dgl.graph((et[0], et[1]))
    graph.ndata['feat'] = F.normalize(ft, 2, 0)

    print('Total Number of Nodes: {}'.format(graph.number_of_nodes()))
    print('Total Number of Edges: {}'.format(graph.number_of_edges()))

    print('Get positive/negative samples')

    all_files = []
    for filename in os.listdir(config['PATHS']['train_datasets_path']):

        if filename.endswith('.csv'):
            all_files.append(filename)

    source_to_fabricated = dict()

    for filename in all_files:
        key = fabricated_to_source(filename, benchmark)
        if key in source_to_fabricated:
            source_to_fabricated[key].append(filename)
        else:
            source_to_fabricated[key] = [filename]

    sampled_datasets = config['PATHS']['sampled_datasets']
    number_of_datasets = config['PARAMETERS'].getint('number_of_datasets')
    number_of_sources = config['PARAMETERS'].getint('number_of_sources')
    epochs = config['PARAMETERS'].getint('epochs')
    dimension = config['PARAMETERS'].getint('dimension')
    learning_rate = config['PARAMETERS'].getfloat('learning_rate')
    margin = config['PARAMETERS'].getfloat('margin')
    model_loss = config['PARAMETERS']['model_loss']
    norm = config['PARAMETERS'].getint('norm')


    # configure stored files
    write_embeddings = config['SAVEFILES'].getboolean('write_embeddings')
    write_results = config['SAVEFILES'].getboolean('write_results')

    results = {'MRR': [], 'Hits@1': [], 'Hits@3': [], 'Hits@10': [], 'Best F1': []}

    
    if model_loss == 'rgcn_margin':
        joins_train, non_join_train_neighbors = get_positive_negative_nodes(join_train_pairs, non_join_train_pairs,
                                                                    columns_to_ids, number_of_datasets,
                                                                    number_of_sources, source_to_fabricated,
                                                                    sampled_datasets, config['PATHS']['samples_path'], benchmark)
        
        model = train_model_triplet_loss(graph, joins_train, non_join_train_neighbors, columns_to_ids,
                                         torch.tensor(edge_types), epochs, dimension, num_features,
                                         margin, norm, rate=learning_rate)
    elif model_loss == 'rgcn_cross_entropy':
            print('Get positive/negative pairs')
            joins_train, non_joins_train = create_join_non_join_lists(join_train_pairs, non_join_train_pairs,
                                                                      number_of_datasets, number_of_sources, source_to_fabricated, 
                                                                      sampled_datasets, config['PATHS']['samples_path'], benchmark)
            print('Train model')
            model = train_model(graph, joins_train, non_joins_train, columns_to_ids, torch.tensor(edge_types), epochs,
                                dimension, num_features, rate=learning_rate)

    elif model_loss == 'mlp_cross_entropy':
        model = train_mlp_model(join_train_pairs, non_join_train_pairs, columns_to_ids, ids_to_features, epochs,
                                rate=learning_rate)
    
    
    print("++++++++TESTING PHASE++++++++")

    join_test_features = get_features(features, config['PATHS']['test_features_path'], pair_type='pos')
    non_join_test_features = get_features(features, config['PATHS']['test_features_path'], pair_type='neg')
    print("Number of all test join pairs: {}".format(len(join_test_features)))
    print("Number of all test non-join pairs: {}".format(len(non_join_test_features)))

    join_test_pairs = list(join_test_features.keys())
    non_join_test_pairs = list(non_join_test_features.keys())

    columns_to_ids, ids_to_columns, ids_to_features = get_column_ids(config['PATHS']['test_node_features'])

    if config['PARAMETERS']['graph_construction'] == 'thresholded':
        edge_dict = create_edge_dict_threshold(join_test_features, non_join_test_features, columns_to_ids, num_features,
                                               config['PARAMETERS'].getfloat('threshold'))

        edge_list, edge_types = edge_lists_types(edge_dict)
    else:
        edge_list, edge_types = create_edge_list_topk(join_test_features, non_join_test_features, columns_to_ids,
                                                      num_features, config['PARAMETERS'].getint('k'))
    
    et = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    ft = torch.tensor(ids_to_features, dtype=torch.float)

    print('Build the graph')
    graph = dgl.graph((et[0], et[1]))
    graph.ndata['feat'] = F.normalize(ft, 2, 0)
    

    print('Total Number of Nodes: {}'.format(graph.number_of_nodes()))
    print('Total Number of Edges: {}'.format(graph.number_of_edges()))

    join_pairs_test = [(columns_to_ids[pair[0]], columns_to_ids[pair[1]]) for pair in join_test_pairs]

    non_join_pairs_test = [(columns_to_ids[pair[0]], columns_to_ids[pair[1]]) for pair in non_join_test_pairs]

    print('Number of positive/negative test pairs: {0}/{1}'.format(len(join_pairs_test), len(non_join_pairs_test)))

    
    results_path = config['PATHS']['results_path']

    if model_loss == 'mlp_cross_entropy':
        embeddings = F.normalize(ft, 2, 0)
    else:
        embeddings = model.gnn(graph, graph.ndata['feat'], torch.tensor(edge_types)).detach()

    if write_embeddings:
        embeddings_file = '{}/rgcn_embeddings_sources_{}_datasets_{}_margin_{}_k_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, int(10*margin), config['PARAMETERS'].getint('k'), epochs, dimension)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    metrics, pos_scores, neg_scores = compute_metrics(join_pairs_test, non_join_pairs_test, columns_to_ids,embeddings, model, model_loss)

    for metric, value in metrics.items():
        results[metric].append(value)

    
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    if model_loss in ['rgcn_cross_entropy', 'mlp_cross_entropy']:
        scores = np.concatenate([np.array(pos_scores), np.array(neg_scores)])
    else:
        scores = np.concatenate([np.array([1 -p for p in pos_scores]), np.array([1 - p for p in neg_scores])])
  
    if model_loss == 'mlp_cross_entropy':
        scores_file = '{}/mlp_scores_sources_{}_datasets_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, epochs, dimension)
        labels_file = '{}/mlp_labels_sources_{}_datasets_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, epochs, dimension)
    elif model_loss == 'rgcn_cross_entropy':
        scores_file = '{}/rgcn_scores_sources_{}_datasets_{}_cross_entropy_k_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, config['PARAMETERS'].getint('k'), epochs, dimension)
        labels_file = '{}/rgcn_labels_sources_{}_datasets_{}_cross_entropy_k_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, config['PARAMETERS'].getint('k'), epochs, dimension)
    else:
        if config['PARAMETERS']['graph_construction'] == 'thresholded':
            scores_file = '{}/rgcn_scores_sources_{}_datasets_{}_margin_{}_t_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, int(10*margin), config['PARAMETERS'].getint('threshold'), epochs, dimension)
            labels_file = '{}/rgcn_labels_sources_{}_datasets_{}_margin_{}_t_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, int(10*margin), config['PARAMETERS'].getint('threshold'), epochs, dimension)
        else:

            features_string = "_".join(str(f) for f in features)
 
            scores_file = '{}/rgcn_scores_sources_{}_datasets_{}_margin_{}_k_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, int(10*margin), config['PARAMETERS'].getint('k'), epochs, dimension, features_string)
            labels_file = '{}/rgcn_labels_sources_{}_datasets_{}_margin_{}_k_{}_epochs_{}_dim_{}.pickle' \
                        .format(results_path, number_of_sources, number_of_datasets, int(10*margin), config['PARAMETERS'].getint('k'), epochs, dimension, features_string)

    if write_results:
        with open(scores_file, 'wb') as f:
            pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if not os.path.isfile(labels_file):
            with open(labels_file, 'wb') as f:
                pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    pr_auc_score = average_precision_score(labels, scores)
    print('PR-AUC: {}'.format(pr_auc_score))

    for metric, values in results.items():

        print('{0} is: {1}'.format(metric, values[0]))