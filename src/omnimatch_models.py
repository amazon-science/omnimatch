import math
import random

import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import itertools


class MLPClassifier(nn.Module):
    """
    Simple 2-layer MLP model that receives the concatenation of the feature sets of a pair of columns as input. Used
    for simple binary classification.
    """

    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x))).squeeze(1)

class RelGCN(nn.Module):
    """
    Relational 2-layer GCN Model
    """

    def __init__(self, in_feats, out_feats, no_types):
        super(RelGCN, self).__init__()
        self.conv1 = dglnn.RelGraphConv(in_feats, out_feats, no_types)
        self.conv2 = dglnn.RelGraphConv(out_feats, out_feats, no_types)

    def forward(self, g, in_feat, edge_types):
        h = F.relu(self.conv1(g, in_feat, edge_types))
        h = F.normalize(self.conv2(g, h, edge_types))
        return h


class MLPPredictor(nn.Module):
    """
        Multi-layer Perceptron predictor:
            - 1 hidden layer
    """

    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.
        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.
        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class GNNModel:
    """
    RGCN model used together with an MLP predictor (coupled with binary cross-entropy loss)
    """

    def __init__(self, in_feats, h_feats, no_types, lr_rate, wd):
        self.gnn = RelGCN(in_feats, h_feats, no_types)
        self.predictor = MLPPredictor(h_feats)
        self.optimizer = torch.optim.Adam(itertools.chain(self.gnn.parameters(), self.predictor.parameters()),
                                          lr=lr_rate, weight_decay=wd)


class MarginLossModel:
    """
    RGCN model used when coupled with a triplet margin loss
    """

    def __init__(self, in_feats, h_feats, no_types, lr_rate, wd):
        self.gnn = RelGCN(in_feats, h_feats, no_types)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=lr_rate, weight_decay=wd)


def  compute_triplet_loss(anchor_nodes, positive_nodes, negative_nodes, h, margin_value=1.0, norm_value=2):
    """
    Function that computes triplet margin loss for a list of anchor, positive and negative nodes
    :param anchor_nodes: The list of anchor nodes
    :param positive_nodes: The list of positive nodes (the ones matching with the anchor nodes)
    :param negative_nodes: The list of negative nodes (the ones not matching with the anchor nodes)
    :param h: The trained model used to fetch embeddings of nodes
    :param margin_value: The margin value used for computing triplet margin loss
    :param norm_value: The norm used to compute distances of node pair embeddings
    :return: Triplet margin loss
    """

    triplet_loss = nn.TripletMarginLoss(margin=margin_value, p=norm_value)

    anchor_embeddings = [h[node] for node in anchor_nodes]

    positive_embeddings = [h[node] for node in positive_nodes]

    negative_embeddings = [h[node] for node in negative_nodes]

    positive_pairs = [(anchor_nodes[i], positive_nodes[i]) for i in range(len(anchor_nodes))]

    positive_pairs = list(set(positive_pairs))

    negative_pairs = [(anchor_nodes[i], negative_nodes[i]) for i in range(len(anchor_nodes))]

    labels = torch.cat([torch.ones(len(positive_pairs)), torch.zeros(len(negative_pairs))])

    labels = labels.int()
    loss = triplet_loss(torch.stack(anchor_embeddings, dim=0), torch.stack(positive_embeddings, dim=0),
                        torch.stack(negative_embeddings, dim=0))

    return loss


def compute_loss(pos_score, neg_score, pos_weight):
    """
    Compute binary cross entropy loss based on predictions on the positive and negative
    edge samples.

    :param pos_score: The scores of positive (join) examples
    :param neg_score: The scores of negative (non-join) examples
    :param pos_weight: The weight used for positive examples
    :return: Binary cross-entropy loss and f1-score
    """
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    weight_positives = torch.FloatTensor([pos_weight])
    return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=weight_positives)


def get_train_val_nodes(join_pairs: list, non_join_neighbors: dict, column_ids: dict, training_prc: float = 0.9):
    """
    Returns anchor, positive and negative nodes for computing triplet margin loss during training
    :param join_pairs: List of column join pairs
    :param non_join_neighbors: Dictionary storing for each column the set of corresponding non-join columns
    :param column_ids: Dict storing for each column the corresponding id
    :param training_prc: The percentage of data that is going to be used for training
    :return: Anchor, positive and negative nodes (ordered identically) for training and validation
    """
    anchor_nodes = []
    positive_nodes = []
    negative_nodes = []

    for pair in join_pairs:

        column1, column2 = pair

        cid1 = column_ids[column1]
        cid2 = column_ids[column2]

        if cid1 in non_join_neighbors:
            anchor = cid1
            pos = cid2
        else:
            anchor = cid2
            pos = cid1
        if anchor in non_join_neighbors:
            neg_neigh = non_join_neighbors[anchor]

            ratio = len(neg_neigh) + 1
            random_neigh = random.sample(neg_neigh, min(ratio, len(neg_neigh)))

            for i in range(len(random_neigh)):
                anchor_nodes.append(anchor)
                positive_nodes.append(pos)

                negative_nodes.append(neg_neigh[i])

    train_len = int(training_prc * len(anchor_nodes))

    return anchor_nodes[:train_len], positive_nodes[:train_len], negative_nodes[:train_len], anchor_nodes[train_len:], positive_nodes[train_len:], negative_nodes[train_len:]


def get_train_val_graphs(no_nodes: int, join_pairs: list, non_join_pairs: list, column_ids: dict,
                         validation_prc: float = 0.1):
    """
    Function that creates positive and negative graphs for training and validation when using cross_entropy loss function
    with a RGCN + MLP.
    :param no_nodes: Total number of nodes
    :param join_pairs: List containing column join pairs
    :param non_join_pairs: List containing column non-join pairs
    :param column_ids: Dict storing for each column its corresponding id
    :param ratio: Negative:Positive sample ratio
    :param validation_prc: Percentage of data to be used for validation
    :return: Positive/Negative graphs for training and validation
    """
    pos_edges = []
    neg_edges = []

    ratio = len(non_join_pairs)  

    non_join_pairs = random.sample(non_join_pairs, min(ratio * len(join_pairs), len(non_join_pairs)))

    for pair in join_pairs:
        column1, column2 = pair

        cid1 = column_ids[column1]
        cid2 = column_ids[column2]

        pos_edges.append((cid1, cid2))
        pos_edges.append((cid2, cid1))

    for pair in non_join_pairs:
        column1, column2 = pair

        cid1 = column_ids[column1]
        cid2 = column_ids[column2]

        neg_edges.append((cid1, cid2))
        neg_edges.append((cid2, cid1))

    val_pos_edges = random.sample(pos_edges, int(validation_prc * len(pos_edges)))
    val_neg_edges = random.sample(neg_edges, int(validation_prc * len(neg_edges)))

    pos_edges = list(set(pos_edges).difference(set(val_pos_edges)))
    neg_edges = list(set(neg_edges).difference(set(val_neg_edges)))

    et_pos = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
    et_neg = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
    positive_graph = dgl.graph((et_pos[0], et_pos[1]), num_nodes=no_nodes)
    negative_graph = dgl.graph((et_neg[0], et_neg[1]), num_nodes=no_nodes)

    et_val_pos = torch.tensor(val_pos_edges, dtype=torch.long).t().contiguous()
    et_val_neg = torch.tensor(val_neg_edges, dtype=torch.long).t().contiguous()
    positive_val_graph = dgl.graph((et_val_pos[0], et_val_pos[1]), num_nodes=no_nodes)
    negative_val_graph = dgl.graph((et_val_neg[0], et_val_neg[1]), num_nodes=no_nodes)

    return positive_graph, negative_graph, positive_val_graph, negative_val_graph


def train_model_triplet_loss(train_graph, join_pairs: list, non_join_neighbors: dict, column_ids: dict,
                             edge_types: torch.tensor, epochs: int,
                             embed_size: int, num_features: int, margin: float, norm: int, 
                             rate: float, wd: float = 0):
    """
    Trains the RGCN model with triplet margin loss.
    :param train_graph: The graph containing feature edges among column/nodes
    :param join_pairs: The list containing join pairs for training/validation
    :param non_join_neighbors: A dict storing for each column in the training/validation its corresponding non join set of columns
    :param column_ids: A dict storing for each column its corresponding id
    :param edge_types: A list storing for each edge its type (the order is equivalent to the order of the edges)
    :param epochs: Number of epochs for which we train our model
    :param embed_size: The size of resulting embeddings
    :param num_features: The number of features used in the feature edge graph
    :param margin: Margin used for triplet margin loss computation
    :param norm: The norm used for triplet margin loss computation
    :param rate: The learning rate
    :param wd: Weight decay parameter for training
    :param write_train_stats: Boolean parameter that dictates whether training, validation losses, f1_scores will be written in files
    :return: The trained model
    """
    model = MarginLossModel(train_graph.ndata['feat'].shape[1], embed_size, num_features, rate, wd)

    train_losses = []
    val_losses = []

    for e in range(epochs):

        model.optimizer.zero_grad()

        h = model.gnn(train_graph, train_graph.ndata['feat'], edge_types)

        anchor_nodes_train, positive_nodes_train, negative_nodes_train, anchor_nodes_val, positive_nodes_val, negative_nodes_val = get_train_val_nodes(
            join_pairs, non_join_neighbors, column_ids)

        train_loss = compute_triplet_loss(anchor_nodes_train, positive_nodes_train, negative_nodes_train, h,
                                          margin_value=margin, norm_value=norm)
        val_loss = compute_triplet_loss(anchor_nodes_val, positive_nodes_val, negative_nodes_val, h,
                                        margin_value=margin, norm_value=norm)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_loss.backward()

        model.optimizer.step()

        if e % 10 == 0:
            print('In epoch {0}, train loss {1}, val loss {2}'.format(e, train_loss, val_loss))
    return model


def train_mlp_model(join_pairs: list, non_join_pairs: list, column_ids: dict, id_features: list, epochs: int,
                    rate: float = 0.001, wd: float = 0):
    """
    Trains a simple MLP classification model
    :param join_pairs: List containing all join column pairs included in training/validation
    :param non_join_pairs: List containing all non-join column pairs included in training/validation
    :param column_ids: Dict storing for each column its corresponding id
    :param id_features: List storing for each column id its corresponding features
    :param epochs: Number of epochs we use to train the model
    :param rate: Learning rate used for training
    :param wd: Weight decay used for training
    :return: The trained MLP model
    """
    positive_features = []

    for pair in join_pairs:
        column1, column2 = pair

        features_cat = id_features[column_ids[column1]] + id_features[column_ids[column2]]

        positive_features.append(features_cat)

    negative_features = []

    for pair in non_join_pairs:
        column1, column2 = pair

        features_cat = id_features[column_ids[column1]] + id_features[column_ids[column2]]

        negative_features.append(features_cat)

    positive_tensors = F.normalize(torch.tensor(positive_features, dtype=torch.float), 2, 0)
    negative_tensors = F.normalize(torch.tensor(negative_features, dtype=torch.float), 2, 0)

    model = MLPClassifier(len(id_features[0]))
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=rate, weight_decay=wd)

    ratio = math.floor(len(non_join_pairs) / len(join_pairs))

    for e in range(epochs):
        optimizer.zero_grad()

        pos_score = model(positive_tensors)
        neg_score = model(negative_tensors)

        train_loss = compute_loss(pos_score, neg_score, ratio)

        train_loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print('In epoch {}, train loss {}'.format(e, train_loss))

    return model

def train_model(train_graph, join_pairs: list, non_join_pairs: list, column_ids: dict,
                edge_types: torch.tensor, epochs: int, embed_size: int,
                num_features: int, rate: float = 0.01, wd: float = 0, write_losses: bool = True):
    """
    Trains the RGCN model when using a binary cross-entropy loss
    :param train_graph: The graph containing feature edges among column/nodes
    :param join_pairs: The list containing join pairs for training/validation
    :param non_join_pairs:The list containing non-join pairs for training/validation
    :param column_ids: A dict storing for each column its corresponding id
    :param edge_types: A list storing for each edge its type (the order is equivalent to the order of the edges)
    :param epochs: Number of epochs for which we train our model
    :param embed_size: The size of resulting embeddings
    :param num_features: The number of features used in the feature edge graph
    :param rate: Learning rate used for training
    :param wd: Weight decay used for training
    :param write_losses: Boolean parameter that dictates whether training, validation losses, f1_scores will be written in files
    :return: Returns the trained model
    """
    model = GNNModel(train_graph.ndata['feat'].shape[1], embed_size, num_features, rate, wd)

    train_losses = []
    val_losses = []

    positive_weight = int(len(non_join_pairs)/len(join_pairs))
    for e in range(epochs):
        model.optimizer.zero_grad()

        graph_pos, graph_neg, graph_val_pos, graph_val_neg = get_train_val_graphs(train_graph.num_nodes(),
                                                                                  join_pairs, non_join_pairs,
                                                                                  column_ids)

        # forward
        h = model.gnn(train_graph, train_graph.ndata['feat'], edge_types)
        pos_score = model.predictor(graph_pos, h)
        neg_score = model.predictor(graph_neg, h)
        train_loss = compute_loss(pos_score, neg_score, positive_weight)

        # backward
        train_loss.backward()
        model.optimizer.step()

        pos_val_score = model.predictor(graph_val_pos, h)
        neg_val_score = model.predictor(graph_val_neg, h)

        val_loss = compute_loss(pos_val_score, neg_val_score, positive_weight)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if e % 10 == 0:
            print('In epoch {}, train loss {} --- val loss {}'.format(e, train_loss, val_loss))
    if write_losses:
        with open("train_losses.txt", "w") as tl_file:
            for loss in train_losses:
                tl_file.write(str(loss.detach().numpy()) + '\n')

        with open("val_losses.txt", "w") as vl_file:
            for loss in val_losses:
                vl_file.write(str(loss.detach().numpy()) + '\n')

    return model