import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def check(name, unique):
    if(name not in unique):
        name = -1
    return name


def load_data(data_dir='./Elliptic/raw/', random_state=28):

    edges = pd.read_csv(data_dir + 'elliptic_txs_edgelist.csv')
    features = pd.read_csv(data_dir + 'elliptic_txs_features.csv', header=None)
    classes = pd.read_csv(data_dir + 'elliptic_txs_classes.csv')
    tx_features = ['tx_feat_' + str(i) for i in range(2, 95)]
    agg_features = ['agg_feat_' + str(i) for i in range(1, 73)]
    features.columns = ['txId', 'time_step'] + tx_features + agg_features
    features = pd.merge(
        features,
        classes,
        left_on='txId',
        right_on='txId',
        how='left')
    features = features[features['class'] != 'unknown']
    X_train, X_test, y_train, y_test = train_test_split(
        features, features['class'], stratify=features['class'], random_state=random_state, test_size=0.50)
    under_sampler = RandomUnderSampler(random_state=random_state)
    X_train, y_train = under_sampler.fit_resample(X_train, y_train)
    idx_train = range(len(X_train))
    idx_test = range(
        len(X_train),
        len(X_train) +
        len(X_test))
    features = pd.concat([X_train, X_test])
    unique = features['txId'].unique()
    edges['txId1'] = edges['txId1'].apply(lambda name: check(name, unique))
    edges['txId2'] = edges['txId2'].apply(lambda name: check(name, unique))
    edges = edges[edges['txId1'] != -1]
    edges = edges[edges['txId2'] != -1]
    class_values = sorted(features['class'].unique())
    features_idx = {
        name: idx for idx,
        name in enumerate(
            sorted(
                features['txId'].unique()))}
    class_idx = {name: id for id, name in enumerate(class_values)}
    features['txId'] = features['txId'].apply(lambda name: features_idx[name])
    edges['txId1'] = edges['txId1'].apply(lambda name: features_idx[name])
    edges['txId2'] = edges['txId2'].apply(lambda name: features_idx[name])
    features['class'] = features['class'].apply(lambda name: class_idx[name])
    labels = features['class']
    classes = sorted(list(set(labels)), reverse=False)
    classes_dict = {
        c: np.identity(
            len(classes))[
            i,
            :] for i,
        c in enumerate(classes)}
    labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    idx_features_labels = features.values[:, :-1]
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)
    edges_unordered = edges.values
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
        idx_features_labels.shape[0], idx_features_labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.FloatTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_test, X_train, X_test, y_train, y_test


def normalize_adj(mx):

    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx


def normalize_features(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)