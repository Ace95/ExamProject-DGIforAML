import numpy as np
import pandas as pd
import torch
from torch import nn
import scipy.sparse as sp
from utils import load_data, preprocess_features, normalize_adj, sparse_mx_to_torch_sparse_tensor
from layers import DGI, LogReg
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


batch_size = 1
nb_epochs = 1000
patience = 100
lr = 0.01
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
nonlinearity = 'prelu'


adj, features, labels, idx_train, idx_val, idx_test = load_data()
features = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[0]


adj = normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()
    
    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) 

    loss = b_xent(logits, lbl)

    #print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

    model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[0, idx_train].type(torch.LongTensor)
    val_lbls = labels[0, idx_val].type(torch.LongTensor)
    test_lbls = labels[0, idx_test].type(torch.LongTensor)

    train_lbls = train_lbls.cuda()
    val_lbls = val_lbls.cuda()
    test_lbls = test_lbls.cuda()

    tot = torch.zeros(1)
    tot = tot.cuda()


# Logistic Regression
accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        logits = logits.cuda()
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)

F1 = f1_score(test_lbls.cpu(),preds.cpu(),pos_label=0)
Recall = recall_score(test_lbls.cpu(),preds.cpu(),pos_label=0)
Precision = precision_score(test_lbls.cpu(),preds.cpu(),pos_label=0)
cm = confusion_matrix(test_lbls.cpu(),preds.cpu())

print('Precision: ', Precision,' Recall: ', Recall, ' F1: ', F1)
print(cm)

# Random Forest

data_ft = pd.read_csv('./Elliptic/raw/elliptic_txs_features.csv', header=None)
data_ed = pd.read_csv('./Elliptic/raw/elliptic_txs_edgelist.csv')
data_lb = pd.read_csv('./Elliptic/raw/elliptic_txs_classes.csv')

dataset = data_ft.merge(data_ed, right_index=True, left_index=True)
dataset = dataset.merge(data_lb, right_index=True, left_index=True) 
dataset.drop(columns=['txId'],inplace=True)
dataset = dataset[dataset['class'] != 'unknown']

x = dataset.iloc[:,1:169]
y = dataset['class']
x_tr,x_val,y_tr,y_val = train_test_split(x,y, test_size=0.30,random_state=28)


#RF on raw data
rfc= RFC(criterion = 'entropy' , n_estimators =100 , random_state = 28 , n_jobs =3)
#rfc= RFC(criterion = 'gini' , n_estimators = 100 , random_state = 28 )
rfc.fit(x_tr,y_tr)

pred = rfc.predict(x_val)
cm = confusion_matrix(y_val,pred)
F1 = f1_score(y_val,pred,pos_label='1')
Recall = recall_score(y_val,pred,pos_label='1')
Precision = precision_score(y_val,pred,pos_label='1')

print('RF on raw data')
print('Precision: ', Precision,' Recall: ', Recall, ' F1: ', F1)
print(cm)

#RF on DGI embeds
rfc.fit(train_embs.cpu(),train_lbls.cpu())

pred_embs = rfc.predict(test_embs.cpu())
cm = confusion_matrix(test_lbls.cpu(),pred_embs)
F1 = f1_score(test_lbls.cpu(),pred_embs,pos_label=0)
Recall = recall_score(test_lbls.cpu(),pred_embs,pos_label=0)
Precision = precision_score(test_lbls.cpu(),pred_embs,pos_label=0)

print('RF on DGI embeds')
print('Precision: ', Precision,' Recall: ', Recall, ' F1: ', F1)
print(cm)