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


# DGI hyperparameters
batch_size = 1
nb_epochs = 1000
patience = 200
lr = 0.01
l2_coef = 0.001
hid_units = 256
sparse = True
nonlinearity = 'prelu'


adj, features, labels, idx_train, idx_test, X_train, X_test, y_train, y_test = load_data()

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
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

#Train DGI encoder
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

model.load_state_dict(torch.load('best_dgi.pkl')) #Select the best performing model

# Get node embeddings 
embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
train_embs = embeds[0, idx_train]
test_embs = embeds[0, idx_test]

train_lbls = labels[0, idx_train].type(torch.LongTensor)
test_lbls = labels[0, idx_test].type(torch.LongTensor)

train_lbls = train_lbls.cuda()
test_lbls = test_lbls.cuda()

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

print('LogReg with Node Embeddings')
print('Precision: ', Precision,' Recall: ', Recall, ' F1: ', F1)
print(cm)

# Random Forest
rfc= RFC(criterion = 'gini' , n_estimators = 200 , random_state = 28, max_features=None,bootstrap=True)

y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

X_train = X_train.drop('class',axis=1)
X_train = X_train.drop('txId',axis=1)
X_test = X_test.drop('class',axis=1)
X_test = X_test.drop('txId',axis=1)

#RF on raw data
rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,pred)
F1 = f1_score(y_test,pred,pos_label=1)
Recall = recall_score(y_test,pred,pos_label=1)
Precision = precision_score(y_test,pred,pos_label=1)

print('RF with raw data')
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

#RF with data augmentation
tr_embs = pd.DataFrame(train_embs.cpu()).astype(float)
tst_embs = pd.DataFrame(test_embs.cpu()).astype(float)
X_train = pd.DataFrame(X_train).astype(float)
X_test = pd.DataFrame(X_test).astype(float)

# Apply data augmentation
train_embs.index = X_train.index
new_train = pd.concat((X_train,tr_embs),axis=1)
tst_embs.index = X_test.index
new_test = pd.concat((X_test,tst_embs),axis=1)

new_train.columns = new_train.columns.astype(str)
new_test.columns = new_test.columns.astype(str)

rfc.fit(new_train,y_train)

pred = rfc.predict(new_test)
cm = confusion_matrix(y_test,pred)
F1 = f1_score(y_test,pred,pos_label=1)
Recall = recall_score(y_test,pred,pos_label=1)
Precision = precision_score(y_test,pred,pos_label=1)