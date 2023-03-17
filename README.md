## Enhancing Money Laundering Detection on the Blockchain with Graph Neural Networks and DGI Embeddings
Exam project for the 2022/23 course in Blockchain & Cryptocurrencies at UNIBO 

This work aims to present a new way to study Bitcoin transactions utilizing Graph Neural Networks (GNNs) in order to detect illicit activities on the blockchain environment. We show how the Deep Graph Infomax (DGI) embeddings allow us to obtain better results in finding illicit transactions when paired with standard classification methods like Random Forest (RF).

In the folder "Papers" you can find articles and papers that have been used to develop this solution and a report describing the implemented methods and the results.  

### Usage

#### Local Run 
1) Clone this repository on your machine;
2) Download the [Elliptic data set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) in the relative folder "data";
3) Run execute.py.
Note: It is higly recommended to run this script on machine with a powerful GPU (possibly Nvidia with CUDA).

#### Colab Run
You can run the full training and test on [Google Colab](https://colab.research.google.com/drive/1lC8aJC7rzri8vndtH1pbcUyWqDfJcGWK?usp=sharing).
Note: you may need to fix some hyper-parameters depending on which plan you are using on Colab in order to avoid CUDA OOM errors.

### References
- Velickovic P., Fedus W. Hamilton W.L, Liò P., Bengio Y. and Hjelm R D.: [Deep Graph Infomax](https://arxiv.org/pdf/1809.10341.pdf) , International Conference on Learning Representations (2019);

- Weng Lo W., Kulatilleke G. K., Sarhan M., Layeghy S., and Portmann M.: [Inspection-L: Self-Supervised GNN Node Embeddings
for Money Laundering Detection in Bitcoin](https://arxiv.org/pdf/2203.10465.pdf), Arxiv (2022);

- Bellei C.: [The Elliptic Data Set: opening up machine learning on the blockchain](https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14), Medium (2019);

- Weber M., Domeniconi G., Chen J., Weidele D. K. I., Bellei C., Robinson T., Leiserson C. E.:[Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics](https://arxiv.org/pdf/1908.02591.pdf), KDD ’19 Workshop on Anomaly Detection in Finance (2019).
