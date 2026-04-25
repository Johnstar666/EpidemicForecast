import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
from datetime import date, timedelta
warnings.filterwarnings('ignore')
import networkx as nx

def generate_new_features(Gs, labels, dates, window=7, scaled=False):
    features = list()
    
    labs = labels.copy()
    nodes = Gs[0].nodes()
  

    #--- one hot encoded the region
    
    for idx,G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region
        
        H = np.zeros([G.number_of_nodes(),window]) #+3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i,node in enumerate(G.nodes()):
            #---- Past cases      
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[1:(idx+1)]] - me[node])/ sd[node]
                else:
                    H[i,(window-idx):(window)] = labs.loc[node, dates[1:(idx+1)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] =  (labs.loc[node, dates[(idx-window+1):(idx+1)]] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window+1):(idx+1)]]
      
            
        features.append(H)
        
    return features



def generate_graphs_tmp(dates, country):
    Gs = []
    for date in dates:
        if country == "IT":
            d = pd.read_csv("./dataset/Italy/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "EN":
            d = pd.read_csv("./dataset/England/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "FR":
            d = pd.read_csv("./dataset/France/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "ES":
            d = pd.read_csv("./dataset/Spain/graphs/"+country+"_"+date+".csv", header=None)
        
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        nodes = sorted(nodes)
        G.add_nodes_from(nodes)

        edge_weights = []
        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
            edge_weights.append(row[1][2])
        
        for node1 in nodes:
            for node2 in nodes:
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=1)
        
        Gs.append(G)
        
    return Gs


def read_meta_datasets_i(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 2, 24)
    edate = date(2020, 4, 24)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"IT")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y

def read_meta_datasets_e(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"EN")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y

def read_meta_datasets_f(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 3, 10)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"FR")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y

def read_meta_datasets_es(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 3, 12)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"ES")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y



class Dataset_EpiGNN(Dataset):
    def __init__(self,root_path, flag='train', size=None, data_path='italy_labels.csv',
                 scale=False, seasonal_patterns=None, drop_short=False):

        self.seq_len = size[0]    
        self.pred_len = size[1]  
        self.window = size[2]        
        if flag == 'test':
            self.token_len = size[3]
        self.root_path = root_path
        self.data_path = data_path
        # 数据校验
        if self.data_path == "italy_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.window,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.token_len,self.root_path,self.data_path)
        elif self.data_path == "england_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.window,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.token_len,self.root_path,self.data_path)
        elif self.data_path == "france_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.window,self.root_path,self.data_path)
            else:   
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.token_len,self.root_path,self.data_path)
        elif self.data_path =='spain_labels.csv':
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.window,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.token_len,self.root_path,self.data_path)
        meta_graphs = torch.tensor(meta_graphs)
        meta_features = torch.tensor(meta_features)
        meta_y = torch.tensor(meta_y)

        
        self.num_nodes = meta_y.shape[1]   
        # print('self.num_nodes',self.num_nodes)
        self.flag = flag         
        total_days = meta_features.shape[0] 


        train_end = total_days-self.window*2
        val_end = total_days-self.window


        border_map = {
            'train': (0, train_end),
            'val': (train_end - self.seq_len, val_end),
            'test': (val_end - self.seq_len, total_days+1)
        }
        
        self.start_idx, self.end_idx = border_map[flag]
        # print("self.start_idx",self.start_idx)
        # print("self.end_idx",self.end_idx)

        self.graphs = meta_graphs[self.start_idx:self.end_idx]     
        # print("self.graphs.shape",self.graphs.shape)
        self.features = meta_features[self.start_idx:self.end_idx]  
        # print("self.features.shape",self.features.shape)
        self.labels = meta_y[self.start_idx:self.end_idx]        
        # print("self.labels.shape",self.labels[12])
    
        self.valid_length = len(self.features) - self.seq_len - self.window + 1

    def __len__(self):
        return self.valid_length

    def __getitem__(self, index):
        t_start = index
        t_end = t_start + self.seq_len
        pred_start = t_end - self.pred_len
        pred_end = pred_start + self.window + self.pred_len

        x_feat1 = self.features[t_start:t_end]  # [seq_len, 20, 3]
        x_feat2 = self.features[pred_start:pred_end]
        
        x_graph1 = self.graphs[t_start:t_end]   # [seq_len, 20, 20]
        x_graph2 = self.graphs[pred_start:pred_end]
        
        y = self.labels[pred_start:pred_end]   # [pred_len, 20]

        x_feat = torch.tensor(x_feat1,dtype=torch.float32)
        x_feat2 = torch.tensor(x_feat2,dtype=torch.float32)
        x_graph = torch.tensor(x_graph1,dtype=torch.float32)
        x_graph2 = torch.tensor(x_graph2,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.float32)

        return x_feat, x_feat2, x_graph, x_graph2, y

    def get_full_graph(self):
        return torch.FloatTensor(self.graphs)