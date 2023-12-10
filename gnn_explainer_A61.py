import torch, os 
import json 
from torch_geometric.data import Data
from gnn_models import *
import datetime
import pickle 
from gnn_data import *
import numpy as  np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch 
import logging

from torch_geometric.explain.algorithm import GNNExplainer

from torch_geometric.utils import to_networkx
import networkx as nx
import pandas 
from torch_geometric.explain import Explainer, GNNExplainer



feature_path="features_doc2vec/features_A61_3y_10p_t0.70_d2v.pt"
feature_path="features_patent_bert/features_A61_3y_10p_patent_bert.pt"
edge_path='edges/edges_A61_3y_10p_d2v_t0.7_f32.87.pt'
label_path="labels/A61_3y_10p.tsv.pt"
resutl_path="results/GNN_explanation.txt"
dim =1024 
lr =0.0001 
nepochs =10 
model ="GCN" 
test_index_path= "test_indexs_future/A61_3y_10p.tsv_test_indexs_2.pt"
edge_factor=-1

try:
   os.mkdir("figures/")
except Exception as e:
   print(e)

def split_indices_by_label(labels, indices):
    label_0_indices = [indices[i] for i in range(len(labels)) if labels[i] == 0]
    label_1_indices = [indices[i] for i in range(len(labels)) if labels[i] == 1]
    return label_0_indices, label_1_indices
    
    
with open(edge_path,'rb') as  f:
      edges=pickle.load(f)

features=torch.load(feature_path)

with open(label_path,'rb') as  f:
     label=pickle.load(f)
if test_index_path: 
    with open(test_index_path,'rb') as f:
        text_indices=pickle.load(f)
        data=prepare_data_future_prediction(features,edges[0], edges[1],label,text_indices )
        labeling_type="Future_prediction"
else:         
    edge_factor=edge_factor
    labeling_type="Random"

    if edge_factor<0:
        data =prepare_data(features, edges[0], edges[1], label)


    else:
        total_edge=len(features)*edge_factor
        data =prepare_data(features, edges[0][0:total_edge], edges[1][0:total_edge], label)
        

nepoches=400
model = GCN(dim=dim, hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
 # Define loss criterion.

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
model, losses=gnn_model_train(model, data, criterion, optimizer, nepoches)
test_acc_val, c_report, c_matrix=gnn_model_test(model,data)



true_indices = np.nonzero(data.test_mask)


new_labels=data.y[data.test_mask]

index_0, index_1=split_indices_by_label(new_labels,true_indices.flatten().numpy())


explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=100),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

if len(index_0)>50:
   selected_0_nodes = index_0[0:50]
else:
   selected_0_nodes = index_0

if len(index_1)>50:
   selected_1_nodes = index_1[0:50]
else:
   selected_1_nodes = index_1





for node_index in selected_1_nodes:
    explanation = explainer(data.x, data.edge_index, index=node_index)
    print(f'Generated explanations in {explanation.available_explanations}')
    path = os.path.join("figures", 'gcn_'+'features_A61_3y_10p_patent_bert_'+'label_1_'+str(node_index)+".png")
    info_path = os.path.join("figures",'gcn_'+'features_A61_3y_10p_patent_bert_'+'label_1_'+str(node_index)+".txt")
    explanation.visualize_feature_importance(path, top_k=10)
    print(f"Feature importance plot has been saved to '{path}'")

    explanation.visualize_graph(path)
    print(f"Subgraph visualization plot has been saved to '{path}'")

    edge_index=explanation.edge_index

    edge_weight=explanation.edge_mask

    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    if edge_weight is not None:  # Discard any edges with zero edge weight:
        mask = edge_weight > 0.65
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    important_nodes=list(set(np.concatenate((edge_index[0],edge_index[1]))))
    important_nodes=[int(node) for node in important_nodes]
    if len(important_nodes)==0:
        continue
    print("Inportant node: ",important_nodes)
    print("node_index", node_index)
    dict={}
    dict[int(node_index)]=important_nodes
    with open(info_path, 'w') as f:
        json.dump(dict, f)
    
    


for node_index in selected_0_nodes:
    explanation = explainer(data.x, data.edge_index, index=node_index)
    print(f'Generated explanations in {explanation.available_explanations}')
    path = os.path.join("figures", 'gcn_'+'features_A61_3y_10p_patent_bert_'+'label_0_'+str(node_index)+".png")
    info_path = os.path.join("figures",'gcn_'+'features_A61_3y_10p_patent_bert_'+'label_0_'+str(node_index)+".txt")
    explanation.visualize_feature_importance(path, top_k=10)
    print(f"Feature importance plot has been saved to '{path}'")

    explanation.visualize_graph(path)
    print(f"Subgraph visualization plot has been saved to '{path}'")

    edge_index=explanation.edge_index

    edge_weight=explanation.edge_mask

    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    if edge_weight is not None:  # Discard any edges with zero edge weight:
        mask = edge_weight > 0.65
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    important_nodes=list(set(np.concatenate((edge_index[0],edge_index[1]))))
    important_nodes=[int(node) for node in important_nodes]
    if len(important_nodes)==0:
        continue
    print("Inportant node: ",important_nodes)
    print("node_index", node_index)
    dict={}
    dict[int(node_index)]=important_nodes
    with open(info_path, 'w') as f:
        json.dump(dict, f)
    
    
    













 
