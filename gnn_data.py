import numpy as  np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch 

def prepare_data(features,edge_indexs_0, edge_indexs_1,labels):
    gnn_length=len(features)
    X, y,indices = (0.1*np.arange(gnn_length*2)).reshape((gnn_length, 2)),range(0,gnn_length),range(gnn_length)
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(X, y,indices, test_size=0.30, random_state=42)
    train_mask=[False for i in range(0,gnn_length)]
    for index in indices_train:
        train_mask[index]=True
        
    test_mask=[False for i in range(0,gnn_length)]
    
    for index in indices_test:
        test_mask[index]=True
    


    edge_index = torch.tensor([edge_indexs_0,
                               edge_indexs_1], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float) 
    labels = torch.tensor(labels, dtype=torch.long)
    print(" labels: ", labels)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    data = Data(x=x,y= labels,edge_index=edge_index,train_mask=train_mask, test_mask=test_mask)
    return data


def prepare_data_future_prediction(features,edge_indexs_0, edge_indexs_1,labels, test_indexs):
    gnn_length=len(features)
    print("feature size: ", gnn_length)
    print("text indexs: ",len( test_indexs))
    train_mask=[True for i in range(0,gnn_length)]
    
    for index in test_indexs:
        #print("test_index: ",index,type(index))
        train_mask[index]=False

        try: 
            train_mask[index]=False
        except Exception as e:
            print("test_index: ",index,type(index))

        
    test_mask=[False for i in range(0,gnn_length)]
    
    for index in test_indexs:
        try:
            test_mask[index]=True
        except Exception as e:
            print("test_index: ",index,type(index))

    

    edge_index = torch.tensor([edge_indexs_0,
                               edge_indexs_1], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float) 
    labels = torch.tensor(labels, dtype=torch.long)
    print(" labels: ", labels)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    data = Data(x=x,y= labels,edge_index=edge_index,train_mask=train_mask, test_mask=test_mask)
    return data