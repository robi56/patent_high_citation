import os, torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from sklearn.metrics import classification_report,confusion_matrix
num_classes=2
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import copy

from torch_geometric.nn import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MultiAggregation,
    SAGEConv,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_channels,num_classes=2):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    

class GCN(torch.nn.Module):
    def __init__(self, dim, hidden_channels,num_classes=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    



class GAT(torch.nn.Module):
    def __init__(self, dim, hidden_channels, heads, num_classes=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dim,hidden_channels, heads)  # TODO
        self.conv2 = GATConv(hidden_channels*heads,num_classes)  # TODO

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGENet(torch.nn.Module):
    def __init__(self, dim, hidden_channels, aggr='mean',num_classes=2, aggr_kwargs=None):
        super().__init__()
        self.conv1 = SAGEConv(
            dim,
            hidden_channels,
            aggr=aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.conv2 = SAGEConv(
            hidden_channels,num_classes,
            aggr=copy.deepcopy(aggr),
            aggr_kwargs=aggr_kwargs,
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Define the Graph Transformer model
class GraphTransformer(torch.nn.Module):
    def __init__(self, dim, hidden_channels, num_classes=2):
        super(GraphTransformer, self).__init__()

        self.conv1 = TransformerConv(dim, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)

        self.fc = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def model_train(model, data,criterion, optimizer, epochs):
    losses=[]
    for epoch in range(1, epochs):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step() 
        if epoch%10==0 or epoch==epochs-1: 
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        losses.append(loss)
    return model, losses

def model_test(model, data,num_classes):
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    # Generate a classification report
    classification_report_str = classification_report(
    data.y[data.test_mask].cpu().numpy(),
    pred[data.test_mask].cpu().numpy(),
    target_names=[str(i) for i in range(num_classes)]  # Replace with your class labels if available
)
    confusion = confusion_matrix(data.y[data.test_mask].cpu().numpy(), pred[data.test_mask].cpu().numpy())

    return test_acc,classification_report_str,confusion


def gnn_model_train(model, data,criterion, optimizer, epochs, checkpoint_path):
    losses=[]
    if os.path.exists(checkpoint_path):
        model=torch.load(checkpoint_path)
    for epoch in range(1, epochs):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x,data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step() 
        if epoch%10==0 or epoch==epochs-1: 
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            torch.save(model, checkpoint_path)

        losses.append(loss)
    return model, losses

def gnn_model_predict(model, data):
    model.eval()
    out = model(data.x,data.edge_index)
    #pred = out.argmax(dim=1)  # Use the class with highest probability.
    
    return out


def gnn_model_test(model, data,num_classes):
    model.eval()
    out = model(data.x,data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        # Generate a classification report
    classification_report_str = classification_report(
    data.y[data.test_mask].cpu().numpy(),
    pred[data.test_mask].cpu().numpy(),
    target_names=[str(i) for i in range(num_classes)]  # Replace with your class labels if available
)
    confusion = confusion_matrix(data.y[data.test_mask].cpu().numpy(), pred[data.test_mask].cpu().numpy())

    
    return test_acc,classification_report_str, confusion

