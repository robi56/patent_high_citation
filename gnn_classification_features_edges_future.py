import argparse 
import torch, os 
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


if os.path.exists("log") is False:
    os.mkdir("log")
level    = logging.INFO
format   = '  %(message)s'


def get_current_time():
    return datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")


def format_output(model_name, data_file, losses, nepoches, test_acc_val, lr):
    output_str= "model_name: "+ model_name +" data_file: "+data_file +"  time:  " + get_current_time()+":  training loss: " + str(losses[-1]) + " epoch : "+ str(nepoches)+ " test_acc: "+ str(test_acc_val)+ " lr: "+str(lr)
    return output_str

def main(args):
    
    log_path =os.path.join("log", "train_future"+".log")

    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    logging.info(str(args))
    
    with open(args.edge_path,'rb') as  f:
          edges=pickle.load(f)
    
    features=torch.load(args.feature_path)
   
    with open(args.label_path,'rb') as  f:
         label=pickle.load(f)
    if args.test_index_path: 
        with open(args.test_index_path,'rb') as f:
            text_indices=pickle.load(f)
            data=prepare_data_future_prediction(features,edges[0], edges[1],label,text_indices )
            labeling_type="Future_prediction"
    else:         
        edge_factor=args.edge_factor
        labeling_type="Random"

        if edge_factor<0:
            data =prepare_data(features, edges[0], edges[1], label)


        else:
            total_edge=len(features)*edge_factor
            data =prepare_data(features, edges[0][0:total_edge], edges[1][0:total_edge], label)
        print(data)
        print("Training samples: ", sum(data.train_mask), " Test samples: ", sum(data.test_mask))
        write_str="Training samples: "+ str( sum(data.train_mask))+ " Test samples: " + str( sum(data.test_mask))
        logging.info(write_str)
    if os.path.exists("results"):
        pass 
    else:
        os.mkdir("results")
    
    dim =args.dim
    model_name=args.model
    nepoches=args.nepochs
    lr=args.lr
    num_class=args.num_class
    if model_name=="MLP": 
        model = MLP(dim=dim, hidden_channels=16,num_classes=num_class)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        model, losses=model_train(model, data, criterion, optimizer, nepoches)
        test_acc_val, c_report, c_matrix=model_test(model,data,num_class)

        with open(args.result_path, "a+") as out_f:
            out_f.write("New Run "+str(args)+ "\n")        
            output_str=format_output(model_name, args.feature_path, losses, nepoches, test_acc_val, lr)  
            print_str=output_str+"\n" +"classification report: "+ c_report + "\n confusion_matrix: " + str(c_matrix)+" labeling_type: "+labeling_type   +"\n"
            logging.info(print_str)         
            out_f.write(print_str)       
    elif model_name=="GCN":
        model = GCN(dim=dim, hidden_channels=16,num_classes=num_class)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
        model, losses=gnn_model_train(model, data, criterion, optimizer, nepoches)
        test_acc_val, c_report, c_matrix=gnn_model_test(model,data,num_class)

        with open(args.result_path, "a+") as out_f:
            out_f.write("New Run "+str(args)+ "\n")        

            output_str=format_output(model_name, args.feature_path, losses, nepoches, test_acc_val, lr) 
            print_str=output_str+"\n" +"classification report: "+ c_report + "\n confusion_matrix: " + str(c_matrix)+" labeling_type: "+labeling_type   +"\n"
        
            logging.info(print_str)                              
            out_f.write(print_str)         

    elif model_name=="GAT":
        model = GAT(dim=dim, hidden_channels=16,heads=8,num_classes=num_class)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.

        model, losses=gnn_model_train(model, data, criterion, optimizer, nepoches)
        test_acc_val, c_report, c_matrix=gnn_model_test(model,data,num_class)

        with open(args.result_path, "a+") as out_f:
            out_f.write("New Run "+str(args)+ "\n")        

            output_str=format_output(model_name, args.feature_path, losses, nepoches, test_acc_val, lr)  
            print_str=output_str+"\n" +"classification report: "+ c_report + "\n confusion_matrix: " + str(c_matrix)+" labeling_type: "+labeling_type   +"\n"
        
            logging.info(print_str)            
            out_f.write(print_str)         
    elif model_name=="GTN":
        model = GraphTransformer(dim=dim, hidden_channels=16,num_classes=num_class)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.

        model, losses=gnn_model_train(model, data, criterion, optimizer, nepoches)
        test_acc_val, c_report, c_matrix=gnn_model_test(model,data,num_class)

        with open(args.result_path, "a+") as out_f:
            out_f.write("New Run "+str(args)+ "\n")        

            output_str=format_output(model_name, args.feature_path, losses, nepoches, test_acc_val, lr)  
            print_str=output_str+"\n" +"classification report: "+ c_report + "\n confusion_matrix: " + str(c_matrix)+" labeling_type: "+labeling_type   +"\n"
        
            logging.info(print_str)            
            out_f.write(print_str)         
    elif model_name=="GSAGE":
        model = GraphSAGENet(dim=dim, hidden_channels=16,num_classes=num_class)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.

        model, losses=gnn_model_train(model, data, criterion, optimizer, nepoches)
        test_acc_val, c_report, c_matrix=gnn_model_test(model,data,num_class)

        with open(args.result_path, "a+") as out_f:
            out_f.write("New Run "+str(args)+ "\n")        

            output_str=format_output(model_name, args.feature_path, losses, nepoches, test_acc_val, lr)  
            print_str=output_str+"\n" +"classification report: "+ c_report + "\n confusion_matrix: " + str(c_matrix)+" labeling_type: "+labeling_type   +"\n"
        
            logging.info(print_str)            
            out_f.write(print_str)         

    
    
    else:
        pass 

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Patent labelling with citation frequencies top x%, bottom x% consideration within a certian time period 3/5/10
            """
    )
    parser.add_argument("--feature_path", type=str, help="gnn data path")
    parser.add_argument("--edge_path", type=str, help="edge data path")
    parser.add_argument("--label_path", type=str, help="label data path")
    parser.add_argument("--result_path", type=str, default="results/result.txt", help="result_path")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--edge_factor", type=int, default=-1)
    parser.add_argument("--test_index_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--model_save", type=str, default="models/default.p")



    args = parser.parse_args()
    
    print(args)
    main(args)

# python gnn_classification.py --data_path gnn_data/A61_5y_10p_t0.68_d2v-v1.pt.pt --result_path results/doc2vec.partial.txt
#python gnn_classification_features_edges.py --feature_path gnn_data/features_A61_10y_10p_t0.68_d2v-v2.pt --edge_path gnn_data/features_A61_10y_10p_t0.68_d2v-v2.pt --label_path gnn_data/label_A61_5_10p.pt --result_path results/patentbert.txt