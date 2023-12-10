import argparse 
import os,json
from datetime import datetime, time
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import pickle
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from gensim.models.doc2vec import Doc2Vec
import logging


if os.path.exists("log") is False:
    os.mkdir("log")
level    = logging.INFO
format   = '  %(message)s'


def main(args):
    if os.path.exists(args.save_dir) is False:
        os.mkdir(args.save_dir)
    log_path =os.path.join("log", args.save_filename+".log")

    handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    logging.info(str(args))

    data_save_path= os.path.join(args.save_dir,args.save_filename)
    edges_save_path= os.path.join(args.save_dir,"edges_"+args.save_filename)
    label_save_path= os.path.join(args.save_dir,"labels_"+args.save_filename)
    if args.feature_path==None: 
        feature_save_path= os.path.join(args.save_dir,"features_"+args.save_filename)
    else:
        if os.path.exists(args.feature_path) is False:
            feature_save_path= os.path.join(args.save_dir,"features_"+args.save_filename)
        else:
            feature_save_path=args.feature_path
    df=pd.read_csv(args.tsv_path, sep='\t')
    #df=df[0:100]
    texts=list(df['text'])
    labels=list(df['label'])
    patent_ids = list(df['patent_id'])
    text_vectors=None 
    if os.path.exists(feature_save_path):
        try:
            text_vectors= torch.load(feature_save_path)
        except Exception as e:
            logging.info(e)
            
    if text_vectors is None: 
        if args.model_type=="doc2vec":
            model = Doc2Vec.load(args.model_path) 

        text_vectors=[model.infer_vector(text.split()) for text in tqdm(texts)]
        text_vectors = np.asarray(text_vectors, dtype=np.float32)

        if args.save_features:
            torch.save(text_vectors,feature_save_path)
        # with open(feature_save_path, 'wb') as f:
        #      pickle.dump(feature_save_path, f)    
    #sims =cosine_similarity(text_vectors,text_vectors)
    edge_indexs=None 
    if os.path.exists(edges_save_path):
        try:
            edge_indexs= torch.load(feature_save_path)
            edge_indexs_0=edge_indexs[0]
            edge_indexs_1=edge_indexs[1]
        except Exception as e:
            logging.info(e)
            
    if edge_indexs is None: 
        sims = dot(text_vectors, text_vectors.T)/(norm(text_vectors,axis=1)*norm(text_vectors,axis=1))

        gnn_length=len(text_vectors)
        edge_indexs_0=[]
        edge_indexs_1=[]

        for i in tqdm(range(0,gnn_length)):
            for j in range(i, gnn_length):
                if i==j:
                    pass
                else:
                    if sims[i,j]>=args.threshold:
                        edge_indexs_0.append(i)
                        edge_indexs_1.append(j)
   
    logging.info("Total number of edges: " + str( len(edge_indexs_0)))
    logging.info("Total number of edges per nodes: " + str( len(edge_indexs_0)/ len(text_vectors)))
    if args.save_edge_info:
        with open(edges_save_path, 'wb') as f:
             pickle.dump([edge_indexs_0,edge_indexs_1], f) 
    with open(label_save_path, 'wb') as f:
         pickle.dump(labels, f) 
#     X, y,indices = (0.1*np.arange(gnn_length*2)).reshape((gnn_length, 2)),range(0,gnn_length),range(gnn_length)
#     X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(X, y,indices, test_size=0.30, random_state=42)
#     train_mask=[False for i in range(0,gnn_length)]
#     for index in indices_train:
#         train_mask[index]=True
        
#     test_mask=[False for i in range(0,gnn_length)]
    
#     for index in indices_test:
#         test_mask[index]=True
    


#     edge_index = torch.tensor([edge_indexs_0,
#                                edge_indexs_1], dtype=torch.long)
#     x = torch.tensor(text_vectors, dtype=torch.float) 
#     labels = torch.tensor(labels, dtype=torch.long)
#     print(" labels: ", labels)
#     train_mask = torch.tensor(train_mask, dtype=torch.bool)
#     test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
#     data = Data(x=x,y= labels,edge_index=edge_index,train_mask=train_mask, test_mask=test_mask)
#     try: 
#         torch.save(data, data_save_path)
#     except Exception as e:
#         print(e)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Patent labelling with citation frequencies top x%, bottom x% consideration within a certian time period 3/5/10
            """
    )
    parser.add_argument("--tsv_path", type=str, help="label dictionary path, format {top: [patent_id], bottom:[patent_ids] ")
    parser.add_argument("--model_path", type=str, help="model path [doc2vec, patent-bert]")
    parser.add_argument("--model_type", type=str, default="doc2vec", help="model path [doc2vec, patent-bert]")
    parser.add_argument("--threshold", type=float,default=0.70,  help=" 0-1")
    parser.add_argument("--save_dir", type=str, default='gnn_data', help="save directory path")
    parser.add_argument("--save_filename", type=str, default="temp_gnn_data.pickel", help="Graph data filename")
    parser.add_argument("--save_features", type=bool, default=True, help="Save_features")
    parser.add_argument("--save_edge_info", type=bool, default=True, help="Save_edge_info")
    parser.add_argument("--feature_path", type=str, default=None, help="feature_path")

    args = parser.parse_args()
    
    print(args)
    main(args)
   
    
    #python gnn_data_preparation_by_feature.py --tsv_path A61_df.tsv  --label_path labeled/A61_5_10.json --model_path doc2vec100d --threshold 0.68 --save_dir gnn_data --save_filename A61_5y_10p_t0.68_d2v-v1.pt 
    
    
    #python gnn_text_data_preparation.py --tsv_path tsvs/A61_5y_10p.tsv   --model_path doc2vec100d --threshold 0.68 --save_dir gnn_data --save_filename A61_5y_10p_t0.68_d2v-v2.pt 
