import argparse 
import os,json
from datetime import datetime, time
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
import pickle
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

def main(args):
    if os.path.exists(args.save_dir) is False:
        os.mkdir(args.save_dir)
    data_save_path= os.path.join(args.save_dir,args.save_filename)

    with open(args.label_path, 'r') as f:
        label_obj=json.load(f)
        
    top_patent_ids=label_obj['top']
    middle_patent_ids=label_obj['middle']    
    bottom_patent_ids=label_obj['bottom']    
    with open(args.title_path) as f:
        title_dict=json.load(f)
    with open(args.abstract_path) as f:
        abstract_dict=json.load(f)
    with open(args.claim_path) as f:
        claim_dict=json.load(f)
            
    

    texts =[]
    labels=[]
    patent_ids=[]
    missed_title=0
    missed_abstract=0
    missed_claims=0
#     for patent_id in top_patent_ids:
#         title=''
#         abstract=''
#         claims=''
#         text=''
#         try:
#             title=title_dict[patent_id]
#             text=title
#         except Exception as e:
#             missed_title=missed_title+1
#         try: 
#             abstract=abstract_dict[patent_id]
#             if len(text)>0:
#                 text=text+abstract
#             else:
#                 text=abstract
            
#         except Exception as e:
#             missed_abstract=missed_abstract+1
#         try: 
#             claims=claim_dict[patent_id]
#             if len(text)>0:
#                 text=text+claims
#             else:
#                 text=claims
            
#         except Exception as e:
#             missed_claims=missed_claims+1
  
#         if len(text)>0:

#             texts.append(text)
#             labels.append(1)
#             patent_ids.append(patent_id)
#     print("Top patent len : ", len(top_patent_ids))
#     print("missed_title,missed_abstract, missed_claims for top : ", missed_title,missed_abstract, missed_claims)
#     missed_title=0
#     missed_abstract=0
#     missed_claims=0
    for patent_id in middle_patent_ids:
        title=''
        abstract=''
        claims=''
        text=''
        try:
            title=title_dict[patent_id]
            text=title
        except Exception as e:
            missed_title=missed_title+1
        try: 
            abstract=abstract_dict[patent_id]
            if len(text)>0:
                text=text+abstract
            else:
                text=abstract
            
        except Exception as e:
            missed_abstract=missed_abstract+1
        try: 
            claims=claim_dict[patent_id]
            if len(text)>0:
                text=text+claims
            else:
                text=claims
                
            
        except Exception as e:
            missed_claims=missed_claims+1
            
  
        if len(text)>0:

            texts.append(text)
            labels.append(1)
            patent_ids.append(patent_id)
    print("Middle patent len  : ", len(middle_patent_ids))
    print("missed_title,missed_abstract, missed_claims for bottom : ", missed_title,missed_abstract, missed_claims)
    missed_title=0
    missed_abstract=0
    missed_claims=0
    for patent_id in bottom_patent_ids:
        title=''
        abstract=''
        claims=''
        text=''
        try:
            title=title_dict[patent_id]
            text=title
        except Exception as e:
            missed_title=missed_title+1
        try: 
            abstract=abstract_dict[patent_id]
            if len(text)>0:
                text=text+abstract
            else:
                text=abstract
            
        except Exception as e:
            missed_abstract=missed_abstract+1
        try: 
            claims=claim_dict[patent_id]
            if len(text)>0:
                text=text+claims
            else:
                text=claims
                
            
        except Exception as e:
            missed_claims=missed_claims+1
            
  
        if len(text)>0:

            texts.append(text)
            labels.append(0)
            patent_ids.append(patent_id)
        
    print("Top patent len bottom : ", len(bottom_patent_ids))
    print("missed_title,missed_abstract, missed_claims for bottom : ", missed_title,missed_abstract, missed_claims)
    df=pd.DataFrame()
    df['patent_id']=patent_ids
    df['text']=texts
    df['label']=labels
    df.to_csv(data_save_path, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Patent labelling with citation frequencies top x%, bottom x% consideration within a certian time period 3/5/10
            """
    )
    parser.add_argument("--title_path", type=str, help=" title file which contains patentid, text")
    parser.add_argument("--abstract_path", type=str, help=" abstract file which contains patentid, text")
    parser.add_argument("--claim_path", type=str, help=" claim file which contains patentid, text")
    parser.add_argument("--label_path", type=str, help="label dictionary path, format {top: [patent_id], bottom:[patent_ids] ")
    parser.add_argument("--save_dir", type=str, default='gnn_data_mclass', help="save directory path")
    parser.add_argument("--save_filename", type=str, default="temp_gnn_data.pickel", help="Graph data filename")

    args = parser.parse_args()
    
    print(args)
    main(args)
    
    
    #python gnn_text_data_preparation_tsv.py --tsv_path A61_df.tsv  --label_path labeled/A61_5_10.json --model_path doc2vec100d --threshold 0.68 --save_dir gnn_data --save_filename A61_5y_10p_t0.68_d2v-v1.pt 
    
    
    #python gnn_text_data_preparation_tsv.py --title_path features/feature_patent_title.json --abstract_path features/feature_patent_abstract.json --claim_path claims_dict/A61_df.json   --label_path labeled/A61_5_10.json --save_dir gnn_data_text --save_filename A61_5y_10p.tsv 
