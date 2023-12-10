import argparse 
import json
from datetime import datetime, time
from tqdm import tqdm 

def check_patent_history(patent_date, no_of_years ):
    patent_publish_year=int(patent_date.split('-')[0])
    today=datetime.combine(datetime.now(), time.min)
    if patent_publish_year<(today.year-no_of_years):
        return True
    else:
        return False 


def main(args):
    with open(args.cite_dict_path, 'r') as f:
         cite_obj=json.load(f)
            
    with open(args.date_dict_path, 'r') as f:
         patent_date_dict=json.load(f)  
    
    factor=args.year_limit
    citation_num_list={}
    
    for i, key in tqdm(enumerate(cite_obj)):
        count=0
        try: 
            patent_date=patent_date_dict[key]
            patent_year= patent_date.split('-')[0]
            if(int(patent_year))<2000:
                continue
            #print(patent_date)
            status=check_patent_history(patent_date, factor )
            #print("status: ", status, "patent_id:",key )
            if status is False:
                print("status: ", status, "patent_id:",key , "date: ", patent_date, "limit: ", factor)
                continue
            patents_cite= cite_obj[key]
            #print(patent_date, patents_cite)
            
            for patent_ob in patents_cite:
                try: 
                    citation_year= patent_ob[1].split('-')[0]
                    patent_year= patent_date.split('-')[0]
                    if factor>=int(citation_year)-int(patent_year) :
                        count =count+1
                except Exception as e:
                        print(patent_ob,patent_date)
            if args.ignore_zero_citation and count==0:
                continue
            citation_num_list[key]=count
        except Exception as e:
            print(e)
            continue
       
    
    sorted_citation_dict_from_top=dict(sorted(citation_num_list.items(), key=lambda x: x[1], reverse=True))
    print("Total selected patents: ", len(sorted_citation_dict_from_top))
    top_N = int(len(sorted_citation_dict_from_top)*(args.percentage/100))
    print("Total patents in a label: ", top_N)
    
    keys=list(sorted_citation_dict_from_top.keys())
    labels={}
    labels['top']=keys[0:top_N]
    labels['bottom']=keys[::-1][0:top_N]
    
    with open(args.save_file_path, 'w') as f:
         f.write(json.dumps(labels))
    print("label is saved into: ", args.save_file_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Patent labelling with citation frequencies top x%, bottom x% consideration within a certian time period 3/5/10
            """
    )
    parser.add_argument("--cite_dict_path", type=str, help="patent citation dictionary path, format:  patent_id: [patent_id,citation_date]")
    parser.add_argument("--date_dict_path", type=str, help="patent date dictionary path, format patent_id: patent_date ")
    parser.add_argument("--year_limit", type=int,default=5,  help="year limit [3/5/10/X] ")
    parser.add_argument("--percentage", type=int, default=10, help=" top and bottom percentage limit based how citation count [5/10] ")
    parser.add_argument("--save_file_path", type=str, default="labels.json", help=" top and bottom percentage limit based how citation count [5/10] ")
    parser.add_argument("--ignore_zero_citation", type=bool, default=False, help="Igore Zero cited patents")

    args = parser.parse_args()
    
    print(args)
    main(args)
    
    


    
#!python patent_labelling.py --cite_dict_path '/home/rabindra_nandi/nandi/patent/p_dict_2000_and_after.json' --date_dict_path '/home/rabindra_nandi/nandi/patent/patent_with_date.json' --year_limit 10 --percentage 5 --save_file_path labels_10_5.json