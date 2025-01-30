# patent_high_citation
Code for the work An Experimental Analysis on Evaluating Patent Citations  "https://aclanthology.org/2024.emnlp-main.23" published in EMNLP 2024



## GNN Explanation with GNNExplainer  

**gnn_explainer_G06.py , gnn_explainer_A61.py gnn_explainer_H04.py** 


The script leverages **GNNExplainer** from **Torch Geometric** to analyze and visualize how the **Graph Convolutional Network (GCN)** makes predictions. It provides insights into important features, key connections, and influential nodes within the graph.  

### **How It Works**  

1️⃣ **Feature Importance Analysis**  
   - Identifies the most influential input features (e.g., embeddings from PatentBERT or Doc2Vec) that drive model predictions.  

2️⃣ **Subgraph Visualization**  
   - Extracts and highlights critical node connections, creating interpretable subgraph representations.  

3️⃣ **Edge Filtering & Refinement**  
   - Removes weak connections using an **edge weight threshold (0.65)** to retain only the strongest influencing edges.  

4️⃣ **Graph & Feature Plots**  
   - Saves **feature importance plots** and **filtered subgraph visualizations** as PNG files in the `figures/` directory.  

5️⃣ **Key Node Influence Extraction**  
   - Logs the most influential neighboring nodes affecting each prediction in a structured JSON format.  

This approach enhances model transparency, making it easier to understand how the GCN classifies nodes based on graph structure and features.  

## **`fidelity_score_cal.py` - Overview**

This script evaluates the fidelity of explanations generated for a Graph Neural Network (GNN) model using **GNNExplainer** and the **fidelity metric** from **Torch Geometric**. It trains a specified GNN model, generates node importance explanations, and calculates fidelity scores for the generated explanations.

### **How It Works**

1️⃣ **Model Training**  
   - The script loads data (edges, features, and labels) from the specified paths and prepares the graph data. It then selects the GNN model (GCN, GAT, GTN, or GSAGE) and trains it on the provided data.  
   - The model's performance is logged, including training loss and test accuracy.  

2️⃣ **Explanation Generation**  
   - Once the model is trained, **GNNExplainer** is used to generate feature importance explanations for the nodes.  
   - The script selects a set of nodes (with class labels 0 and 1) and computes explanations for both the entire set of nodes and a subset of nodes.  

3️⃣ **Fidelity Calculation**  
   - The **fidelity** metric is calculated for the full explanation (for all nodes) and partial explanation (for selected nodes). The fidelity score measures how well the explanations align with the actual model behavior.  

4️⃣ **Logging and Saving Results**  
   - The fidelity scores and model information (including training details) are logged and saved in a result file.  


###  **Sample Usage**

Run the script to compute fidelity scores for a trained GNN model:

```bash
python fidelity_score_cal.py --feature_path gnn_data/features_A61_10y_10p_t0.68_d2v-v2.pt --edge_path gnn_data/features_A61_10y_10p_t0.68_d2v-v2.pt --label_path gnn_data/label_A61_5_10p.pt --result_path results/patentbert.txt
```





