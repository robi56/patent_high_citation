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





