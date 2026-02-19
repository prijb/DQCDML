# Convert a DQCD event into a graph format.
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import os
import sys
# Deep learning
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx

# Storage
import pickle

# Aesthetic
from tqdm import tqdm
import matplotlib.pyplot as plt


# Additional imports
# Add the project path
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vectors import vec4

# Open the root file
f = uproot.open("/home/pb4918/Physics/Projects/DarkQCD/JoeUROP/scenarioA_mpi_4_mA_1p33_ctau_1p0.root")
t = f["Events"]

var_dict_jagged = {
    "Jet": ["pt", "eta", "phi", "mass"],
    "Muon": ["pt", "eta", "phi", "charge", "dxy"],
    "muonSV": ["x", "y", "z", "chi2", "dxy", "dxySig", "mass"],
}

var_list = ["nJet", "nMuon", "nSV", "dimuon_trigger"]
for key, value in var_dict_jagged.items():
    var_list += [f"{key}_{v}" for v in value]

print("\nVariables to load:")
print(var_list)

events = t.arrays(var_list, library="ak", how="zip", entry_stop=100)

# Filter events
events = events[events["dimuon_trigger"] == 1]

event = events[0]

#print("\nFirst event:")
#print(event.tolist())

M_MUON = 0.105658  # GeV
M_PION = 0.139     # GeV

"""
Variables:
num_node_features: Dictionary for the number of node features for each type
node_features_hetero_ind: For a given node type, it gives the relevant indices the features occupy
    For example: Say we have jets with 3 features and muons with 4 features. Then for a given node, there are 7 possible features in total
    -> Muon node = [0, 0, 0, Muon_feature_0[0], Muon_feature_1[0], Muon_feature_2[0], Muon_feature_3[0]]
    -> Jet node = [Jet_feature_0[0], Jet_feature_1[0], Jet_feature_2[0], 0, 0, 0, 0]
    Here the node_features_hetero_ind dictionary will be:
    {   
        "Jet": [0, 1, 2],
        "Muon": [3, 4, 5, 6],
    }
"""
node_features = {
    "Jet": ["pt", "eta", "phi", "mass"],
    "Muon": ["pt", "eta", "phi", "charge", "dxy"],
    "muonSV": ["x", "y", "z", "chi2", "dxy", "dxySig", "mass"],
}
num_node_features = {}
node_features_hetero_ind = {}
k = 0
for key in node_features.keys():
    num_node_features[key] = len(node_features[key])
    node_features_hetero_ind[key] = np.arange(k, k + num_node_features[key])
    k += num_node_features[key]

#print("\nNode features:")
#print(node_features)
#print("Number of node features:")
#print(num_node_features)
#print("Node features hetero ind:")
#print(node_features_hetero_ind)

# Jagged variable name
jvname = {}
for key in node_features.keys():
    jvname[key] = []
    for var in node_features[key]:
        jvname[key].append(var)

print("Jagged variable name:")
print(jvname)

# Count the number of nodes per type
nums = {}
for key in node_features.keys():
    # Taking the first jagged variable to count the number of nodes
    nums[key] = len(event[key][jvname[key][0]])
num_nodes = sum(nums[key] for key in nums)
# The number of edges for n nodes is n^2 - n
num_edges = (num_nodes ** 2) - num_nodes
total_features = sum(num_node_features[key] for key in num_node_features)

print("Number of nodes per type:")
print(nums)
print("Total number of nodes:")
print(num_nodes)
print("Number of edges:")
print(num_edges)

# The node feature matrix for the event is of dimension (num_nodes, total_features)
x = np.zeros((num_nodes, total_features), dtype=float)
tot_nodes = 0
# Vector of four momenta 
p4vec = []

# Loop over the node types
for key in jvname.keys():
    print(f"Node type: {key}")
    if nums[key] > 0:
        for i_node_feature in range(num_node_features[key]):
            feature_name = jvname[key][i_node_feature]
            feature_index = node_features_hetero_ind[key][i_node_feature]
            x[tot_nodes:tot_nodes + nums[key], feature_index] = event[key][feature_name]

        # Fill the four momentum vector
        for i_node in range(nums[key]):
            v = vec4()
            if key == "Jet":
                v.setPtEtaPhiM(event[key]["pt"][i_node], 
                               event[key]["eta"][i_node], 
                               event[key]["phi"][i_node], 
                               event[key]["mass"][i_node])
            elif key == "Muon":
                v.setPtEtaPhiM(event[key]["pt"][i_node], 
                               event[key]["eta"][i_node], 
                               event[key]["phi"][i_node], 
                               M_MUON)
            elif key == "muonSV":
                v.setXYZM(event[key]["x"][i_node], 
                          event[key]["y"][i_node], 
                          event[key]["z"][i_node], 
                          event[key]["mass"][i_node])
            p4vec.append(v)
        tot_nodes += nums[key]

print(f"num_nodes = {num_nodes}, tot_nodes = {tot_nodes}")


# Making edge features
num_edge_features = 1
edge_attr = np.zeros((num_edges, num_edge_features), dtype=float)
# Shows which node pair corresponds to edge i
indexlist = np.zeros((num_nodes, num_nodes), dtype=int) 

n = 0
for i_node in range(num_nodes):
    for j_node in range(num_nodes):
        if i_node != j_node:
            p4_i = p4vec[i_node]
            p4_j = p4vec[j_node]

            # Get the dR 
            dR_ij = p4_i.deltaR(p4_j)

            #print(f"Node {i_node} and {j_node} have dR = {dR_ij}")
            edge_attr[n, 0] = dR_ij

            indexlist[i_node, j_node] = n
            n += 1

# Get the edge index
edge_index = np.zeros((2, num_edges))
n = 0
for i_node in range(num_nodes):
    for j_node in range(num_nodes):
        if i_node != j_node:
            edge_index[0, n] = i_node
            edge_index[1, n] = j_node
            n += 1
        

# Tensor conversion
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor([0], dtype=torch.float)
w = torch.tensor([1], dtype=torch.float)
# No global features
u = torch.tensor([], dtype=torch.float)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Print the data
#print("\nData:")
#print(f"x: {x}")
#print(f"y: {y}")
#print(f"w: {w}")
#print(f"u: {u}")
#print(f"edge_attr: {edge_attr}")
#print(f"edge_index: {edge_index}")

# Create the data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u)

print("\nData object:")
print(data)

############# Plotting the graph #############
# Create a NetworkX graph
G = nx.Graph()

# Assign node types (already computed in your code)
jet_count = nums["Jet"]
muon_count = nums["Muon"]
muonSV_count = nums["muonSV"]

node_type_by_index = {}
for i in range(num_nodes):
    if i < jet_count:
        node_type_by_index[i] = "Jet"
    elif i < jet_count + muon_count:
        node_type_by_index[i] = "Muon"
    else:
        node_type_by_index[i] = "muonSV"

# Add nodes to G with an attribute for the object type
for i in range(num_nodes):
    G.add_node(i, obj_type=node_type_by_index[i])

# Add edges using the edge_index tensor
edge_index_np = edge_index.numpy()
num_edges = edge_index_np.shape[1]
for i in range(num_edges):
    src = int(edge_index_np[0, i])
    dst = int(edge_index_np[1, i])
    G.add_edge(src, dst)

# Map each node type to a color
color_map = {"Jet": "red", "Muon": "blue", "muonSV": "green"}
node_colors = [color_map[node_type_by_index[i]] for i in G.nodes()]

# Create a dictionary for node labels: here we show object type and, for example, the first feature as pT
node_labels = {}
for i in G.nodes():
    pT_val = 0.0
    if i < jet_count:
        pT_val = data.x[i, 0].item()
        print(f"Jet {i} has pT = {pT_val}")
        node_labels[i] = f"{node_type_by_index[i]}\npt={pT_val:.1f}"
    elif i < jet_count + muon_count:
        pT_idx = node_features_hetero_ind["Muon"][0]
        pT_val = data.x[i, pT_idx].item()
        print(f"Muon {i} has pT = {pT_val}")
        node_labels[i] = f"{node_type_by_index[i]}\npt={pT_val:.1f}"
    elif i >= jet_count + muon_count:
        pT_idx = node_features_hetero_ind["muonSV"][-1]
        pT_val = data.x[i, pT_idx].item()
        print(f"muonSV {i} has mass = {pT_val}")
        node_labels[i] = f"{node_type_by_index[i]}\nmass={pT_val:.1f}"


# Create a dictionary for edge labels: here we show the edge_attr's first column (e.g. dR)
edge_labels = {}
for (u, v) in G.edges():
    e_idx = indexlist[u, v]  # from your code
    dR_val = data.edge_attr[e_idx, 0].item()  # The 0th feature is dR
    edge_labels[(u, v)] = f"{dR_val:.2f}"

# Layout for the graph
pos = nx.spring_layout(G, seed=42)  # seed for reproducible layout

# Draw the graph
plt.figure(figsize=(8, 6))
# Draw nodes and edges
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=900, edge_color='gray')

# Draw node labels (we turned off "with_labels" above so we can customize them here)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black', font_size=9)

# Draw edge labels (can get crowded if there are many edges!)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12, label_pos=0.2)

plt.title("DQCD Event Graph: Jets (red), Muons (blue), muonSV (green)")
plt.tight_layout()
plt.savefig("plots/graph.png")
