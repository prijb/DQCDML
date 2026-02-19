import numpy as np
import torch
import awkward as ak
from tqdm import tqdm

# --- Dummy helper classes and functions ---

# A simple 4-vector class to hold kinematics.
class vec4:
    def __init__(self):
        self.data = [0, 0, 0, 0]
    def setPtEtaPhiM(self, pt, eta, phi, M):
        self.data = [pt, eta, phi, M]
    def setXYZM(self, x, y, z, M):
        self.data = [x, y, z, M]
    def __repr__(self):
        return f"vec4({self.data})"

# A dummy Data class simulating torch_geometric Data object.
class Data:
    def __init__(self, num_nodes, x, edge_index, edge_attr, y, w, u):
        self.num_nodes = num_nodes
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.w = w
        self.u = u
    def __repr__(self):
        return (f"Data(num_nodes={self.num_nodes}, "
                f"x={tuple(self.x.shape)}, "
                f"edge_index={tuple(self.edge_index.shape)}, "
                f"edge_attr={tuple(self.edge_attr.shape)}, "
                f"y={self.y.tolist()}, w={self.w.tolist()}, u={self.u.tolist()})")

# Dummy "analytic" module that computes number of edges, connectivity, and edge features.
class analytic:
    @staticmethod
    def count_simple_edges(num_nodes, directed, self_loops):
        if directed:
            return num_nodes * (num_nodes - 1)
        else:
            return num_nodes * (num_nodes - 1) // 2
    @staticmethod
    def get_simple_edge_index(num_nodes, num_edges, self_loops, directed):
        edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edges.append([i, j])
                if directed:
                    edges.append([j, i])
        return np.array(edges).T  # shape [2, num_edges]
    @staticmethod
    def get_Lorentz_edge_features(p4vec, num_nodes, num_edges, num_edge_features, self_loops, directed):
        # For illustration, return random edge features.
        return np.random.rand(num_edges, num_edge_features)

# Dummy auxiliary functions.
class aux:
    @staticmethod
    def slice_range(start, stop, N):
        if start is None:
            start = 0
        if stop is None:
            stop = N
        return start, stop, stop - start
    @staticmethod
    def split_start_end(indices, parts):
        indices = list(indices)
        chunk_size = int(np.ceil(len(indices) / parts))
        chunks = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]
        return [(chunk[0], chunk[-1] + 1) for chunk in chunks]

# --- Simplified parsing function (non-Ray version) ---
def parse_graph_data(X, ids, features, node_features, graph_param,
                     Y=None, weights=None, entry_start=None, entry_stop=None, 
                     null_value=float(-999.0), EPS=1e-12):
    
    # Constants for masses
    M_MUON = 0.105658  # GeV
    M_PION = 0.139     # GeV

    global_on = graph_param['global_on']
    directed = graph_param['directed']
    self_loops = graph_param['self_loops']

    entry_start, entry_stop, num_events = aux.slice_range(entry_start, entry_stop, N=len(X))
    dataset = []
    num_global_features = len(features)
    
    # Build mapping from node type to the number of features and column indices.
    num_node_features = {}
    node_features_hetero_ind = {}
    k = 0
    for key in node_features.keys():
        print("Debug: parse_graph_data: key =", key)
        print(node_features[key])
        num_node_features[key] = len(node_features[key])
        node_features_hetero_ind[key] = np.arange(k, k + num_node_features[key])
        k += num_node_features[key]

    print(num_node_features)
    print(node_features_hetero_ind)
    
    # For each node type, split the variable name into two parts (e.g. "muon_pt" -> ["muon", "pt"]).
    jvname = {}
    for key in node_features.keys():
        jvname[key] = []
        for var in node_features[key]:
            jvname[key].append(var.split('_', 1))

    print("\nDebug: parse_graph_data: jvname =", jvname)
    
    num_edge_features = 4  # fixed number of edge features

    # Loop over events.
    for ev in tqdm(range(entry_start, entry_stop), desc="Parsing events"):
        # Count nodes for each type using the first variable as a proxy.
        nums = {}
        print("\nDebug: parse_graph_data: ev =", ev)
        for key in jvname.keys():
            #print(f"jvname[{key}][0] =", jvname[key][0])
            nums[key] = len(X[ev][jvname[key][0][0]][jvname[key][0][1]])
            # Counts number of nodes for each type
            print(f"Number of nodes of type {key}:", nums[key])
        
        num_nodes = sum(nums[key] for key in nums)
        num_edges = analytic.count_simple_edges(num_nodes, directed, self_loops)
        print(f"Debug: parse_graph_data: num_nodes = {num_nodes}, num_edges = {num_edges}")
        
        # Initialize node feature matrix.
        total_features = sum(num_node_features[key] for key in num_node_features)
        print("Debug: parse_graph_data: num_node_features =", num_node_features)
        print("Debug: parse_graph_data: total_features =", total_features)
        x = np.zeros((num_nodes, total_features), dtype=float)
        tot_nodes = 0
        p4vec = []  # list for 4-momentum vectors
        
        print("Debug: Looping over node types...")
        # Loop over each node type.
        for key in jvname.keys():
            if nums[key] > 0:
                print(f"Debug: parse_graph_data: key = {key}")
                # Fill in the corresponding columns of x.
                for j in range(num_node_features[key]):
                    feat_name = jvname[key][j]
                    print(f"Debug: parse_graph_data: feat_name = {feat_name}")
                    x[tot_nodes:tot_nodes+nums[key], node_features_hetero_ind[key][j]] = \
                        X[ev][feat_name[0]][feat_name[1]]
                # Build 4-vectors for each node.
                for i in range(nums[key]):
                    v = vec4()
                    if key == 'muon':
                        v.setPtEtaPhiM(X[ev]['muon']['pt'][i],
                                       X[ev]['muon']['eta'][i],
                                       X[ev]['muon']['phi'][i],
                                       M_MUON)
                    elif key == 'jet':
                        v.setPtEtaPhiM(X[ev]['jet']['pt'][i],
                                       X[ev]['jet']['eta'][i],
                                       X[ev]['jet']['phi'][i],
                                       X[ev]['jet']['mass'][i])
                    elif key == 'sv':
                        v.setXYZM(X[ev]['sv']['x'][i],
                                  X[ev]['sv']['y'][i],
                                  X[ev]['sv']['z'][i],
                                  X[ev]['sv']['mass'][i])
                    p4vec.append(v)
                tot_nodes += nums[key]

        print("Debug x:", x)
        
        # Build target and weight tensors.
        if Y is None:
            y = torch.tensor([0], dtype=torch.long)
        else:
            y = torch.tensor([Y[ev]], dtype=torch.long)
        if weights is None:
            w = torch.tensor([1.0], dtype=torch.float)
        else:
            w = torch.tensor([weights[ev]], dtype=torch.float)
        
        # Process node features.
        x[~np.isfinite(x)] = null_value
        x = torch.tensor(x, dtype=torch.float)
        
        # Construct edge features and connectivity.
        edge_attr = analytic.get_Lorentz_edge_features(p4vec, num_nodes, num_edges, num_edge_features, self_loops, directed)
        print("Debug: edge_attr =", edge_attr)
        edge_attr[~np.isfinite(edge_attr)] = null_value
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        print("Debug: edge_attr =", edge_attr)
        
        edge_index = analytic.get_simple_edge_index(num_nodes, num_edges, self_loops, directed)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        print("Debug: edge_index =", edge_index)
        
        # Build global feature vector.
        if not global_on:
            u = torch.tensor([], dtype=torch.float)
        else:
            u_mat = np.zeros(len(features), dtype=float)
            for j, feat in enumerate(features):
                # Assume global feature is a scalar.
                u_mat[j] = X[ev][feat] if isinstance(X[ev][feat], (int, float)) else X[ev][feat][0]
            u_mat[~np.isfinite(u_mat)] = null_value
            u = torch.tensor(u_mat, dtype=torch.float)
        
        # Create the Data object.
        data = Data(num_nodes=x.shape[0], x=x, edge_index=edge_index, 
                    edge_attr=edge_attr, y=y, w=w, u=u)
        dataset.append(data)
    
    return dataset

# --- Setting up example inputs ---

# Define node feature names for each type.
muon_vars = ['muon_pt', 'muon_eta', 'muon_phi']
jet_vars  = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_mass']
sv_vars   = ['sv_x', 'sv_y', 'sv_z', 'sv_mass']
node_features = {'muon': muon_vars, 'jet': jet_vars, 'sv': sv_vars}

# Global (scalar) feature names.
scalar_vars = ['event_energy']

# Graph construction parameters.
graph_param = {'global_on': True, 'coord': 'cartesian', 'directed': False, 'self_loops': False}

# Construct two dummy events.
# Event 0: 2 muons, 1 jet, 1 secondary vertex, global event energy.
event0 = {
    'muon': {
        'pt':  [10.0, 20.0],
        'eta': [0.1, -0.2],
        'phi': [0.5, -0.5]
    },
    'jet': {
        'pt':   [50.0],
        'eta':  [1.0],
        'phi':  [0.3],
        'mass': [5.0]
    },
    'sv': {
        'x':    [0.2],
        'y':    [0.1],
        'z':    [0.3],
        'mass': [1.0]
    },
    'event_energy': 100.0
}

# Event 1: 1 muon, 2 jets, 0 secondary vertices, global event energy.
event1 = {
    'muon': {
        'pt':  [15.0],
        'eta': [0.2],
        'phi': [0.4]
    },
    'jet': {
        'pt':   [40.0, 60.0],
        'eta':  [0.9, 1.1],
        'phi':  [0.2, 0.6],
        'mass': [4.5, 6.0]
    },
    'sv': {
        'x':    [],
        'y':    [],
        'z':    [],
        'mass': []
    },
    'event_energy': 150.0
}

X = [event0, event1]
ids = ['event0', 'event1']
Y = [0, 1]
weights = [1.0, 1.0]

# --- Run the parser and inspect outputs ---
data_graph = parse_graph_data(X, ids, scalar_vars, node_features, graph_param,
                              Y, weights, entry_start=0, entry_stop=len(X))

for i, data in enumerate(data_graph):
    print(f"\nGraph for event {i}:")
    print(data)
