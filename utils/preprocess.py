# This module preprocesses files and caches them
#Processing
import yaml
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
#Deep learning
import torch
#Storage
import pickle

#Aesthetic
from tqdm import tqdm

class Preprocessor():
    def __init__(self, file_list, vars_yaml, tree_name="Events", label=0, transform=None, cache_dir="cachedir", use_existing_cache=False, batch_size=10000):
        self.file_list = file_list
        self.vars_yaml = vars_yaml
        self.tree_name = tree_name
        self.label = label
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_existing_cache = use_existing_cache
        self.batch_size = batch_size

        # Attributes filled
        self.vars_yaml_content = None
        self.scalar_vars = None
        self.jagged_vars = None
        self.weight_var = None
        self.read_vars = None
        self.train_vars = None
        self.multiplicities = None
        self.feature_vars = None
        self.plot_vars = None

        # Save cutflow efficiencies
        self.total_events = 0
        self.passed_events = 0

        # Define branches to read and multiplicities
        vars_yaml_content = yaml.load(open(self.vars_yaml, "r"), Loader=yaml.FullLoader)
        scalar_vars = vars_yaml_content["scalar_vars"]
        scalar_vars_aux = vars_yaml_content["scalar_vars_aux"]
        jagged_vars = vars_yaml_content["jagged_vars"]
        if vars_yaml_content["weight_var"] is not None: self.weight_var = vars_yaml_content["weight_var"]
        jagged_vars_aux = vars_yaml_content["jagged_vars_aux"]
        multiplicities = vars_yaml_content["jagged_multiplicity"]

        self.vars_yaml_content = vars_yaml_content
        self.scalar_vars = scalar_vars
        self.jagged_vars = jagged_vars

        # Get the list of variables to read and train on
        read_vars = []
        train_vars = []
        for var in scalar_vars:
            read_vars.append(var)
            train_vars.append(var)

        for var in scalar_vars_aux:
            read_vars.append(var)
        
        for obj in jagged_vars.keys():
            for var in jagged_vars[obj]:
                read_vars.append(f"{obj}_{var}")
                train_vars.append(f"{obj}_{var}")

        for obj in jagged_vars_aux.keys():
            for var in jagged_vars_aux[obj]:
                read_vars.append(f"{obj}_{var}")

        if self.weight_var is not None:
            read_vars.append(self.weight_var)

        # Get the list of long-vector variables
        var_list = []
        var_list_plot = []
        var_list += scalar_vars
        n_vars = len(scalar_vars)
        for obj in jagged_vars.keys():
            for var in jagged_vars[obj]:
                n_vars += multiplicities[obj]
                for i in range(multiplicities[obj]):
                    var_list.append(f"{obj}_{var}_{i}")

        # Add plot variables
        plot_vars = vars_yaml_content["plot_vars"]
        for var in plot_vars["scalar_vars"]:
            var_list_plot.append(var)
        
        for obj in plot_vars["jagged_vars"].keys():
            for var in plot_vars["jagged_vars"][obj]:
                for i in range(multiplicities[obj]):
                    var_list_plot.append(f"{obj}_{var}_{i}")

        print(f"Plotting variables: {var_list_plot}")

    
        self.read_vars = read_vars
        self.train_vars = train_vars
        self.multiplicities = multiplicities
        self.feature_vars = var_list
        self.plot_vars = var_list_plot

        #print(f"Reading variables: {read_vars}")
        #print(f"Training variables: {train_vars}")
        #print(f"Number of training variables: {n_vars}")
        #print(f"Feature variables: {var_list}")


    def __len__(self):
        return len(self.file_list)
    
    # Load the data from an existing cache
    def get_X_y_w(self):
        X = None
        y = None
        w = None
        print("Loading cache from", self.cache_dir)
        for i, file in enumerate(tqdm(os.listdir(self.cache_dir), total=len(os.listdir(self.cache_dir)))):
            cache_file = os.path.join(self.cache_dir, file)

            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                X_i = data["X"]
                y_i = data["y"]
                w_i = data["w"]
                self.total_events += data["total_events"]
                self.passed_events += data["passed_events"]
                if i == 0:
                    X = X_i
                    y = y_i
                    w = w_i
                else:
                    #X = torch.cat((X, X_i), dim=0)
                    #y = torch.cat((y, y_i), dim=0)
                    #w = torch.cat((w, w_i), dim=0)
                    X = pd.concat([X, X_i])
                    y = pd.concat([y, y_i])
                    w = pd.concat([w, w_i])
        
        return X, y, w
    
    # Cache a single file (Does preprocessing including flattening)
    def cache_file(self, input_path, output_path):

        # Open the file
        f = uproot.open(input_path)
        t = f[self.tree_name]
        # Get total number of events
        n_entries_file = t.num_entries
        n_batches = n_entries_file//self.batch_size + 1
        print(f"Splitting file into {n_batches} batches")

        for i_batch in tqdm(range(n_batches)):
            #events = t.arrays(self.read_vars, library="ak", how="zip")
            data_dict = {}

            events = t.arrays(self.read_vars, library="ak", entry_start=i_batch*self.batch_size, entry_stop=(i_batch+1)*self.batch_size, how="zip")

            # Count total events
            total_events = len(events)

            # Filter out events
            events = events[events["dimuon_trigger"] == 1]

            # Filter out entries (jagged)
            filter_vars = self.vars_yaml_content["filter_vars"]
            if filter_vars["scalar_filter"] is not None:
                for scalar_filter in filter_vars["scalar_filter"]:
                    mask = eval(f"events.{scalar_filter}")
                    events = events[mask]
            if filter_vars["jagged_filter"] is not None:
                collections_filtered = []
                for jagged_filter in filter_vars["jagged_filter"]:
                    collection = jagged_filter.split(".")[0]
                    mask = eval(f"events.{jagged_filter}")
                    events[collection] = events[collection][mask]
                    if collection not in collections_filtered:
                        collections_filtered.append(collection)
                # Recount multiplicities after filtering
                for collection in collections_filtered:
                    events[f"n{collection}"] = ak.num(events[collection])

            """
            mask_jagged = events["muonSV"]["charge"] == 0
            events["muonSV"] = events["muonSV"][mask_jagged]   
            mask_jagged = events["SV"]["charge"] == 0
            events["SV"] = events["SV"][mask_jagged]
            mask_jagged = events["fourmuonSV"]["charge"] == 0
            events["fourmuonSV"] = events["fourmuonSV"][mask_jagged]


            # Count after jagged filtering 
            events["nmuonSV"] = ak.num(events["muonSV"])
            events["nSV"] = ak.num(events["SV"])
            events["nfourmuonSV"] = ak.num(events["fourmuonSV"])"
            """

            # Count passed events
            passed_events = len(events)

            # Reorder the jagged arrays
            sort_vars = self.vars_yaml_content["sort_vars"]
            for obj in sort_vars:
                sort_var = sort_vars[obj]["var"]
                #print(f"Sorting {obj} by {sort_var}")
                sort_mask = ak.argsort(events[obj][sort_var], axis=-1, ascending=sort_vars[obj]["ascending"])
                events[obj] = events[obj][sort_mask]

            # Make an array of long-vectors (Fill using masking)
            events_flat = np.full((len(events), len(self.feature_vars)), -999.0)

            for i, scalar_var in enumerate(self.scalar_vars):
                events_flat[:, i] = events[scalar_var]

            var_index = len(self.scalar_vars)
            for obj in self.jagged_vars.keys():
                for var in self.jagged_vars[obj]:
                    for i in range(self.multiplicities[obj]):
                        mask = ak.num(events[obj]) > i
                        events_flat[mask, var_index] = (events[mask])[obj][var][:, i]
                        var_index += 1

            # Then finally fill with plot variables 
            events_plot = np.full((len(events), len(self.plot_vars)), -999.0)
            plot_var_index = 0
            plot_vars = self.vars_yaml_content["plot_vars"]

            for scalar_var in plot_vars["scalar_vars"]:
                events_plot[:, plot_var_index] = events[scalar_var]
                plot_var_index += 1
            
            for obj in plot_vars["jagged_vars"].keys():
                for var in plot_vars["jagged_vars"][obj]:
                    for i in range(self.multiplicities[obj]):
                        mask = ak.num(events[obj]) > i
                        events_plot[mask, plot_var_index] = (events[mask])[obj][var][:, i]
                        plot_var_index += 1

            # Concatenate columnwise
            events_flat = np.concatenate((events_flat, events_plot), axis=1)

            # Save as pandas dataframe
            X_df = pd.DataFrame(events_flat, columns=self.feature_vars+self.plot_vars)
            y_df = pd.DataFrame({"label": self.label*np.ones((X_df.shape[0],))})
            if self.weight_var is not None:
                w_df = pd.DataFrame({self.weight_var: events[self.weight_var]})
            else:
                w_df = pd.DataFrame({"weight": np.ones((X_df.shape[0],))})

            data_dict["X"] = X_df
            data_dict["y"] = y_df
            data_dict["w"] = w_df
            # Save cutflow efficiencies
            data_dict["total_events"] = total_events
            data_dict["passed_events"] = passed_events


            #X = torch.tensor(events_flat, dtype=torch.float32)
            #y = torch.full((X.shape[0],), self.label, dtype=torch.float32)
            #if self.weight_var is not None:
            #    w = torch.tensor(events[self.weight_var], dtype=torch.float32)
            #else:
            #    w = torch.ones((X.shape[0],), dtype=torch.float32)

            #data_dict["X"] = X
            #data_dict["y"] = y
            #data_dict["w"] = w

            # Save the data
            #with open(output_path, "wb") as f:
            #    pickle.dump(data_dict, f)

            # Replace the file name with a batch number: file_0.pkl -> file_0_0.pkl
            output_path_batch = output_path.replace(".pkl", f"_{i_batch}.pkl")
            with open(output_path_batch, "wb") as f:
                pickle.dump(data_dict, f)

        return None
    
    # Cache all files
    def cache_files(self):
        if self.use_existing_cache:
            print(f"Using existing cache at {self.cache_dir}")
            return None
        
        else:
            print("Caching files to ", self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            #Clear cache directory
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))

            for i, file in enumerate(tqdm(self.file_list)):
                input_path = file
                output_path = os.path.join(self.cache_dir, f"file_{i}.pkl")
                self.cache_file(input_path, output_path)







