# This module preprocesses files and saves them in the graph format
#Processing
import yaml
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os
import sys
#Deep learning
import torch
import torch_geometric
from torch_geometric.data import Data
#Storage
import pickle

#Aesthetic
from tqdm import tqdm

class PreprocessorGraph():
    def __init__(self, file_list, tree_name="Events"):
        self.file_list = file_list
        self.tree_name = tree_name

    def __len__(self):
        return len(self.file_list)
    
    # Does the preprocessing into graphs
    def cache_file(self, input_path, output_path):

        # Open the root file
        f = uproot.open(input_path)
        t = f[self.tree_name]
        