# This code test preprocessing and caching of files
import uproot   
import awkward as ak
import numpy as np
import pandas as pd
import os, sys
import glob
import pickle
#Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler, RandomSampler
#Plotting 
import matplotlib.pyplot as plt
import mplhep as hep
import hist
plt.style.use(hep.style.CMS)
plt.rcParams["figure.figsize"] = (12.5, 10)

cwd = os.getcwd()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.makedirs("plots", exist_ok=True)

# Import project modules
from utils.preprocess import Preprocessor

# Directories
input_dir = "data"
cache_dir = "cache"
vars_yaml_file = "config/vars.yml"

file_list_signal = glob.glob(f"{input_dir}/*ctau_10.root")
file_list_bkg = glob.glob(f"{input_dir}/DileptonMinBias*.root")
#file_list_bkg = glob.glob(f"{input_dir}/data*.root")

print(f"Number of signal files: {len(file_list_signal)}")
print(file_list_signal)
print(f"Number of background files: {len(file_list_bkg)}")
print(file_list_bkg)

# Preprocess signal files
preprocessor_signal = Preprocessor(file_list_signal, vars_yaml_file, tree_name="Events", label=1, cache_dir=f"{cache_dir}/cache_signal", use_existing_cache=False, batch_size=100000)
preprocessor_bkg = Preprocessor(file_list_bkg, vars_yaml_file, tree_name="Events", label=0, cache_dir=f"{cache_dir}/cache_background", use_existing_cache=True, batch_size=100000)

# Cache the files
preprocessor_signal.cache_files()
preprocessor_bkg.cache_files()

# Get the X, y and weights
X_signal, y_signal, w_signal = preprocessor_signal.get_X_y_w()
X_bkg, y_bkg, w_bkg = preprocessor_bkg.get_X_y_w()

# Sum w should be 1 (this keeps the log loss fixed)
#w_signal = w_signal/np.sum(w_signal)
#w_bkg = w_bkg/np.sum(w_bkg)

# Equal probability of signal and background
w_signal = (1/np.sum(w_signal))*w_signal
w_bkg = (1/np.sum(w_bkg))*w_bkg

sum_w_signal = np.sum(w_signal)
sum_w_bkg = np.sum(w_bkg)
scale_pos_weight = (sum_w_bkg/sum_w_signal).item()

# Inverse weighting
#w_signal = 1/(w_signal)
#w_bkg = 1/(w_bkg)

# xs weight
#w_signal = np.ones_like(w_signal)*1.0
#w_bkg = np.ones_like(w_bkg)*1e7

# Print cutflow efficiencies
print("\nCutflow efficiencies")
print(f"Signal: {preprocessor_signal.passed_events}/{preprocessor_signal.total_events} = {preprocessor_signal.passed_events/preprocessor_signal.total_events:.2e}")
print(f"Background: {preprocessor_bkg.passed_events}/{preprocessor_bkg.total_events} = {preprocessor_bkg.passed_events/preprocessor_bkg.total_events:.2e}")

print(f"Initial signal weights: {w_signal}")
print(f"Initial background weights: {w_bkg}")
print(f"Scale pos weight factor: {scale_pos_weight:.2e}")

# Force weights (to be equal)
#w_signal = np.ones(len(X_signal))/len(X_signal)
#w_bkg = np.ones(len(X_bkg))/len(X_bkg)
#print(f"Signal weights (forced to equal): {w_signal}")
#print(f"Background weights (forced to equal): {w_bkg}")


# Split into train and test
from sklearn.model_selection import train_test_split
X_train_signal, X_test_signal, y_train_signal, y_test_signal, w_train_signal, w_test_signal = train_test_split(X_signal, y_signal, w_signal, test_size=0.5, random_state=42)
X_train_bkg, X_test_bkg, y_train_bkg, y_test_bkg, w_train_bkg, w_test_bkg = train_test_split(X_bkg, y_bkg, w_bkg, test_size=0.8, random_state=42)

# Get the plot variables
X_train_signal_plot = X_train_signal[preprocessor_signal.plot_vars]
X_test_signal_plot = X_test_signal[preprocessor_signal.plot_vars]
X_train_bkg_plot = X_train_bkg[preprocessor_signal.plot_vars]
X_test_bkg_plot = X_test_bkg[preprocessor_signal.plot_vars]

# Restrict entries to just the feature variables
X_train_signal = X_train_signal[preprocessor_signal.feature_vars]
X_test_signal = X_test_signal[preprocessor_signal.feature_vars]
X_train_bkg = X_train_bkg[preprocessor_signal.feature_vars]
X_test_bkg = X_test_bkg[preprocessor_signal.feature_vars]

print(f"Number of signal events in training: {len(X_train_signal)}")
print(f"Number of signal events in testing: {len(X_test_signal)}")
print(f"Number of background events in training: {len(X_train_bkg)}")
print(f"Number of background events in testing: {len(X_test_bkg)}")

print("Feature variables:", preprocessor_signal.feature_vars)
print("Plot variables:", preprocessor_signal.plot_vars)

# Train with a BDT (Pandas dataframe is enough)
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from xgboost import plot_importance
model = XGBClassifier(n_estimators=50)
model.set_params(eval_metric="logloss")
# Scale pos weight option
model.set_params(scale_pos_weight=scale_pos_weight)

# Concatenate the signal and background dataframes
X_train = pd.concat([X_train_signal, X_train_bkg])
y_train = np.concatenate([y_train_signal, y_train_bkg])
w_train = np.concatenate([w_train_signal, w_train_bkg])
X_test = pd.concat([X_test_signal, X_test_bkg])
y_test = np.concatenate([y_test_signal, y_test_bkg])
w_test = np.concatenate([w_test_signal, w_test_bkg])

# Concatenate the signal and background dataframes (plot variables)
X_train_plot = pd.concat([X_train_signal_plot, X_train_bkg_plot])
X_test_plot = pd.concat([X_test_signal_plot, X_test_bkg_plot])

# Change the flat weight to be equal
w_train_signal_flat = np.ones(len(X_train_signal))/len(X_train_signal)
w_train_bkg_flat = np.ones(len(X_train_bkg))/len(X_train_bkg)
w_train_flat = np.concatenate([w_train_signal_flat, w_train_bkg_flat])
w_test_signal_flat = np.ones(len(X_test_signal))/len(X_test_signal)
w_test_bkg_flat = np.ones(len(X_test_bkg))/len(X_test_bkg)
w_test_flat = np.concatenate([w_test_signal_flat, w_test_bkg_flat])

# Normalize to total number of training events => Otherwise XGB is stuck at first epoch (logloss = 0.88120)
w_train = (w_train/np.sum(w_train))*len(w_train)

# Train the model
print(f"\nTraining model with xs weights and {len(preprocessor_signal.feature_vars)} variables")
model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], sample_weight_eval_set=[w_test], verbose=True)

# Plot the MVA score curve for signal and background
h_mva_score_signal = (
    hist.Hist.new.Regular(100, 0, 1, label="MVA score", name="mva_score")
    .Weight()
)
h_mva_score_signal_scenarioA = (
    hist.Hist.new.Regular(100, 0, 1, label="MVA score", name="mva_score")
    .Weight()
)
h_mva_score_bkg = (
    hist.Hist.new.Regular(100, 0, 1, label="MVA score", name="mva_score")
    .Weight()
)
h_mva_score_bkg_scenarioA = (
    hist.Hist.new.Regular(100, 0, 1, label="MVA score", name="mva_score")
    .Weight()
)
y_pred_signal = model.predict_proba(X_test_signal)[:, 1]
y_pred_signal_scenarioA = X_test_signal_plot["bdt_scenarioA"]
y_pred_bkg = model.predict_proba(X_test_bkg)[:, 1]
y_pred_bkg_scenarioA = X_test_bkg_plot["bdt_scenarioA"]

h_mva_score_signal.fill(y_pred_signal, weight=w_test_signal)
h_mva_score_signal_scenarioA.fill(y_pred_signal_scenarioA, weight=w_test_signal)
h_mva_score_bkg.fill(y_pred_bkg, weight=w_test_bkg)
h_mva_score_bkg_scenarioA.fill(y_pred_bkg_scenarioA, weight=w_test_bkg)

fig, ax = plt.subplots()
hep.histplot(h_mva_score_bkg, label="Background", histtype="step", ax=ax, color="blue", density=True, flow=None)
hep.histplot(h_mva_score_signal, label="Signal", histtype="step", ax=ax, color="red", density=True, flow=None)
#ax.set_ylim(1e-2, 2.0)
ax.set_yscale("log")
ax.set_xlabel("MVA score")
ax.set_ylabel("Density")
ax.legend()
plt.savefig("plots/mva_score.png")

fig, ax = plt.subplots()
hep.histplot(h_mva_score_bkg_scenarioA, label="Background", histtype="step", ax=ax, color="blue", density=True, flow=None)
hep.histplot(h_mva_score_signal_scenarioA, label="Signal", histtype="step", ax=ax, color="red", density=True, flow=None)
#ax.set_ylim(1e-2, 2.0)
ax.set_yscale("log")
ax.set_xlabel("MVA score (sc A BDT)")
ax.set_ylabel("Density")
ax.legend()
plt.savefig("plots/mva_score_scenarioA.png")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1], sample_weight=w_test)
fpr_scenarioA, tpr_scenarioA, thresholds_scenarioA = roc_curve(y_test, X_test_plot["bdt_scenarioA"], sample_weight=w_test)
roc_auc = auc(fpr, tpr)
roc_auc_scenarioA = auc(fpr_scenarioA, tpr_scenarioA)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, lw=2, color="orange", label=f"ROC AUC = {roc_auc:.4f}")
ax.plot(fpr_scenarioA, tpr_scenarioA, lw=2, color="green", label=f"ROC AUC (sc A BDT) = {roc_auc_scenarioA:.4f}")
ax.plot(np.logspace(-5, 0, 100), np.logspace(-5, 0, 100), linestyle="--", lw=2, color="k", label="Random")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xscale("log")
ax.set_xlim(1e-5, 1)
ax.set_ylim(0, 1)
ax.legend()
plt.savefig("plots/roc_curve.png")

# Plot the feature importance
fig, ax = plt.subplots()
plot_importance(model, ax=ax, importance_type="total_gain", max_num_features=20)
plt.tight_layout()
plt.savefig("plots/feature_importance.png")

## Plotting
# Plot the testing leading muonSV_mass_0 with weight variable
h_muonsv_mass_0_signal = (
    hist.Hist.new.Regular(100, 0, 10, label="MuonSV mass 0", name="muonsv_mass_0")
    .Weight()
)
h_muonsv_mass_0_bkg = (
    hist.Hist.new.Regular(100, 0, 10, label="MuonSV mass 0", name="muonsv_mass_0")
    .Weight()
)
