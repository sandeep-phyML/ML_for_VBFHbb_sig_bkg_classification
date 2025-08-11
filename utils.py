import os
import math
import keras
import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#from update_branch_root import ROOT_pd_ROOT
from keras.losses import SparseCategoricalCrossentropy
loss=SparseCategoricalCrossentropy()
from tensorflow.keras.activations import swish
from sklearn.metrics import roc_curve, auc
from typing import Tuple, List, Optional , Dict
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE, SelectFromModel
from tensorflow.keras import layers
from tensorflow.keras.models import clone_model
#import subprocess  , use this if you want to run a script from command line
import ROOT
import array
from tensorflow.keras.layers import Layer
import yaml
def save_output(out_data , ofile_path: str = "model_summary.txt"):
    with open(ofile_path, "a") as f:
        f.write(out_data + "\n")

class PrepareDataset():
    def __init__(self , train_feature: List[str] , weight_feature: List[str],output_log_path , nsample , is_resampling: bool,mass_norm : bool = True):
        self.train_feature = train_feature
        self.weight_features = weight_feature
        self.nsample = nsample
        self.output_file_path = output_log_path
        self.total_weight_branch = "weight_"
        self.norm_branch = "norm_weight"
        self.label_branch_name = "label"
        self.event_branch = "T_event"
        self.tree_name = "tree"
        self.is_resample_all = is_resampling
        self.normalise_mass = mass_norm 
        
    def convert_tree_to_pd(self, input_file: str) -> pd.DataFrame:
        file_data = uproot.open(f"{input_file}:{self.tree_name}")
        pd_data = file_data.arrays(file_data.keys(), library="pd")
        save_output(f"Converted {input_file} to pandas DataFrame with shape {pd_data.shape}",ofile_path = self.output_file_path)
        return pd_data
        
    def get_weight_branch_(self, pd_data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        new_pd_data = pd_data.copy()
        new_pd_data[self.total_weight_branch] = np.ones(len(new_pd_data))
        avail_weight_branches = [branch for branch in self.weight_features if branch in pd_data.columns ]
        save_output(f"Available weight features : {avail_weight_branches}",ofile_path = self.output_file_path)
        for branch in avail_weight_branches:
            if branch in new_pd_data.columns:
                new_pd_data[self.total_weight_branch] *= new_pd_data[branch]
            else:
                raise KeyError(f"Column '{branch}' not found in DataFrame.")
        return new_pd_data 
        
    def normalise_the_weight(self, pd_data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        scale_weight = 1.0 /( pd_data[self.total_weight_branch].sum())
        median_weights_ = 1.0/np.median(pd_data[self.total_weight_branch]*scale_weight)
        pd_data[self.norm_branch] = pd_data[self.total_weight_branch] * scale_weight * median_weights_
        save_output(f"Normalised weights with median scaling applied",ofile_path = self.output_file_path)
        return pd_data 
  
    def add_label_branch(self, pd_data: pd.DataFrame, label: int) -> pd.DataFrame:
        pd_data[self.label_branch_name] = np.full(len(pd_data), label)
        save_output(f"Added label branch with value {label}",ofile_path = self.output_file_path)
        return pd_data

    def resample_events(self, pd_data: pd.DataFrame) -> pd.DataFrame:
        replace = self.nsample > len(pd_data)
        resampled_data = pd_data.sample(n=self.nsample, replace=replace, random_state=42).reset_index(drop=True)
        save_output(f"Resampled data to {self.nsample} events with replacement={replace}",ofile_path = self.output_file_path)
        return resampled_data

        
    def replace_nan_inf_with_zero(self, pd_data: pd.DataFrame) -> pd.DataFrame:
        pd_data = pd_data.replace([np.nan, np.inf, -np.inf], 0)
        save_output("Replaced NaN and Inf values with zero",ofile_path = self.output_file_path)
        return pd_data
        
    def get_np_feaweilabel_odd_even_train(self,folder_path:str, file_label_list: List[Dict[str, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_data = []
        for item in file_label_list:
            file_path = os.path.join(folder_path, item["file_name"])
            label_value = item["label"]
            pd_data_ = self.convert_tree_to_pd(file_path)
            if self.is_resample_all:
                pd_data_ = self.resample_events(pd_data_)
            pd_data_ = self.add_label_branch(pd_data_, label = label_value)
            pd_data_= self.get_weight_branch_(pd_data_)
            pd_data_ = self.normalise_the_weight(pd_data_ )
            pd_data_ = pd_data_[self.train_feature + [self.event_branch,self.norm_branch, self.label_branch_name] +[ "T_reg_mbb"]]
            all_data.append(pd_data_)
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = self.replace_nan_inf_with_zero(combined_data)
        combined_data = combined_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        feature_odd = combined_data[combined_data[self.event_branch] % 2 == 1].reset_index(drop=True).filter(self.train_feature).to_numpy()
        feature_even = combined_data[combined_data[self.event_branch] % 2 == 0].reset_index(drop=True).filter(self.train_feature).to_numpy()
        weight_odd = combined_data[combined_data[self.event_branch] % 2 == 1].reset_index(drop=True)[self.norm_branch].to_numpy()
        weight_even = combined_data[combined_data[self.event_branch] % 2 == 0].reset_index(drop=True)[self.norm_branch].to_numpy()
        label_odd = combined_data[combined_data[self.event_branch] % 2 == 1].reset_index(drop=True)[self.label_branch_name].to_numpy()
        label_even = combined_data[combined_data[self.event_branch] % 2 == 0].reset_index(drop=True)[self.label_branch_name].to_numpy()
        mass_all = combined_data["T_reg_mbb"].to_numpy()
     
        min_mass = mass_all.min()
        max_mass = mass_all.max()
        if self.normalise_mass:
            save_output(f"Normalising mass with min: {min_mass}, max: {max_mass}", ofile_path=self.output_file_path)
            mass_odd = ((combined_data[combined_data[self.event_branch] % 2 == 1]
                        .reset_index(drop=True)["T_reg_mbb"].to_numpy()) - min_mass) / (max_mass - min_mass)
            
            mass_even = ((combined_data[combined_data[self.event_branch] % 2 == 0]
                        .reset_index(drop=True)["T_reg_mbb"].to_numpy()) - min_mass) / (max_mass - min_mass)
        else:
            save_output(f"Not normalising mass, using raw values", ofile_path=self.output_file_path)
            mass_odd = combined_data[combined_data[self.event_branch] % 2 == 1].reset_index(drop=True)["T_reg_mbb"].to_numpy()
            mass_even = combined_data[combined_data[self.event_branch] % 2 == 0].reset_index(drop=True)["T_reg_mbb"].to_numpy()
        save_output(f"Processed data - Odd features shape: {feature_odd.shape}, Even features shape: {feature_even.shape}", ofile_path=self.output_file_path)
        return feature_odd, feature_even , weight_odd , weight_even , label_odd , label_even , mass_odd , mass_even
            
    def get_even_odd_predict_feature_df(self, file_list: List[str]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        all_data = {}
        save_output(f"Processing files for predictions : {file_list}", ofile_path=self.output_file_path)
        for file in file_list:
            all_data[file] = {"odd_data": {}, "even_data": {}}
            pd_data_ = self.convert_tree_to_pd(file)
            odd_data_ = pd_data_[pd_data_[self.event_branch] % 2 == 1].reset_index(drop=True)
            even_data_ = pd_data_[pd_data_[self.event_branch] % 2 == 0].reset_index(drop=True)
            all_data[file]["odd_data"]["full_data_"] = odd_data_
            all_data[file]["even_data"]["full_data_"] = even_data_
            all_data[file]["odd_data"]["features"] = self.replace_nan_inf_with_zero(odd_data_.filter(self.train_feature)).to_numpy()
            all_data[file]["even_data"]["features"] = self.replace_nan_inf_with_zero(even_data_.filter(self.train_feature)).to_numpy()
            save_output(f"Processed file {file} - Odd features shape: {all_data[file]['odd_data']['features'].shape}, Even features shape: {all_data[file]['even_data']['features'].shape}", ofile_path=self.output_file_path)
        return all_data
    def create_five_perc_data(self,infile_path):
        pd_data = self.convert_tree_to_pd(infile_path)
        selected_data = pd_data.sample(frac=0.05, random_state=42)
        with uproot.recreate(infile_path.replace(".root","_5perc.root")) as f:
            f["tree"] = selected_data
        save_output(f"Created 5% sample data from {infile_path} , selected event {len(selected_data)} from {len(pd_data)} ", ofile_path=self.output_file_path)
        return True 
    


class Plot():
    def __init__(self , history_out_path: str ):
        self.tree_name = "tree"
        self.history_out_path = history_out_path
    def helper(self):
        print("\n=== Plot Class Methods ===")
        print("1. plot_tprofile(file_path, save_path, is_mclass=True, is_data=False,")
        print("                 mbb_range=(110.0, 140.0), vbfbclass='BiClassANN',")
        print("                 vbf_dnn='VBF_DNN', qcd_dnn='QCD_DNN')")
        print("   → Plots TProfile of Higgs reconstructed mass vs DNN score.")
        
        print("\n2. plot_var_distribution(branch_names, files_colour_legend_dict, folder_path, save_path,")
        print("                         range_=(0, 1), nbin=25)")
        print("   → Plots normalized variable distribution from ROOT trees.")
        
        print("\n3. plot_roc_curve(y_true, y_score, save_path, label='Model')")
        print("   → Plots ROC curve with AUC score from given predictions.")
        print("==========================\n")
    def plot_tprofile(self,file_path ,save_path, is_mclass = True ,is_data = False,nbins = 50,mbb_range=(110.0, 140.0),vbfbclass="BiClassANN",vbf_dnn = "VBF_DNN",qcd_dnn = "QCD_DNN"):
        save_output(f"Plotting TProfile from file: {file_path}, is_mclass = {is_mclass} , vbfbclass={vbfbclass},vbf_dnn = {vbf_dnn},qcd_dnn = {qcd_dnn} ", ofile_path=self.history_out_path)
        ROOT.gStyle.SetPalette(112)
        # Define bin edges (uniform bins from 0 to 1)
        nbins = nbins
        edges = np.linspace(0.0, 1.0, nbins + 1)
        # Open the ROOT file
        file = ROOT.TFile.Open(file_path)
        if not file or file.IsZombie():
            print("Error: Could not open file.")
            return
        # Access the tree
        tree = file.Get(self.tree_name )
        if not tree:
            print("Error: Could not find TTree named 'tree'")
            return
        # Define scalar variable holders using array module
        dnn_vbfb = array.array('f', [0.])
        dnn_qcd = array.array('f', [0.])
        higgs_reco_mass1 = array.array('f', [0.])
        T_weight = array.array('f', [0.])
        T_HLTweight = array.array('f', [0.])
        T_PUweight = array.array('f', [0.])
        LUMI = array.array('f', [0.])
        # Set branch addresses
        if is_mclass:
            tree.SetBranchAddress(vbf_dnn, dnn_vbfb)
            tree.SetBranchAddress(qcd_dnn, dnn_qcd)
        else:
            tree.SetBranchAddress(vbfbclass, dnn_vbfb)
        tree.SetBranchAddress("T_reg_mbb", higgs_reco_mass1)
        if not is_data:
            tree.SetBranchAddress("T_weight", T_weight)
            tree.SetBranchAddress("T_HLTweight", T_HLTweight)
            tree.SetBranchAddress("T_PUweight", T_PUweight)
            tree.SetBranchAddress("LUMI", LUMI)
        # Create TProfile and 2D histogram
        profile = ROOT.TProfile("profile", "TProfile of Higgs Reco Mass vs DNN Score;DNN Score;Higgs Reco Mass [GeV]",
                                nbins, edges)
        profile.Sumw2()
        h = ROOT.TH2F("h", "", 20, 0.0, 1.0, 25, mbb_range[0], mbb_range[1])
        # Loop over entries
        nEntries = tree.GetEntries()
        for i in range(nEntries):
            tree.GetEntry(i)
            if is_mclass:
                x = dnn_vbfb[0]/ (dnn_vbfb[0] + dnn_qcd[0]) if (dnn_vbfb[0] + dnn_qcd[0]) != 0 else 0.0
            else:
                x = dnn_vbfb[0]
            
            y = higgs_reco_mass1[0]
            weight = 1.0
            if not is_data:
                weight = T_weight[0] * T_HLTweight[0] * T_PUweight[0] * LUMI[0]
            h.Fill(x, y, weight)
            profile.Fill(x, y, weight)
        print("Correlation coefficient:", h.GetCorrelationFactor())
        save_output(f"Correlation coefficient: {h.GetCorrelationFactor()}", ofile_path=self.history_out_path)
        # Create h_ from h
        h_ = ROOT.TH2F("h_", "", 20, 0.0, 1.0, 25, 110.0, 140.0)
        for i in range(1, 21):
            print(f"Bin error {i}: {profile.GetBinError(i)}")
            for j in range(1, 26):
                h_.SetBinContent(i, j, h.GetBinContent(i, j))

        # Create TProfile from TH2
        profile_ = h_.ProfileX("profile", 0, 20, "s")

        # Plotting
        c1 = ROOT.TCanvas("c1", "", 800, 702)
        c1.SetLeftMargin(0.1115)
        c1.SetRightMargin(0.1441)
        h_.SetStats(0)
        h_.GetXaxis().SetTitle("DNN score")
        h_.GetYaxis().SetTitle("Higgs mass (GeV)")
        h_.Draw("COLZ")

        profile_.SetLineColor(ROOT.kRed)
        profile_.SetMarkerStyle(20)
        profile_.SetMarkerColor(ROOT.kRed)
        profile_.SetLineWidth(2)
        profile_.SetErrorOption("S")
        profile_.Draw("SAME")

        # Add CMS label
        latex = ROOT.TLatex(0.48, 0.9, "#bf{CMS} #it{Simulation Preliminary}")
        latex.SetNDC()
        latex.SetTextFont(42)
        latex.SetTextSize(0.04)
        latex.SetTextAlign(20)
        latex.Draw("same")

        # Save plot
        c1.SaveAs(save_path)
        save_output(f"Saved TProfile plot to {save_path}",ofile_path = self.history_out_path)

        # Cleanup
        file.Close()
    
    def plot_var_distribution(self, branch_names:List[str], files_colour_legend_dict, folder_path , save_path , range_=(0, 1),nbin = 25):
        plt.figure(figsize=(8, 6))
        for file_colour_legend in files_colour_legend_dict:
            file_name, color, label = file_colour_legend
            file_path = os.path.join(folder_path, file_name)
            try:
                df = uproot.open(f"{file_path}:tree").arrays(branch_names, library="pd")
                clean_data = df.replace([np.inf, -np.inf], np.nan).dropna()
                num_data = clean_data[branch_names[0]]
                if len(branch_names) == 1:
                    deno_data = 1.0
                else:
                    deno_data = clean_data[branch_names[0]]
                    for branch_name in branch_names[1:]:
                        deno_data += clean_data[branch_name]
                save_output(f"Processing file {file_path} for branches {branch_names} with label {label}", ofile_path = self.history_out_path)
                selected_data = num_data / deno_data
                final_data = selected_data.replace([np.inf, -np.inf], np.nan).dropna()
                if final_data.empty:
                    print(f"No valid data for branch {branch_names} in {label}")
                    continue
                counts, bins = np.histogram(final_data , bins=nbin, range=range_, density=False)
                total = counts.sum()
                if total == 0:
                    print(f"Empty histogram for {label}")
                    continue
                norm_counts = counts / total  # Normalize so sum of bins content = 1
                plt.step(bins[:-1], norm_counts, where="mid", label=label, color=color)
                plt.margins(x=0)
            except Exception as e:
                print(f"Failed for {file_path}: {e}")
        plt.xlabel(branch_name)
        plt.ylabel("Normalized Entries (Sum = 1)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title(f"Distribution of {branch_name}")
        plt.tight_layout()
        plt.savefig(f"{save_path}")
        plt.show()
        save_output(f"Saved variable distribution plot to {save_path}",ofile_path = self.history_out_path)

    def plot_roc_curve(self, y_true, y_score , save_path , label="Model"):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        save_output(f"Saved ROC curve plot to {save_path}",ofile_path = self.history_out_path)



class DNNModel():
    def __init__(self, input_shape: int,learning_rate: float,loss_function: str, activation: str = 'sigmoid', nclass: int = 1):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.activation = activation
        self.nclass = nclass
        self.base_classifier = self.build_classifier()
        self.odd_classifier = clone_model(self.base_classifier)
        self.even_classifier = clone_model(self.base_classifier)
        self.odd_classifier.set_weights(self.base_classifier.get_weights())
        self.even_classifier.set_weights(self.base_classifier.get_weights())
        self.optimizer_odd = keras.optimizers.AdamW(learning_rate=learning_rate)
        self.optimizer_even = keras.optimizers.AdamW(learning_rate=learning_rate)
    def build_classifier(self):
        inputs = keras.Input(shape=(self.input_shape,))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='elu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.nclass, activation=self.activation)(x)
        return keras.Model(inputs, output, name="classifier")
    def compile_mclass_model(self, odd_model, even_model ):
        odd_model.compile(optimizer= self.optimizer_odd,loss=self.loss_function,
                    metrics=[
                        'accuracy'
                    ])
        even_model.compile(optimizer = self.optimizer_even,loss=self.loss_function,
                    metrics=[
                        'accuracy'
                    ])
        return True 
    def is_compiled(self, model):
        return model.optimizer is not None


class DNNModelTraining():
    def __init__(self,learning_rate: float,loss_function: str ):
        self.optimizer_odd = keras.optimizers.AdamW(learning_rate=learning_rate)
        self.optimizer_even = keras.optimizers.AdamW(learning_rate=learning_rate)
        self.loss_function = loss_function
    def compile_mclass_model(self, odd_model, even_model ):
        odd_model.compile(optimizer= self.optimizer_odd,loss=self.loss_function,
                    metrics=[
                        'accuracy'
                    ])
        even_model.compile(optimizer = self.optimizer_even,loss=self.loss_function,
                    metrics=[
                        'accuracy'
                    ])
        return True 


class GradientReversalLayer(Layer):
    def __init__(self, lambd=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = lambd

    def call(self, x):
        @tf.custom_gradient
        def reverse(x):
            def grad(dy):
                return -self.lambd * dy
            return x, grad
        return reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({"lambd": self.lambd})
        return config

class BasicMethods():
    def __init__(self, ):
        pass
    def read_config_file(self, file_name ) -> Dict[str, any]:
        with open(file_name, 'r') as file:
            config = yaml.safe_load(file)
        return config.copy()
    def create_log_folder(self, log_path: str) -> None:
        with open(log_path, 'w') as f:
            f.write("Log file created.\n")
        return True
# ---------- 2. Classifier Network ----------
def build_classifier(input_shape, activation='sigmoid', nclass=1):
    inputs = keras.Input(shape=(input_shape,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256, activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(nclass, activation=activation)(x)
    return keras.Model(inputs, output, name="classifier")


# ---------- 3. Adversary Network ----------
def build_adversary():
    inputs = keras.Input(shape=(1,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='linear')(x)
    return keras.Model(inputs, output, name="adversary")


# ---------- 4. Adversarial Combined Model ----------
def build_adversarial_model(classifier, adversary, input_shape):
    inputs = keras.Input(shape=(input_shape,))
    classifier_output = classifier(inputs)
    reversed_output = GradientReversalLayer()(classifier_output)
    adversary_output = adversary(reversed_output)

    # Naming outputs for loss targeting
    classifier_output_named = layers.Lambda(lambda x: x, name="classifier_output")(classifier_output)
    adversary_output_named = layers.Lambda(lambda x: x, name="mass_output")(adversary_output)

    return keras.Model(inputs=inputs, outputs=[classifier_output_named, adversary_output_named], name="adversarial_model")

