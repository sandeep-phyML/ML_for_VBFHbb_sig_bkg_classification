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
    # print(out_data)
    with open(ofile_path, "a") as f:
        f.write(out_data + "\n")

class PrepareDataset():
    def __init__(self , config: Dict[str, any], output_log_path: str, is_biclass: bool = True):
        self.config = config
        self.train_feature = self.config['train_features']
        self.weight_features = self.config["weight_features"]
        self.nsample = self.config["nsamples"]
        self.output_file_path = output_log_path
        self.total_weight_branch = "weight_"
        self.norm_branch = "norm_weight"
        self.label_branch_name = "label"
        self.event_branch = self.config['event_branch']
        self.tree_name = self.config['tree_name']
        self.is_resample_all = self.config['is_resampling']
        self.normalise_mass = self.config['mass_norm']
        if is_biclass:
            self.train_folder_path = self.config["train_biclass_files_labels"]["folder_path"]
            self.file_label_list = self.config["train_biclass_files_labels"]["file_names_labels"]
        else:
            self.train_folder_path = self.config["train_mclass_files_labels"]["folder_path"]
            self.file_label_list = self.config["train_mclass_files_labels"]["file_names_labels"]
        self.pred_paths = [os.path.join(self.config["prediction_files"]["folder_path"], file_name) for file_name in self.config["prediction_files"]["file_names"]]
    def convert_tree_to_pd(self, input_file: str) -> pd.DataFrame:
        file_data = uproot.open(f"{input_file}:{self.tree_name}")
        pd_data = file_data.arrays(file_data.keys(), library="pd")
        save_output(f"Converted {input_file} to pandas DataFrame with shape {pd_data.shape}",ofile_path = self.output_file_path)
        return pd_data
    
    def resample_events(self, pd_data: pd.DataFrame) -> pd.DataFrame:
        replace = self.nsample > len(pd_data)
        resampled_data = pd_data.sample(n=self.nsample, replace=replace, random_state=42).reset_index(drop=True)
        save_output(f"Resampled data to {self.nsample} events with replacement={replace}",ofile_path = self.output_file_path)
        return resampled_data
    def add_label_branch(self, pd_data: pd.DataFrame, label: int) -> pd.DataFrame:
        pd_data[self.label_branch_name] = np.full(len(pd_data), label)
        save_output(f"Added label branch with value {label}",ofile_path = self.output_file_path)
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
        
    def replace_nan_inf_with_zero(self, pd_data: pd.DataFrame) -> pd.DataFrame:
        pd_data = pd_data.replace([np.nan, np.inf, -np.inf], 0)
        save_output("Replaced NaN and Inf values with zero",ofile_path = self.output_file_path)
        return pd_data
        
    def get_np_feaweilabel_odd_even_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_data = []
        for item in self.file_label_list:
            file_path = os.path.join(self.train_folder_path, item["file_name"])
            label_value = item["label"]
            pd_data_ = self.convert_tree_to_pd(file_path)
            pd_data_ = self.add_event_filter(pd_data_, self.config["event_branches"] , self.config["cuts"])
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
            
    def prepare_pred_data(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        all_data = {}
        save_output(f"Processing files for predictions : {self.pred_paths}", ofile_path=self.output_file_path)
        for file in self.pred_paths:
            all_data[file] = {"odd_data": {}, "even_data": {}}
            pd_data_ = self.convert_tree_to_pd(file)
            print(f"Processing file: {file} with shape {pd_data_.shape}")
            odd_data_ = pd_data_[pd_data_[self.event_branch] % 2 == 1].reset_index(drop=True)
            even_data_ = pd_data_[pd_data_[self.event_branch] % 2 == 0].reset_index(drop=True)
            all_data[file]["odd_data"]["full_data_"] = odd_data_
            all_data[file]["even_data"]["full_data_"] = even_data_
            all_data[file]["odd_data"]["features"] = self.replace_nan_inf_with_zero(odd_data_.filter(self.train_feature)).to_numpy()
            all_data[file]["even_data"]["features"] = self.replace_nan_inf_with_zero(even_data_.filter(self.train_feature)).to_numpy()
            save_output(f"Processed file {file} - Odd features shape: {all_data[file]['odd_data']['features'].shape}, Even features shape: {all_data[file]['even_data']['features'].shape}", ofile_path=self.output_file_path)
            print(f"Processed file {file} - Odd features shape: {all_data[file]['odd_data']['features'].shape}, Even features shape: {all_data[file]['even_data']['features'].shape}")
        return all_data
    def create_five_perc_data(self,infile_path):
        pd_data = self.convert_tree_to_pd(infile_path)
        selected_data = pd_data.sample(frac=0.05, random_state=42)
        with uproot.recreate(infile_path.replace(".root","_5perc.root")) as f:
            f["tree"] = selected_data
        save_output(f"Created 5% sample data from {infile_path} , selected event {len(selected_data)} from {len(pd_data)} ", ofile_path=self.output_file_path)
        return True 
    def prepare_data_for_bdt_training(self, input_file: str, output_file: str,nsample:int,filter_branches,cuts ,weight_branch:str = "train_weight"  ) -> bool:
        print(f"Preparing data for BDT training from {input_file} to {output_file} with nsample {nsample}, filter branches {filter_branches}, cuts {cuts}, weight branch {weight_branch}")
        self.norm_branch = weight_branch
        self.nsample = nsample
        pd_data_ = self.convert_tree_to_pd(input_file)
        pd_data_ = self.add_event_filter(pd_data_,filter_branches, cuts)
        pd_data_ = self.resample_events(pd_data_)
        pd_data_= self.get_weight_branch_(pd_data_)
        pd_data_ = self.normalise_the_weight(pd_data_ )
        pd_data_ = self.replace_nan_inf_with_zero(pd_data_)
        print(f"Saving processed data for bdt training  to {output_file} with shape {pd_data_.shape}")
        with uproot.recreate(output_file) as f:
            f["tree"] = pd_data_
            return True
    def filter_nan_with_zero_event_sel(self, input_file: str ,filter_branches,cuts) -> bool:
        print(f"Filtering NaN and Inf values in {input_file} with branches {filter_branches} and cuts {cuts}")
        pd_data_ = self.convert_tree_to_pd(input_file)
        pd_data_ = self.replace_nan_inf_with_zero(pd_data_)
        pd_data_ = self.add_event_filter(pd_data_,filter_branches, cuts)
        with uproot.recreate(input_file) as f:
            f["tree"] = pd_data_
            return True
    def add_event_filter(self,data: pd.DataFrame, event_branches , cuts) -> pd.DataFrame:
        # here cuts are applied and ,
        for index , branch in enumerate(event_branches):
            print(f"Applying cut on branch {branch} with value {cuts[index]}")
            data = data[data[branch]>=cuts[index]]
            print(f"Data shape after cut on {branch}: {data.shape}")
        return data
    


class Plot():
    def __init__(self , config ,log_path: str ):
        self.config = config
        self.tree_name = config["tree_name"]
        self.history_out_path = log_path
        self.train_features = config["train_features"]
        
        
    
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
        T_btag_weight_central = array.array('f', [0.])
        T_online_btag_weight = array.array('f', [0.])
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
            tree.SetBranchAddress("T_btag_weight_central", T_btag_weight_central)
            tree.SetBranchAddress("T_online_btag_weight", T_online_btag_weight)
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
                weight = T_weight[0] *  LUMI[0] * T_HLTweight[0] * T_PUweight[0] * T_btag_weight_central[0] * T_online_btag_weight[0]
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
        c1.SaveAs(save_path.replace(".png", f"{h.GetCorrelationFactor()}.png"))
        save_output(f"Saved TProfile plot to {save_path}",ofile_path = self.history_out_path)

        # Cleanup
        file.Close()
    
    def plot_var_distribution(self, branch_names:List[str],   range_=(0, 1),nbin = 25):
        out_file_name = "_".join(branch_names) + "_distribution.png"
        input_file_info = self.config["variable_distribution_files"]
        save_path = os.path.join(input_file_info["save_folder_path"], out_file_name)
        folder_path = input_file_info["input_folder_path"]
        plt.figure(figsize=(8, 6))
        for file_colour_legend in input_file_info["file_names_legend_colors"]:
            file_name, color, label = file_colour_legend["file_name"], file_colour_legend["color"], file_colour_legend["legend"]
            file_path = os.path.join(folder_path, file_name)
            try:
                df = uproot.open(f"{file_path}:tree").arrays(branch_names, library="pd")
                clean_data = df
                num_data = clean_data[branch_names[0]]
                if len(branch_names) != 1:
                    df_deno = sum([df[branch] for branch in branch_names])
                    df_deno = df_deno.replace(0, np.nan)
                else :
                    df_deno = 1.0
                save_output(f"Processing file {file_path} for branches {branch_names} with label {label}", ofile_path = self.history_out_path)
                selected_data = num_data / df_deno
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
        label_name = " / ".join(branch_names)
        plt.xlabel(label_name)
        
        plt.ylabel("Normalized Entries (Sum = 1)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title(f"Distribution of {label_name}")
        plt.tight_layout()
        plt.savefig(f"{save_path}")
        print(f"Saved variable distribution plot to {save_path}")
        save_output(f"Saved variable distribution plot to {save_path}",ofile_path = self.history_out_path)



    def plot_roc_curve(self, branch_names: List[str]):
        input_file_info = self.config["roc_curve_files"]
        save_folder = input_file_info["save_folder_path"]
        input_folder = input_file_info["input_folder_path"]
        file_entries = input_file_info["file_names_labels"]

        y_true_even, y_score_even = [], []
        y_true_odd, y_score_odd = [], []

        for entry in file_entries:
            file_name = entry["file_name"]
            label = int(entry["label"])  
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing file: {file_path} for branches {branch_names} with label {label}")

            try:
                df = uproot.open(f"{file_path}:tree").arrays(branch_names + ["T_event"], library="pd")
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                num = df[branch_names[0]]
                if len(branch_names) != 1:
                    df_deno = sum([df[branch] for branch in branch_names])
                    df_deno = df_deno.replace(0, np.nan)
                else :
                    df_deno = 1.0
                score = (num / df_deno).replace([np.inf, -np.inf], np.nan).dropna()
                df["score"] = score
                df["label"] = label

                even_mask = (df["T_event"] % 2 == 0)
                odd_mask = ~even_mask

                y_true_even.extend(df.loc[even_mask, "label"].tolist())
                y_score_even.extend(df.loc[even_mask, "score"].tolist())

                y_true_odd.extend(df.loc[odd_mask, "label"].tolist())
                y_score_odd.extend(df.loc[odd_mask, "score"].tolist())

                save_output(f"Processed file {file_name} for branches {branch_names} with label {label}", 
                            ofile_path=self.history_out_path)

            except Exception as e:
                print(f"Failed for {file_name}: {e}")
                continue

        fpr_even, tpr_even, _ = roc_curve(y_true_even, y_score_even)
        auc_even = auc(fpr_even, tpr_even)

        fpr_odd, tpr_odd, _ = roc_curve(y_true_odd, y_score_odd)
        auc_odd = auc(fpr_odd, tpr_odd)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr_even, tpr_even, label=f"Even Events (AUC = {auc_even:.2f})", color='blue')
        plt.plot(fpr_odd, tpr_odd, label=f"Odd Events (AUC = {auc_odd:.2f})", color='green')
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Even vs Odd Events) for var {'_'.join(branch_names)}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_file_name = "_".join(branch_names) + "_even_odd_roc_curve.png"
        save_path = os.path.join(save_folder, out_file_name)
        plt.savefig(save_path)
        print(f"Saved ROC curve plot to {save_path}")
        save_output(f"Saved ROC curve plot to {save_path}", ofile_path=self.history_out_path)



class DNNModel():
    def __init__(self, config: Dict[str, any],log_path: str,is_biclass:bool, is_bdt: bool = False):
        self.is_biclass = is_biclass
        self.config = config
        self.log_path = log_path
        self.input_shape = None
        self.learning_rate = None
        self.loss_function = None
        self.activation =  None
        self.nclass = None
        self.features_odd_data = None
        self.features_even_data = None
        self.labels_odd_data = None
        self.labels_even_data = None
        self.weights_odd_data = None
        self.weights_even_data = None
        self.train_odd_dataset = None
        self.train_even_dataset = None
        if self.is_biclass and is_bdt:
            self.odd_model_save_path = os.path.join(self.config['output_plot_models']["folder_path"], self.config['output_plot_models']["biclass_bdt_file_names"]["odd_model"])
            self.even_model_save_path = os.path.join(self.config['output_plot_models']["folder_path"], self.config['output_plot_models']["biclass_bdt_file_names"]["even_model"])
        elif is_biclass and not is_bdt:
            self.odd_model_save_path = os.path.join(self.config['output_plot_models']["folder_path"], self.config['output_plot_models']["biclass_dnn_file_names"]["odd_model"])
            self.even_model_save_path = os.path.join(self.config['output_plot_models']["folder_path"], self.config['output_plot_models']["biclass_dnn_file_names"]["even_model"])
        else:
            self.odd_model_save_path = os.path.join(self.config['output_plot_models']["folder_path"], self.config['output_plot_models']["mclass_dnn_file_names"]["odd_model"])
            self.even_model_save_path = os.path.join(self.config['output_plot_models']["folder_path"], self.config['output_plot_models']["mclass_dnn_file_names"]["even_model"])
        if is_biclass:
            self.train_folder_path = self.config["train_biclass_files_labels"]["folder_path"]
            self.file_label_list = self.config["train_biclass_files_labels"]["file_names_labels"]
        else:
            self.train_folder_path = self.config["train_mclass_files_labels"]["folder_path"]
            self.file_label_list = self.config["train_mclass_files_labels"]["file_names_labels"]
        self.get_inshape_nclass_activ_loss()
        self.base_classifier = self.build_classifier()
        self.odd_classifier = clone_model(self.base_classifier)
        self.even_classifier = clone_model(self.base_classifier)
        self.odd_classifier.set_weights(self.base_classifier.get_weights())
        self.even_classifier.set_weights(self.base_classifier.get_weights())
        self.optimizer_odd = keras.optimizers.AdamW(learning_rate=self.learning_rate)
        self.optimizer_even = keras.optimizers.AdamW(learning_rate=self.learning_rate)
        self.pred_data_dict = None
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
    def compile_dnn_model(self):
        self.odd_classifier.compile(optimizer= self.optimizer_odd,loss=self.loss_function,
                    metrics=[
                        'accuracy'
                    ])
        self.even_classifier.compile(optimizer = self.optimizer_even,loss=self.loss_function,
                    metrics=[
                        'accuracy'
                    ])
        print("odd_model_classifier compiled? :", self.is_compiled(self.odd_classifier))
        print("even_model_classifier compiled? :", self.is_compiled(self.even_classifier))
        self.odd_classifier.summary()  
        self.even_classifier.summary()
        save_output("Compiled odd and even classifiers", ofile_path=self.log_path )
        return True 
    def is_compiled(self, model):
        return model.optimizer is not None
    def prepare_tf_dataset(self):
        odd_dataset = tf.data.Dataset.from_tensor_slices((
            self.features_odd_data,
            self.labels_odd_data,
            self.weights_odd_data
        )).shuffle(1000).batch(self.config['batch_size'])

        even_dataset = tf.data.Dataset.from_tensor_slices((
            self.features_even_data,
            self.labels_even_data,
            self.weights_even_data
        )).batch(self.config['batch_size'])

        odd_dataset = odd_dataset.map(lambda x, y, w: ((x), y, w))
        even_dataset = even_dataset.map(lambda x, y, w: ((x), y, w))
        self.train_odd_dataset = odd_dataset
        self.train_even_dataset = even_dataset
        save_output("Prepared TensorFlow datasets for training", ofile_path=self.log_path )
        return True

    def train_dnn_model(self):
        self.prepare_tf_dataset()
        lr_scheduler = keras.callbacks.LearningRateScheduler(self.scheduler)
        early_stopping = self.early_stopping()
        history_odd = self.odd_classifier.fit(
            self.train_odd_dataset,
            validation_data=self.train_even_dataset,
            epochs=self.config['num_epochs'],
            callbacks=[lr_scheduler, early_stopping]
        )
        
        history_even = self.even_classifier.fit(
            self.train_even_dataset,
            validation_data=self.train_odd_dataset,
            epochs=self.config['num_epochs'],
            callbacks=[lr_scheduler, early_stopping]
        )
        if not self.is_biclass:
            self.analyze_overfitting(history_odd, loss_threshold=self.config['accuracy_threshold'], acc_threshold=self.config['loss_threshold'] , is_even= False,is_biclass= False, is_bdt = False)
            self.analyze_overfitting(history_even, loss_threshold=self.config['accuracy_threshold'], acc_threshold=self.config['loss_threshold'] , is_even= True,is_biclass= False, is_bdt = False)
        else :
            self.analyze_overfitting(history_odd, loss_threshold=self.config['accuracy_threshold'], acc_threshold=self.config['loss_threshold'] , is_even= False,is_biclass= True, is_bdt = False)
            self.analyze_overfitting(history_even, loss_threshold=self.config['accuracy_threshold'], acc_threshold=self.config['loss_threshold'] , is_even= True,is_biclass= True, is_bdt = False)
        save_output("Training completed for both odd and even classifiers", ofile_path=self.log_path )
    def predict_dnn_score(self):
        even_model = None
        odd_model = None
        try:
            even_model = keras.models.load_model(self.even_model_save_path) # read file path for biclass or mclass from config based on is_biclass
            odd_model = keras.models.load_model(self.odd_model_save_path)
            save_output(f"Loaded odd classifier model from {self.odd_model_save_path}", ofile_path=self.log_path )
            save_output(f"Loaded even classifier model from {self.even_model_save_path}", ofile_path=self.log_path )
        except Exception as e:
            save_output(f"Error loading models: {e}", ofile_path=self.log_path )
            return False
        if even_model  and odd_model :
            for file_path in self.pred_data_dict.keys():
                print(f"Predicting scores for file: {file_path}")
                odd_data = self.pred_data_dict[file_path]['odd_data']['full_data_']
                even_data = self.pred_data_dict[file_path]['even_data']['full_data_']
                odd_feature_data = self.pred_data_dict[file_path]['odd_data']['features']
                even_feature_data = self.pred_data_dict[file_path]['even_data']['features']
                print("Odd feature data shape:", odd_feature_data.shape , "Odd data shape:", odd_data.shape)
                print("Even feature data shape:", even_feature_data.shape , "Even data shape:", even_data.shape)
                odd_scores = even_model.predict(odd_feature_data, batch_size=256).squeeze()
                even_scores = odd_model.predict(even_feature_data, batch_size=256).squeeze()
                if self.is_biclass:
                    branch_name = self.file_label_list[0]["branch_name"]
                    odd_data[branch_name] = odd_scores
                    even_data[branch_name] = even_scores
                else:
                    for branch_index_info in self.file_label_list:
                        index = branch_index_info["label"]
                        branch_name = branch_index_info['branch_name']
                        odd_data[branch_name] = odd_scores[:,index]
                        even_data[branch_name] = even_scores[:,index]
                combined_data = pd.concat([odd_data, even_data], ignore_index=True)
                print(combined_data.dtypes)
                print(combined_data.max())
                print(combined_data.min())

                with uproot.recreate(file_path) as ofile:
                    ofile["tree"] = combined_data
                save_output(f"Saved predictions to {file_path}", ofile_path=self.log_path )
                print(f"Saved predictions to {file_path}")
        else:
            save_output("Models not loaded successfully, cannot predict", ofile_path=self.log_path )
            print("Models not loaded successfully, cannot predict")
            return False
        
    def save_model_weights(self):
        odd_model_path = self.odd_model_save_path 
        even_model_path = self.even_model_save_path 
        self.odd_classifier.save(odd_model_path)
        self.even_classifier.save(even_model_path)
        print(f"Saved odd classifier model to {odd_model_path}")
        print(f"Saved even classifier model to {even_model_path}")
        save_output(f"Saved odd classifier model to {odd_model_path}", ofile_path=self.log_path )
        save_output(f"Saved even classifier model to {even_model_path}", ofile_path=self.log_path )
    
    def early_stopping(self) -> keras.callbacks.EarlyStopping:
        return keras.callbacks.EarlyStopping(
            monitor=self.config["monitor"],
            verbose=self.config['verbose'],
            patience=self.config['patience'],
            restore_best_weights=self.config['restore_best_weights'],
            start_from_epoch=self.config['start_from_epoch']
        )
    def scheduler(self, epoch, lr, decay_rate=None, start_decay=None):
        if decay_rate is None:
            decay_rate = self.config["learning_rate"]  # ✅ now self is defined
        if start_decay is None:
            start_decay = self.config["start_decay"]
        if epoch < start_decay:
            return lr
        else:
            return lr * math.exp(-decay_rate * (epoch - start_decay))
    def get_inshape_nclass_activ_loss(self) -> bool:
        input_shape = len(self.config['train_features'])
        if  self.is_biclass:
            nclass = 1
            activation = self.config["act_output_layer_binary"]
            loss_function = self.config['loss_function_binary']
        else:
            nclass = len(self.config['train_mclass_files_labels']['file_names_labels'])
            activation = self.config['act_output_layer_mclass']
            loss_function = self.config['loss_function_mclass']
        self.input_shape = input_shape
        self.learning_rate = self.config['learning_rate']
        self.loss_function = loss_function
        self.activation =  activation
        self.nclass = nclass
        save_output(f"Input shape: {input_shape}, Number of classes: {nclass}, Activation: {activation}, Loss function: {loss_function}", ofile_path=self.log_path )
        return True
    def analyze_overfitting(self , history, loss_threshold=0.1, acc_threshold=0.1 , is_even:bool = True,is_biclass:bool = True, is_bdt:bool = False):
        if is_biclass and is_bdt and is_even:
            file_path = os.path.join(self.config['roc_curve_files']["save_folder_path"], self.config['roc_curve_files']["file_names_loss_acc_train_val"]["bdt_biclass_even_file_name"])
        elif is_biclass and is_bdt and not is_even:
            file_path = os.path.join(self.config['roc_curve_files']["save_folder_path"], self.config['roc_curve_files']["file_names_loss_acc_train_val"]["bdt_biclass_odd_file_name"])
        elif not is_biclass and not is_bdt and  is_even:
            file_path = os.path.join(self.config['roc_curve_files']["save_folder_path"], self.config['roc_curve_files']["file_names_loss_acc_train_val"]["dnn_mclass_even_file_name"])
        elif not is_biclass and not is_bdt and not is_even:
            file_path = os.path.join(self.config['roc_curve_files']["save_folder_path"], self.config['roc_curve_files']["file_names_loss_acc_train_val"]["dnn_mclass_odd_file_name"])
        elif is_biclass and not is_bdt and is_even:
            file_path = os.path.join(self.config['roc_curve_files']["save_folder_path"], self.config['roc_curve_files']["file_names_loss_acc_train_val"]["dnn_biclass_even_file_name"])
        elif is_biclass and not is_bdt and not is_even:
            file_path = os.path.join(self.config['roc_curve_files']["save_folder_path"], self.config['roc_curve_files']["file_names_loss_acc_train_val"]["dnn_biclass_odd_file_name"])
        else:
            raise ValueError("Invalid combination of is_biclass and is_bdt flags.")
        history_dict = history.history
        epochs = range(1, len(history_dict['loss']) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_dict['loss'], label='Training Loss')
        plt.plot(epochs, history_dict['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        acc_key = 'accuracy' if 'accuracy' in history_dict else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'
    
        if acc_key in history_dict and val_acc_key in history_dict:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, history_dict[acc_key], label='Training Accuracy')
            plt.plot(epochs, history_dict[val_acc_key], label='Validation Accuracy')
            plt.title('Accuracy over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
    
        plt.tight_layout()
        plt.savefig(file_path)
        print(f"Saved loss and accuracy plots to {file_path}")
        save_output(f"Saved loss and accuracy plots to {file_path}", ofile_path=self.log_path )
        # plt.show()
    
        # Check for overfitting
        final_train_loss = history_dict['loss'][-1]
        final_val_loss = history_dict['val_loss'][-1]
        loss_gap = final_val_loss - final_train_loss
    
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Loss Gap: {loss_gap:.4f}")
    
        if loss_gap > loss_threshold:
            print("⚠️ Potential overfitting detected (validation loss is significantly higher).")
        else:
            print("✅ No significant overfitting based on loss.")
    
        if acc_key in history_dict and val_acc_key in history_dict:
            final_train_acc = history_dict[acc_key][-1]
            final_val_acc = history_dict[val_acc_key][-1]
            acc_gap = final_train_acc - final_val_acc
    
            print(f"Final Training Accuracy: {final_train_acc:.4f}")
            print(f"Final Validation Accuracy: {final_val_acc:.4f}")
            print(f"Accuracy Gap: {acc_gap:.4f}")
    
            if acc_gap > acc_threshold:
                print("⚠️ Potential overfitting detected (training accuracy much higher than validation).")
            else:
                print("✅ No significant overfitting based on accuracy.")





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



