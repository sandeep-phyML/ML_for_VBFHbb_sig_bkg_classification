// define all the variables files here 
#pragma once
#include <iostream>
#include <vector>
#include "TString.h"

// feature names
std::vector<std::string> features = {"T_mqq","T_dETAqq","T_dPHIqq", "T_btgb1","T_btgb2","T_qglq1",
 "T_qglq2", "T_NJ_30", "T_ptAll", "T_pzAll", "T_E_rest_30", 
 "T_HTT_rest_30", "T_phiA_bb_qq","T_alphaqq", "T_dR_subleadqH"};

// weight variables 
TString sig_weight = "T_weight*LUMI*T_HLTweight*T_PUweight*T_btag_weight_central";
TString bkg_weight = "T_weight";
TString modelName = "BDTG";

// branches and trees
TString tree_name = "tree";
TString biclass_branch = "BDTG_BiClass";


// file paths and names for training sample 
TString train_folder_path = "/Users/sandeeppradhan/Desktop/VBF_Analysis_Folder/2022_pre_EE_Ntuples/DNN_Ntuples_Train/";

// file names for the biclass classification 
TString sig_file = train_folder_path + "tree_VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8_2022.root";
TString bkg_file = train_folder_path + "tree_JetMET_2022_5perc.root";


// file paths and names for prediction sample



// output file names and path for results 


TString output_folder_path = "/Users/sandeeppradhan/Desktop/VBF_Analysis_Folder/DNN_RESULTS/MClass_DNN_VBF/output_plot_models/TMVA_Output/";
TString result_even_path = output_folder_path + "TMVA_result_even.root";
TString result_odd_path = output_folder_path + "TMVA_result_odd.root";
TString weights_even_path = output_folder_path + "weights_even.xml";
TString weights_odd_path = output_folder_path + "weights_odd.xml";
TString class_odd_path = output_folder_path + "class_odd.C";
TString class_even_path = output_folder_path + "class_even.C";
