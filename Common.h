// define all the variables files here 
#pragma once
#include <iostream>
#include <vector>
#include "TString.h"

// feature names and variables ---------------------------------------
Float_t T_mqq, T_dETAqq, T_dPHIqq, T_btgb1, T_btgb2;
Float_t T_qglq1, T_qglq2, T_NJ_30, T_ptAll, T_pzAll;
Float_t T_E_rest_30, T_HTT_rest_30, T_phiA_bb_qq, T_alphaqq, T_dR_subleadqH;
std::vector<std::pair<std::string, Float_t*>> vars = {
    {"T_mqq", &T_mqq},
    {"T_dETAqq", &T_dETAqq},
    {"T_dPHIqq", &T_dPHIqq},
    {"T_btgb1", &T_btgb1},
    {"T_btgb2", &T_btgb2},
    {"T_qglq1", &T_qglq1},
    {"T_qglq2", &T_qglq2},
    {"T_NJ_30", &T_NJ_30},
    {"T_ptAll", &T_ptAll},
    {"T_pzAll", &T_pzAll},
    {"T_E_rest_30", &T_E_rest_30},
    {"T_HTT_rest_30", &T_HTT_rest_30},
    {"T_phiA_bb_qq", &T_phiA_bb_qq},
    {"T_alphaqq", &T_alphaqq},
    {"T_dR_subleadqH", &T_dR_subleadqH}
};
// ----------------------------------------------------------------------

// weights for training -------------------------------------------------
TString mc_weight = "T_weight*LUMI*T_HLTweight*T_PUweight*T_btag_weight_central";
TString data_weight = "T_weight";
std::vector<double> classWeights = {1.0, 1.0, 1.0, 1.0, 1.0}; // if you assymetric train data class , change here 
TString sig_weight = mc_weight;
TString bkg_weight = data_weight;
std::vector<TString> weightBranches = {
        data_weight, mc_weight, mc_weight, mc_weight, mc_weight
    };
// ----------------------------------------------------------------------

// file paths for the training bDTG -------------------------------------
TString train_folder_path = "/Users/sandeeppradhan/Desktop/VBF_Analysis_Folder/2022_pre_EE_Ntuples/DNN_Ntuples_Train/";
TString sig_file = train_folder_path + "tree_VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8_2022.root";
TString bkg_file = train_folder_path + "tree_JetMET_2022_5perc.root";
std::vector<TString> mclass_train_paths = {
    train_folder_path + "tree_JetMET_2022_5perc.root",
    train_folder_path + "tree_VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8_2022.root",
    train_folder_path + "tree_GluGluHto2B_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8_2022.root",
    train_folder_path + "tree_TTto4Q_TuneCP5_13p6TeV_powheg-pythia8_2022.root",
    train_folder_path + "tree_Zto2Q-4Jets_HT_TuneCP5_13p6TeV_madgraphMLM-pythia8_2022.root"
};

// ----------------------------------------------------------------------

// files path for training output / results -----------------------------
TString output_folder_path = "/Users/sandeeppradhan/Desktop/VBF_Analysis_Folder/DNN_RESULTS/MClass_DNN_VBF/output_plot_models/TMVA_Output/";
TString mclass_result_even_path = output_folder_path + "mclass_TMVA_result_even.root";
TString mclass_result_odd_path = output_folder_path + "mclass_TMVA_result_odd.root";
TString mclass_weights_even_path = output_folder_path + "mclass_weights_even.xml";
TString mclass_weights_odd_path = output_folder_path + "mclass_weights_odd.xml";
TString mclass_odd_path = output_folder_path + "mclass_odd.C";
TString mclass_even_path = output_folder_path + "mclass_even.C";
// biclass training output / results  ----------------------------------------
TString biclass_result_even_path = output_folder_path + "biclass_TMVA_result_even.root";
TString biclass_result_odd_path = output_folder_path + "biclass_TMVA_result_odd.root";
TString biclass_weights_even_path = output_folder_path + "biclass_weights_even.xml";
TString biclass_weights_odd_path = output_folder_path + "biclass_weights_odd.xml";
TString biclass_odd_path = output_folder_path + "bclass_odd.C";
TString biclass_even_path = output_folder_path + "bclass_even.C";
// ----------------------------------------------------------------------

// files for prediction -----------------------------------------
TString pred_file = train_folder_path + "tree_VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8_2022.root";
//---------------------------------------------------------------

// output branch names -------------------------------------------------
Float_t BDTG_QCD, BDTG_VBF, BDTG_GGH, BDTG_TT, BDTG_Z2Q;
Float_t BDTG_BiClass; // for biclass classification
std::vector<std::tuple<std::string, Float_t*, std::string>> mclass_pred_vars = {
    {"BDTG_QCD", &BDTG_QCD ,"BDTG_QCD/F"},
    {"BDTG_VBF", &BDTG_VBF ,"BDTG_VBF/F"},
    {"BDTG_GGH", &BDTG_GGH ,"BDTG_GGH/F"},
    {"BDTG_TT",  &BDTG_TT  ,"BDTG_TT/F"},
    {"BDTG_Z2Q", &BDTG_Z2Q ,"BDTG_Z2Q/F"}
};

std::tuple<std::string, Float_t*, std::string> biclass_pred_vars =
    {"BDTG_BiClass", &BDTG_BiClass, "BDTG_BiClass/F"};

// ----------------------------------------------------------------------

// common variables -----------------------------------------------------
TString modelName = "BDTG";
TString tree_name = "tree";