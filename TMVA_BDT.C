#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"
#include "TCut.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include <iostream>
#include <vector>
#include "Common.h"
#include "tmva_library.h"
using namespace TMVA;

void TMVA_BDT(int nclasses = 2) {
    // --- Init TMVA
    TMVA::Tools::Instance();
    // --- Input signal and background
    TFile *fsig = TFile::Open(sig_file);
    TFile *fbkg = TFile::Open(bkg_file);
    TTree *tsig_raw = (TTree*)fsig->Get(tree_name);
    TTree *tbkg_raw = (TTree*)fbkg->Get(tree_name);

    // Clean: remove NaN/Inf in T_alphaqq
    TString cleanCut = "!isnan(T_alphaqq) && !isinf(T_alphaqq)";
    TTree *tsig = tsig_raw->CopyTree(cleanCut);
    TTree *tbkg = tbkg_raw->CopyTree(cleanCut);
    // --- Cuts for training/testing split
    TCut cut_even("T_event % 2 == 0");
    TCut cut_odd ("T_event % 2 == 1");

    // --- Train even/odd models
    TrainModel(tsig, tbkg, cut_even,result_even_path, "Even", weights_even_path,class_even_path);
    TrainModel(tsig, tbkg, cut_odd,result_odd_path ,"Odd", weights_odd_path,class_odd_path);

    std::cout << "==> Training finished, weights saved as  " <<weights_even_path << " and "<<  weights_odd_path << std::endl;
    predict_bdt_score(pred_file);

};


