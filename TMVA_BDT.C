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
using namespace TMVA;


auto TrainModel = [](TTree *sig, TTree *bkg, TCut cut, TString out_result_path ,TString model_fold , TString weight_path , TString class_path) {
        TFile *outFile = TFile::Open(out_result_path, "RECREATE");

        TMVA::Factory *factory = new TMVA::Factory("MyJob", outFile,
            "!V:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification");

        TMVA::DataLoader *loader = new TMVA::DataLoader("dataset_" + model_fold);

        // --- Define training variables
        for (const auto& feature : features) {
            loader->AddVariable(feature, 'F');
        }
        // loader->AddVariable("T_mqq", 'F');
        // loader->AddVariable("T_dETAqq", 'F');
        // loader->AddVariable("T_dPHIqq", 'F');
        // loader->AddVariable("T_btgb1", 'F');
        // loader->AddVariable("T_btgb2", 'F');
        // loader->AddVariable("T_qglq1", 'F');
        // loader->AddVariable("T_qglq2", 'F');
        // loader->AddVariable("T_NJ_30", 'F');
        // loader->AddVariable("T_ptAll", 'F');
        // loader->AddVariable("T_pzAll", 'F');
        // loader->AddVariable("T_E_rest_30", 'F');
        // loader->AddVariable("T_HTT_rest_30", 'F');
        // loader->AddVariable("T_phiA_bb_qq", 'F');
        // loader->AddVariable("T_alphaqq", 'F');
        // loader->AddVariable("T_dR_subleadqH", 'F');

        loader->AddSignalTree(sig, 1.0);
        loader->AddBackgroundTree(bkg, 1.0);

        loader->SetSignalWeightExpression(sig_weight);
        loader->SetBackgroundWeightExpression(bkg_weight);

        loader->PrepareTrainingAndTestTree(cut, cut,
            "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V");

        factory->BookMethod(loader, TMVA::Types::kBDT, modelName,
            "!H:!V:NTrees=850:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=15");

        factory->TrainAllMethods();
        factory->TestAllMethods();
        factory->EvaluateAllMethods();

        outFile->Close();
        delete factory;
        delete loader;

        // rename weight file
        TString src  = "dataset_" + model_fold + "/weights/MyJob_BDTG.weights.xml";
        TString dest = weight_path;   // directly use the user-provided path
        gSystem->Rename(src, dest);
        // rename the class path 
        TString src_class  = "dataset_" + model_fold + "/weights/MyJob_BDTG.class.C";
        TString dest_class = class_path;
        gSystem->Rename(src_class, dest_class);

    };

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

    // --- Helper lambda: train a model
    
    // --- Train even/odd models
    TrainModel(tsig, tbkg, cut_even,result_even_path, "Even", weights_even_path,class_even_path);
    TrainModel(tsig, tbkg, cut_odd,result_odd_path ,"Odd", weights_odd_path,class_odd_path);

    std::cout << "==> Training finished, weights saved as weights_even.xml and weights_odd.xml" << std::endl;
    /*
    // --- Apply model to prediction file
    TFile *fpred = TFile::Open("tree_VBFHto2B_M-125_dipoleRecoilOn_TuneCP5_13p6TeV_powheg-pythia8_2022.root", "UPDATE"); // just reuse signal.root for demo
    TTree *tpred = (TTree*)fpred->Get("tree");

    TMVA::Reader *readerEven = new TMVA::Reader("!Color:!Silent");
    TMVA::Reader *readerOdd  = new TMVA::Reader("!Color:!Silent");

    // --- Variable setup
    Float_t T_mqq, T_dETAqq, T_dPHIqq, T_btgb1, T_btgb2;
    Float_t T_qglq1, T_qglq2, T_NJ_30, T_ptAll, T_pzAll;
    Float_t T_E_rest_30, T_HTT_rest_30, T_phiA_bb_qq, T_alphaqq, T_dR_subleadqH;

    tpred->SetBranchAddress("T_mqq", &T_mqq);
    tpred->SetBranchAddress("T_dETAqq", &T_dETAqq);
    tpred->SetBranchAddress("T_dPHIqq", &T_dPHIqq);
    tpred->SetBranchAddress("T_btgb1", &T_btgb1);
    tpred->SetBranchAddress("T_btgb2", &T_btgb2);
    tpred->SetBranchAddress("T_qglq1", &T_qglq1);
    tpred->SetBranchAddress("T_qglq2", &T_qglq2);
    tpred->SetBranchAddress("T_NJ_30", &T_NJ_30);
    tpred->SetBranchAddress("T_ptAll", &T_ptAll);
    tpred->SetBranchAddress("T_pzAll", &T_pzAll);
    tpred->SetBranchAddress("T_E_rest_30", &T_E_rest_30);
    tpred->SetBranchAddress("T_HTT_rest_30", &T_HTT_rest_30);
    tpred->SetBranchAddress("T_phiA_bb_qq", &T_phiA_bb_qq);
    tpred->SetBranchAddress("T_alphaqq", &T_alphaqq);
    tpred->SetBranchAddress("T_dR_subleadqH", &T_dR_subleadqH);

    // add vars to reader
    auto addVars = [&](TMVA::Reader* reader){
        reader->AddVariable("T_mqq", &T_mqq);
        reader->AddVariable("T_dETAqq", &T_dETAqq);
        reader->AddVariable("T_dPHIqq", &T_dPHIqq);
        reader->AddVariable("T_btgb1", &T_btgb1);
        reader->AddVariable("T_btgb2", &T_btgb2);
        reader->AddVariable("T_qglq1", &T_qglq1);
        reader->AddVariable("T_qglq2", &T_qglq2);
        reader->AddVariable("T_NJ_30", &T_NJ_30);
        reader->AddVariable("T_ptAll", &T_ptAll);
        reader->AddVariable("T_pzAll", &T_pzAll);
        reader->AddVariable("T_E_rest_30", &T_E_rest_30);
        reader->AddVariable("T_HTT_rest_30", &T_HTT_rest_30);
        reader->AddVariable("T_phiA_bb_qq", &T_phiA_bb_qq);
        reader->AddVariable("T_alphaqq", &T_alphaqq);
        reader->AddVariable("T_dR_subleadqH", &T_dR_subleadqH);
    };

    addVars(readerEven);
    addVars(readerOdd);

    readerEven->BookMVA("BDTG", "dataset_ModelEven/weights/MyJob_BDTG.weights.xml");
    readerOdd ->BookMVA("BDTG", "dataset_ModelOdd/weights/MyJob_BDTG.weights.xml");

    Float_t BDT_Score;
    TBranch *b_bdt = tpred->Branch("BDT_Score", &BDT_Score, "BDT_Score/F");

    Long64_t nentries = tpred->GetEntries();
    for (Long64_t i=0; i<nentries; i++) {
        tpred->GetEntry(i);
        if ( (int)tpred->GetLeaf("T_event")->GetValue() % 2 == 0 )
            BDT_Score = readerOdd->EvaluateMVA("BDTG");
        else
            BDT_Score = readerEven->EvaluateMVA("BDTG");
        b_bdt->Fill();
    }

    fpred->Write("", TObject::kOverwrite);
    fpred->Close();

    std::cout << "==> Applied BDT scores stored in signal.root" << std::endl; 
*/
}

