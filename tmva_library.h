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

auto addVars = [](TMVA::Reader* reader){
    for (auto &v : vars) {
        reader->AddVariable(v.first.c_str(), v.second);
    }
};

auto predict_bdt_score = [](TString pred_path) {
    TFile *fpred = TFile::Open(pred_path, "UPDATE");
    TTree *tpred = (TTree*)fpred->Get(tree_name);

    TMVA::Reader *readerEven = new TMVA::Reader("!Color:!Silent");
    TMVA::Reader *readerOdd  = new TMVA::Reader("!Color:!Silent");

    for (auto &v : vars) {
        tpred->SetBranchAddress(v.first.c_str(), v.second);
    }

    addVars(readerEven);
    addVars(readerOdd);

    readerEven->BookMVA(modelName, weights_even_path);
    readerOdd ->BookMVA(modelName, weights_odd_path);

    Float_t BDT_Score;
    tpred->Branch("BDTG_BiClass", &BDT_Score, "BDTG_BiClass/F");

    Long64_t nentries = tpred->GetEntries();
    for (Long64_t i=0; i<nentries; i++) {
        tpred->GetEntry(i);

        if ((int)tpred->GetLeaf("T_event")->GetValue() % 2 == 0)
            BDT_Score = readerOdd->EvaluateMVA(modelName);
        else
            BDT_Score = readerEven->EvaluateMVA(modelName);


        tpred->Fill();
    }

    fpred->Write("", TObject::kOverwrite);
    fpred->Close();

    std::cout << "==> Applied BDT scores stored in " << pred_path << std::endl;
};


auto TrainModel = [](TTree *sig, TTree *bkg, TCut cut, TString out_result_path ,TString model_fold , TString weight_path , TString class_path) {
        TFile *outFile = TFile::Open(out_result_path, "RECREATE");

        TMVA::Factory *factory = new TMVA::Factory("MyJob", outFile,
            "!V:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification");

        TMVA::DataLoader *loader = new TMVA::DataLoader("dataset_" + model_fold);

        // --- Define training variables
        for (const auto& feature : features) {
            loader->AddVariable(feature, 'F');
        }

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

