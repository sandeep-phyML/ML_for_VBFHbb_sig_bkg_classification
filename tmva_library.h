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


auto TrainBiclassModel_ = [](TTree *sig, TTree *bkg, TCut cut, TString out_result_path ,TString model_fold , TString weight_path , TString class_path) {
        TFile *outFile = TFile::Open(out_result_path, "RECREATE");
        TMVA::Factory *factory = new TMVA::Factory("MyJob", outFile,
            "!V:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification");
        TMVA::DataLoader *loader = new TMVA::DataLoader("dataset_" + model_fold);
        // --- Define training variables
        for (const auto& feature : vars) {
            loader->AddVariable(feature.first.c_str(), 'F');
        }
        loader->AddSignalTree(sig, 1.0);
        loader->AddBackgroundTree(bkg, 1.0);
        loader->SetSignalWeightExpression(sig_weight);
        loader->SetBackgroundWeightExpression(bkg_weight);
        loader->PrepareTrainingAndTestTree(cut, cut,
            "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V");
        factory->BookMethod(loader, TMVA::Types::kBDT, modelName,
            "!H:!V:NTrees=850:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=15");
        // --- Train, test and evaluate the model
        factory->TrainAllMethods();
        factory->TestAllMethods();
        factory->EvaluateAllMethods();
        // --- Save the output
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

auto TrainBiclassBDTG = [](){
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
    TrainBiclassModel_(tsig, tbkg, cut_even,biclass_result_even_path, "Even", biclass_weights_even_path,biclass_even_path);
    TrainBiclassModel_(tsig, tbkg, cut_odd,biclass_result_odd_path ,"Odd", biclass_weights_odd_path,biclass_odd_path);
    std::cout << "==> Training finished, weights saved as  " <<biclass_weights_even_path << " and "<<  biclass_weights_odd_path << std::endl;

};
// ================= MULTICLASS TRAINING ===================
auto TrainModelMClass = [](std::vector<TTree*> classTrees, TCut cut,TString out_result_path, TString model_fold,
                            TString weight_path,TString class_path)                                                   
{
    TFile *outFile = TFile::Open(out_result_path, "RECREATE");
    TMVA::Factory *factory = new TMVA::Factory("MyJob", outFile,
        "!V:Color:DrawProgressBar:Transformations=I:AnalysisType=Multiclass");

    TMVA::DataLoader *loader = new TMVA::DataLoader("dataset_" + model_fold);
    // --- Define training variables
    for (const auto& feature : vars) {
        loader->AddVariable(feature.first.c_str(), 'F');
    }
    // --- Add all classes with their trees and weight expressions
    for (size_t i = 0; i < classTrees.size(); ++i) {
        loader->AddTree(classTrees[i], std::get<0>(mclass_pred_vars[i]), classWeights[i]);
        loader->SetWeightExpression(weightBranches[i], std::get<0>(mclass_pred_vars[i]));
    }
    loader->PrepareTrainingAndTestTree(cut, cut,
        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V");
    factory->BookMethod(loader, TMVA::Types::kBDT, modelName,
        "!H:!V:NTrees=850:BoostType=Grad:Shrinkage=0.10:"
        "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=15");
    // --- Train, test and evaluate the model
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();
    // --- Save the output
    outFile->Close();
    delete factory;
    delete loader;
    // --- Rename output weight files
    TString src  = "dataset_" + model_fold + "/weights/MyJob_BDT.weights.xml";
    gSystem->Rename(src, weight_path);
    TString src_class = "dataset_" + model_fold + "/weights/MyJob_BDT.class.C";
    gSystem->Rename(src_class, class_path);
};

auto TrainMClassBDTG = [](){
    std::vector<TTree*> trees;
    // ---- Open files for 5 classes ----
    for (auto &train_class_path : mclass_train_paths) {
        std::cout << "Opening file: " << train_class_path << std::endl;
        TFile *f = TFile::Open(train_class_path);
        if (!f || f->IsZombie()) {
            std::cerr << "Error opening file: " << train_class_path << std::endl;
            return;
        }
        TTree *t_raw = (TTree*)f->Get(tree_name);
        TString cleanCut = "!isnan(T_alphaqq) && !isinf(T_alphaqq)";
        TTree *t = t_raw->CopyTree(cleanCut);
        t->SetDirectory(0);
        f->Close();
        trees.push_back(t);
    }
    TCut cut_even("T_event % 2 == 0");
    TCut cut_odd ("T_event % 2 == 1");
    // --- Train models for even and odd events
    TrainModelMClass(trees,  cut_even, mclass_result_even_path, "Even",
                     mclass_weights_even_path, mclass_even_path); 
    TrainModelMClass(trees,cut_odd,mclass_result_odd_path, "Odd",
                     mclass_weights_odd_path, mclass_odd_path);
    // --- Rename the class files 
    std::cout << "==> Multi-class training finished. Weights saved as "
              << mclass_weights_even_path << " and "
              << mclass_weights_odd_path << std::endl;
};

auto predict_bdt_score = [](TString pred_path , bool isMulticlass = false) {
    // Open the prediction file and get the tree
    TFile *fpred = TFile::Open(pred_path, "UPDATE");
    TTree *tpred = (TTree*)fpred->Get(tree_name);

    TMVA::Reader *readerEven = new TMVA::Reader("!Color:!Silent");
    TMVA::Reader *readerOdd  = new TMVA::Reader("!Color:!Silent");
    for (auto &v : vars) {
        tpred->SetBranchAddress(v.first.c_str(), v.second);
    }
    addVars(readerEven);
    addVars(readerOdd);

    if (isMulticlass) {
        readerEven->BookMVA(modelName, mclass_weights_even_path);
        readerOdd ->BookMVA(modelName, mclass_weights_odd_path);
        for (const auto& var : mclass_pred_vars) {
            tpred->Branch(std::get<0>(var).c_str(), std::get<1>(var), std::get<2>(var).c_str());
        }
    }
    else {
        readerEven->BookMVA(modelName, biclass_weights_even_path);
        readerOdd ->BookMVA(modelName, biclass_weights_odd_path);
        tpred->Branch(std::get<0>(biclass_pred_vars).c_str(),
                    std::get<1>(biclass_pred_vars),
                    std::get<2>(biclass_pred_vars).c_str());
    }

    Long64_t nentries = tpred->GetEntries();
    for (Long64_t i=0; i<nentries; i++) {
        tpred->GetEntry(i);
        
        if (isMulticlass) {
            std::vector<Float_t> probs;
            if ((int)tpred->GetLeaf("T_event")->GetValue() % 2 == 0)
                probs = readerOdd->EvaluateMulticlass(modelName);
            else
                probs = readerEven->EvaluateMulticlass(modelName);

            for (size_t j = 0; j < mclass_pred_vars.size(); ++j) {
                *std::get<1>(mclass_pred_vars[j]) = probs[j];  // ✅ store prob in variable
            }
        } else {
            if ((int)tpred->GetLeaf("T_event")->GetValue() % 2 == 0) 
                *std::get<1>(biclass_pred_vars) = readerOdd->EvaluateMVA(modelName);
            else 
                *std::get<1>(biclass_pred_vars) = readerEven->EvaluateMVA(modelName);
        }
         tpred->Fill();
    }

       
    
    fpred->Write("", TObject::kOverwrite);
    fpred->Close();
    std::cout << "==> Applied BDT scores stored in " << pred_path << "is multiclass " << isMulticlass << std::endl;
};
