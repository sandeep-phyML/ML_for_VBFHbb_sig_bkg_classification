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

void TMVA_BDT(int nclasses = 2,bool train = true , bool predict = true) {
    // --- Init TMVA
    TMVA::Tools::Instance();
    // --- Input signal and background
    if (nclasses == 2) {
        if (train) TrainBiclassBDTG();
        if (!predict) return; // if not predicting, exit early
        for (const auto& pred_file : pred_files) {
            predict_bdt_score(pred_file, false);
            // break;
        }
    }
    else{
        if (train ) TrainMClassBDTG();
        if (!predict) return; // if not predicting, exit early
        for (const auto& pred_file : pred_files) {
            predict_bdt_score(pred_file, true);
            // break;
        }
    
        
    }
};


