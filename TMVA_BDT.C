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
    if (nclasses == 2) {
        TrainBiclassBDTG();
        predict_bdt_score(pred_file, false);
    }
    else{
        //TrainMClassBDTG();
        predict_bdt_score(pred_file, true);
    }
};


