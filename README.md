This repository contains , three scripts 
1) config.yml :, it contins all the files path , tree name , branch , features  , numbers etc used for the DNN training etc
2) utils.py : this is library where all necessary functions and classes are defined , there three types of a) Plot , PrepareDataset , DNNModel , BasicMethods
   a) Plot contanis a collections of function will be called for Tprofiles ( corelation check) , distribution of the dnn score , roc curve
   b) PrepareDataset will , read the input root files , process (take care Nan etc ) and normalise , calculate weights for training ,it convert it np.arrays or pandas dataframe etc finally get the dataready for train and prdiction purpose .
   c) DNNModel : has a collection of methods for DNN biclass and mclass model , initialisation , compilation , training, saving models , checking model perfermance like over fitting etc  and predicting
   d) BasicMethods : this class will have all the other functions which are not in a fit to the other classes purpose
Finally ,
3) main.py , this is the main scipt that we will run , first read few command line arguments like , dnn or bdt training , biclass or mclass training , only train or train with prediction or only plotting validation etc

for only train DNN binary model run ,

python main.py --train --biclass

for mclass only training and saving models 

python main.py --train 
