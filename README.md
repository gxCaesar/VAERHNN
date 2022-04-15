# VAERHNN
VAERHNN is an AI-based drug repurposing protocol to investigate potent leads against CRC. VAERHNN can comprehensively integrate the information of the target and its inhibitors or agonists for drug repurposing. We built a voting-averaged ensemble regression (VAER) model based on ensemble learning algorithm for molecular activity prediction. Moreover, we also assemble a hybrid neural network (HNN) consisting of multiple neural networks to predict the drug-target affinity (DTA). Our proposed protocol can be used to identify potentially potent leads for repurposing against CRC or other diseases.

The overall flow chart is as follows:

<div align="center">
<p><img src="https://user-images.githubusercontent.com/57307280/163565613-6fdb9272-613f-470f-b3cf-bd2b914d2445.jpg" width="600"></p>
</div>

## Reproducibility
The analysis in the paper can be completely reproduced. In the directory, we have set up three folders: VAER and HNN. You can reproduce or train your own data in the folder. In addition, you can also use it in a targeted manner. Folders to perform activity predictions and DTA predictions.
You may need the dataset indicated below to reproduce all results correctly.
Also, if you want to leverage the VAERHNN protocol for drug repurposing for a disease, or if you want to retrain the model, check out the appropriate sections in this README.

## Installation & Dependencies
The code of VAER and HNN is written in Python 3, which is mainly tested on Python 3.7 and Linux OS. It's faster to train on a GPU, but you can also work on a standard computer.

VAER has the following dependencies:
* feature selector (https://github.com/WillKoehrsen/feature-selector)
* DeepPurpose (https://github.com/kexinhuang12345/DeepPurpose)
* torch
* Numpy
* Pandas
* xgboost
* catboost
* Matplotlib
* Seaborn
* Sklearn
* Scipy
* pandas

Installing all of the packages should take roughly several minutes.
To install the feature selector package, you have to clone and install the feature selector package using:
```
git clone https://github.com/WillKoehrsen/feature-selector.git
cd feature-selector
python setup.py install
```
To install the DeepPurpose package, you have to clone and install the DeepPurpose package using:
```
git clone https://github.com/kexinhuang12345/DeepPurpose.git
cd DeepPurpose
python setup.py install
```
The list of required environments is in the requirements.txt file. You can also download and unzip the feature selector and DeepPurpose packages first, and then install the required environment using pip or conda as follows:
```
Pip:
conda create -n vaerhnn python=3.7
conda activate vaerhnn
pip install -r requirements.txt
conda:
conda env create -f environment.yaml
conda activate vaerhnn
```


## Dataset:
+ VAER: molecular activity prediction.
	+ GFA_features.csv: pIC50 and GFA features of inhibitors against the target.
	+ inhibitor.csv: pIC50 and features of inhibitors against the target after feature selection.
	+ candidate.csv: features of the candidates for drug repurposing.

+ HNN: DTA prediction.
	+ chembl.txt: SMILES and DTA (pIC50) of inhibitors, as well as the amino acid sequence of the target.
	+ candidate.txt: SMILES of the candidates for drug repurposing and the amino acid sequence of the target.

## Training VAER with Your Own Data
First of all, please treat your data as our data. Then we perform dimensionality reduction.
```
cd VAER
python Feature_selection.py
```
The selected features and pIC50 then constitute a dataset, which is used as the training for VAER. Please perform the training of VAER with the following command:
```
python VAER.py
```
Then you can get the training performance of the ensemble learning and the pIC50 predictions of the candidates by VAER. For comparison with the baseline ML model, you should perform the following:
```
python Comparison_boosting.py
```

## Training HNN with Your Own Data
Please process your inhibitor or agonist, target amino acid sequence and candidate data for training as our data. Then, please perform the training of the hybrid neural network consisting of 72 encoders for drugs and targets, as follows:
```
cd HNN
python RunALL.py
```
This process can take anywhere from an hour to several hours (depending on your data). At the end of training, you will have the C-index and MSE performance for 72 model combinations, and DTA predictions for the candidates. Then organize the data of C-index and MSE, and draw a heat map of performance, which can easily see the distribution of performance and select some top-performing model combinations, as follows:
```
python Plot_CI_MSE.py
```
Please select a combination of models with both C-index and MSE in red in the heatmap. Then, the calculation of voting average is performed with CI as the weight factor. The prediction of DTA by HNN is obtained as follows:
```
python Voting_averaged.py
```

##  Contact
Please contact chengx48@mail2.sysu.edu.cn for help or submit an issue.














