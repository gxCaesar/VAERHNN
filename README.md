# VAERHNN
VAERHNN is an AI-based drug repurposing protocol to investigate potent leads against CRC. VAERHNN can comprehensively integrate the information of the target and its inhibitors or agonists for drug repurposing. We built a voting-averaged ensemble regression (VAER) model based on ensemble learning algorithm for molecular activity prediction. Moreover, we also assemble a hybrid neural network (HNN) consisting of multiple neural networks to predict the drug-target affinity (DTA).  Molecular dynamics (MD) simulations are performed to verify the binding stability of the lead compound and target as the final step of the virtual screening. Our proposed protocol can be used to identify potentially potent leads for repurposing against CRC or other diseases.

The overall flow chart is as follows:

<div align="center">
<p><img src="https://user-images.githubusercontent.com/57307280/163565613-6fdb9272-613f-470f-b3cf-bd2b914d2445.jpg" width="600"></p>
</div>

## Reproducibility
The analysis in the paper can be completely reproduced. In the directory, we have set up three folders: VAER, HNN and MD. You can reproduce or train your own data in the folder. In addition, you can also use it in a targeted manner. Folders to perform activity predictions, DTA predictions or perform MD simulations.
You may need the dataset indicated below to reproduce all results correctly.
Also, if you want to leverage the VAERHNN protocol for drug repurposing for a disease, or if you want to retrain the model, check out the appropriate sections in this README.

## Installation & Dependencies
The code of VAER and HNN is written in Python 3, which is mainly tested on Python 3.7 and Linux OS. It's faster to train on a GPU (install tensorflow GPU), but you can also work on a standard computer.
MD simulations need to install GROMACS version 2019 software and be carried out on Linux operating system.

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
The list of required environments is in the requirements.txt file. The other packages can be mostly installed using pip or conda.
```
conda create -n vaerhnn python=3.7
```


https://manual.gromacs.org/documentation/current/install-guide













