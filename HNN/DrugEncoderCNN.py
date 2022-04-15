
# coding: utf-8


import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *


#CNN_AAC
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'AAC'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 200
                        )
generate_config()
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_AAC.csv',y_pred,delimiter=',')


#CNN_PseAAC
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'PseudoAAC'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 200 
                        )
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_PseAAC.csv',y_pred,delimiter=',')


#CNN_ConTriad
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'Conjoint_triad'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 200
                        )
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_ConTriad.csv',y_pred,delimiter=',')


#CNN_Quasi-seq
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'Quasi-seq'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 200
                        )
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_Quasi-seq.csv',y_pred,delimiter=',')



#CNN_CNN
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'CNN'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 200
                        )
model = models.model_initialize(**config)
model.train(train, val, test)


X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_cnn.csv',y_pred,delimiter=',')


#CNN_GRU
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'CNN_RNN'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 200,
                         rnn_Use_GRU_LSTM_target = 'GRU'
                        )
model = models.model_initialize(**config)
model.train(train, val, test)


X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_Gru.csv',y_pred,delimiter=',')


#CNN_Lstm
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'CNN_RNN'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 200, 
                         rnn_Use_GRU_LSTM_target = 'LSTM'
                        )
model = models.model_initialize(**config)
model.train(train, val, test)


X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_Lstm.csv',y_pred,delimiter=',')


#CNN_Transformer
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'CNN', 'Transformer'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 200
                        )
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_CNN_Transformer.csv',y_pred,delimiter=',')