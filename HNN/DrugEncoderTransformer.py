# coding: utf-8

import DeepPurpose.DTI as model
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

#Transformer_AAC
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'AAC'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)


config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 100
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
np.savetxt('result/results_Transformer_AAC.csv',y_pred,delimiter=',')

#Transformer_PseAAC
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'PseudoAAC'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 100
                        )
model = models.model_initialize(**config)
model.train(train, val, test)


X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_Transformer_PseAAC.csv',y_pred,delimiter=',')


#Transformer_ConTriad
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'Conjoint_triad'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 100
                        )
model = models.model_initialize(**config)
model.train(train, val, test)


X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_Transformer_ConTriad.csv',y_pred,delimiter=',')


#Transformer_Quasi-seq
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'Quasi-seq'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)


config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 100
                        )
model = models.model_initialize(**config)
model.train(train, val, test)


X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_Transformer_Quasi-seq.csv',y_pred,delimiter=',')


#Transformer_CNN
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'CNN'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 100
                        )
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_Transformer_CNN.csv',y_pred,delimiter=',')


#Transformer_RNN_GRU
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'CNN_RNN'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 100,
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
np.savetxt('result/results_Transformer_Gru.csv',y_pred,delimiter=',')


#Transformer_RNN_LSTM
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'CNN_RNN'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         train_epoch = 100   , 
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
np.savetxt('result/results_Transformer_lstm.csv',y_pred,delimiter=',')


#Transformer_Transformer
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Transformer', 'Transformer'
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2], random_seed = 1)

config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding,
                         train_epoch = 100
                        )
model = models.model_initialize(**config)
model.train(train, val, test)

X_drug_zinc, X_target, y = read_file_training_dataset_drug_target_pairs('data/candidate.txt')
X_pred = data_process(X_drug_zinc, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='no_split')
y_pred = model.predict(X_pred)
print('The predicted score is ' + str(y_pred))
np.savetxt('result/results_Transformer_Transformer.csv',y_pred,delimiter=',')