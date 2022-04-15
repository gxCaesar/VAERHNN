# coding: utf-8

import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

#Morgan_AAC
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'AAC'
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
np.savetxt('result/results_Morgan_AAC.csv',y_pred,delimiter=',')



#Morgan_PseAAC
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'PseudoAAC'
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
np.savetxt('result/results_Morgan_PseAAC.csv',y_pred,delimiter=',')

#Morgan_ConTriad
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'Conjoint_triad'
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
np.savetxt('result/results_Morgan_ConTriad.csv',y_pred,delimiter=',')


#Morgan_Quasi-seq
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'Quasi-seq'
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
np.savetxt('result/results_Morgan_Quasi-seq.csv',y_pred,delimiter=',')

#Morgan_CNN
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'CNN'
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
np.savetxt('result/results_Morgan_CNN.csv',y_pred,delimiter=',')


#Morgan_GRU
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'CNN_RNN'
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
np.savetxt('result/results_Morgan_Gru.csv',y_pred,delimiter=',')


#Morgan_LSTM
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'CNN_RNN'
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
np.savetxt('result/results_Morgan_Lstm.csv',y_pred,delimiter=',')

#Morgan_Transformer
X_drug, X_target, y = read_file_training_dataset_drug_target_pairs('data/chembl.txt')
drug_encoding, target_encoding = 'Morgan', 'Transformer'
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
np.savetxt('result/results_Morgan_Transformer.csv',y_pred,delimiter=',')

