# Training and predicting DTA for candidates
import os
os.system("python ./DrugEncoderMorgan.py")
os.system("python ./DrugEncoderPubchem.py")
os.system("python ./DrugEncoderDaylight.py")
os.system("python ./DrugEncoderRdkit.py")
os.system("python ./DrugEncoderCNN.py")
os.system("python ./DrugEncoderGru.py")
os.system("python ./DrugEncoderTransformer.py")
os.system("python ./DrugEncoderMPNN.py")