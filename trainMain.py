
import numpy as np
import torch
import json
from tqdm import tqdm
from encoders import *
from trainFunctions import *
from utils import *


train_data, val_data, test_data, TEXT = get_data()
data = {"train":train_data, "val": val_data, "test": test_data}

print("Data loaded, starting train")

################### default params and metadata

metadata = {
    "vector_size" : 300,
    "vocab_size" : len(TEXT.vocab),
    "pretrained" : TEXT.vocab.vectors,
    "pad_idx" : TEXT.vocab.stoi[TEXT.pad_token]
}

default_params = {
    "lr_decrease_factor":5,
    "lr_stopping" : 1e-5,
    "layer_num" : 1,
    "layer_size" : 500,
    "lr" : 0.001,
}

################### parameter ranges for sweep
param_ranges = {
    "learning rates":[0.01, 0.001, 0.0001],
    "lr_decrease_factors":[3, 5],
    "lr_stoppings": [1e-5, 1e-6], 
    "layer nums":[1,2],
    "layer sizes":[500,1000,2000],
}
######################################################

print("--------- Fitting models and testing on set-aside data ------------")
for encoderClass in [BiLSTMEncoder, MaxBiLSTMEncoder]:
    # searching for best params
    best_params_for_model = paramSweep(encoderClass, data, default_params, param_ranges, metadata)
    # training model with best params (and saving training plots)
    best_model = construct_and_train_model_with_config(encoderClass, data, best_params_for_model, metadata)
    # testing the best model
    best_model_results = testModel(best_model, data)
    # saving best model and results
    save_model_and_res(best_model, best_model_results)


