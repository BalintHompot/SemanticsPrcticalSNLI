# Pretraining language models on SNLI
This repository contains code for pre-training and evaluating 4 different language models pre-trained on the SNLI (inference) task. The inference is performed by a shallow linear classifier that works on the output of one of the 4 encoder models:
* The baseline encoder averages the word vectors of a sentence
* A unidiractional LSTM encoder
* A bidirectional LSTM encoder
* A bidirectional LSTM encoder with max pooling


## Pre-requisits
The reuired packages can be installed using the requirements.txt. Besides that, the SentEval package has to be installed for evaluation. For that, follow the installation and download guide in https://github.com/facebookresearch/SentEval.
Note: the SentEval package has to be in the root directory of this project

## Usage
The main entry point of the project is trainMain which calls the separate functions from trainFunctions and performs the whole task end-to-end from parameter sweeping through model trainig and storing to SentEval results. When called, the full task is performed with the arguments coded into the script, however it can be customized and the train functions can be called separately (e.g. sweeping can be skipped to directly train a model).

For customization, the easiest way to change which encoders are investigated, what parameter ranges are used for sweeping, and what are the default parameters is to edit this main script at the corresponding sections. The easiest way to construct a model with custom parameters is to define a config file and use it at constructions.

The examples.ipynb closely follows the trainMain to show how the separate functions of the project can be used. It also gives easy to use examples to show how customization can be done. It also contains the analysis of the project results. I recommend to start at this file and the usage will quickly become clear.

Note: the usage of the project is centered around "run"-s. The user can pass runName to the functions that involve storing, loading or checking for files: the outputs will be stored in the corresponding directory (created automatically), loaded from there, and the functions will check in those directories if there are already stored params/models for the run with the given name. In all functions this runName defaults to "best" (the idea is that the sweeping is performed so we are storing the best).

For further information on the files and functions, see the documentation below.

## Documentation
The scripts and folders in the project's root directory are listed here.
### trainMain.py
This is the main function of the project which calls or imports the rest. In this, we import the data, define a set of default encoder parameters and ranges for the parameters to be optimized. Then for each encoder class the code will:
* Run parameter sweeping and store the best parameters (this is skipped if best params are found, and sweep is not forced)
* Construct a model with the found parameters and train it (this is skipped if stored trained best model is found and retrain is not forced)
* Test the model on the test set and save the model and results
* Test the model using SentEval.
The details for these are in the following files.
### encoder.py
Contains the classes of the 4 encoder models and their parent class
### model.py
Contains the SNLI predictor class. It takes an encoder as an argument and uses it as it's encoder, extending it with the shallow classifier
### utils.py
Contains data loading and accuracy calculation script as well as the results printing function which can be called separately:
**printResults**:
* input:
    * encoderNames: list of encoder names to be included in the table
    * resultType: the results to be included, SNLI for dev and test, SentEval, or SNLI+transfer, the latter contains micro and macro avg of senteval tasks
    * runName: the name of the run for which results are printed
* output: None, only printing
### trainFunctions.py
This file contains the model training functions that are called when performing the pretraining. The most important ones that are directly called:
**paramSweep**
Runs parameter sweeping and stores the best ones in the corresponding folder.
Note: the parameters are searched one by one while leaving the rest at default. This is to reduce search time, although in principle it does not neccessarily give the best parameter setting as they are not properly searched on a grid.
* input: 
    * encoderClass: an encoder class (not an instance)
    * data: the data as imported using the utils
    * default_config: a dictionary of default parameters, containing the ones that are sweeped: lr for starting learning rate, lr_decrease_factor for the decrease factor when val accuracy drops, lr_stopping for stopping the training when lr drops below this, layer_num for the number of hidden layers (not used in baseline encoder), layer_size for the dimensionality of hidden layers (not used in baseline encoder)
    * param_ranges: a dictionary of parameter ranges to be investigated. Same parameters as in the default config.
    * metadata: a dictionary of metadata, params that are not optimized: vector_size, vocab_size, pretrained for pretrained embeddings and pad_idx for the index of the padding token
    * forceOptimize: boolean, when set to true, the sweeping will be performed een if there is already a stored best param for the encoder. Checked for each param separately.
    * runName: string, the name of the run. The script creates a folder for _configs with this prefix to store the output. Defaults to "best"
* output: a dict with the best params 
**construct_and_train_model_with_config**
Constructs the SNLI classifier model with a given config, then calls train.
* input
    * encoderClass: the encoder class to be used (not an instance)
    * data: the data as imported using utils
    * config: dict with the params that are sweeped
    * metadata: dict with the params that are not searched
    * forceRetrain: boolean, when set to true, construction and training is performed even if a best model is stored for the given encoder
    * runName: string, the name of the run. The script checks if there is a stored model in the folder _models with this prefix and the given encoderClass's name. Defaults to "best"
* output: the trained model
**trainModel**
The function that performs the actual training
* input 
    * model: an SNLI classifier model instance
    * data: the data as imported using utils
    * optimizer: the optimizer for the training
    * lr_threshold: training stops when lr drops below this
    * lr_decrease_factor: lr is decreased by this factor when an epoch drops in validation accuracy
    * plotting: boolean, when True, a plot will be saved with the losses and accuracies (not tested on the current version, leave it False)
* output: the trained model and the last validation accuracy
**testModel**
The function that performs testing on the set aside data.
* input
    * model: a trained SNLI encoder instance
    * data: the data as imported using utils
* output: dev and test accuracy in a dict
**save_model_and_res**
Stores the model and the results in the corresponding folders (folders are hard coded, edit script to change)
* input
    * model: a trained SNLI encoder instance
    * results: a dict with the results
    * runName: string, the name of the run. The script creates folders _models and _model_results  with this prefix to store the outputs. Defaults to "best"
* output: none
**testExample**
Prompts the user to give premises and hypotheses (in a loop). For each pair, prints the model's verdict on entailment
* input:
    * modelName: string, the model's name to be tested
    * texField: torchtext Field that contains the preprocessing. If nothing givem, the code creates automatically
    * labelField: torchtext Field that contains the mapping from output index to verdict. If nothing givem, the code creates automatically.
    * runName: the name of the run from which the model is loaded
* output: None, only printing

### sentEval.py
Contains the code for running the sentEval benchmark. The function runSentEval can be called with passing a model, a torchtext Field that contains the text preprocessing pipeline (this is normally obtained when getting the data with utils), and optionally a list of tests that will be performed. This defaults to 'all' which runs all tests, but the string 'paper' can also be passed which performs the tasks reported in [the Conneau et al paper](https://arxiv.org/pdf/1705.02364.pdf). The runName can be passed similarly to the other functions to store the output.

### example.ipynb
This is a jupyter notebook to present the usage and results of the project. It follows trainMain closely with some added examples on how to customize the pipeline. It also contains the analysis of results

### trainMain.job
The job file that was submitted to the Lisa cluster

### requirements.txt
The packages needed to run the project

### runs
The user can give names to runs when performing the functions. For each run, a directory will be placed under the the runs directory with the given name. The new directory will contain 3 subdirectories: runName_configs with the used configs (either created by user or found during sweeping), runName_models with the trained models for each encoder, and runName_model_results with the SNLI and SentEval results for each encoder.
Currently there are two run outputs: the "best" was done with the reported params in the paper with Adam and appropriately changed lr, the "sgd" is the same but with SGD optimizer and lr changed back.
### logs
slurm output of the runs on the Lisa cluster


