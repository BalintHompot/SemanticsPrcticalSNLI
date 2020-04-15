# Pretraining language models on SNLI
This repository contains code for pre-training and evaluating 4 different language models pre-trained on the SNLI (inference) task. The inference is performed by a shallow linear classifier that works on the output of one of the 4 encoder models:
* The baseline emcoder averages the word vectors of a sentence
* A unidiractional LSTM encoder
* A bidirectional LSTM encoder
* A bidirectional LSTM encoder with max pooling


## Pre-requisits
The reuired packages can be installed using the requirements.txt. Besides that, the SentEval package has to be installed for evaluation. For that, follow the installation and download guide in https://github.com/facebookresearch/SentEval.
Note: the SentEval package has to be in the root directory of this project

## Files and folders
### trainMain.py
This is the main function of the project which calls or imports the rest. In this, we import the data, define a set of default encoder parameters, ranges for each parameter. Then for each encoder class the code will:
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
Contains data loading and accuracy calculation script
### trainFunctions.py
This file contains the model training functions that are called when performing the pretraining. The most important ones that are directly called:
**paramSweep**
Runs parameter sweeping and stores the best ones in the corresponding folder
* input: 
    * encoderClass: an encoder class (not an instance)
    * data: the data as imported using the utils
    * default_config: a dictionary of default parameters, containing the ones that are sweeped: lr for starting learning rate, lr_decrease_factor for the decrease factor when val accuracy drops, lr_stopping for stopping the training when lr drops below this, layer_num for the number of hidden layers (not used in baseline encoder), layer_size for the dimensionality of hidden layers (not used in baseline encoder)
    * param_ranges: a dictionary of parameter ranges to be investigated. Same parameters as in the default config.
    * metadata: a dictionary of metadata, params that are not optimized: vector_size, vocab_size, pretrained for pretrained embeddings and pad_idx for the index of the padding token
    * forceOptimize: boolean, when set to true, the sweeping will be performed een if there is already a stored best param for the encoder. Checked for each param separately.
* output: a dict with the best params 
**construct_and_train_model_with_config**
Constructs the SNLI classifier model with a given config, then calls train.
* input
    * encoderClass: the encoder class to be used (not an instance)
    * data: the data as imported using utils
    * config: dict with the params that are sweeped
    * metadata: dict with the params that are not searched
    * forceRetrain: boolean, when set to true, construction and training is performed even if a best model is stored for the given encoder
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
* output: none

### sentEval.py
Contains the code for running the sentEval benchmark. The function runSentEval can be called with passing a model, a torchtext Field that contains the text preprocessing pipeline (this is normally obtained when getting the data), and optionally a list of tests that will be performed. This defaults to 'all' which runs all tests, but the string 'paper' can be passed which performs the tasks reported in [the Conneau et al paper](https://arxiv.org/pdf/1705.02364.pdf).

### example.ipynb
This is jupyter notebook to present the usage and results of the project. It follows trainMain closely with some added examples on how to customize the pipeline.

### best_configs
Folder containing the jsons with the best configs found for the encoders

### best_models
Trained models with the different encoders using pickle serialization

### best_model_results
Contains two json per encoder: one is the SNLI dev and test accuracy, the other is the SentEval results

## Usage
The usage can be easily understood by taking a look at the trainMain.py or the example notebook.\\
To easily perform the full pipeline one can directly call the trainMain.py. See above what trainMain performs when called. \\


The easiest way to change which encoders are investigated, what parameter ranges are used for sweeping, and what are the default parameters is to edit this script at the corresponding sections. The easiest way to construct a model with custom parameters is to define a config file and use it at constructions. Examples can be found in the example notebook.

