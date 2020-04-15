#Pretraining language models on SNLI
This repository contains code for pre-training and evaluating 4 different language models pre-trained on the SNLI (inference) task. The inference is performed by a shallow linear classifier that works on the output of one of the 4 encoder models:
Markup : * The baseline emcoder averages the word vectors of a sentence
        * A unidiractional LSTM encoder
        * A bidirectional LSTM encoder
        * A bidirectional LSTM encoder with max pooling


##Pre-requisits
The reuired packages can be installed using the requirements.txt. Besides that, the SentEval package has to be installed for evaluation. For that, follow the installation and download guide in https://github.com/facebookresearch/SentEval.
Note: the SentEval package has to be in the root directory of this project

##Files
###encoder.py
Contains the classes of the 4 encoder models and their parent class
###model.py
Contains the SNLI predictor class. It takes an encoder as an argument and uses it as it's encoder, extending it with the shallow classifier
###utils.py
Contains data loading and accuracy calculation script
###trainFunctions.py
This file contains the functions that are called when performing the pretraining:
Markup : * paramsweep
        * A unidiractional LSTM encoder
        * A bidirectional LSTM encoder
        * A bidirectional LSTM encoder with max pooling

##Usage

