
from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import numpy as np
from torchtext import data
from torchtext import datasets
import logging


PATH_SENTEVAL = ''
PATH_TO_DATA = 'data'



sys.path.insert(0, PATH_SENTEVAL)
import senteval

def prepare(params, samples):
    train = data.Dataset(samples, params.textField)
    senteval_iter = data.BucketIterator.splits(
        (train), batch_size=300, device="cuda")
    return senteval_iter

def batcher(params, batch):

    return params.model(batch)


# params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'seed':1234}
# params_senteval['classifier'] = {'nhid': 600, 'optim': 'adam,lr=0.0005', 'batch_size': 64,
#                                  'tenacity': 5, 'epoch_size': 4, 'dropout': 0.2}

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def runSentEval(model, originalTEXTfield):

    params_senteval["model"] = model
    params_senteval['textField'] = originalTEXTfield

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ['CR', 'MR']
    # define transfer tasks
    '''
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    '''
    #['MR', 'CR', 'SUBJ', 'MPQA', 'STSBenchmark', 'SST2', 'SST5', 'TREC', 'MRPC', 
    #'SICKRelatedness', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)