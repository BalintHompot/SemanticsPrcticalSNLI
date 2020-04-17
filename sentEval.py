from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import numpy as np
from torchtext import data
from torchtext import datasets
import logging
from utils import get_data
import pickle as pkl
import json

PATH_SENTEVAL = ''
PATH_TO_DATA = 'SentEval/data'

sys.path.insert(0, PATH_SENTEVAL)
import senteval


def batcher(params, batch):
    sentences = []
    for s in batch:
        if s == []:
            s = ["-"]
        sentence = params.inputs.preprocess(s)
        sentences.append(sentence)
    sentences = params.inputs.process(sentences, device = "cuda")
    emb = params.model.forward(sentences)
    embeddings = []
    for sent in emb:
      sent = sent.cpu()
      embeddings.append(sent.data.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings


# params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'seed':1234}
# params_senteval['classifier'] = {'nhid': 600, 'optim': 'adam,lr=0.0005', 'batch_size': 64,
#                                  'tenacity': 5, 'epoch_size': 4, 'dropout': 0.2}

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def runSentEval(model, textfield, tasks = "all", runName = "best"):

    params_senteval['model'] = model.encoder
    params_senteval['inputs'] = textfield

    se = senteval.engine.SE(params_senteval, batcher)

    # define transfer tasks
    if tasks == "all":
        ## run all available tasks
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                        'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                        'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                        'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                        'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    elif tasks == "paper":
        ## run tasks reported in the paper
        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC','MRPC', 'SICKEntailment', 'STS14']
    else:
        ## you can pass a list as arg
        transfer_tasks = tasks

    results = se.eval(transfer_tasks)

    with open("./runs/" + runName + "/" + runName + "_model_results/"+model.name+"_SentEval_results.json", "w+") as writer:
        json.dump(results, writer, indent=1, default=lambda o: '<not serializable>')
