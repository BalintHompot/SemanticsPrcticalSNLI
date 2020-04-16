from torchtext import data, datasets
from torchtext.vocab import GloVe
import numpy as np
from tabulate import tabulate
from json import load

def get_data():
    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize='spacy')
    LABEL = data.Field(sequential=False)

    # make splits for data
    print("Accessing raw input and preprocessing")
    train, val, test = datasets.SNLI.splits(TEXT, LABEL,  root='.data', train='snli_1.0_train.jsonl', validation='snli_1.0_dev.jsonl', test='snli_1.0_test.jsonl')
    print("done")

    # build the vocabulary
    print("Building vocabulary with GloVe")
    TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=300))
    LABEL.build_vocab(train)
    print("done")

    # make iterator for splits
    print("Loading data into iterables")
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train,val, test), batch_size=64, device="cuda")
    print("done, returning data")
    ### text contains metadata, returning it
    return train_iter, val_iter, test_iter, TEXT



def accuracy(output, target):
    output = np.argmax(output.cpu().detach().numpy(), axis = 1)
    target = target.cpu().detach().numpy()
    correct = np.sum(output == target)
    return correct/len(target)
    
def printResults(encoderNames, resultType, runName = "best"):
    tabs = []
    headers = ["Model"]
    headersSet = False
    for encoder in encoderNames:
        if resultType == "SNLI":
            with open("./" + runName + "_model_results/" + encoder + " SNLI_best_config_results.json", "rb") as filehandler:
                enc_res = load(filehandler)
            filehandler.close()
            enc_tab_res = [encoder, enc_res["dev accuracy"], enc_res["test accuracy: "]]
        elif resultType == "SentEval":
            with open("./" + runName + "_model_results/" + encoder + " SNLI_SentEval_results.json", "rb") as filehandler:
                enc_res = load(filehandler)
            filehandler.close()
            enc_tab_res = [encoder]
            for task in enc_res.keys():
                if task == "MRPC":
                    enc_tab_res.append(str(enc_res[task]["acc"]) + "/" + str(enc_res[task]["f1"]))
                elif task == "STS14":
                    enc_tab_res.append(str(round(enc_res[task]["all"]["pearson"]["mean"], 2)) + "/" + str(round(enc_res[task]["all"]["spearman"]["mean"], 2)))
                else:
                    enc_tab_res.append(enc_res[task]["acc"])

        else:
            print("not a valid result type (SNLI|SentEval)")
            return
        tabs.append(enc_tab_res)
        if not headersSet:
            headers.extend(list(enc_res.keys()))
            headersSet = True

    print(tabulate(tabs, headers=headers, tablefmt='orgtbl'))


printResults(["Vector mean", "LSTM", "BiLSTM", "Pooled BiLSTM"], resultType = "SentEval")