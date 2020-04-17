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
        (train,val, test), batch_size=300, device="cuda")
    print("done, returning data")
    ### text contains metadata, returning it
    return train_iter, val_iter, test_iter, TEXT, LABEL



def accuracy(output, target):
    output = np.argmax(output.cpu().detach().numpy(), axis = 1)
    target = target.cpu().detach().numpy()
    correct = np.sum(output == target)
    return correct/len(target)
    
def printResults(encoderNames, resultType, runName = "best"):
    tabs = []
    headers = ["Model"]
    headersSet = False
    micro = 0
    microCount = 0
    macro = 0
    macroCount = 0
    for encoder in encoderNames:
        enc_tab_res = [encoder]
        if resultType == "SNLI" or resultType == "SNLI+transfer":
            with open("./runs/" + runName + "/" + runName + "_model_results/" + encoder + " SNLI_best_config_results.json", "rb") as filehandler:
                enc_res = load(filehandler)
            filehandler.close()
            enc_tab_res.extend([enc_res["dev accuracy"]*100, enc_res["test accuracy: "]*100])
            if not headersSet:
                headers.extend(list(enc_res.keys()))
                headersSet = True

        if resultType == "SentEval" or resultType == "SNLI+transfer":
            with open("./runs/" + runName + "/" + runName + "_model_results/" + encoder + " SNLI_SentEval_results.json", "rb") as filehandler:
                enc_res = load(filehandler)
            filehandler.close()
            
            for task in enc_res.keys():
                if task == "MRPC":
                    enc_tab_res.append(str(enc_res[task]["acc"]) + "/" + str(enc_res[task]["f1"]))
                elif task == "STS14":
                    enc_tab_res.append(str(round(enc_res[task]["all"]["pearson"]["mean"], 2)) + "/" + str(round(enc_res[task]["all"]["spearman"]["mean"], 2)))
                else:
                    enc_tab_res.append(enc_res[task]["acc"])
                    macro += enc_res[task]["acc"]
                    macroCount += 1
                    micro += enc_res[task]["acc"] * enc_res[task]["ntest"]
                    microCount += enc_res[task]["ntest"]
            if not headersSet:
                headers.extend(list(enc_res.keys()))
                headersSet = True
        
        if resultType == "SNLI+transfer":
            enc_tab_res = enc_tab_res[0:3]
            enc_tab_res.extend([macro / macroCount, micro/microCount])

        tabs.append(enc_tab_res)

    if resultType == "SNLI+transfer":
        headers.extend(["transfer macro", "transfer micro"])

    print(tabulate(tabs, headers=headers, tablefmt='orgtbl'))

