from torchtext import data, datasets
from torchtext.vocab import GloVe
import numpy as np

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
    return train_iter, val_iter, test_iter, TEXT



def accuracy(output, target):
    output = np.argmax(output.cpu().detach().numpy(), axis = 1)
    target = target.cpu().detach().numpy()
    correct = np.sum(output == target)
    return correct/len(target)
    