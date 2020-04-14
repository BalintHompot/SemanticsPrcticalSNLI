
import numpy as np
import torch
import json
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
from model import SNLIModel
from utils import accuracy
DEVICE = "cuda"

def lr_is_bigger(optimizer, threshold):
    for g in optimizer.param_groups:
        if g['lr'] <= threshold:
            return False
    return True

def trainModel(model, data, optimizer, lr_threshold, lr_decrease_factor, plotting = False):

    print("======================== " + model.name + "========================")
    model = model.to(DEVICE)
    ###for plotting
    if plotting:
        losses = {"ticks":[], "values":[], "seriesName": "train loss"}
        accuracies = {"ticks":[], "values":[], "seriesName": "train accuracy"}
        accuracies_val = {"ticks":[], "values":[], "seriesName": "validation accuracy"}
    
    last_epoch_val_acc = 0.4 ### if after the first epoch we are not better then 0.4 we should immediately decrease
    epoch = 0
    while lr_is_bigger(optimizer, lr_threshold):
        avg_acc = 0
        total_loss = 0
        batch_count = 0

        print("epoch " + str(epoch))
        for batch in tqdm(data["train"]):
            labels = (batch.label -1).to(DEVICE)
            model.train()
            optimizer.zero_grad()
            output = model(batch.premise, batch.hypothesis)
            loss = model.loss_fn(output, labels)
            avg_acc += accuracy(output, labels)
            total_loss += loss.item()
            batch_count += 1
            loss.backward()
            optimizer.step()


        avg_acc /= batch_count
        print("epoch: " + str(epoch) + " total loss: " + str(total_loss) + " avg acc: " + str(avg_acc))
        if plotting:
            losses["ticks"].append(epoch)
            losses["values"].append(loss.item())
            accuracies["ticks"].append(epoch)
            accuracies["values"].append(avg_acc)
        if epoch%1 == 0:
            val_acc = 0
            val_batch_count = 0
            for batch_val in iter(data["val"]):
                model.eval()
                validation_out = model(batch_val.premise, batch_val.hypothesis)
                val_acc += accuracy(validation_out, batch_val.label - 1)
                val_batch_count += 1
            val_acc /= val_batch_count
            if plotting:
                accuracies_val["ticks"].append(epoch)
                accuracies_val["values"].append(val_acc)
            ## decreasing the learning rate when worse performance
            if val_acc <= last_epoch_val_acc:
                print("decreasing lr")
                for g in optimizer.param_groups:
                    g['lr'] /= lr_decrease_factor
            last_epoch_val_acc = val_acc
            print("-------------- validation average acc: " + str(val_acc))
        epoch += 1
        

    if plotting:
        savePlot([losses, accuracies, accuracies_val], model.name)
    print("--Train finished, learning rate decreased below threshold--")
    return model, val_acc

def savePlot(dataSeries, modelName):
    f, a = plt.subplots(1, len(dataSeries), squeeze = False)
    for seriesInd, series in enumerate(dataSeries):
        a[0][seriesInd].plot(series["ticks"], series["values"])
        a[0][seriesInd].set_title(series["seriesName"])
        a[0][seriesInd].set_xlabel('epoch')
        a[0][seriesInd].set_ylabel('value')
    plt.savefig("./plots/" + modelName + "_training.png")


def testModel(model, data, scorePlotting = True):
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    print('Final performance of ' + model.name + ' with optimal params on test partition.')
    test_acc = 0
    test_batch_count = 0
    for batch_test in iter(data["test"]):
        model.eval()
        test_out = model(batch_test.premise, batch_test.hypothesis)
        test_acc += accuracy(test_out, batch_test.label - 1)
        test_batch_count += 1
    test_acc /= test_batch_count
    results = {"test accuracy: ": test_acc}
    print("test accuracy: "+str(test_acc))
    return results

def save_model_and_res(model, results):
    with open("./best_model_results/"+model.name+"_best_config_results.json", "w+") as writer:
        json.dump(results, writer, indent=1)

    filehandler = open("./best_models/" + model.name + ".model", 'wb') 
    pkl.dump(model, filehandler)
    filehandler.close()
    

def construct_and_train_model_with_config(encoderClass, data, config, metadata, forceRetrain = False):

    encoder = encoderClass(metadata, config["number of neurons per layer"], config["number of layers"])
    model = SNLIModel(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning rate"])

    model_save_path = "./best_models/" + model.name + ".model"
    if os.path.exists(model_save_path) and not forceRetrain:
        print("-----------------")
        print("best model already stored and force retrain is false")
        print("retrieving model")
        print("-----------------")
        filehandler = open(model_save_path, "rb")
        trained_model = pkl.load(filehandler)
        filehandler.close()
    else:

        print("++++++++++++++++++++++++++ Training model " + model.name + " with best params +++++++++++++++++++++++++++++++")

        trained_model, train_results = trainModel(model, data, optimizer,  config["lr_stopping"], config["lr_decrease_factor"], plotting = False)
    return trained_model



def paramSweep(encoderClass, data, default_config, param_ranges, metadata, forceOptimize = False):
    '''
    '''

    #retrieve the previously optimised model config
    model_name = encoderClass(metadata, 1,1).name + " SNLI"
    save_file = "./best_configs/" + model_name + "_best_config.json"
    if os.path.exists(save_file) and not forceOptimize:
        print(f"{model_name} optimized parameters already exists, retrieving")
        with open(save_file, "r") as f:
            best_config =  json.load(f)
    else:
        best_config = {} 
    
    optimized_keys = best_config.keys()

    best_acc = -10000 #default starting point mea

    learning_rates = param_ranges["learning rates"]
    lr_stoppings = param_ranges["lr_stoppings"]
    lr_decrease_factors = param_ranges["lr_decrease_factors"]
    layer_nums = param_ranges["layer nums"]
    layer_neurons = param_ranges["layer sizes"]

    ### to save time we optimize for the params separately (which might not be optimal, but we don't have to run hundreds of models)
    if not "learning rate" in optimized_keys:
        print("............................")
        print("optimizing starting learning rate")
        print("............................")
        for lr in learning_rates:
            encoder = encoderClass(metadata, default_config["layer_size"], default_config["layer_num"])
            model = SNLIModel(encoder)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            current_model, current_acc = trainModel(model, data, optimizer, default_config["lr_stopping"], default_config["lr_decrease_factor"])
            if(current_acc > best_acc):
                best_acc = current_acc
                best_config["learning rate"] = lr
        print("............................")
        print("finished sweeping for learning rate, best value: " + str(best_config["learning rate"]))
        print("............................")
    else:
        print("best learning rate retrieved from stored param file")

    if not "lr_stopping" in optimized_keys:
        best_acc = -1000
        print("optimizing lr stopping")
        print("............................")
        for lr_stopping in lr_stoppings:
            encoder = encoderClass(metadata, default_config["layer_size"], default_config["layer_num"])
            model = SNLIModel(encoder)
            optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
            current_model, current_acc = trainModel(model, data, optimizer, lr_stopping, default_config["lr_decrease_factor"])
            if(current_acc > best_acc):
                best_acc = current_acc
                best_config["lr_stopping"] = lr_stopping
        print("............................")
        print("finished sweeping for lr stopping criterion, best value: " + str(best_config["lr_stopping"]))
        print("............................")
    else:
        print("best lr stopping criterion retrived from stored param file")

    if not "lr_decrease_factor" in optimized_keys:
        best_acc = -1000
        print("optimizing lr decrease factor")
        print("............................")
        for lr_decrease_factor in lr_decrease_factors:
            encoder = encoderClass(metadata, default_config["layer_size"], default_config["layer_num"])
            model = SNLIModel(encoder)
            optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
            current_model, current_acc = trainModel(model, data, optimizer, default_config["lr_stopping"], lr_decrease_factor)
            if(current_acc > best_acc):
                best_acc = current_acc
                best_config["lr_decrease_factor"] = lr_decrease_factor
        print("............................")
        print("finished sweeping for lr decrease factor, best value: " + str(best_config["lr_decrease_factor"]))
        print("............................")
    else:
        print("best lr decrease factor retrived from stored param file")


    best_acc = -1000
    if model_name != "Vector mean SNLI":
        ### mean model does not have these params
        if not "number of layers" in optimized_keys:

            print("optimizing layer num")
            print("............................")
            for layer_num in layer_nums:
                encoder = encoderClass(metadata, default_config["layer_size"],layer_num)
                model = SNLIModel(encoder)
                optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
                current_model, current_acc = trainModel(model, data, optimizer, default_config["lr_stopping"], default_config["lr_decrease_factor"])
                if(current_acc > best_acc):
                    best_acc = current_acc
                    best_config["number of layers"] = layer_num
            print("............................")
            print("finished sweeping for number of layers, best value: " + str(best_config["number of layers"]))
            print("............................")
        else:
            print("best layer number retrived from stored param file")

        if not "number of neurons per layer" in optimized_keys:
            best_acc = -1000
            print("optimizing layer size")
            print("............................")
            for layer_neuron in layer_neurons:
                encoder = encoderClass(metadata, layer_neuron,default_config["layer_num"])
                model = SNLIModel(encoder)
                optimizer = torch.optim.Adam(model.parameters(), lr=default_config["lr"])
                current_model, current_acc = trainModel(model, data, optimizer, default_config["lr_stopping"], default_config["lr_decrease_factor"])
                if(current_acc > best_acc):
                    best_acc = current_acc
                    best_config["number of neurons per layer"] = layer_neuron
            print("............................")
            print("finished sweeping for neurons per layer, best value: " + str(best_config["number of neurons per layer"]))
            print("............................")
        else:
            print("best layer number retrived from stored param file")

    else:
        ## setting to 1 so the initialization of the model works, they are not used though
        best_config["number of layers"] = 1
        best_config["number of neurons per layer"] = 1

    with open("./best_configs/" + model_name + "_best_config.json", "w+") as writer:
        json.dump(best_config, writer, indent=1)

    return best_config

