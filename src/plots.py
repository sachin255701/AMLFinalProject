import numpy as np
import matplotlib.pyplot as plt
from train_network import *


def classifcationAccuracyPlot(new_labels, ref_labels, title, plot=True):
    if(new_labels.shape != ref_labels.shape):
        print("Input array and reference array dimensions doesn't match. Please check!")
        return
    unique_elements, counts_elements = np.unique(new_labels == ref_labels, return_counts=True)

    if(plot):
        plt.subplots(figsize=(5, 5))
        plt.bar(unique_elements, counts_elements)
        plt.xlabel('Classes', fontsize=10)
        plt.ylabel('Number of Examples', fontsize=10)
        plt.xticks(unique_elements, unique_elements, fontsize=10, rotation=0)
        plt.title(title)
        plt.show()

    accuracy = ((counts_elements[np.where(unique_elements == True)])/(np.sum(counts_elements))) * 100

    print("Accuracy of this model with given predicted values is " + str(accuracy[0]) + " %")
    return str(accuracy[0])

def outputClassesPlot(xelement, yelement, classcount, title):
    index = np.arange(classcount)
    plt.subplots(figsize=(18, 5))
    plt.bar(xelement, yelement)
    plt.xlabel('Classes', fontsize=15)
    plt.ylabel('Number of examples', fontsize=15)
    plt.xticks(index, index, fontsize=10, rotation=0)
    plt.title(title, fontsize=15)
    plt.show()

def verifyAttack(adv, random_sample, y_val, attack):
    # list of distilled models at diffrent temperatures
    distilled_models_with_different_T = ["models/distilled_model_T_1.h5",
                                         "models/distilled_model_T_2.h5",
                                         "models/distilled_model_T_5.h5",
                                         "models/distilled_model_T_10.h5",
                                         "models/distilled_model_T_35.h5",
                                         "models/distilled_model_T_50.h5",
                                         "models/distilled_model_T_100.h5"]

    failed = 0
    for model_t in distilled_models_with_different_T:
        distill_model = train_distillation(([], []), ([], []), 0, model_t)
        y_val_adversarial_distill_pred = distill_model.predict(adv.reshape(1, 28, 28, 1))
        output = np.argmax(y_val_adversarial_distill_pred, axis=1)
        if (output == y_val[random_sample]):
            failed = 1
            print(attack + " attack failed for model " + model_t)
        else:
            print("Distillation Model " + str(model_t.split("/")[-1]) + " predicted label: " + str(
                output) + " True Label: " + str(y_val[random_sample]) + ". Distillation Model failed to correctly classify adversarial image.")

    if (failed == 0):
        print("Defensive Distillation neural network model failed against "+ attack + " attack at all Distillation Temperatures.")