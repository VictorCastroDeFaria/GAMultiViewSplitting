import random
import numpy as np
import os

import time

from hardnessFunctions import *
from geneticFunctionsInteger import *
from fileUtilities import *


from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier

import csv

from joblib import Parallel, delayed

###############################

import warnings
warnings.filterwarnings("ignore")


import openml

def extractFilenameCore(filename):
    # Remove the "_clean.csv" suffix from the filename
    return filename.replace('_clean.csv', '')

def appendFilenameCore(dataset_name: str):
    return (dataset_name + "_clean.csv")

study_id = 15
study = openml.study.get_study(study_id)


def trainingFolds(filename, fold, fold_iteration, train_index_original, test_index):

    working_dataset_df, target_variable = processCsvToDataFrame(filename)

    # Data extraction and specific parameters:
    n_genes = working_dataset_df.shape[1]
    n_instances = working_dataset_df.shape[0]


    print("-----------")
    print("Fold:",fold)

    # Split data into Original Training and Test data:
    X_train_original = working_dataset_df.iloc[train_index_original]
    y_train_original = target_variable.iloc[train_index_original]

    X_test = working_dataset_df.iloc[test_index]
    y_test = target_variable.iloc[test_index]
    

    y_test_values = y_test.values.ravel()
    y_train_original_values = y_train_original.values.ravel()
  
    # Determining the random seed for the entire experiment (depends on the fold iteration)
    random.seed(fold_iteration) 

    print("------------------------")
    print('Fold:', fold, 'Fold Iteration:', fold_iteration)
        
    # Classification method of choice
    clf = KNeighborsClassifier(n_neighbors = 7)
        
    # Standard Scaling for Original Training and Testing data (must be a different scaler because the training data is different from the validation steps)
    scaler_train_original = StandardScaler()
    scaler_train_original.fit(X_train_original)
    X_train_original_scaled = pd.DataFrame(scaler_train_original.transform(X_train_original))
    X_test_scaled = pd.DataFrame(scaler_train_original.transform(X_test))
    y_test_values = y_test.values.ravel()

    # Single-view assessment
    single_view_model = clf.fit(X_train_original_scaled, y_train_original_values)
    single_view_accuracy = single_view_model.score(X_test_scaled,y_test_values)
    print("Single view accuracy: ",single_view_accuracy)


    # Initializing random splits
    random_split = initializeIndividual(n_genes, n_views)
    # Make sure to assess the objective function with the original training data, along with the test data
    random_mean_accuracy = fitnessFunction(random_split, n_views, X_train_original_scaled, y_train_original, X_test_scaled, y_test, clf)
    random_split_oldOF = oldFitnessFunction(random_split, n_views, X_train_original_scaled, y_train_original, alpha)

    print("Random Split Score (accuracy): ", random_mean_accuracy)

    # There must be a different random state each time the GA is initialized, to be used in the Stratified Shuffle Split (Validation data reshuffling)
    random_shuffle_state = random.randint(1,100)
    print("Random Shuffle State", random_shuffle_state)

    # Initializing optimized view splits    
    individual, best_individual_score, mean_scores, max_scores, std_scores, last_population, mean_rand_scores, std_rand_scores = geneticAlgorithmWithValidation(random_split,
                                                                                                                     n_views,
                                                                                                                    pop_size,
                                                                                                                    num_iter,
                                                                                                                    prob_crossover,
                                                                                                                    prob_mutation,
                                                                                                                    setting="maximize",
                                                                                                                    X_train_original = X_train_original,
                                                                                                                    y_train_original = y_train_original,
                                                                                                                    classifier = clf,
                                                                                                                    number_of_points = number_of_points,
                                                                                                                    random_shuffle_state = random_shuffle_state)
        


    # Make sure to assess the objective function with the original training data, along with the test data
    optimized_split_oldOF = oldFitnessFunction(individual, n_views, X_train_original_scaled, y_train_original, alpha)
    optimized_mean_accuracy = fitnessFunction(individual, n_views, X_train_original_scaled, y_train_original, X_test_scaled, y_test, clf)
        
        
    last_pop_scores = []
    for solution in last_population:
        last_pop_solution = fitnessFunction(solution, n_views, X_train_original_scaled, y_train_original, X_test_scaled, y_test, clf)
        last_pop_scores.append(last_pop_solution)
            
    with open('AGMonitorIntegerAccuracy.txt','a') as file:
        file.write('Dataset: ' + each_file + '\n')
        file.write('Fold: ' +  str(fold) + ' Fold Iteration: ' + str(fold_iteration) + '\n')
        file.write("Objective Function (last population acc on test set): " + str(last_pop_scores) + '\n')
        file.write('Number of Views: ' +  str(n_views) + '\n')
        file.write("Best OF score after halting: " + str(best_individual_score) + '\n')
        file.write("Mean OFs through AG execution: " + str(mean_scores) + '\n')
        file.write("Standard Deviation of OFs through AG execution" + str(std_scores) + "\n")
        file.write("Max OFs through AG execution: " + str(max_scores) + '\n')
        file.write("Mean Rand Scores through AG execution" + str(mean_rand_scores) + "\n")
        file.write("Std Rand Scores through AG execution" + str(std_rand_scores) + "\n")
        file.write('\n')
        file.close()

    print("Mean Accuracies: (Optimized) ", optimized_mean_accuracy, " (Random) ",  random_mean_accuracy)


    random_minKDN_train = minimumTrainKDN(random_split, n_views, X_train_original_scaled, y_train_original)
    opt_minKDN_train = minimumTrainKDN(individual, n_views, X_train_original_scaled, y_train_original)
        
    # Calculate how many instances are easier in the easiest view of the optmimized split
    # rather than in the easiest view of the random split (Train set)
    opt_easier_train = calculateSurpassing(random_minKDN_train, opt_minKDN_train)
        
    random_minKDN_test = minimumTestKDN(random_split, n_views, X_train_original_scaled, y_train_original, X_test_scaled, y_test, clf)
    opt_minKDN_test = minimumTestKDN(individual, n_views, X_train_original_scaled, y_train_original, X_test_scaled, y_test, clf)
                
    # Calculate how many instances are easier in the easiest view of the optmimized split
    # rather than in the easiest view of the random split (Test set, estimated kDN)
    opt_easier_test = calculateSurpassing(random_minKDN_test, opt_minKDN_test)

    mean_kdn_random_train = np.mean(random_minKDN_train)
    mean_kdn_opt_train = np.mean(opt_minKDN_train)

    mean_kdn_random_test = np.mean(random_minKDN_test)
    mean_kdn_opt_test = np.mean(opt_minKDN_test)

    # Writing the results to save later
    with open('foldPerformancesIntegerAccuracy.csv','a') as file:
        data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator = '\n')
        
        data_writer.writerow([folder,
                              each_file,
                  fold,
                  fold_iteration,
                  alpha,
                  n_genes,
                  n_views,
                  n_instances,
                  num_iter,
                  single_view_accuracy,
                  random_mean_accuracy,
                  optimized_mean_accuracy,
                  best_individual_score,
                  random_split_oldOF,
                  optimized_split_oldOF,
                  mean_kdn_random_train,
                mean_kdn_opt_train,
                opt_easier_train,
                mean_kdn_random_test,
                mean_kdn_opt_test,
                opt_easier_test])



    # Training Random Hardness profile
    hardness_dict_random_train = hardnessProfileTrain(random_split, n_views, X_train_original_scaled, y_train_original, alpha)

    # Training Optimized Hardness profile
    hardness_dict_opt_train = hardnessProfileTrain(individual, n_views, X_train_original_scaled, y_train_original, alpha)

    # Testing Random Hardness profile
    hardness_dict_random_test = hardnessProfileTest(random_split, n_views, X_train_original_scaled, y_train_original, X_test_scaled, alpha)

    # Testing Optimized Hardness profile
    hardness_dict_opt_test = hardnessProfileTest(individual, n_views, X_train_original_scaled, y_train_original, X_test_scaled, alpha)

    with open('hardnessDistributionIntegerAccuracy.txt','a') as file:
        file.write('-------------------------- \n')
        file.write('Dataset: ' +  filename + ' Folder ' + folder + '\n')
        file.write('Fold: ' +  str(fold) + ' Fold Iteration: ' + str(fold_iteration) + '\n')
        file.write('Number of Views: ' +  str(n_views) + '\n')

        names = ["Distribution Random Split Train", 
                    "Distribution Optimized Split Train", 
                    "Distribution Random Split Test", 
                    "Distribution Optimized Split Test"]

        for index, item in enumerate([hardness_dict_random_train,
                                        hardness_dict_opt_train,
                                            hardness_dict_random_test,
                                            hardness_dict_opt_test]):
            file.write(names[index] + ' (Total Number of Views: ' +  str(n_views) + ") \n")
            for key in sorted(item.keys()):
                file.write("Easy in " + str(key) + " Views: " + str(item[key]) + " \n")
            file.write(" \n")

        file.write('\n')
        file.close()    

# Folder(s) where the datasets are stored:
folders = ['datasets']

# Mutation and Crossover Rates:
prob_mutation = 0.05
prob_crossover = 0.50

# Generations
num_iter = 50

# Population size
pop_size = 20

# Number of Crossover points
number_of_points = 2

# Instance Hardness Threshold
alpha = 0.5 


# Random State seting
random_state = 15

# Initialize the random state
random.seed(random_state)

fold_iterations = 10

# How many different views are allowed
n_views_possbilities = [5, 10, 20]

#----------------------------------------------

with open('foldPerformancesIntegerAccuracy.csv','a') as file:
    data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,lineterminator = '\n')

    data_writer.writerow(["folder",
                          "file",
                              "fold",
                              "fold_iteration",
                              "alpha",
                              "n_genes",
                              "n_views",
                           "n_instances",
                              "n_generations",
                              "single_view_acc",
                              "random_split_acc",
                              "opt_split_acc",
                              "opt_split_val_acc",
                              "random_split_old_OF",
                              "opt_split_old_OF",
                            "mean_kdn_random_train",
                            "mean_kdn_opt_train",
                            "opt_easier_train",
                            "mean_kdn_random_test",
                            "mean_kdn_opt_test",
                            "opt_easier_test"])
#----------------------------------------------


print("Alpha: ", alpha)

register_list = []

for folder in folders:
    
    files = os.listdir(folder)

    print("Exploring folder", folder)

    print(files)

    datasets_names = [extractFilenameCore(filename) for filename in files]

    for task_id in study.tasks:

        task = openml.tasks.get_task(task_id)
        
        dataset = openml.datasets.get_dataset(task.dataset_id)
    
        if dataset.name in datasets_names:
        
            each_file = appendFilenameCore(dataset.name)
            filename = f'{folder}/{each_file}'

            # Para cada view
            for n_views in n_views_possbilities:
    
                for fold in range(10):
                
                    train_index_original, test_index = task.get_train_test_split_indices(fold = fold)

                    print(fold, dataset.name,"Train splits:", train_index_original[:5],"Test splits:", test_index[:5])
                    for fold_iteration in range(10):
                        register_list.append([filename, fold, fold_iteration, train_index_original, test_index, n_views])

    
    # How many iterations will each fold have to collect learning results from optimized and random splits

    Parallel(n_jobs = 1)(delayed(trainingFolds)(filename,
                                                 fold,
                                                 fold_iteration,
                                                 train_index_original, 
                                                 test_index) for [filename, fold, fold_iteration, train_index_original, test_index, n_views] in register_list)
