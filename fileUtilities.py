import numpy as np
import pandas as pd
import os

def transformToDataFrame(npy_dataset):
    data = np.load(npy_dataset)
    data = pd.DataFrame(data)
    return data

def processCsvToDataFrame(csv_dataset):
    data = pd.read_csv(csv_dataset, header = None)
    print(data.shape)
    X_data, y_data = data.iloc[:,:-1], data.iloc[:,-1:]
    return X_data, y_data

def determineViews(folder):
    views = []
    
    files = os.listdir(folder)

    for file in files:
        filename = f'{folder}/{file}'
        
        if ".npy" not in filename[-5:]: # Checks the file formats
            break

        data = transformToDataFrame(filename)

        # Not class column vector
        if data.shape[1] > 1:
            views.append(data)
        else: # Class column Vector
            target_variable = data
            target_variable.columns = ['tgt']

    return views, target_variable

def concatenateViews(views):
    working_dataset_df = pd.concat([view for view in views],axis = 1)
    working_dataset_df.columns = [int(_) for _ in range(working_dataset_df.shape[1])]
    return working_dataset_df

def output_mean_and_std(list_of_lists):
    mean_list, std_list = [], []
    for item in list_of_lists:
        mean_list.append(np.mean(item))
        std_list.append(np.std(item))
    return mean_list, std_list