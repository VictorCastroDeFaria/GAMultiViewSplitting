from collections import Counter
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, rand_score

import scipy.special

import itertools

from hardnessFunctions import *

def checkFeasibility(solution:list, n_views):
    # For the integer encoding, a feasible solution is the one that has all views
    solution_decoded = solutionDecoder(solution, n_views)
    return hasAllViews(solution_decoded)

def hasAllViews(solution_dict:dict):
    return not np.any([item == [] for item in solution_dict.values()])

# Transforms a solution structure vectors in a dictionary that maps features to views
def solutionDecoder(solution: list, n_views: int):
    # Check the number of different views in the soluction structure:
    solution_dict = {}

    # Create a list for every key (linear complexity):
    for view in range(n_views):
        solution_dict[view] = []

    # Link column indexes to each key (linear complexity):
    for item, index in zip(solution, range(len(solution))):
        solution_dict[item].append(index)
    
    return solution_dict

def kdnPerView(solution: list, n_views: int, dataset, target_variable):
    solution_decoded = solutionDecoder(solution, n_views)
        
    kDN_series = []
    for key in solution_decoded.keys():
        columns = solution_decoded[key]
        X_data = dataset[columns].values
        df_meta_feat, _ = kdn_score(X_data, target_variable,7)

        
        kDN_series.append(df_meta_feat)

    return kDN_series


def oldFitnessFunction(solution: list, n_views: int, dataset, target_variable, alpha:float):
    # If a view is missing:
    if not checkFeasibility(solution, n_views):
        return 0 # Arbitrarily low value
    
    target_variable_values = target_variable.values.ravel()
    
    kdn_per_view = kdnPerView(solution, n_views, dataset, target_variable_values)
    return easyInstancesCount(kdn_per_view, alpha)/dataset.shape[0]


def fitnessFunction(solution: list, n_views, X_train, y_train, X_test, y_test, classifier):

    decoded_splits = solutionDecoder(solution,n_views)

    # If a view is missing:
    if not hasAllViews(decoded_splits):
        return 0 # Arbitrarily low value

    fitness_accuracy, _ = faramarzLearning(X_train, X_test, y_train, y_test, decoded_splits, classifier)
    return fitness_accuracy

def faramarzLearning(X_train, X_test, y_train, y_test, decoded_splits, classifier):
    query_kDN_per_view = []
    outputs_per_view = []
    
    # For each view
    for key in decoded_splits.keys():

        columns = decoded_splits[key]
        X_train_view = X_train[columns].values
        X_test_view = X_test[columns].values
        
        y_train_values = y_train.values.ravel()
        y_test_values = y_test.values.ravel()
        query_estimate_kDN = querySamplekDN(X_train_view, y_train_values, 7, X_test_view)
        query_estimate_kDN = np.round(query_estimate_kDN,2)
        
        query_kDN_per_view.append(query_estimate_kDN)
        
        # Training step - needs to be done to allow access to different models for each instance
        model = classifier.fit(X_train_view, y_train_values)
        outputs_view = model.predict(X_test_view)
        outputs_per_view.append(outputs_view)
       
    dynamically_selected_outputs = []
    each_instance_min_kdn = []

    for instance_index in range(len(outputs_per_view[0])):
        instance_kDN_per_view = [query_kDN_per_view[_][instance_index] for _ in range(len(outputs_per_view))]
        easiest_view_kDN = np.min(instance_kDN_per_view)
        easiest_view_index = instance_kDN_per_view.index(easiest_view_kDN)

        dynamically_selected_outputs.append(outputs_per_view[easiest_view_index][instance_index])
        each_instance_min_kdn.append(easiest_view_kDN)

    faramarz_accuracy = accuracy_score(y_test_values, dynamically_selected_outputs)
    
    return faramarz_accuracy, each_instance_min_kdn

# Calculates how many instances are easy (below an IHM threshold) in at least one view:
def easyInstancesCount(IH_series:list, alpha:float):
    
    easy_instances_mapping = np.array(IH_series) <= alpha

    easy_in_one_or_more_views = np.sum(easy_instances_mapping, axis = 0) > 0

    return np.sum(easy_in_one_or_more_views)

# =============================================================================
# 
# =============================================================================

def minimumTestKDN(solution: list, n_views, X_train, y_train, X_test, y_test, classifier):

    decoded_splits = solutionDecoder(solution, n_views)

    # If a view is missing:
    if not hasAllViews(decoded_splits):
        return [1 for item in range(len(X_test))] # Arbitrarily big value

    _, minimum_test_kdn = faramarzLearning(X_train, X_test, y_train, y_test, decoded_splits, classifier)
    return minimum_test_kdn

def minimumTrainKDN(solution:list, n_views: int, dataset, target_variable):
    target_variable_values = target_variable.values.ravel()
    solution_decoded = solutionDecoder(solution, n_views)
    
    # If a view is missing:
    if not hasAllViews(solution_decoded):
        return [1 for item in range(len(target_variable_values))] # Arbitrarily big value
        
    kDN_series = kdnPerView(solution, n_views, dataset, target_variable_values)

    return np.min(kDN_series, axis = 0)

# How many elements from list_1 are greater than their corresponding elements in list_2
def calculateSurpassing(list_1,list_2):
    assert len(list_1) == len(list_2)
    boolean_vector = np.array(list_1) > np.array(list_2)
    return np.sum(boolean_vector)/len(boolean_vector)

# In how many views each instance is considered to be easy in Training Data
def hardnessProfileTrain(solution: list, n_views: int, dataset, target_variable, alpha:float):

    target_variable_values = target_variable.values.ravel()
    
    kdn_per_view = kdnPerView(solution, n_views, dataset, target_variable_values)
    easy_instances_mapping = np.array(kdn_per_view) <= alpha
    how_many_easy = np.sum(easy_instances_mapping, axis = 0)
    hardness_profile = {}
    for item in how_many_easy:
        if not item in hardness_profile:
            hardness_profile[item] = 1
        else:
            hardness_profile[item] += 1
    return hardness_profile

# In how many views each instance is considered to be easy in Test Data
def hardnessProfileTest(solution: list, n_views: int, X_train, y_train, X_test, alpha:float):

    y_train_values = y_train.values.ravel()

    decoded_splits = solutionDecoder(solution, n_views)
    
    query_kDN_per_view = []

    # For each view
    for key in decoded_splits.keys():

        columns = decoded_splits[key]
        X_train_view = X_train[columns].values
        X_test_view = X_test[columns].values
        
        y_train_values = y_train.values.ravel()

        query_estimate_kDN = querySamplekDN(X_train_view, y_train_values, 7, X_test_view)
        query_estimate_kDN = np.round(query_estimate_kDN,2)

        query_kDN_per_view.append(query_estimate_kDN)

    easy_instances_mapping = np.array(query_kDN_per_view) <= alpha
    how_many_easy = np.sum(easy_instances_mapping, axis = 0)
    hardness_profile = {}
    for item in how_many_easy:
        if not item in hardness_profile:
            hardness_profile[item] = 1
        else:
            hardness_profile[item] += 1
    return hardness_profile


# =============================================================================
# 
# =============================================================================

# Majority voting for a list of predictive outputs
def majority_vote(predictions):
    # Count occurrences of each prediction
    vote_count = Counter(predictions)
    
    # Find the maximum count
    max_count = max(vote_count.values())
    
    # Get predictions with maximum count (in case of ties)
    majority_votes = [prediction for prediction, count in vote_count.items() if count == max_count]
    
    # Return one of the tied predictions randomly
    return random.choice(majority_votes)

# Initialize Solutions and check if they are feasible
def initializeIndividual(n_genes: int, n_views: int):
    flag_feasible = False
    
    while not flag_feasible:
        solution = [random.choice(range(n_views)) for gene in range(n_genes)]
        flag_feasible = checkFeasibility(solution, n_views)
    
    return solution

# Mutation function:
def mutate(individual: list, 
           probability: float, 
           n_views: int):
    # This method is problem-dependent, since mutation dynamics vary with solution structures
    for item in range(len(individual)):
        if random.random() < probability:
            flag_changed = False
            while not flag_changed:
                new_gene = random.choice(range(n_views))
                if new_gene != individual[item]:
                    individual[item] = new_gene
                    flag_changed = True
    return individual

# Creep Mutation function:
def creep_mutation(vector, mutation_rate=0.1):
    vector = np.array(vector)
    direction_vector =  np.zeros(len(vector))

    for _ in range(len(vector)):
        if random.random() < mutation_rate:
            direction_vector[_] = np.random.choice([-1, 1])
    
    # Create a copy of the vector to perform mutations
    mutated_vector = np.array(vector.copy())

    upper_limit = np.max(mutated_vector)
    lower_limit = np.min(mutated_vector) 

    flag_no_more_carryovers = False

    # Apply creep mutation
    while not flag_no_more_carryovers:

        mutated_vector = mutated_vector + direction_vector
        flag_no_more_carryovers = True
        direction_vector =  np.zeros(len(vector))
        if mutated_vector[0] > upper_limit:
            mutated_vector[0] = lower_limit
        if mutated_vector[0] < lower_limit:
            mutated_vector[0] = upper_limit
        
        for index in range(1, len(mutated_vector)):
            # Check the mutation carry-ons (there cannot be an element that is greater than the upper limit or lower than the lower limit at the same time):
            if mutated_vector[index] > upper_limit:
                mutated_vector[index] = lower_limit
                direction_vector[index-1] = 1
                flag_no_more_carryovers = False
            elif mutated_vector[index] < lower_limit:
                mutated_vector[index] = upper_limit
                direction_vector[index-1] = -1
                flag_no_more_carryovers = False
    
    # Change elements for better display
    mutated_vector = [int(item) for item in mutated_vector]
    
    return mutated_vector

# Multiple-point Crossover Operation:
def crossover(parent1: list,
              parent2: list,
             number_of_points: int,
             probability: float):
    child1, child2 = parent1.copy(), parent2.copy()
    if random.random() < probability:
        # Two-point crossover
        assert number_of_points > 0
        crossover_points = random.sample(range(1,len(parent1)-2),number_of_points)
        crossover_points.sort()

        # First crossover (at the zero-indesx point):
        child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:]
        child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:]
        
        for index in range(1, len(crossover_points)):
            child1, child2 = child1[:crossover_points[index]] + child2[crossover_points[index]:], child2[:crossover_points[index]] + child1[crossover_points[index]:]   
    return [child1, child2]

# GA's maximize selection step
def selection_maximize(population: list, 
                       scores: list, 
                       samples_amount:int = 2):
    random_individual_index = random.randint(0,len(population)-1)
    for opponent_index in random.sample(range(len(population)),samples_amount):
        if scores[opponent_index] > scores[random_individual_index]:
            random_individual_index = opponent_index
    return population[random_individual_index]

# GA's minimize selection step
def selection_minimize(population: list, 
                       scores: list, 
                       samples_amount:int = 2):
    random_individual_index = random.randint(0,len(population)-1)
    for opponent_index in random.sample(range(len(population)),samples_amount):
        if scores[opponent_index] < scores[random_individual_index]:
            random_individual_index = opponent_index
    return population[random_individual_index]


def initializePopulation(random_individual, population_size, mutation_rate, n_views):
	population = []
	for _ in range(population_size):
		flag_feasible = False
		while not flag_feasible:
			new_individual = random_individual.copy()
			new_individual = creep_mutation(new_individual, mutation_rate)
			flag_feasible = checkFeasibility(new_individual,n_views)
		population.append(new_individual)
	return population


def scaleTrainValidation(X_train_original, y_train_original, train_index, validation_index):
    # Determining the actual test set and the validation set
    X_train = X_train_original.iloc[train_index]
    y_train = y_train_original.iloc[train_index]
        
    X_validation = X_train_original.iloc[validation_index]
    y_validation = y_train_original.iloc[validation_index]
                    
    scaler_train_validation = StandardScaler()
    scaler_train_validation.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler_train_validation.transform(X_train))
    X_validation_scaled = pd.DataFrame(scaler_train_validation.transform(X_validation))
    
    return X_train_scaled, X_validation_scaled, y_train, y_validation

def find_two_greatest_indexes(numbers):
    if len(numbers) < 2:
        return "List must contain at least two numbers"

    max1_index = 0
    max2_index = 1

    # Flip indexes if list[0] < list[1] (1 should come before 0)
    if numbers[max1_index] < numbers[max2_index]:
        max1_index, max2_index = max2_index, max1_index
    
    for i in range(2, len(numbers)):
        if numbers[i] > numbers[max1_index]:
            max2_index = max1_index
            max1_index = i
        elif numbers[i] > numbers[max2_index]:
            max2_index = i
    return max1_index, max2_index

def find_two_lowest_indexes(numbers):
    if len(numbers) < 2:
        return "List must contain at least two numbers"

    max1_index = 0
    max2_index = 1

    # Flip indexes if list[0] > list[1] (1 should after 0)
    if numbers[max1_index] > numbers[max2_index]:
        max1_index, max2_index = max2_index, max1_index
    
    for i in range(2, len(numbers)):
        if numbers[i] < numbers[max1_index]:
            max2_index = max1_index
            max1_index = i
        elif numbers[i] < numbers[max2_index]:
            max2_index = i
    return max1_index, max2_index

# =============================================================================
# 
# =============================================================================

# Genetic Aglorithm (GA) - Optimizing for accuracy in the validation set
def geneticAlgorithmWithValidation(first_solution:list,
                                   n_views:int,
                     pop_size: int,
                     num_iter: int,
                     prob_crossover: float,
                     prob_mutation: float,
                     setting: str,
                     X_train_original,
                     y_train_original,
                     classifier,
                    number_of_points: int,
                    random_shuffle_state: int):
        
        mean_scores = []
        max_scores = []
        std_scores = []
        mean_rand_scores = []
        std_rand_scores = []
    
        # Building the first generation
        population = initializePopulation(first_solution, pop_size, prob_mutation, n_views)

        
        # Initialize first best score variables 
        if setting == 'minimize':
            best_individual, best_score = population[0], 1000

        elif setting == 'maximize':
            best_individual, best_score = population[0], -1000
        

        # Determining the validation set (This is important to prevent overfitting)
        # This is generated for each GA generation
        strat_shuffle_split = StratifiedShuffleSplit(n_splits=num_iter, test_size=0.3,random_state=random_shuffle_state)


        for generation, (train_index, validation_index) in enumerate(strat_shuffle_split.split(X_train_original, y_train_original)):
            print("Generation %d ----------------------------------------------" % generation)
            
            # Determining the actual test set and the validation set
            X_train_scaled, X_validation_scaled, y_train, y_validation = scaleTrainValidation(X_train_original, y_train_original, train_index, validation_index)
            
            # Calculating fitnesses
            scores = [fitnessFunction(individual, n_views, X_train_scaled, y_train, X_validation_scaled, y_validation, classifier) for individual in population]
            # WARNING: Validation sets change between generations, so fitness functions of a same view split can change in a following generation

            # Obtaining Rand Scores for the population
            simple_rand = 0
            quadratic_rand = 0

            for combination in itertools.combinations(population, 2):
                solution_A, solution_B = list(combination)
                rand_index = rand_score(solution_A, solution_B)
                simple_rand += rand_index
                quadratic_rand += rand_index**2

            combinations_size = scipy.special.comb(pop_size,2)
            
            mean_rand = simple_rand/combinations_size
            std_rand = np.sqrt(quadratic_rand/combinations_size - (simple_rand/combinations_size)**2)
            mean_rand_scores.append(round(mean_rand,3))
            std_rand_scores.append(round(std_rand,4))
            
            # Statistics steps - for monitoring purposes
            print("Mean Score:", round(np.mean(scores),3))
            print("Max Score: ", round(np.max(scores),3))
            print("Std Score: ", round(np.std(scores),4))
            mean_scores.append(float(round(np.mean(scores),3)))
            max_scores.append(float(round(np.max(scores),3)))
            std_scores.append(float(round(np.std(scores),4)))

            
            # Searching for all-time best among the newly found solutions:
            for individual_index in range(pop_size):

                if setting == 'minimize':
                    if scores[individual_index] < best_score:
                        best_individual = population[individual_index]
                        best_score = scores[individual_index]
                        print("Generation %d, new best = %.3f" % (generation,
                        best_score))

                elif setting == 'maximize':
                    if scores[individual_index] > best_score:
                        best_individual = population[individual_index]
                        best_score = scores[individual_index]
                        print("Generation %d, new best = %.3f" % (generation, 
                        best_score))

                else:
                    raise ValueError("You should either input 'maximize' or 'minimize' for argument 'setting'")
            
            # Elitism step - always the two better-performing individuals
            # Tournament step (competition or selection)
            if setting == 'minimize':
                elite_1, elite_2 = find_two_lowest_indexes(scores)
                selected = [selection_minimize(population,scores) for _ in range(pop_size)]
                
            elif setting == 'maximize':
                elite_1, elite_2 = find_two_greatest_indexes(scores) 
                selected = [selection_maximize(population,scores) for _ in range(pop_size)]
            
            print("Elite individuals: ", elite_1, elite_2)
            print("With values: ", round(scores[elite_1],3), round(scores[elite_2],3))
            
            
            offspring = []
            offspring.append(population[elite_1])
            offspring.append(population[elite_2])

            # Offspring step
            if generation < num_iter-1: # Only if it is not the final generation
                while len(offspring) < len (population):
                    parent_index_1, parent_index_2 = random.sample(range(len(population)),2)
                    [parent1, parent2] = selected[parent_index_1], selected[parent_index_2]
                    for child in crossover(parent1, parent2, number_of_points, prob_crossover):

                        # While generating offspring, mutate each individual with a probability
                        child = creep_mutation(child, prob_mutation)
                        offspring.append(child)

                population = offspring
    
        return best_individual, best_score, mean_scores, max_scores, std_scores, population, mean_rand_scores, std_rand_scores


# Genetic Algorithm (GA) - Optimizing for amount of easy instances
def geneticAlgorithm(first_solution:list, 
                     alpha:int,
                     n_views: int,
                     pop_size: int,
                     num_iter: int,
                     prob_crossover: float,
                     prob_mutation: float,
                     setting: str,
                     X_train_original,
                     y_train_original,
                    number_of_points: int):
                        
        
        
        mean_scores = []
        max_scores = []
        std_scores = []
    
        # Building the first generation
        population = initializePopulation(first_solution, pop_size, prob_mutation, n_views)

        # Initialize first best score variables 
        if setting == 'minimize':
            best_individual, best_score = population[0], 1000

        elif setting == 'maximize':
            best_individual, best_score = population[0], -1000

        for generation in range(num_iter):
            print("Generation %d ----------------------------------------------" % generation)
            
            # Calculating fitnesses
            scores = [oldFitnessFunction(individual, n_views, X_train_original, y_train_original, alpha) for individual in population]

            
            # Statistics steps - for monitoring purposes
            print("Mean Score:", round(np.mean(scores),3))
            print("Max Score: ", round(np.max(scores),3))
            print("Std Score: ", round(np.std(scores),4))
            mean_scores.append(float(round(np.mean(scores),3)))
            max_scores.append(float(round(np.max(scores),3)))
            std_scores.append(float(round(np.std(scores),4)))
            
            # Searching for all-time best among the newly found solutions:
            for individual_index in range(pop_size):

                if setting == 'minimize':
                    if scores[individual_index] < best_score:
                        best_individual = population[individual_index]
                        best_score = scores[individual_index]
                        print("Generation %d, new best = %.3f" % (generation,
                        best_score))

                elif setting == 'maximize':
                    if scores[individual_index] > best_score:
                        best_individual = population[individual_index]
                        best_score = scores[individual_index]
                        print("Generation %d, new best = %.3f" % (generation, 
                        best_score))

                else:
                    raise ValueError("You should either input 'maximize' or 'minimize' for argument 'setting'")

            # Elitism step - always the two better-performing individuals
            # Tournament step (competition or selection)
            if setting == 'minimize':
                elite_1, elite_2 = find_two_lowest_indexes(scores)
                selected = [selection_minimize(population,scores) for _ in range(pop_size)]
                
            elif setting == 'maximize':
                elite_1, elite_2 = find_two_greatest_indexes(scores) 
                selected = [selection_maximize(population,scores) for _ in range(pop_size)]
            
            print("Elite individuals: ", elite_1, elite_2)
            print("With values: ", round(scores[elite_1],3), round(scores[elite_2],3))
            
            
            offspring = []
            offspring.append(population[elite_1])
            offspring.append(population[elite_2])

            # Offspring step
            if generation < num_iter-1: # Only if it is not the final generation
                while len(offspring) < len (population):

                    parent_index_1, parent_index_2 = random.sample(range(len(population)),2)
                    [parent1, parent2] = selected[parent_index_1], selected[parent_index_2]
                    for child in crossover(parent1, parent2, number_of_points, prob_crossover):
                        # While generating offspring, mutate each individual with a probability
                        child = creep_mutation(child, prob_mutation)
                        offspring.append(child)

                population = offspring

    
        return best_individual, best_score, mean_scores, max_scores, std_scores, population