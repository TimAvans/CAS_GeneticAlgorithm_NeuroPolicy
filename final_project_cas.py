'''
Assignment:
Deep Reinforcement Learning via Evolution
Investigate to what extend evolutionary methods, such as genetic algorithms,
the Cross Entropy Method or Evolution Strategies, can be used to optimize a
neural policy.
'''

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from concurrent.futures import ProcessPoolExecutor

'''
Simple implementation: https://medium.com/@yauheniya.ai/solving-the-lunar-lander-with-genetic-algorithms-ef80c1376cfa
Generational genetic algorithm: https://danyagordin.com/static/media/dqn.9579cf0e6e41fa46cf8c.pdf
environment used: https://www.gymlibrary.dev/
General information: https://en.wikipedia.org/wiki/Evolutionary_algorithm

Maybe compare the results of the simple and a true neural policy?

Use package openai gym for the environment
Objective:
Invesitgate how genetic algorithms can be used to optimize a neural policy.
Metrics:
Converging speed, Efficiency (amount of episodes needed to train),
Performance (average reward of the final policy over fixed nuber of evaluation episodes)
Baseline?:
We might need a baseline to compare against
(simple: maybe just a random policy to show improvement)
(harder: gradient method: reinforce or deep q-learning)
Parameters:
Population size, mutation rate, crossover-prob
Complete procedure (the complete experiment):
Initialization -> randomly init wieghts
evaluation -> run the starting policy in the chosen environment for baseline
iterate ->
GeneticAlgorithm:
Use a selection to choose only the top percentage of high-performing policies
Use a crossover to combine parameters of the selected top policies
Add a random noise to mutate the "offspring" policies

evaluate more-> evaluate the policy every e.g. 5 iterations for results

use multiple seeds (test multiple) for reproducibility and testing more accurately
'''

'''
LunarLander actions:
0: do nothing,
1: fire left orientation engine,
2: fire main engine,
3: fire right orientation engine
'''

class NeuralPolicyModel(nn.Module):
  '''
  This is a simple model that will function as policy.
  The input size will be 8 for the LunarLander since it's observation vector is 8 long
  the output size will be 4 since there are 4 possible actions
  '''
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralPolicyModel, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


def initialize_initial_population(population_size, model):
  '''
  This function initializes the population.
  Determines the genome length by summing the total number of trainable parameters in the policy model.
  :param population_size: The number of policies in the population.
  :param model: The neural network model.
  :return: A 2d numpy array (population_size, genome_length) of random genomes, where each row is a genome.
  '''
  #The length of the genome is the sum of all trainable model parameters
  length = sum(p.numel() for p in model.parameters())
  #Generate random genomes of the length determined before and with weights between -1 and 1
  genomes = np.random.uniform(low=-1.0, high=1.0, size=(population_size, length))
  return genomes

def genome_to_policymodel(genome, model):
  '''
  Take a genome used to initialize the population and put it into the model.
  This function directly alters the model parameters as a reference.
  :param genome: The 1d array representing the genome of a policy.
  :param model: The model representing the policy.
  '''
  #Cast the genome to a torch tensor
  genome_as_tensor = torch.tensor(genome)
  #Set the genome_tensor as the model parameter weights
  torch.nn.utils.vector_to_parameters(genome_as_tensor, model.parameters())

def kpoint_crossover(parent1, parent2, k):
  '''
  Perform k-point crossover between two parents.
  :param parent1: The first parent.
  :param parent2: The second parent.
  :param k: The number of crossover points.
  :return: The offspring genome.
  '''
  if k < 2:
    raise ValueError("k must be greater than 1")
  if len(parent1) != len(parent2):
    raise ValueError("Parents must have the same length")

  crossover_points = np.random.choice(len(parent1), k, replace=False)
  crossover_points.sort()

  offspring = []
  start = 0
  #Loop through the randomly generated crossover points
  for i, point in enumerate(crossover_points):
    #If the k-value index is an even number take from parent 1
    if i % 2 == 0:
      offspring.extend(parent1[start:point])
    #If it is an uneven number take from parent 2
    else:
      offspring.extend(parent2[start:point])
    #Reset the starting point to the end point
    start = point

  #Add the end part of the appropriate parent
  if len(crossover_points) % 2 == 0:
    offspring.extend(parent1[start:])
  else:
    offspring.extend(parent2[start:])

  return np.array(offspring)

def singlepoint_crossover(parent1, parent2):
  '''
  Perform single-point crossover between two parents.
  :param parent1: The first parent.
  :param parent2: The second parent.
  :return: The offspring genome.
  '''
  if len(parent1) != len(parent2):
    raise ValueError("Parents must have the same length")

  #Determine the random crossover point between 1 and the length of the array
  crossover_point = np.random.randint(1, len(parent1))

  #Get the 2 sub-arrays from the parents
  p1_parent1 = parent1[:crossover_point]
  p2_parent2 = parent2[crossover_point:]

  #Concatenate them and return the offspring array
  return np.concatenate((p1_parent1, p2_parent2))

def mutate(genome, rate, scale):
  '''
  Mutate the genome with a given rate.
  :param genome: The genome to mutate.
  :param rate: The mutation rate.
  :param scale: The mutation scale (intensity or magnitude).
  :return: The mutated genome.
  '''
  #Determine a mutation mask using the mutation rate
  mask = np.random.rand(len(genome)) < rate
  #Use the mutation mask to alter the genome by a normal distribution where scale is the mutation scale
  genome[mask] += np.random.normal(scale=scale, size=np.sum(mask))
  return genome


def selection(population, fitness, n_selected):
  '''
  Select the top n_selected individuals from the population based on their fitness.
  :param population: The population to select from.
  :param fitness: The fitness of each individual in the population.
  :param n_selected: The number of individuals to select.
  :return: The selected individuals.
  '''
  #Get the indices from the fitness array values high to low
  indices = np.argsort(fitness)[::-1]
  #Return from the population n_selected genomes with high fitness scores
  return population[indices[:n_selected]]

def evaluate(env, genome, model, n_episodes=5):
  '''
  Evaluate the performance of a specific genome in the environment
  :param env: The environment to evaluate the genome in.
  :param genome: The genome to evaluate.
  :param model: The model representing the policy.
  :param n_episodes: The number of episodes to evaluate.
  :return: The average reward of the policy over the evaluation episodes.
  '''

  #Set the given genome as the parameter weights in the given model
  genome_to_policymodel(genome, model)
  total_r = 0 #Total reward set to 0

  #Loop for the length of the evaluation
  for _ in range(n_episodes):
    #Get the first observation
    observation = env.reset()
    observation = observation[0]
    while True:
      #Cast the observation as a tensor
      observation_tensor = torch.tensor(observation, dtype=torch.double)
      #Get the actions and their probabilities
      actions = model(observation_tensor)
      #Pick the most probable/best action
      action = torch.argmax(actions).item()
      #Perform the best action
      observation, reward, terminated, truncated, info = env.step(action)
      #Add the reward
      total_r += reward
      #Check if simulation is finished
      if terminated or truncated:
        break
  #Return the average reward
  return total_r / n_episodes

def evaluate_worker(genome, n_episodes):
  #Each process creates its own environment
  temp_env = gym.make('LunarLander-v3')

  #Each process also has to make its own model
  model = NeuralPolicyModel(8, 16, 4)
  genome_to_policymodel(genome, model)
  
  fitness = evaluate(temp_env, genome, model, n_episodes)
  temp_env.close()  # Clean up the environment

  return fitness
  
def evaluate_population_parallel(population, n_episodes = 5, n_workers = 5):
  '''
  '''
  with ProcessPoolExecutor(max_workers=n_workers) as executor:
        fitness_scores = list(executor.map(evaluate_worker, population, [n_episodes] * len(population)))
  return np.array(fitness_scores)

# Training
def genetic_algorithm(env, model, population_size, generations, mutation_rate, mutation_scale, n_eval_episodes = 15, n_top_genomes_selected = 15, crossover_mode = 'singlepoint', n_kpoints = 3, RUN_ID = None, n_parallelization_workers = 10):
  '''
  '''
  if crossover_mode not in ['singlepoint', 'kpoint']:
    raise ValueError("crossover mode must be \'singlepoint\' or \'kpoint\'")
  
  #Multiple lists for plotting insights after training
  best_fitness = []
  average_fitness = []

  #Initial population (random genomes)
  population = initialize_initial_population(population_size=population_size, model=model)
  #Use elitism to keep the best performing genome over the generations
  elite_genome = (None, -float('inf'))

  #Training loop
  for generation in range(generations):
    #Determine the fitness scores for the entire population
    fitness_scores = []
    # for genome in population:
    #   fitness_scores.append(evaluate(env=env, genome=genome, model=model, n_episodes=15))
    fitness_scores = evaluate_population_parallel(population, n_eval_episodes, n_parallelization_workers)

    fitness_scores = np.array(fitness_scores)

    #Preserver statistics for plotting
    best_fitness.append(np.max(fitness_scores))
    average_fitness.append(np.mean(fitness_scores))

    #Use selection to select the top genomes of the population
    top_genomes = selection(population=population, fitness=fitness_scores, n_selected=n_top_genomes_selected)

    #Select the elite genome we want to carry over to the next generation and save until a better one emerges
    elite_index = np.argmax(fitness_scores)
    if fitness_scores[elite_index] > elite_genome[1]:
      elite_genome = (population[elite_index], fitness_scores[elite_index])

    print(f"{crossover_mode} {RUN_ID} Generation {generation}, Best Fitness: {fitness_scores[elite_index]}")

    #Generate the next population using crossover, mutation and top selection
    next_population = []
    while len(next_population) < population_size -1:
      idx_parent1, idx_parent2 = np.random.choice(len(top_genomes), 2, replace=False)
      genome_parent1 = top_genomes[idx_parent1]
      genome_parent2 = top_genomes[idx_parent2]

      if crossover_mode == 'singlepoint':
        child_genome = singlepoint_crossover(genome_parent1, genome_parent2)
      elif crossover_mode == 'kpoint':
        child_genome = kpoint_crossover(genome_parent1, genome_parent2, n_kpoints)

      mutated_child_genome = mutate(child_genome, mutation_rate, mutation_scale)
      next_population.append(mutated_child_genome)

    next_population.append(elite_genome[0])
    population = np.array(next_population)
  return {'elite_genome': elite_genome, 'best_fitness': best_fitness, 'average_fitness': average_fitness}

if __name__ == "__main__":
  RUN_ID = None
  RUN = ['kpoint', 'singlepoint']
  population_size = 100
  generations = 200
  mutation_rate = 0.15
  mutation_scale = 0.75
  n_eval_episodes = 25
  n_kpoints = 3
  n_top_genomes = int(population_size*0.2)
  for i in range(2):
    if RUN_ID is None:
      RUN_ID = "RUN_" + str(np.random.randint(1, np.iinfo(np.int32).max))

    save_folder = RUN[i] + '/' + RUN_ID + "/"
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
      os.makedirs(save_folder + 'videos/')

    #The initialization of the environment
    env = gym.make('LunarLander-v3')
    observation, info = env.reset()

    with open(save_folder + RUN_ID + "_settings.txt", "w") as file:
        file.write(f"population_size = {population_size} \n generations = {generations} \n  mutation_rate = {mutation_rate} \n mutation_scale={mutation_scale}\n number_evaluation_episodes={n_eval_episodes}\n number_kpoints={n_kpoints} \n number_top_genomes = {n_top_genomes}")

    model = NeuralPolicyModel(8, 16, 4)
    torch.save(model.state_dict(), save_folder + "untrained_" + RUN_ID)

    results = genetic_algorithm(env, model, population_size, generations, mutation_rate, mutation_scale, n_top_genomes_selected=n_top_genomes, n_eval_episodes = n_eval_episodes, n_kpoints=n_kpoints, crossover_mode = RUN[i], RUN_ID = RUN_ID)
    genome = results['elite_genome']
    genome_to_policymodel(genome[0], model)

    torch.save(model.state_dict(), save_folder + "trained_" + RUN_ID)

    #Plot and save the graph for insights in the training
    plt.figure()
    plt.plot(results["best_fitness"], label=f"{RUN[i]} best fitness")
    plt.plot(results["average_fitness"], label=f"{RUN[i]} average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness progress over generations")
    plt.legend()
    plt.savefig(save_folder + RUN_ID + "training_fitness_plot_" + RUN_ID + '.png', bbox_inches='tight')

    # Evaluation
    env_eval = gym.make('LunarLander-v3', render_mode='rgb_array')
    env_video = gym.wrappers.RecordVideo(env_eval, video_folder=save_folder+"videos/")
    obs, info  = env_video.reset(seed=42)
    model.eval()

    for _ in range(500):
      observation_tensor = torch.tensor(observation, dtype=torch.double)
      actions = model(observation_tensor)
      action = torch.argmax(actions).item()
      observation, reward, terminated, truncated, info = env_video.step(action)  

      if terminated or truncated:
        obs, info  = env_video.reset(seed=42)
    
    env_video.close()