##
# Tries to find a mapping between Tangut character components and phonetics using genetic algorithms
#

# Directory for the .pkl files
PHON_DIR = ""


import codecs
import sys
import lib_text_utils
import numpy as np
from deap import base
from deap import creator
from deap import tools
import re
import random
import datetime
import pickle
from collections import Counter
import itertools
import math

def jaccard_similarity(set1, set2):
  """
  Calculates the Jaccard similarity between two sets
  """
  firstSet = set(set1)
  secondSet = set(set2)
  noUnion = len(firstSet.union(secondSet))
  noInt = len(firstSet.intersection(secondSet))
  sim = 1 - float(noInt) / noUnion
  return sim


def levenshtein(s, t):
  """
  Calculates the Levenshtein distance between two strings s and t
  """
  n = len(s)
  m = len(t)
  if n==0:
    return m
  if m==0:
    return n
  v0 = range(m+1)
  for i in range(1,n+1):
    v1 = [0] * (m+1)
    v1[0] = i
    for j in range(1,m+1):
      cost = 1
      if s[i-1] == t[j-1]:
        cost = 0
      v1[j] = min(v1[j-1]+1, v0[j]+1, v0[j-1]+cost)
    v0 = v1
  return v1[m]


"""
Calculates the levenshtein distance after sorting the characters in the string
This measure is concerned with the presence/absence of characters and their multiplicity,
but not their order
"""
def sorted_levenshtein(s, t):
  s1 = sorted(s[:])
  t1 = sorted(t[:])
  return levenshtein(s1, t1)


"""
Extracts phonetic information to a text file
phon - A dictionary of lists as produced by extract_phonetic_info
"""
def phon_info_to_txt(phon):
  outFile = open("/media/alan/DATAPART1/Data/Ed/Lang/Xixia/PhD/Phonetics/XixiaPhon.txt", "w")
  outFile.write("Lfw;RecNum;PhonClass;Phonetics\n");
  for charCntr in range(len(phon["Lfw"])):
    outFile.write(str(phon["Lfw"][charCntr]) + ";" + phon["RecNum"][charCntr] + ";" + phon["PhonClass"][charCntr] + ";" + phon["Phonetics"][charCntr] + "\n");
  outFile.close();

"""
Calculates the predicted phonetics for a character given the recursive code (as a list) and the recursive code/phonetic mapping
Returns the predicted phonetics as a list
"""
def phon_predict(recurs_code, rec_phon_map):
  pred = [rec_phon_map[x-1] for x in recurs_code]
  return filter(None, pred)

"""
Calculates the fitness of a mapping between recursive radicals and phonetics 
"""
def phon_map_fitness(rec_codes, phon_codes, saved_fitness, rec_phon_map):
  # Calculate the predicted phonetics for each character given the mapping
  pred = [phon_predict(x, rec_phon_map) for x in rec_codes]

  # Use an average relative Levenshtein measure, i.e. average of (Levenshtein / length of dictionary phonetic code)
  # Raw Levenshtein:
  fitness = [levenshtein(act, est) for act, est in zip(phon_codes, pred) if act != ""]
  # Relative Levenshtein:
  # fitness = [levenshtein(act, est) / float(len(act)) for act, est in zip(phon_codes, pred)]
  return np.mean(fitness)


"""
Calculates the fitness of a mapping between recursive radicals and phonetics
Uses a sorted Levenshtein measure - reading order doesn't matter 
"""
def phon_map_fitness_sorted(rec_codes, phon_codes, saved_fitness, rec_phon_map):
  pred = [phon_predict(x, rec_phon_map) for x in rec_codes]
  fitness = [sorted_levenshtein(act, est) for act, est in zip(phon_codes, pred) if act != ""]
  return np.mean(fitness)


"""
Mutate a recursive code - phonetic map by swapping two elements
Will keep trying until get a different mapping to the original
"""
def phon_mutation_swap(rec_phon_map):
  # Work out the elements to swap - keep going until find two different elements
  swapPos = random.sample(range(len(rec_phon_map)), 2)
  while rec_phon_map[swapPos[0]] == rec_phon_map[swapPos[1]]:
    swapPos = random.sample(range(len(rec_phon_map)), 2)
  # Swap the elements
  swap = rec_phon_map[swapPos[0]]
  rec_phon_map[swapPos[0]] = rec_phon_map[swapPos[1]]
  rec_phon_map[swapPos[1]] = swap


"""
Mutates a recursive code by replacing the mapping for an element at random
"""
def phon_mutation_replace(phon_elements, prob_non_phon, rec_phon_map):
  replPos = random.randrange(len(rec_phon_map))
  if random.random() <= prob_non_phon:
    newPhon = ""
  else:
    newPhon = random.choice(phon_elements)
  rec_phon_map[replPos] = newPhon


"""
Mutates a recursive code by mutating each element with a given probability
This version doesn't create a deep copy - mutates the original
"""
def phon_mutation_replace_all(phon_elements, mutate_prob, prob_non_phon, rec_phon_map):
  for pos in range(len(rec_phon_map)):
    if random.random() <= mutate_prob:
      if random.random() <= prob_non_phon:
        rec_phon_map[pos] = ""
      else:
        rec_phon_map[pos] = random.choice(phon_elements)


"""
Finds the best single mutation for a given mapping
"""
def find_best_single_mutation(phon_elements, rec_phon_map, toolbox):
  bestFitness = toolbox.evaluate(rec_phon_map)
  bestMap = rec_phon_map
  foundBetter = False
  for i in range(len(rec_phon_map)):
    for phon in phon_elements:
      mutPhonMap = rec_phon_map[:]
      mutPhonMap[i] = phon
      mutFitness = toolbox.evaluate(mutPhonMap)
      if mutFitness < bestFitness:
        bestMap = mutPhonMap
        bestFitness = mutFitness
        foundBetter = True
  return (bestMap, bestFitness, foundBetter)
      


"""
Performs a crossover (breeding) between two recursive code - phonetic maps
Allows for the case where the same element codes for two different items in the two maps
Start with half of map 1, try to merge in other half from map 2, randomly allocate non-merged items to empty positions
Choose half to keep from map 1 at random
"""
def phon_crossover(map1, map2):
  map1Vals = map1[:]
  map2Vals = map2[:]
  map1Pos = [i for i,x in enumerate(map1) if x != ""]
  noKeepMap1 = len(map1Pos) / 2
  map1Keep = random.sample(map1Pos, noKeepMap1)
  # Positions and values to copy from map1 to map2 
  map1Copy = list(set(map1Pos).difference(set(map1Keep)))
  map1CopyVals = [x for i,x in enumerate(map1Vals) if i in map1Copy]
  # Positions to copy from map2 to map1
  # Values will be the same as map1CopyVals
  map2Copy = [i for i,x in enumerate(map2Vals) if x in map1CopyVals]

  # Set all non-kept map1 values to blank
  for i in range(len(map1)):
    if i not in map1Keep:
      map1[i] = ""
  # Set values in map2 to blank where they are to be copied from map1 into new positions
  for i, x in enumerate(map2):
    if x in map1CopyVals:
      map2[i] = ""

  # Copy values from map1 to map2 if possible
  map2NotInserted = []
  for i in map1Copy:
    if map2[i] == "":
      map2[i] = map1Vals[i]
    else:
      map2NotInserted.append(map1Vals[i])
  # Copy values from map2 to map1 if possible
  map1NotInserted = []
  for i in map2Copy:
    if map1[i] == "":
      map1[i] = map2Vals[i]
    else:
      map1NotInserted.append(map2Vals[i])

  # Randomly allocate the items that were not inserted due to position clashes
  if len(map1NotInserted) > 0:
    insPos = random.sample([i for i,x in enumerate(map1) if x == ""], len(map1NotInserted))
    random.shuffle(insPos)
    for i,x in zip(insPos, map1NotInserted):
      map1[i] = x
  if len(map2NotInserted) > 0:
    insPos = random.sample([i for i,x in enumerate(map2) if x == ""], len(map2NotInserted))
    random.shuffle(insPos)
    for i,x in zip(insPos, map2NotInserted):
      map2[i] = x 


# Performs a crossover without worrying about duplicate mappings
# Just takes half the sounds from one mapping (at random) and half from the other
# THIS VERSION DOES NOT USE DEEP COPIES
def phon_crossover_norestrict2(map1, map2):
  cpMap1 = map1[:]
  noParts = len(map1)
  noKeepMap1 = noParts / 2
  keepPos = random.sample(range(noParts), noKeepMap1)
  # Update map1
  for pos in range(noParts):
    if pos not in keepPos:
      map1[pos] = map2[pos]
  # Update map2
  for pos in range(noParts):
    if pos not in keepPos:
      map2[pos] = cpMap1[pos]




"""
Builds a random mapping between character parts and phonetics
This is an individual in the genetic algorithm
In this version each phonetic value maps to only one character part
"""
def phon_mapping_uniq(noCharParts, phonList):
  recPhonMap = [""] * noCharParts
  noPhonCodes = len(phonList)
  positions = random.sample(range(noCharParts), noPhonCodes)
  for pos, phon in zip(positions, phonList):
    recPhonMap[pos] = phon
  return recPhonMap

"""
Builds a random mapping between character parts and phonetics
This is an individual in the genetic algorithm
In this version each phonetic value can map to more than one character part
probNonPhon - The probability that a given part won't map to a phonetic value
"""
def phon_mapping_random(noCharParts, phonList, probNonPhon):
  recPhonMap = [""] * noCharParts
  for pos in range(len(recPhonMap)):
    if random.random() > probNonPhon:
      recPhonMap[pos] = random.choice(phonList)
  return recPhonMap


"""
Simulates character phonetics from a given recursive radical-phonetic component mapping

Has non-coding components and a given error rate where a mapping will map to a different component at random
Use this to test how well genetic algorithms can pick up structure
"""
def simulate_phon_mapping(mapping, rec_codes, phon_codes, error):
  phonSim = []
  for rec_code in rec_codes:
    pred = [mapping[x-1] if random.random() > error else random.choice(phon_codes) for x in rec_code]
    pred = filter(None, pred)
    # Add in a single random sound if there is no phonetics left for this character
    if len(pred) == 0:
      pred = [random.choice(phon_codes)]
    phonSim.append(pred)
  return phonSim

"""
Loads an object from a file using pickle
"""
def load_pickle_file(file_name):
  loadFile = open(file_name, 'rb')
  obj = pickle.load(loadFile)
  return obj


"""
Calculates the frequencies of items in a list
"""
def calc_frequencies(in_list):
  counts = Counter(in_list)
  counts = [(k,v) for k,v in counts.iteritems()]
  counts = sorted(counts, key=lambda x: x[0])
  keys = [x[0] for x in counts]
  values = [x[1] for x in counts]
  sumVals = sum(values)
  values = [x / float(sumVals) for x in values]
  return (keys, values)


"""
Calculates the entropy of the elements in a list
"""
def calc_entropy(in_list):
  keys, freqs = calc_frequencies(in_list)
  entropy = -sum([p * math.log(p) for p in freqs])
  return entropy


"""
Calculates the conditional entropy of a list of lists
"""
def cond_entropy(in_list):
  fullList = list(itertools.chain(*in_list))
  items = list(set(fullList))
  items = sorted(items)
  noElems = len(items)
  itemDict = {items[x]: x for x in range(noElems)}

  # Calculate the conditional counts - the number of times each element follows a specified element
  pairCounts = np.zeros((noElems, noElems))
  for elem in in_list:
    for cntr in range(len(elem)-1):
      pos1 = itemDict[elem[cntr]]
      pos2 = itemDict[elem[cntr+1]]
      pairCounts[pos1, pos2] = pairCounts[pos1, pos2] + 1

  # Calculate the conditional probabilities - the probability of a given element given a specified previous element
  rowSums = sum(pairCounts.T)
  pairProbs = np.zeros((noElems, noElems))
  # Probably a more streamlined way to do this:
  for row in range(noElems):
    if rowSums[row] > 0:
      pairProbs[row,:] = pairCounts[row,:] / float(rowSums[row])
    
  # Calculate the individual item (unconditional) probabilities
  keys, singleProbs = calc_frequencies(fullList)
  # Both items and keys should match up as sorted, but check this
  assert keys == items, "cond_entropy - keys don't match"
  
  # Calculate the conditional entropy
  condEnt = np.array(pairProbs)
  for row in range(noElems):
    for col in range(noElems):
      if condEnt[row,col] > 0:
        condEnt[row,col] = condEnt[row,col] * math.log(condEnt[row,col])
  rowSums = sum(condEnt.T)
  condEntropy = - sum([x * y for x,y in zip(rowSums, singleProbs)])
  return condEntropy


"""
Converts a list of lists to a 2D NP array of indicator variables
"""
def list_of_list_to_ind(in_list):
  # Determine the unique set of elements in the list of lists
  elems = set([])
  for subList in in_list:
    elems = elems.union(set(subList))
  elems = list(elems)
  # Calculate the indicator vector for each sublist
  arrIndic = np.array([])
  for subList in in_list:
    setSubList = set(subList)
    subIndic = np.array([1 if x in setSubList else 0 for x in elems])
    if len(arrIndic) == 0:
      arrIndic = subIndic
    else:
      arrIndic = np.vstack([arrIndic, subIndic])
  return arrIndic.T
  
      
  

NO_REC_CODES = 176
SIMULATION = False
SIMULATION_ERROR = 0.6
ANY_READING_ORDER = False  # Set to True to use a sorted Levenshtein fitness measure
CALC_CORNER_FREQUENCIES = False

# Load objects from pickle files
chars = load_pickle_file(PHON_DIR + "chars.pkl")
phon = load_pickle_file(PHON_DIR + "phon.pkl")
phonCodes = load_pickle_file(PHON_DIR + "phonCodes.pkl")
phonCodeLists = load_pickle_file(PHON_DIR + "phonCodeLists.pkl")

# Work out the unique set of phonetic items
phonCodeUniq = reduce(lambda x,y: x.union(y), phonCodeLists, set())
phonCodeUniq = list(phonCodeUniq)
noPhonCodes = len(phonCodeUniq)
recCodeLists = [re.split("[\{\[\(\]\}\),]*", x["RecNum"]) for x in chars]
recCodeLists = [map(int, filter(None,x)) for x in recCodeLists]
revRecCodes = recCodeLists[:]
tmp = [x.reverse() for x in revRecCodes]  # Reversing occurs in-place - tmp is a dummy variable
flatRecCodes = list(itertools.chain(*recCodeLists))
recCodeKeys, recCodeFreqs = calc_frequencies(flatRecCodes)
noRecCodes = len(flatRecCodes)
recCodeCounts = recCodeFreqs[:]
recCodeCounts = map(lambda x: x * noRecCodes, recCodeCounts)
recEntropy = calc_entropy(flatRecCodes)
print("The entropy is: ", recEntropy)
condEntropy = cond_entropy(recCodeLists)
print("The conditional entropy is: ", condEntropy)



fitness = []
savedFitness = {}
for iter in range(1):
  # Setup a random mapping of parts to phonetics
  #recPhonMap = phon_mapping_random(NO_REC_CODES, phonCodeUniq, 0.5)
  recPhonMap = phon_mapping_uniq(NO_REC_CODES, phonCodeUniq)
  # Calculate the fitness of this random mapping
  fitness.append(phon_map_fitness(recCodeLists, phonCodeLists, savedFitness, recPhonMap))
print fitness
print min(fitness)
print recPhonMap
#print phon_mutation_swap(recPhonMap)
print phon_mutation_replace_all(phonCodeUniq, 0.2, 0.5, recPhonMap)
# print phon_crossover(["a","b","c","d","e"], ["e","d","c","b","a"])

# Simulate a phonetic mapping
if SIMULATION:
  simMap = simulate_phon_mapping(recPhonMap, recCodeLists, phonCodeUniq, SIMULATION_ERROR)
  # REPLACE THE ACTUAL PHONETICS WITH THE SIMULATED PHONETICS TO TEST GA
  phonCodeLists = simMap

"""
Initialises an individual with a random mapping
"""
def indiv_init_mapping(icls, no_char_parts, phon_list):
  ind = icls(phon_mapping_random(no_char_parts, phon_list, 0.5))
  return ind

# Set up Deap objects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("individual", indiv_init_mapping, creator.Individual, NO_REC_CODES, phonCodeUniq)
#toolbox.register("individual", tools.initRepeat, creator.Individual, 
#    toolbox.attr_bool, 200) # Originally 100
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

CXPB, MUTPB, NGEN = 0.5, 0.2, 100

# Operators
if ANY_READING_ORDER:
  toolbox.register("evaluate", phon_map_fitness_sorted, recCodeLists, phonCodeLists, savedFitness)  # Reverse is revRecCodes
else:
  toolbox.register("evaluate", phon_map_fitness, recCodeLists, phonCodeLists, savedFitness)  # Reverse is revRecCodes
toolbox.register("mate", phon_crossover_norestrict2)
#toolbox.register("mate", phon_crossover)
toolbox.register("mutate", phon_mutation_replace_all, phonCodeUniq, 0.1, 0.5)
#toolbox.register("mutate", phon_mutation_swap)
toolbox.register("select", tools.selTournament, tournsize=10)

pop = toolbox.population(n=2000)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
  ind.fitness.values = (fit,)
bestFitness = [] # Stores the best fitness value for each generation

for g in range(NGEN):
  print("-- Generation %i --" % g)
  print("savedFitness no. of entries: ", len(savedFitness))

  # After the 30th generation, change to just mutating one element at a time
  if NGEN > 30:
    toolbox.register("mutate", phon_mutation_replace, phonCodeUniq, 0.5)
        
  # Select the next generation individuals
  offspring = toolbox.select(pop, len(pop))
  # Clone the selected individuals
  offspring = list(map(toolbox.clone, offspring))

  # Apply crossover and mutation on the offspring
  for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
      toolbox.mate(child1, child2)
      del child1.fitness.values
      del child2.fitness.values

  for mutant in offspring:
    if random.random() < MUTPB:
      toolbox.mutate(mutant)
      del mutant.fitness.values

  # Evaluate the individuals with an invalid fitness
  print("Starting fitness evaluation")
  invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
  fitnesses = map(toolbox.evaluate, invalid_ind)
  for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = (fit,)
  print("  Evaluated %i individuals" % len(invalid_ind))

  # The population is entirely replaced by the offspring
  pop[:] = offspring
        
  # Gather all the fitnesses in one list and print the stats
  fits = [ind.fitness.values[0] for ind in pop]
        
  length = len(pop)
  mean = sum(fits) / length
  sum2 = sum(x*x for x in fits)
  std = abs(sum2 / length - mean**2)**0.5
        
  print("  Min %s" % min(fits))
  print("  Max %s" % max(fits))
  print("  Avg %s" % mean)
  print("  Std %s" % std)

  best_ind = tools.selBest(pop, 1)[0]
  print("No. correctly found (only works with simulation testing):")
  print(sum([1 for b,p in zip(best_ind, recPhonMap) if b==p]))
  print("No. of non-blanks correctly found:")
  print(sum([1 for b,p in zip(best_ind, recPhonMap) if b==p and p!=""]))
  print("Rec code frequency correctly found:")
  print(sum([f for b,p,f in zip(best_ind, recPhonMap, recCodeFreqs) if b==p]))
  bestFitness.append(best_ind.fitness.values[0])
    
  print("-- End of (successful) evolution --")
    
best_ind = tools.selBest(pop, 1)[0]


print("Best individual has fitness %s:" % (best_ind.fitness.values)) 
print(best_ind)
print("The phonetic map being used for simulation is:")
print(recPhonMap)
print("Best fitness levels across the iterations:")
print(bestFitness)
print("No. correctly found (only works with simulation testing):")
print(sum([1 for b,p in zip(best_ind, recPhonMap) if b==p]))
print("No. of non-blanks correctly found:")
print(sum([1 for b,p in zip(best_ind, recPhonMap) if b==p and p!=""]))
print("Rec code frequency correctly found:")
print(sum([f for b,p,f in zip(best_ind, recPhonMap, recCodeFreqs) if b==p]))

