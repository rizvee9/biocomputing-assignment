import random
import numpy as np


NUM_OF_GENES = 64
GENE_SIZE = 7
TOTAL_GENES = NUM_OF_GENES * GENE_SIZE

# The class Individual is a blueprint for creating objects that have a rule and a fitness
class Individual():
    rule = None
    fitness = 0

    def __init__(self, rule):
        self.rule = "".join(str(item) for item in rule)
        self.fitness = 0

    def reset(self):
        self.fitness = 0


def geneSplitter(point):
    """
    It takes a string of length $ and returns a list of strings of length $ where $ is divisible
    by $

    :param point: The point to be split into individual genes
    :return: A list of genes.
    """
    return [point[i:i + GENE_SIZE] for i in range(0, len(point), GENE_SIZE)]


def crossover(parents, crossoverType=None):
    """
    It takes two parents and returns a child

    :param parents: a list of two parents
    :param crossoverType: 1 for single point cross over, 2 for double point cross over
    :return: an individual.
    """
    parent1, parent2 = parents[0], parents[1]
    if crossoverType == 1:  # Single point cross over
        crossoverPoint = random.randint(1, NUM_OF_GENES)
        point1, point2 = geneSplitter(
            parent1.rule), geneSplitter(parent2.rule)
        ind = Individual(
            "".join(point1[:crossoverPoint]+point2[crossoverPoint:]))

    elif crossoverType == 2:  # Double point cross over
        crossoverPoint1 = random.randint(1, NUM_OF_GENES)
        crossoverPoint2 = random.randint(1, NUM_OF_GENES)
        while True:
            if crossoverPoint1 >= crossoverPoint2:
                crossoverPoint1 = random.randint(1, NUM_OF_GENES)
                crossoverPoint2 = random.randint(1, NUM_OF_GENES)
            else:
                break

        point1, point2 = geneSplitter(
            parent1.rule), geneSplitter(parent2.rule)

        point1Of1, point2Of1, point3Of1 = point1[:crossoverPoint1], point1[
            crossoverPoint1:crossoverPoint2], point1[crossoverPoint2:]
        point1Of2, point2Of2, point3Of2 = point2[:crossoverPoint1], point2[
            crossoverPoint1:crossoverPoint2], point2[crossoverPoint2:]

        partSelection = random.randint(0, 1)
        if partSelection == 0:
            ind = Individual("".join(point1Of1 + point2Of2 + point3Of1))
        else:
            ind = Individual("".join(point1Of2 + point2Of1 + point3Of2))

    return ind


def fitnessCalculator(trainData, newIndividual):
    """
    It takes in a training set and an individual, and returns the fitness of the individual

    :param trainData: The training data
    :param newIndividual: The individual whose fitness is to be calculated
    :return: The fitness of the individual.
    """
    fitness = 0
    for k in list(set(geneSplitter(newIndividual.rule))):
        for j in trainData:
            if k == j:
                fitness += 1
                break
    return (fitness / NUM_OF_GENES) * 100


def mutation(offSpring, train):
    """
    It takes an individual, randomly selects a gene, and flips it
    :param offSpring: offspring
    :return: A new individual with a mutated rule.
    """
    randomPosition = random.randint(0, TOTAL_GENES - 1)
    offSpringListFormat = list(offSpring.rule)

    if offSpringListFormat[randomPosition] == "0":
        offSpringListFormat[randomPosition] = "1"
    else:
        offSpringListFormat[randomPosition] = "0"

    newOffSpring = Individual("".join(offSpringListFormat))
    newOffSpring.fitness = fitnessCalculator(train, newOffSpring)
    return newOffSpring


def holdoutMethodSplitter(df, ratio):
    """
    It shuffles the dataset, then splits it into two parts, one for training and one for testing

    :param df: the dataframe you want to split
    :param ratio: the ratio of the train set to the test set
    :return: A tuple of two dataframes, one for the training set and one for the test set.
    """

    shuffle_df = df.sample(frac=1)
    train_size = int(ratio * len(df))
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]
    return (train_set, test_set)


def sort(pop):
    """
    It sorts the population by fitness

    :param pop: the population of individuals
    :return: The sorted population.
    """

    n = len(pop)
    for i in range(n):
        already_sorted = True
        for j in range(n - i - 1):
            if pop[j].fitness > pop[j + 1].fitness:
                pop[j], pop[j + 1] = pop[j + 1], pop[j]
                already_sorted = False
        if already_sorted:
            break
    return pop


def tournmentSelection(population, NUM_OF_INDI):
    """
    The above function is used to select two parents from the population.

    :param population: the population of individuals
    :param NUM_OF_INDI: The number of individuals in the population
    :return: The parent with the higher fitness value.
    """

    p1 = population[np.random.randint(0, NUM_OF_INDI)]
    p2 = population[np.random.randint(0, NUM_OF_INDI)]
    while p1 == p2:
        p1 = population[np.random.randint(0, NUM_OF_INDI)]
        p2 = population[np.random.randint(0, NUM_OF_INDI)]
    if p1.fitness > p2.fitness:
        return p1
    else:
        return p2


def findBestWorstFitness(population, worst):
    """
    If the worst parameter is true, return the worst fitness in the population, otherwise return the
    best fitness in the population

    :param population: The population of the current generation
    :param worst: True if you want to get the worst fitness, False if you want to get the best fitness
    :return: The best or worst fitness value in the population.
    """

    if worst:
        fitness = 100
        for i in population:
            if i.fitness < fitness:
                fitness = i.fitness
        return fitness
    else:
        fitness = 0
        for i in population:
            if i.fitness > fitness:
                fitness = i.fitness
        return fitness
