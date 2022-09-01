import random
import numpy as np
import pandas as pd
import galib as ga
from matplotlib import pyplot as plt



NUM_OF_INDIVIDUAL = 30
TOTAL_GENERATION = 3000
MUTATION_RATE = 0.4
HOLDOUT_METHOD_RATIO = 0.80
CROSS_OVER_TYPE = 1 # 1 for single point, # 2 for double point


print('Calculating Please wait......')

# Read data from a file and split it into two columns.
df = pd.read_csv("data1.txt").iloc[:, 0].str.split(" ", expand=True)
df.columns = ["input", "output"]
df = df.input + df.output

worstFitness = []
bestFitness = []
averageFitness = []
generations = []

train, test = ga.holdoutMethodSplitter(df, HOLDOUT_METHOD_RATIO)

# Creating an individual population and calculating their fitness.
population = [ga.Individual(np.random.randint(2, size=ga.TOTAL_GENES))
              for i in range(NUM_OF_INDIVIDUAL)]
for k in population:
    k.fitness = ga.fitnessCalculator(train, k)

# The main loop that runs for the number of generations
for l in range(TOTAL_GENERATION):
    averageFitness.append(
        sum([i.fitness for i in population])/NUM_OF_INDIVIDUAL)
    bestFitness.append(ga.findBestWorstFitness(population, worst=False))
    worstFitness.append(ga.findBestWorstFitness(population, worst=True))

    generations.append(l)
    newPopulation = []

    mt = random.uniform(0, 1)

    while len(newPopulation) < NUM_OF_INDIVIDUAL:

        population1 = ga.tournmentSelection(population, NUM_OF_INDIVIDUAL)
        population2 = ga.tournmentSelection(population, NUM_OF_INDIVIDUAL)

        newIndividual = ga.crossover([population1, population2], CROSS_OVER_TYPE)
        newIndividual.fitness = ga.fitnessCalculator(train, newIndividual)

        if mt > MUTATION_RATE:
            newPopulation.append(ga.mutation(newIndividual, train))
        else:
            newPopulation.append(newIndividual)

    population = newPopulation
    newPopulation = []

#Graphing the population's average, best, and worst fitness over the number of generations.
print('Calculations completed and plotting inprogress. Please wait......\n')
de = pd.DataFrame({"Generations": generations,
                  "Best": bestFitness, "Average": averageFitness, "Worst": worstFitness})
de.plot(title=f"Data Set 1\n (Mutation Rate: {MUTATION_RATE} | Train Data Set Size: {train.shape[0]} | No. of Generations:{TOTAL_GENERATION})", y=[
        "Best", "Average", "Worst"], color=['blue', 'green', 'black'], x="Generations", kind="line", linewidth=1)
plt.show()

