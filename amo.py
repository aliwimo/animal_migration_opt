from random import random, randint, uniform, seed
from copy import deepcopy
from math import floor
import numpy as np


POP_SIZE        = 90
DIMENSION       = 30
GENERATIONS     = 50
BOUND           = 100
UB              = BOUND
LB              = -BOUND

PA = [0] * POP_SIZE
BEST = []

for i in range(POP_SIZE):
    PA[i] = (POP_SIZE - i) / (POP_SIZE)

def init_p():
    pop = np.zeros(shape = (POP_SIZE, DIMENSION))
    for i in range(POP_SIZE):
        for j in range(DIMENSION):
            pop[i][j] = LB + random() * (UB - LB)
    return pop

# Sphere problem as objective function ..
def fitness(pop):
    fitnesses = np.zeros(POP_SIZE)
    for i in range(POP_SIZE):
        fitnesses[i] = sum(x**2 for x in pop[i])
    return fitnesses

# finding neighbour
def rand_neighbour(i):
    l = POP_SIZE
    if i == (l - 1):
        candidates = [i-2, i-1, i, 0, 1]
        return candidates[randint(0, 4)]
    elif i == (l - 2):
        candidates = [i-2, i-1, i, i+1, 0]
        return candidates[randint(0, 4)]
    else:
        candidates = [i-2, i-1, i, i+1, i+2]
        return candidates[randint(0, 4)]

def find_bounds(x):
    if x < LB:
        x = random() * (UB - LB) + LB
    if x > UB:
        x = random() * (UB - LB) + LB
    return x

def run(pop_now):
    pop_next = np.zeros(shape = (POP_SIZE, DIMENSION))
        
    for i in range(POP_SIZE):
        ng = rand_neighbour(i)
        mu, sigma = 0, 1 # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)
        for j in range(DIMENSION):
            pop_next[i][j] = pop_now[i][j] + (s * (pop_now[ng][j] - pop_now[i][j]))
            # pop_next[i][j] = pop_now[i][j] + (random() * (pop_now[ng][j] - pop_now[i][j]))
            pop_next[i][j] = find_bounds(pop_next[i][j])
    
    # calculating Fitness for this Generation and the next one
    fit_now = fitness(pop_now)
    fit_next = fitness(pop_next)

    # Evaluate
    for i in range(POP_SIZE):
        if fit_next[i] < fit_now[i]:
            pop_now[i] = pop_next[i]
    
    # recalculate fitness for pop_now
    fit_now = fitness(pop_now)


    index_of_fittest = np.argmin(fit_now)
    best = pop_now[index_of_fittest]

    PA_indexes = np.argsort(fit_now)

    for i in range(POP_SIZE):
        for j in range(DIMENSION):
            state = False
            while not state:
                r1 = randint(0, POP_SIZE - 1)
                r2 = randint(0, POP_SIZE - 1)
                state = False if r1 == i or r2 == i or r1 == r2 else True
            if random() > PA[PA_indexes[i]]:
                pop_next[i][j] = pop_now[r1][j] + (random() * (best[j] - pop_now[i][j])) + (random() * (pop_now[r2][j] - pop_now[i][j]))
                pop_next[i][j] = find_bounds(pop_next[i][j])

    # calculating Fitness for this Generation and the next one
    fit_now = fitness(pop_now)
    fit_next = fitness(pop_next)

    # Evaluate
    for i in range(POP_SIZE):
        if fit_next[i] < fit_now[i]:
            pop_now[i] = pop_next[i]
    
    # recalculate fitness for pop_now
    fit_now = fitness(pop_now)
    
    BEST.append(min(fit_now))
    return pop_now
# end of run

POP = init_p()
for gen in range(GENERATIONS):
    pop_now = run(POP)
    POP = pop_now.copy()
    print(f"Generateion {gen} - Best: {BEST[gen]}")
