import pandas as pd
import numpy as np


class Chromosome:

    def __init__(self):
        pass

    def score_chromosome(self):
        pass


class GeneticAlgorithm:
    # Constructor:
    #   - File: an int corresponding to the file number in the 'data' folder (1-40)
    #   - Selection: string either 'Rank', 'Tournament', or 'Roulette'
    #   - Crossover: string for crossover method ...
    #   - Mutation: string for mutation method ...
    #   - Mutation Rate: float for the mutation rate
    def __init__(self, file, selection, crossover, mutation, mutation_rate):
        # Load test data and store info
        print()
        with open(f'data\\pmed{file}.txt', 'r') as file:
            data_string = file.read().split('\n')
            data_info = data_string[0].lstrip().rstrip().split(' ')
            self.data = [data_string[i].lstrip().rstrip().split(' ') for i in range(1, len(data_string))]

        # Initialize parameter variables
        self.num_vertices = int(data_info[0])
        self.num_edges = int(data_info[1])
        self.p = int(data_info[2])
        self.selection_mech = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        self.mutation_rate = mutation_rate

        # Initialize chromosome
        self.chromosome = np.zeros(shape=(self.num_vertices, 1))
        print()


if __name__ == "__main__":

    test = GeneticAlgorithm(1, 'Tournament', 'MX1', 'Single-Point', 0.05)
    print()