import pandas as pd
import numpy as np
import random
import time


class GeneticAlgorithm:
    # Constructor:
    #   - File: int corresponding to the file number in the 'data' folder (1-40)
    #   - Selection: string either 'Rank', 'Tournament', or 'Roulette'
    #   - Crossover: string for crossover method ...
    #   - Mutation: string for mutation method ...
    #   - Mutation Rate: float for the mutation rate
    def __init__(self, file, selection, crossover, mutation, mutation_rate):
        # Load test data and store info
        with open(f'data\\pmed{file}.txt', 'r') as file:
            data_string = file.read().split('\n')
            data_info = data_string[0].lstrip().rstrip().split(' ')
            data = [data_string[i].lstrip().rstrip().split(' ') for i in range(1, len(data_string))]

        # Initialize parameter variables
        self.num_vertices = int(data_info[0])
        self.vertices = list(range(1, self.num_vertices+1))
        self.num_edges = int(data_info[1])
        self.p = int(data_info[2])

        self.selection_mech = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        self.mutation_rate = mutation_rate

        # Create cost matrix
        init_matrix = np.zeros((self.num_vertices, self.num_vertices))
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                if i != j:
                    init_matrix[i, j] = 99999
        self.cost_matrix = pd.DataFrame(init_matrix, index=list(range(1,self.num_vertices+1)), columns=list(range(1,self.num_vertices+1)))
        for line in data:
            self.cost_matrix.loc[int(line[0])][int(line[1])] = int(line[2])
            self.cost_matrix.loc[int(line[1])][int(line[0])] = int(line[2])

        # Apply Floyd's algorithm to cost matrix
        for k in range(1, self.num_vertices+1):
            for i in range(1, self.num_vertices+1):
                for j in range(1, self.num_vertices+1):
                    self.cost_matrix.loc[i][j] = min(self.cost_matrix.loc[i][j], self.cost_matrix.loc[i][k]+self.cost_matrix.loc[k][j])
            print(k)
        print()

        # Initialize chromosome
        self.chromosome = pd.DataFrame(np.zeros(shape=(100, self.num_vertices)), columns=list(range(1,self.num_vertices+1)))
        for i in range(100):
            selected_vertices = random.sample(self.vertices, self.p)
            for j in range(1,self.num_vertices+1):
                if j in selected_vertices:
                    self.chromosome.loc[i][j] = 1

    def start(self):
        chromosome_scores = []
        for idx in self.chromosome.index:
            chromosome_scores.append(self.score_chrsomosome(self.chromosome.loc[[idx]]))

    def score_chrsomosome(self, chromosome):
        centers = []
        for i in chromosome.columns:
            if chromosome.loc[0][i] == 1:
                centers.append(i)

        # Next: Assign points to centers and calculate distance

        print()



if __name__ == "__main__":

    GA = GeneticAlgorithm(1, 'Tournament', 'MX1', 'Single-Point', 0.05)
    GA.start()