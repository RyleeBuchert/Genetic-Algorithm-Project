import pandas as pd
import numpy as np
import random
import time


# Method to open file and get data
#   - Returns dictionary of p, num_vertices, and coordinates
def open_file(file_path):
    with open(file_path, 'r') as file:
        data_string = file.read().split('\n')

    data_info = data_string[0].split(' ')
    p = int(data_info[0])
    num_vertices = int(data_info[1])

    raw_data = [data_string[i].split(',') for i in range(1, len(data_string))]
    data = [(float(item[0]), float(item[1])) for item in raw_data if item[0] != '']
    # centroids = [data[i] for i in range(p)]
    # data = data[p:]
    return {'p': p, 'num_vertices': num_vertices, 'data': data}


class GeneticAlgorithm:
    # Constructor:
    #   - File: file object from 'open_file' method
    #   - Selection: string either 'Rank', 'Tournament', or 'Roulette'
    #   - Crossover: string for crossover method ...
    #   - Mutation: string for mutation method ...
    #   - Mutation Rate: float for the mutation rate
    def __init__(self, data_dict, selection, crossover, mutation, mutation_rate):
        # Initialize parameter variables
        self.p = data_dict['p']
        self.num_vertices = data_dict['num_vertices']
        self.vertices = list(range(1, self.num_vertices+1))
        self.vertice_coords = data_dict['data']

        self.selection_mech = selection
        self.crossover_method = crossover
        self.mutation_method = mutation
        self.mutation_rate = mutation_rate

        # Initialize chromosome
        self.chromosome_df = pd.DataFrame(np.zeros(shape=(100, self.num_vertices)), columns=self.vertices)
        for i in range(100):
            selected_vertices = random.sample(self.vertices, self.p)
            for j in range(1, self.num_vertices+1):
                if j in selected_vertices:
                    self.chromosome_df.loc[i][j] = 1

    # Method to begin genetic algorithm
    def start(self):
        # Score each chromosome in the chromosome_df
        chromosome_scores = []
        for idx in self.chromosome_df.index:
            chromosome_scores.append(self.score_chrsomosome(self.chromosome_df.loc[[idx]], idx))

        # Initialize dataframe for the mutated children
        mutated_children = pd.DataFrame(np.zeros(shape=(100, self.num_vertices)), columns=self.vertices)

        # Elitism --- place two fittest chromosomes in mutated children pool
        top_scores = sorted(chromosome_scores)[:2]
        top_indexes = [chromosome_scores.index(score) for score in top_scores]
        for i, idx in enumerate(top_indexes):
            mutated_children.iloc[[i]] = self.chromosome_df.iloc[[idx]]

        # Selection --- call roulette, tournament, or rank method
        if self.selection_mech == "Roulette":
            selected_chromosomes = self.roulette_selection(chromosome_scores)
        elif self.selection_mech == "Tournament":
            selected_chromosomes = self.tournament_selection(chromosome_scores)
        elif self.selection_mech == "Rank":
            pass
        else:
            print("Invalid selection mechanism")
            return

        print()


    # Method for Roulette selection
    def roulette_selection(self, scores):
        # Get total sum and normalize scores
        total_score = sum(scores)
        normalized_scores = [total_score/item for item in scores]

        # Get total normalized sum and percentages
        normalized_sum = sum(normalized_scores)
        percentages = [item/normalized_sum for item in normalized_scores]

        # Select and return chromosomes based on percentages
        selections = [scores.index(np.random.choice(scores, p=percentages)) for i in range(100)]
        return [self.chromosome_df.iloc[[i]] for i in selections]

    # Method for tournament selection
    def tournament_selection(self, scores):
        r = 0.75
        selected_indexes = []
        for i in range(100):
            # Randomly select two chromosome scores
            better = None
            worse = None
            the_two = random.sample(scores, 2)

            # Rank the scores by fitness
            if the_two[1] < the_two[0]:
                better = 1
                worse = 0
            else: 
                better = 0
                worse = 1

            # Pick the better 75% of the time, worse 25% of the time
            chance = random.uniform(0, 1)
            if chance < r:
                selected_indexes.append(scores.index(the_two[better]))
            else:
                selected_indexes.append(scores.index(the_two[worse]))
        
        # Return chromosomes from index list
        return [self.chromosome_df.iloc[[i]] for i in selected_indexes]

    # Method for rank selection
    def rank_selection(self):
        pass



    # Score chromosome based on total distance from points to centers
    def score_chrsomosome(self, chromosome, idx):
        # Get list of centers and remaining points to be assigned
        centers = [i for i in chromosome.columns if chromosome.loc[idx][i] == 1]
        rem_points = [i for i in self.vertices if i not in centers]

        # Assign points to centers and calculate distance
        center_dict = {idx: self.vertice_coords[idx - 1] for idx in centers}
        rem_point_dict = {idx: self.vertice_coords[idx - 1] for idx in rem_points}
        total_distance = 0
        point_assigments = {}
        for key, val in rem_point_dict.items():
            closest_center = self.get_closest_center(val, center_dict)
            total_distance += closest_center[2]
            point_assigments.update({key: {closest_center[0], closest_center[1]}})
        return total_distance

    # Method to find closest center to a point
    def get_closest_center(self, point, centers):
        closest = None
        closest_dist = None
        for key, val in centers.items():
            dist = self.get_distance(point, val)
            if closest is None:
                closest = key
                closest_dist = dist
                continue
            if dist < closest_dist:
                closest = key
                closest_dist = dist
        return [closest, centers[closest], closest_dist]

    # Calculate Euclidean distance between two points
    def get_distance(self, point, center):
        return np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)



if __name__ == "__main__":

    data_dict = open_file('data\\toy_data.txt')
    GA = GeneticAlgorithm(data_dict, 'Tournament', 'MX1', 'Single-Point', 0.05)
    GA.start()