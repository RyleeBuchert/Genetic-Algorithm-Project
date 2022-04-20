import pandas as pd
import numpy as np
import random
import time


class GeneticAlgorithm:
    # Constructor:
    #   - File: file object from 'open_file' method
    #   - Selection: string either 'Rank', 'Tournament', or 'Roulette'
    #   - Crossover: string either 'Single Point', 'Double Point, 'N-Point', or 'Uniform'
    #   - Mutation Rate: float for the chance of a random mutation
    #   - Iter: int for the number of iterations GA should run for
    def __init__(self, data_dict, selection, crossover, mutation_rate, iter):
        # Initialize parameter variables
        self.p = data_dict['p']
        self.num_vertices = data_dict['num_vertices']
        self.vertices = list(range(1, self.num_vertices+1))
        self.vertice_coords = data_dict['data']

        self.selection_mech = selection
        self.crossover_method = crossover
        self.mutation_rate = mutation_rate
        self.num_iterations = iter

        # Initialize chromosome
        self.chromosome_df = pd.DataFrame(np.zeros(shape=(100, self.num_vertices)), columns=self.vertices)
        for i in range(100):
            selected_vertices = random.sample(self.vertices, self.p)
            for j in range(1, self.num_vertices+1):
                if j in selected_vertices:
                    self.chromosome_df.loc[i][j] = 1


    # Method to run genetic algorithm
    def start(self):
        for i in range(self.num_iterations):
            # Score each chromosome in the chromosome_df
            chromosome_scores = [self.score_chrsomosome(self.chromosome_df.iloc[[i]], i) for i in self.chromosome_df.index]
            print(i+1 + ': ' + min(chromosome_scores))

            # Elitism --- place two fittest chromosomes in next generation with no changes
            top_chromosomes = pd.DataFrame(np.zeros(shape=(2, self.num_vertices)), columns=self.vertices)
            top_indexes = [chromosome_scores.index(score) for score in sorted(chromosome_scores)[:2]]
            for i, idx in enumerate(top_indexes):
                top_chromosomes.iloc[[i]] = self.chromosome_df.iloc[[idx]]

            # Selection --- wrapper method calls either roulette, tournament, or rank mechanism
            chromosome_pool = self.selection(self.selection_mech, chromosome_scores)
            if chromosome_pool == -1:
                return

            # Crossover --- wrapper method calls either single, double, n-point, or uniform crossover
            chromosome_pool = self.crossover(self.crossover_method, chromosome_pool)
            if chromosome_pool == -1:
                return

            # Mutation --- randomly flip a bit with 5% probability
            chromosome_pool = self.mutate(chromosome_pool)

            # Update chromosome pool with mutated children
            drop_indexes = random.sample(list(range(99)), 2)
            self.chromosome_df = pd.concat([top_chromosomes, pd.DataFrame(chromosome_pool, columns=self.vertices).drop(drop_indexes)])
            self.chromosome_df.reset_index(drop=True, inplace=True)
        
        # Return chromosome df
        return self.chromosome_df.iloc[[chromosome_scores.index(min(chromosome_scores))]]


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

        # Return sum of Euclidean distances
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


    # Wrapper method for selection
    def selection(self, mech, scores):
        if mech == "Roulette":
            return self.roulette_selection(scores)
        elif mech == "Tournament":
            return self.tournament_selection(scores)
        elif mech == "Rank":
            return self.rank_selection(scores)
        else:
            print("Invalid selection mechanism, must be 'Roulette', 'Tournament', or 'Rank'")
            return -1


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
            if chance < 0.75:
                selected_indexes.append(scores.index(the_two[better]))
            else:
                selected_indexes.append(scores.index(the_two[worse]))
        
        # Return chromosomes from index list
        return [self.chromosome_df.iloc[[i]] for i in selected_indexes]


    # Method for rank selection
    def rank_selection(self, scores):
        # Sort scores and rank
        sorted_scores = sorted(scores)
        score_ranks = [sorted_scores.index(val)+1 for val in scores]

        # Compute sum of ranks and normalize scores
        rank_sum = sum(score_ranks)
        rank_normalized = [rank_sum/val for val in score_ranks]

        # Sum normalized scores and get percentages
        normalized_sum = sum(rank_normalized)
        score_percentages = [val/normalized_sum for val in rank_normalized]

        # Select and return chromosomes based on percentages
        selections = [scores.index(np.random.choice(scores, p=score_percentages)) for i in range(100)]
        return [self.chromosome_df.iloc[[i]] for i in selections]


    # Wrapper method for crossovers
    def crossover(self, mech, parents):
        if mech == "Single Point":
            return self.single_point_crossover(parents)
        elif mech == "Double Point":
            return self.double_point_crossover(parents)
        elif mech == "Uniform":
            return self.uniform_crossover(parents)
        else:
            print("Invalid crossover method, must be 'Single Point', 'Double Point', 'N-Point', or 'Uniform'")
            return -1

            
    # Method for single point crossover, returns child pool
    def single_point_crossover(self, parents):
        child_pool = []
        for i in range(50):
            # Randomly select crossover points and parents
            point = random.randint(1, self.num_vertices)
            selected_parents = random.sample(parents, 2)

            # Create variables for each parent
            parent1 = selected_parents[0].to_numpy()[0]
            parent2 = selected_parents[1].to_numpy()[0]

            # Crossover parents to produce child chromosomes
            child1 = np.append(parent1[:point], parent2[point:]).reshape((1, self.num_vertices))
            child2 = np.append(parent2[:point], parent1[point:]).reshape((1, self.num_vertices))

            # Fix up children if infeasible
            child1, child2 = self.fix_up(child1, child2)

            # Add children to pool
            child_pool.append(child1[0])
            child_pool.append(child2[0])

        # Return pool of children
        return child_pool


    # Method for double point crossover, returns child pool
    def double_point_crossover(self, parents):
        child_pool = []
        for i in range(50):
            # Randomly select two crossover points
            point1 = random.randint(1, self.num_vertices)
            point2 = random.randint(1, self.num_vertices)

            # If points are the same, choose a new point2
            while point1 == point2:
                point2 = random.randint(1, self.num_vertices)

            # Swap points if point2 < point1
            if point2 < point1:
                point1, point2 = point2, point1

            # Randomly select two parents and create arrays
            selected_parents = random.sample(parents, 2)
            parent1 = selected_parents[0].to_numpy()[0]
            parent2 = selected_parents[1].to_numpy()[0]

            # Crossover parents to produce children chromosomes
            child1 = np.concatenate((parent1[:point1],parent2[point1:point2],parent1[point2:])).reshape((1, self.num_vertices))
            child2 = np.concatenate((parent2[:point1],parent1[point1:point2],parent2[point2:])).reshape((1, self.num_vertices))

            # Fix up children if infeasible
            child1, child2 = self.fix_up(child1, child2)

            # Add children to pool
            child_pool.append(child1[0])
            child_pool.append(child2[0])

        # Return pool of children
        return child_pool


    # Method for uniform crossover, returns child pool
    def uniform_crossover(self, parents):
        child_pool = []
        for i in range(50):
            # Generate array of 0's and 1's randomly
            u = [random.randint(0, 1) for i in range(self.num_vertices)]

            # Initialize child arrays
            child1 = np.zeros(shape=(1, self.num_vertices))
            child2 = np.zeros(shape=(1, self.num_vertices))

            # Randomly select two parents and create arrays
            selected_parents = random.sample(parents, 2)
            parent1 = selected_parents[0].to_numpy()[0]
            parent2 = selected_parents[1].to_numpy()[0]

            # Use uniform crossover algorithm to generate children
            for idx, bit in enumerate(u):
                if bit == 0:
                    child1[0, idx] = parent1[idx]
                    child2[0, idx] = parent2[idx]
                elif bit == 1:
                    child1[0, idx] = parent2[idx]
                    child2[0, idx] = parent1[idx]

            # Fix up children if infeasible
            child1, child2 = self.fix_up(child1, child2)

            # Add children to pool
            child_pool.append(child1[0])
            child_pool.append(child2[0])
        
        # Return pool of children
        return child_pool


    # Method to fix up infeasible chromosomes
    def fix_up(self, child1, child2):
        # Fix up children if infeasible
        while child1.sum() < self.p:
            child1[0, random.choice(np.where(child1 == 0)[1])] = 1

        while child1.sum() > self.p:
            child1[0, random.choice(np.where(child1 == 1)[1])] = 0

        while child2.sum() < self.p:
            child2[0, random.choice(np.where(child2 == 0)[1])] = 1

        while child2.sum() > self.p:
            child2[0, random.choice(np.where(child2 == 1)[1])] = 0

        return child1, child2


    # Method for mutating chromosomes in pool
    def mutate(self, pool):
        for child in pool:
            chance = random.uniform(0, 1)
            if chance < self.mutation_rate:
                # Randomly choose index and flip bit
                index = random.randint(0, self.num_vertices-1)
                if child[index] == 0:
                    child[index] = 1
                elif child[index] == 1:
                    child[index] = 0

                # Fix up if infeasible
                while child.sum() < self.p:
                    child[random.choice(np.where(child == 0)[0])] = 1
                while child.sum() > self.p:
                    child[random.choice(np.where(child == 1)[0])] = 0

        # Return mutated child pool
        return pool


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


if __name__ == "__main__":
    # Open data and run genetic algorithm
    data_dict = open_file('data\\toy_data2.txt')
    GA = GeneticAlgorithm(data_dict, 'Roulette', 'Single Point', 0.05, 150)
    print(GA.start())