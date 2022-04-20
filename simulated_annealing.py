import pandas as pd
import numpy as np
import random


class SimulatedAnnealing:
    # Constructor:
    #   - Data_Dict: file object from 'open_file' method
    #   - Perturb_Mech: ...
    #   - Iter: int for the number of iterations annealing runs for
    #   - Alpha: ...
    #   - Beta: ...
    def __init__(self, data_dict, perturb_mech, iter, alpha=None, beta=None):
        # Get dataset variables
        self.p = data_dict['p']
        self.num_vertices = data_dict['num_vertices']
        self.vertices = list(range(1, self.num_vertices+1))
        self.vertice_coords = data_dict['data']
        
        # Generate initial solution randomly
        self.S = pd.DataFrame(np.zeros(shape=(1, self.num_vertices)), columns=self.vertices)
        selected_vertices = random.sample(self.vertices, self.p)
        for i in range(1, self.num_vertices+1):
            if i in selected_vertices:
                self.S[i] = 1

        # Initialize SA variables
        self.perturb_mech = perturb_mech
        self.T = 10
        self.iterations = iter
        self.alpha = alpha if alpha else 0.98
        self.beta = beta if beta else 1.02


    # Method to begin annealing process
    def start(self):
        i = 0
        while self.T > 1:
            while i < self.iterations:
                new_S = self.perturb()
                if (self.score(new_S) < self.score(self.S)):
                    self.S = new_S
                i += 1
            self.T = self.alpha * self.T
            self.iterations = self.beta * self.iterations


    # Method to perturb solution
    def perturb(self):
        # Randomly choose index and flip bit
        solution = self.S[:]
        index = random.randint(0, self.num_vertices-1)
        if solution[index][0] == 0:
            solution[index] = 1
        elif solution[index][0] == 1:
            solution[index] = 0

        # Fix up solution if infeasible
        print()
        while solution.sum(axis=1)[0] < self.p:
            solution[random.choice(np.where(solution == 0)[1])] = 1
        while solution.sum(axis=1)[0] > self.p:
            solution[random.choice(np.where(solution == 1)[1])] = 0
        
        # Return new solution
        return solution

    
    # Method to score solutions
    def score(self, solution):
        # Get list of centers and remaining points to be assigned
        centers = [i for i in solution.columns if solution.loc[i] == 1]
        rem_points = [i for i in self.vertices if i not in centers]

        # Assign points to centers and calculate distance
        center_dict = {i: self.vertice_coords[i - 1] for i in centers}
        rem_point_dict = {i: self.vertice_coords[i - 1] for i in rem_points}
        total_distance = 0
        point_assigments = {}
        for key, val in rem_point_dict.items():
            closest_center = self.get_closest_center(val, center_dict)
            total_distance += closest_center[2]
            point_assigments.update({key: {closest_center[0], closest_center[1]}})


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
    data_dict = open_file('data\\toy_data.txt')
    SA = SimulatedAnnealing(data_dict, 'Mutation', 100)
    print(SA.start())