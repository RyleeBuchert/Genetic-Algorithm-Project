import matplotlib.pyplot as plt
import numpy as np
import random
import copy


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
        self.vertices = list(range(0, self.num_vertices))
        self.vertice_coords = data_dict['data']
        
        # Generate initial solution randomly
        self.S = np.zeros(self.num_vertices)
        selected_vertices = random.sample(self.vertices, self.p)
        for i in range(self.num_vertices):
            if i+1 in selected_vertices:
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
        count = 0
        while self.T > 1:
            while i < self.iterations:
                new_S = self.perturb(self.S)
                if (self.score(new_S, 'N') < self.score(self.S, 'N')):
                    self.S = copy.deepcopy(new_S)
                print(str(count+1) + ': ' + str(self.score(self.S, 'N')))
                i += 1
                count += 1
            self.T = self.alpha * self.T
            self.iterations = self.beta * self.iterations


    # Method to perturb solution
    def perturb(self, S):
        # Randomly choose index and flip bit
        solution = copy.deepcopy(S)
        index = random.randint(0, self.num_vertices-1)
        if solution[index] == 0:
            solution[index] = 1
        elif solution[index] == 1:
            solution[index] = 0

        # Fix up solution if infeasible
        while np.sum(solution) < self.p:
            solution[random.choice(np.where(solution == 0)[0])] = 1
        while np.sum(solution) > self.p:
            solution[random.choice(np.where(solution == 1)[0])] = 0
        
        # Return new solution
        return solution

    
    # Method to score solutions
    def score(self, solution, mode):
        # Get list of centers and remaining points to be assigned
        centers = np.where(solution == 1)[0]
        rem_points = np.where(solution == 0)[0]

        # Assign points to centers and calculate distance
        center_dict = {i: self.vertice_coords[i] for i in centers}
        rem_point_dict = {i: self.vertice_coords[i] for i in rem_points}
        total_distance = 0
        point_assigments = {}
        for key, val in rem_point_dict.items():
            closest_center = self.get_closest_center(val, center_dict)
            total_distance += closest_center[2]
            point_assigments.update({key: closest_center[1]})

        # Return total distance
        if mode == 'N':
            return total_distance
        elif mode == 'P':
            return [center_dict, rem_point_dict, point_assigments]


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


    # Method to plot graph with connections
    def plot(self):
        # Get centers and assign points
        assignments = self.score(self.S, 'P')

        # Plot points on graph
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        for val in assignments[0].values():
            plt.plot(val[0], val[1], 'ro')
        for val in assignments[1].values():
            plt.plot(val[0], val[1], 'bo')

        # Connect points to centroids and show graph
        for key, val in assignments[2].items():
            point = self.vertice_coords[key]
            center = val
            plt.plot([point[0], center[0]], [point[1], center[1]], 'c')
        plt.show()


# Method to open file and get data
#   - Returns dictionary of p, num_vertices, and coordinates
def open_file(file_path):
    with open(file_path, 'r') as file:
        data_string = file.read().split('\n')

    data_info = data_string[0].split(' ')
    p = int(data_info[0])
    num_vertices = int(data_info[1])

    raw_data = [data_string[i].split(',') for i in range(1, len(data_string))]
    data = [[float(item[0]), float(item[1])] for item in raw_data if item[0] != '']
    # centroids = [data[i] for i in range(p)]
    # data = data[p:]
    return {'p': p, 'num_vertices': num_vertices, 'data': data}


if __name__ == "__main__":
    # Open data and run simulated annealing
    data_dict = open_file('data\\toy_data3.txt')
    SA = SimulatedAnnealing(data_dict, 'Mutation', 100)
    SA.start()
    print(SA.S)
    SA.plot()