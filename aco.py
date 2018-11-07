import math
import matplotlib.pyplot as plt
from random import randint

import numpy as np

num_of_ants = 10  # int(input('Enter the number of ants: '), 16)
city_coordinates = []
distance_matrix = []
num_of_cities = 0
tour_matrix = []
tabu_matrix = []
pheromone_matrix = []
best_tour = []
best_min_distance = 0
initial_pheromone_level = 0.15


def init():
    global city_coordinates, distance_matrix, num_of_ants, num_of_cities, initial_pheromone_level
    # evaporation_rate = float(input('Enter the evaporation rate: [0,1]: '))
    # max_iterations = int(input('Enter the max iterations: '), 16)
    initial_pheromone_level = 0.15  # float(input('Enter the initial pheromone level: '))
    # read the input file
    city_coordinates = read_file("Cities")
    num_of_cities = len(city_coordinates)
    distance_matrix = calculate_distance_between_cities()
    init_aco()
    x = city_coordinates[tour_matrix[:, best_tour], 0]
    y = city_coordinates[tour_matrix[:, best_tour], 1]
    plt.plot(x, y)

    plt.plot(city_coordinates[:, 0], city_coordinates[:, 1], 'ro')
    plt.title('Ant Colony Optimization: Bays29, Cost: {}'.format(best_min_distance))
    plt.show()


def read_file(input_file):
    return np.genfromtxt(input_file, dtype=float, delimiter=",")
    # print(distance_matrix)


def calculate_distance_between_cities():
    distance_m = np.zeros(shape=(num_of_cities, num_of_cities))
    i = 0
    for coordinate1 in city_coordinates:
        j = 0
        for coordinate2 in city_coordinates:
            x_distance = coordinate1[0] - coordinate2[0]
            y_distance = coordinate1[1] - coordinate2[1]
            distance = math.sqrt(math.pow(x_distance, 2) + math.pow(y_distance, 2))
            distance_m[i][j] = distance
            j += 1
        i += 1
    return distance_m


def init_aco():
    global tabu_matrix, tour_matrix, pheromone_matrix
    # step 1: apply initial quantity of pheromone to all the edges
    pheromone_matrix = apply_initial_pheromones()

    for i in range(1, 10):
        # step 2: set a random start node for all the ants
        tour_matrix, tabu_matrix = add_ants_initially()
        # step 3: traverse through the graph to find a path
        find_solution(1, 1)
        # step 4: update the pheromone levels - evaporate and add new pheromone
        update_pheromone(0.15)
        print("------------------------------------")
    # print(tour_matrix)


def apply_initial_pheromones():
    pheromone_m = np.zeros(shape=(num_of_cities, num_of_cities))
    for i in range(0, num_of_cities):
        for j in range(0, num_of_cities):
            if i != j:
                pheromone_m[i][j] = initial_pheromone_level
    return pheromone_m


def add_ants_initially():
    tour_m = np.zeros(shape=(num_of_cities, num_of_ants), dtype=int)
    tabu_m = np.zeros(shape=(num_of_cities, num_of_ants), dtype=int)
    for i in range(0, num_of_ants):
        random_city = randint(0, num_of_cities - 1)
        tour_m[0][i] = random_city
        tabu_m[random_city][i] = -1
    return tour_m, tabu_m


def find_solution(alpha, beta):
    cities_counter = 1
    for tour in tour_matrix:
        # if cities_counter == 1:
        tour_counter = 0
        for city in tour:
            # print(city)
            next_best_cities = selection_probability(city, alpha, beta)
            i = 0
            next_best_city = next_best_cities[i]
            # print(next_best_city)
            while isTabu(next_best_city, tour_counter):
                i += 1
                if i >= num_of_cities:
                    break
                next_best_city = next_best_cities[i]
            if i < num_of_cities:
                tour_matrix[cities_counter][tour_counter] = next_best_city
                tabu_matrix[next_best_city][tour_counter] = -1
                tour_counter += 1
        cities_counter += 1


def selection_probability(city, alpha, beta):
    probability = np.zeros(shape=(num_of_cities, 1))
    for i in range(0, num_of_cities):
        if i == city:
            continue
        else:
            probability[i] = pheromone_matrix[city][i] / distance_matrix[city][i]
    sum_of_probabilities = 1
    probability = np.divide(probability, sum_of_probabilities)
    return sorted(range(len(probability)), key=lambda k: probability[k])[::-1]


def isTabu(city, ant):
    if tabu_matrix[city][ant] == -1:
        return True
    else:
        return False


def update_pheromone(evaporation_coefficient):
    evaporate(evaporation_coefficient)
    add_new_pheromone()


def add_new_pheromone():
    global pheromone_matrix, best_tour, best_min_distance
    distance_per_ant = calculated_distance_per_tour()
    best_min_distance = np.min(distance_per_ant)
    best_tour = np.argmin(distance_per_ant)
    pheromone_matrix += (1 / best_min_distance)
    print(best_min_distance, " - tour: ", tour_matrix[:, best_tour])
    return pheromone_matrix


def calculated_distance_per_tour():
    distance = np.zeros(shape=(1, num_of_ants))
    counter = 0
    for tour in np.transpose(tour_matrix):
        for i in range(0, len(tour)):
            if i == len(tour) - 1:
                break
            distance[0][counter] += distance_matrix[tour[i]][tour[i + 1]]
        counter += 1
    return distance


def evaporate(evaporation_coefficient):
    global pheromone_matrix
    pheromone_matrix *= (1 - evaporation_coefficient)
    return pheromone_matrix


if __name__ == '__main__':
    init()
