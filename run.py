import copy
import math
from traffic_simulation import TrafficSimulation
from constants import *
import random
from intersection import Intersection, StopLight, StopSign
from typing import List

QUEUE_THRESHOLD_HIGH = 3  # Arbitrary threshold for long queues
QUEUE_THRESHOLD_LOW = 1   # Arbitrary threshold for short queues

# AI function will go in this file where we will simulate the game. We can think of the game as a black box.
# We will simply pass in inputs to the simulation and we will get some answer that we then try to minimize.

#TODO 
#Inputs: 
# # the AI inputs are the matrix of stoplights and stopsigns
# for a 3 by 3 intersection
# example [0 ,1, 0]
#        [0, 1, 1]
#        [0, 1  0]
# where 1 represents a stoplight and 0 represents a stopsign
# Output: Avg time to get to destination

# function for runnning the game

# instantiates a traffic simulation and runs it until all cars are at their destination. 
# routes are randomized so the simulation is different each time
# returns the average time to destination

def simulate(ai_inputs):
    print(f"matrix size: {len(ai_inputs)}x{len(ai_inputs[0])}")

    ts = TrafficSimulation(matrix=ai_inputs, num_of_cars=30)
    while not ts.done():
        ts.update_car_positions()
    return ts.result()

def random_intersection_placement(width: int, height: int) -> List[List[Intersection]]:
    return [
        [
            StopLight(duration=random.choice([5, 10])) if random.randint(0, 1) else StopSign()
            if x > 0 and y > 0 else None
            for x in range(width)
        ]
        for y in range(height)
    ]


def optimize_intersection_placement(width: int, height: int, iterations: int = 1000) -> List[List[Intersection]]:
    current_grid = random_intersection_placement(width, height)
    best_grid, best_time = current_grid, simulate(current_grid)
    print("curr grid")
    for row in current_grid:
        print(" ".join(['1' if isinstance(cell, StopLight) else '0' for cell in row]))

    temperature, cooling_rate = 100.0, 0.95

    for _ in range(iterations):
        new_grid = copy.deepcopy(current_grid)
        # 50% chance of changing to stop sign or stop light.
        for y, row in enumerate(new_grid):
            for x, cell in enumerate(row):
                if isinstance(cell, StopSign) and random.random() < 0.5:
                    new_grid[y][x] = StopLight(duration=random.choice([5, 10]))
                elif isinstance(cell, StopLight) and random.random() < 0.5:
                    new_grid[y][x] = StopSign()

        print("NEW grid")
        for row in new_grid:
            print(" ".join(['1' if isinstance(cell, StopLight) else '0' for cell in row]))

        new_time = simulate(new_grid)

        # Accept new grid on simulated annealing prob
        if new_time < best_time or random.random() < acceptance_probability(best_time, new_time, temperature):
            current_grid = new_grid
            if new_time < best_time:
                best_grid, best_time = new_grid, new_time

        temperature *= cooling_rate  # Reduce temperature

    return best_grid


def acceptance_probability(old_time, new_time, temperature):
    if new_time < old_time:
        return 1.0
    return math.exp((old_time - new_time) / temperature)


def optimize_intersection_placement_v2(width: int, height: int, iterations: int = 1000) -> List[List[Intersection]]:
    """
    Optimize intersection placement using traffic queue heuristics and simulated annealing.
    """
    current_grid = random_intersection_placement(width, height)
    print("Initial Grid:")
    for row in current_grid:
        print(" ".join(['1' if isinstance(cell, StopLight) else '0' for cell in row]))
    
    # Initialize TrafficSimulation once and reuse
    best_grid = current_grid
    best_time = simulate(current_grid)  # Updated function to reuse TrafficSimulation
    
    temperature = 200.0
    cooling_rate = 0.99

    for i in range(iterations):
        # Create a deep copy of the current grid for modifications
        new_grid = copy.deepcopy(current_grid)

        # Evaluate intersections and modify based on queue lengths
        queue_data = evaluate_intersections(new_grid)
        for (x, y), queue_length in queue_data.items():
            if queue_length >= QUEUE_THRESHOLD_HIGH and isinstance(new_grid[x][y], StopSign):
                print(f"changed at ({x}, {y}) to light")
                new_grid[x][y] = StopLight(duration=random.choice([5, 10]))
            elif queue_length <= QUEUE_THRESHOLD_LOW and isinstance(new_grid[x][y], StopLight):
                print(f"changed at ({x}, {y}) to sign")
                new_grid[x][y] = StopSign()

        print(" Updated Grid:")
        for row in new_grid:
            print(" ".join(['1' if isinstance(cell, StopLight) else '0' for cell in row]))
        
        # Update the TrafficSimulation instance with the new grid
        new_time = simulate(new_grid)

        # Determine if the new grid should be accepted
        if new_time < best_time or random.random() < acceptance_probability(best_time, new_time, temperature):
            current_grid = new_grid
            if new_time < best_time:
                best_grid = new_grid
                best_time = new_time

        # Cool down the temperature
        temperature *= cooling_rate
        print(f"{i+1} out of {iterations}, Best Time: {best_time:.2f}")

    return best_grid


def evaluate_intersections(grid: List[List[Intersection]]) -> dict:
    """
    Evaluate intersections for queue metrics and return a dictionary of (x, y): queue_length.
    """
    queue_data = {}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if isinstance(cell, (StopSign, StopLight)):
                queue_length = cell.max_queue_length()  # Ensure this is a method call or property access
                # print(f"Queue at ({x}, {y}): {queue_length}")  # Debugging output
                queue_data[(x, y)] = queue_length
    return queue_data


def genetic_algorithm(width: int, height: int, population_size: int = 10, generations: int = 5, mutation_rate: float = 0.1) -> List[List[Intersection]]:
    """
    Optimizes intersection placement using a genetic algorithm.
    """
    def create_individual():
        return random_intersection_placement(width, height)

    def mutate(individual):
        x, y = random.randint(1, width - 1), random.randint(1, height - 1)
        individual[y][x] = StopLight(duration=random.choice([5, 10])) if isinstance(individual[y][x], StopSign) else StopSign()

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, height - 2)
        return parent1[:crossover_point] + parent2[crossover_point:]

    # Initialize population and evaluate fitness
    population = [create_individual() for _ in range(population_size)]
    fitness_scores = [simulate(ind) for ind in population]

    for _ in range(generations):
        # Select top individuals based on fitness
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        population = [ind for ind, _ in sorted_population[:population_size // 2]]

        # Generate new population with crossover and mutation
        while len(population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            population.append(child)

        # Recalculate fitness
        fitness_scores = [simulate(ind) for ind in population]

    # Return the best individual
    return population[fitness_scores.index(min(fitness_scores))]



if __name__ == "__main__":
    # Example setup
    width, height = 9, 9
    num_iterations = 10

    # Run optimization
    best_matrix = optimize_intersection_placement_v2(width, height, num_iterations)

    # Print the results
    print("Optimal grid:")
    for row in best_matrix:
        print(" ".join(['1' if isinstance(cell, StopLight) else '0' for cell in row]))
