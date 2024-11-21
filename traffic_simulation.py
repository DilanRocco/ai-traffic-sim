from typing import List
from car import Car
from pos import Pos
from direction import Direction
import random
import time
import math
from intersection import StopLight, Intersection, StopSign
from routefinder import RouteFinder
from constants import *

def random_intersection_placement(width: int, height: int) -> List[List[Intersection]]:
    matrix = [[None for i in range(width)] for i in range(height)]
    # print(matrix, 'matrix creation')
    for y in range(height):
        for x in range(width):
            if y == 0 or x == 0:
                continue
            random_intersection = random.randint(0, 1)
            if random_intersection == 1:
                matrix[y][x] = StopLight(duration = random.choice([5,10]))
            else:
                matrix[y][x] = StopSign()
    return matrix 

# Main logic for the Traffic Simulation 
# Pygame should not be in this file
class TrafficSimulation:
    def __init__(self, matrix: List[List[Intersection]] = random_intersection_placement(10, 8), num_of_cars: int = 10):
        self.height = len(matrix)
        self.width = len(matrix[0])
        if self.height < 2 or self.width < 2 or num_of_cars <= 0:
            raise Exception("Invalid input to Traffic Simulation")
        self.matrix = matrix
        self.initialize_intersections()  # Ensure all intersections are properly initialized
        self.cars = self.initialize_cars(num_of_cars)
        self.start_time = time.time()
        self.times = [math.inf]*num_of_cars
        self.setup_intersection_threads()


    # Initialize the queues and states for all intersections in the matrix
    def initialize_intersections(self):
        for y in range(self.height):
            for x in range(self.width):
                intersection = self.matrix[y][x]
                if isinstance(intersection, StopLight):
                    intersection.queue = []  # Reset StopLight queue
                elif isinstance(intersection, StopSign):
                    intersection.queue = []  # Reset StopSign queue

    # moves all cars which are currently free to move
    def update_car_positions(self):
        cars_to_be_updated = [car for car in self.cars if not car.in_queue and not car.finished]
        for car in cars_to_be_updated:
            next_move = car.get_next_move()
            if next_move:
                self.move_car(car.id, next_move)

    # moves a car from it's current position to its new position, and joins the queue for whatever intersection is at the new position
    def move_car(self, car_index: int, direction: Direction):

        # Check the bounds
        if car_index >= len(self.cars):
            raise Exception(f"move_car: Invalid index of car {car_index}, max index is {len(self.cars)-1}")

        car = self.cars[car_index]
        # print(f"Car{car.id}, from dir {car.on_side}, from{car.curr_pos}, to {car.dest}")

        if car_index >= len(self.cars):
            raise Exception("move_car: Invalid index of car") 
        curr_position = car.curr_pos
        direction_math = direction.math_dirs()
        new_x = curr_position.x + direction_math[0]
        new_y = curr_position.y + direction_math[1]
        car.curr_pos = Pos(new_x, new_y)
        if car.at_destination():
            car.color = 'black'
            car.finished = True
            self.times[car_index] = time.time() - self.start_time
            # print(f"Car{car.id}, at dest, Car{car.route}, {car.in_queue}")
            return

        if not (0 < new_x < self.width and 0 < new_y < self.height):
            # print("moving car out of bounds")
            return

        car.in_queue = True
        intersection = self.matrix[new_y][new_x]
        intersection.join(car_index, direction, self)

    def release_car_from_queue(self, car_index, direction):
        car = self.cars[car_index]
        car.route_index += 1
        car.in_queue = False
        car.on_side = direction

    def setup_intersection_threads(self):
        # Initialize threads for all intersections
        for y in range(1, self.height):
            for x in range(1, self.width):
                self.matrix[y][x].start_timer(self)

    def result(self):
        filtered_times = [time for time in self.times if time != math.inf]
        return sum(filtered_times) / len(filtered_times)

    def done(self):
        return all(car.finished for car in self.cars)

    # place cars randomly on the map
    def initialize_cars(self, num_of_cars):
        cars = []
        for i in range(num_of_cars):
            source = Pos(random.randint(1, self.width-1), random.randint(1, self.height-1))
            destination = Pos(random.randint(1, self.width-1), random.randint(1, self.height-1))
            coming_from = Direction(random.randint(0, 3))
            color = random.choice(CAR_COLORS)
            route = RouteFinder().generate_route(source, destination, self.matrix)
            car = Car(coming_from, source, destination, color, route)
            car.id = i
            cars.append(car)

        '''print(f"we have {num_of_cars} cars")
        for car in cars:
            print(f"Car {car.id} at {car.curr_pos}, goal: {car.dest}")
'''
        # manual routes for debugging
        # route = RouteFinder().generate_route(Pos(4,8), Pos(4,8), self.matrix)
        # c1 = Car(Direction(0), Pos(4,6), Pos(4,6), 'red', route)
        # c2 = Car(Direction(0), Pos(4,7), Pos(4,2), 'blue', [Direction.up]*5)
        # c3 = Car(Direction(0), Pos(4,7), Pos(4,2), 'green', [Direction.up]*5)
        # c4 = Car(Direction(0), Pos(4,7), Pos(4,2), 'yellow', [Direction.up]*5)
        # c5 = Car(Direction(0), Pos(4,7), Pos(4,2), 'blue', [Direction.up]*5)
        # c6 = Car(Direction(0), Pos(4,7), Pos(4,2), 'green', [Direction.up]*5)
        # cars = [c2, c3, c4, c5, c6]
        # print('route', c1.route)

        return cars