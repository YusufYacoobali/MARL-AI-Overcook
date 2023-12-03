import random

class Map:
    def _init_(self, file_path, num_objects, arglist):
        self.width, self.height = map(int, arglist.grid_size.split('x'))
        self.arglist = arglist
        self.num_objects = num_objects
        self.file_path = file_path
        self.layout = None
        self.object_chars = "tlp-/*"
        self.layout = None
        self.population_size = 3
        self.num_generations = 3
        self.population = None

    def generate_random_layout(self):
        characters = list(self.object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
        random.shuffle(characters)
        layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]
        return layout

    def evaluate_fitness(self):
        fitness_score = 0
        if self.type == 'r':
            #fully random map
            fitness_score = self.calculate_random_fitness()
        elif self.type == 's':
            #spread out map
            fitness_score = self.calculate_spread_fitness()
        elif self.type == 'a':
            #all blocked map
            fitness_score = self.calculate_collab_optional_fitness()
        elif self.type == 't':
            #trapped map
            fitness_score = self.calculate_collab_fitness()
        else:
            print(f"Invalid grid type: {self.type}")

        return fitness_score
    
    def calculate_collab_fitness():
            # Calculate fitness based on collaboration requirements in trapped map
        collaboration_score = 0

        # Iterate through the map layout to identify collaboration opportunities
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == '-':
                    # Found a wall, check for collaboration opportunities on either side
                    left_side = self.layout[y][:x]
                    right_side = self.layout[y][x + 1:]

                    # Check if there's a workstation (ingredient, slicing counter, etc.) on one side
                    if any(obj in left_side for obj in ['t', 'l', 'p', '*']) and any(obj in right_side for obj in ['t', 'l', 'p', '*']):
                        # Collaboration opportunity found, increase the collaboration score
                        collaboration_score += 1

        # You can adjust the scoring based on the importance of collaboration in your game
        return collaboration_score

    def crossover(self, other_map):
        # Implement crossover operation between two maps
        # Exchange segments of genetic information (map layout)
        pass

    def mutate(self):
        # Implement mutation operation on the map
        # Introduce random changes to encourage diversity
        pass

    def evolve_population(self, population):
        # Select maps for reproduction based on fitness
        selected_maps = self.select_maps_for_reproduction(population)

        # Create the next generation
        next_generation = []

        while len(next_generation) < len(population):
            # Choose parents for crossover
            parent1 = random.choice(selected_maps)
            parent2 = random.choice(selected_maps)

            # Perform crossover to create offspring
            child = parent1.crossover(parent2)

            # Apply mutation to the child
            child.mutate()

            # Add the child to the next generation
            next_generation.append(child)

        return next_generation

    def select_maps_for_reproduction(self):
        # Implement selection mechanism based on fitness
        # Higher fitness maps have a higher chance of being selected
        pass

    def genetic_algorithm(self):
        # Generate an initial population
        self.population = [self.generate_random_layout() for _ in range(self.population_size)]

        for generation in range(self.num_generations):
            # Evaluate fitness for each map in the population
            for map_instance in self.population:
                map_instance.evaluate_fitness()

            # Create the next generation
            self.population = self.evolve_population()

            # Optionally, you can keep track of the best map in each generation
            best_map = max(self.population, key=lambda x: x.fitness)
            print(f"Generation {generation + 1}, Best Fitness: {best_map.fitness}")

        # Get the optimal map from the final generation
        self.layout = max(self.population, key=lambda x: x.fitness)
        # Optionally, return or save the best map from the final generation
        #return max(population, key=lambda x: x.fitness)

    def generate_best_map(self):
        pass

    def generate_map(self):
        self.type = self.arglist.grid_type.lower()
        self.genetic_algorithm()
        self.place_players_and_objects()

    def place_players_and_objects(self):
        self.layout.append(["\n", "SimpleTomato", "\n"])

        lettuce_coordinates = [
            (5, 1),
            (4, 1),
            (4, 4),
            (2, 4)
        ]

        for x, y in lettuce_coordinates:
            self.layout.append([str(x), " ", str(y)])
        # Write the generated layout to the specified file
        with open(self.file_path, 'w') as f:
            for row in self.layout:
                f.write("".join(row) + '\n')
        print("file made")
