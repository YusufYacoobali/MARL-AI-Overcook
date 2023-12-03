import random

class Map:
    def __init__(self, file_path, num_objects, arglist):
        self.width, self.height = map(int, arglist.grid_size.split('x'))
        self.arglist = arglist
        self.num_objects = num_objects
        self.file_path = file_path
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

    def evaluate_fitness(self, map):
        fitness_score = 0

        if self.type == 'r':
            fitness_score = self.calculate_random_fitness()
        elif self.type == 's':
            fitness_score = self.calculate_spread_fitness()
        elif self.type == 'a':
            fitness_score = self.calculate_collab_optional_fitness()
        elif self.type == 't':
            fitness_score = self.calculate_collab_fitness(map)
        else:
            print(f"Invalid grid type: {self.type}")

        return fitness_score

    def calculate_collab_fitness(self, map):
        blocked_score = 0
        rows, cols = self.height, self.width

        def flood_fill(row, col, visited):
            if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or map[row][col] != ' ':
                return 0
        
            visited[row][col] = True
        
            count = 1
            count += flood_fill(row + 1, col, visited)
            count += flood_fill(row - 1, col, visited)
            count += flood_fill(row, col + 1, visited)
            count += flood_fill(row, col - 1, visited)
            # count += flood_fill(row + 1, col + 1, visited)  # Diagonal movement
            # count += flood_fill(row - 1, col - 1, visited)  # Diagonal movement
        
            return count

        visited = [[False for _ in range(cols)] for _ in range(rows)]

        def find_object_positions(layout):
            object_positions = {obj: [] for obj in ['p', 't', 'l', '/', '*']}
            for row_idx, row in enumerate(layout):
                for col_idx, char in enumerate(row):
                    if char in object_positions:
                        object_positions[char].append((row_idx, col_idx))
            return object_positions

        def calculate_distance_score(object_positions):
            distance_score = 0

            # Define the weights for each distance
            weight_t_to_slash = 0.3
            weight_l_to_slash = 0.3
            weight_slash_to_p = 0.4
            weight_p_to_star = 0.3

            # Calculate distances and update the distance score
            if 't' in object_positions and '/' in object_positions:
                for tomato_pos in object_positions['t']:
                    for slash_pos in object_positions['/']:
                        distance_t_to_slash = abs(tomato_pos[0] - slash_pos[0]) + abs(tomato_pos[1] - slash_pos[1])
                        distance_score += weight_t_to_slash * distance_t_to_slash
            
            if 'l' in object_positions and '/' in object_positions:
                for lettuce_pos in object_positions['l']:
                    for slash_pos in object_positions['/']:
                        distance_l_to_slash = abs(lettuce_pos[0] - slash_pos[0]) + abs(lettuce_pos[1] - slash_pos[1])
                        distance_score += weight_l_to_slash * distance_l_to_slash
            
            if '/' in object_positions and 'p' in object_positions:
                for slash_pos in object_positions['/']:
                    for plate_pos in object_positions['p']:
                        distance_slash_to_p = abs(slash_pos[0] - plate_pos[0]) + abs(slash_pos[1] - plate_pos[1])
                        distance_score += weight_slash_to_p * distance_slash_to_p
            
                        if 't' in object_positions and 'l' in object_positions and '*' in object_positions:
                            for delivery_pos in object_positions['*']:
                                distance_p_to_star = abs(plate_pos[0] - delivery_pos[0]) + abs(plate_pos[1] - delivery_pos[1])
                                distance_score += weight_p_to_star * distance_p_to_star

            return distance_score
        
        object_positions = find_object_positions(map)
        distance_score = calculate_distance_score(object_positions)
        separated_regions = 0
        collaboration_score = 0

        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and map[row][col] == ' ':
                    # If an unvisited empty space is found, perform flood-fill from that position
                    count = flood_fill(row, col, visited)
                    if count > 0:
                        separated_regions += 1
        self.print_map(map)
    
        if separated_regions > 1:
            separated_regions = 10
        collaboration_score = separated_regions + distance_score
        # print("Collaboration Fitness:", collaboration_score)
        # print("separated regions:", separated_regions)
        # print("dist Score:", distance_score)
        return collaboration_score, blocked_score, distance_score  # Su


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
                self.evaluate_fitness(map_instance)

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

    def print_map(self, map):
        for row in map:
            print(''.join(row))
