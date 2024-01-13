import random

class BaseMap:
    def __init__(self, file_path, num_objects, arglist):
        self.width, self.height = map(int, arglist.grid_size.split('x'))
        self.arglist = arglist
        self.num_objects = num_objects
        self.file_path = file_path
        self.object_chars = "tlp----/* "
        self.layout = None
        self.population_size = 10
        self.num_generations = 3
        self.population = None
        self.region_starting_points = []
        random.seed()

    def generate_random_layout(self):
        characters = list(self.object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
        random.shuffle(characters)
        layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]
        return layout
    
    def flood_fill(self, row, col, visited, layout):
        """
        Perform a flood-fill from the given position and mark all directly connected empty spaces.
        """
        rows, cols = self.height, self.width
        if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or layout[row][col] != ' ':
            return 0  # Return 0 if the current position is out of bounds or not an empty space

        visited[row][col] = True

        # Recursively perform flood-fill in all directions
        count = 1  # Initialize count to 1 for the current empty space
        count += self.flood_fill(row + 1, col, visited, layout)
        count += self.flood_fill(row - 1, col, visited, layout)
        count += self.flood_fill(row, col + 1, visited, layout)
        count += self.flood_fill(row, col - 1, visited, layout)
        return count

    def evaluate_fitness(self, map):
        pass

    def crossover(self, map1, map2):
        # Perform crossover between two parents to create a child
        child_layout = [self.crossover_row(row1, row2) for row1, row2 in zip(map1, map2)]
        #print(random.randint(1,10))
        return child_layout

    def crossover_row(self, row1, row2):
        # Select a random crossover point along the row length
        crossover_point = random.randint(1, min(len(row1), len(row2)) - 1)
        # Combine the left part of row1 and the right part of row2
        child_row = row1[:crossover_point] + row2[crossover_point:]
        return child_row

    def mutate(self, layout):
        pass

    def generate_map(self):
        #self.type = self.arglist.grid_type.lower()
        self.genetic_algorithm()
        self.place_players_and_objects()

    def start(self):
        map_type = self.arglist.grid_type.lower()
        if map_type == 'r':
            fitness_score = self.calculate_random_fitness()
        elif map_type == 's':
            fitness_score = self.calculate_spread_fitness()
        elif map_type == 'a':
            fitness_score = self.calculate_collab_optional_fitness()
        elif map_type == 't':
            collab_map_instance = CollabMap(self.file_path, self.num_objects, self.arglist)
            collab_map_instance.generate_map()
            #fitness_score = self.calculate_collab_fitness(map)
        else:
            print(f"Invalid grid type: {self.type}")

    def evolve_population(self, selected_maps):
        next_generation = []
        crossover_children = []
        mutated_children = []
        #make next gen same size as prev pop
        while len(next_generation) < len(self.population):
            # Choose parents for crossover
            parent1 = random.choice(selected_maps)
            parent2 = random.choice(selected_maps)

            print(parent1[0])
            print(parent1[1])
            print(parent1)
            #self.print_horizontal_maps(parent1[0])
            # Perform crossover to create offspring
            child = self.crossover(parent1[0], parent2[0])
            crossover_children.append(child)
            # Apply mutation to the child
            mutated_child = self.mutate(child)
            mutated_children.append(mutated_child)
            # Add the child to the next generation
            next_generation.append(mutated_child)

        print("CROSSOVER CHILDREN:")
        self.print_horizontal_maps(crossover_children)

        print("\nMUTATED CHILDREN:")
        self.print_horizontal_maps(mutated_children)
        return next_generation
    
    def print_horizontal_maps(self, maps):
        # Find the maximum number of rows in the maps
        max_rows = max(len(map_) for map_ in maps)

        # Print the maps horizontally with extra spaces between each map
        for i in range(max_rows):
            for map_ in maps:
                if i < len(map_):
                    print(''.join(map_[i]), end="     ")
                else:
                    print(" " * len(map_[0]), end="     ")
            print()

    def select_maps_for_reproduction(self):
        
        #best_maps = [self.evaluate_fitness(map_instance) for map_instance in self.population]
        best_maps = [(map_instance, self.evaluate_fitness(map_instance)[0]) for map_instance in self.population]
        # Sort the list based on fitness in descending order
        sorted_best_maps = sorted(best_maps, key=lambda x: x[1], reverse=True)

        # Select the top 1/3 of the best maps
        top_third = int(len(sorted_best_maps) / 3)
        selected_maps = sorted_best_maps[:top_third]
        print("Best map had score: ", selected_maps[-1][1])
        return selected_maps

    def genetic_algorithm(self):
        # Generate an initial population
        self.population = [self.generate_random_layout() for _ in range(self.population_size)]

        for generation in range(self.num_generations):

            selected_maps = self.select_maps_for_reproduction()
            # Create the next generation
            self.population = self.evolve_population(selected_maps)
            print("NEW POPULATION: ")
            self.print_horizontal_maps(self.population)

            # Optionally, you can keep track of the best map in each generation
            #best_map = max(self.population, key=lambda x: x.fitness)
            #print(f"Generation {generation + 1}, Best Fitness: {best_map.fitness}")

        # Get the optimal map from the final generation
        self.layout = max(self.population, key=lambda x: self.evaluate_fitness(x))
        #self.print_map(self.layout)
        print("FINAL MAP: ")
        self.print_map(self.layout)
        # Optionally, return or save the best map from the final generation
        #return max(population, key=lambda x: x.fitness)

    def generate_map(self):
        self.type = self.arglist.grid_type.lower()
        self.genetic_algorithm()
        self.place_players_and_objects()

    def find_empty_coordinates(self, region):
        rows, cols = len(self.layout)-1, len(self.layout[0])
        empty_coordinates = []
        print (rows, cols)

        for row in range(rows):
            for col in range(cols):
                if 0 <= row < rows and 0 <= col < cols and self.layout[row][col] == ' ':
                    # Add the empty coordinate if it's within the valid range
                    empty_coordinates.append((row, col))

        return empty_coordinates[:4]  # Return the first four empty coordinates

    def place_players_and_objects(self):
        self.layout.append(["\n", "SimpleTomato", "\n"])

        # chef_region1 = self.find_empty_coordinates("Region 1")
        # chef_region2 = self.find_empty_coordinates("Region 2")
        # chef2_region1 = self.find_empty_coordinates("Region 1")
        # chef2_region2 = self.find_empty_coordinates("Region 2")

        # chef_coordinates = chef_region1 + chef_region2 + chef2_region1 + chef2_region2

        rows, cols = self.width, self.height
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        separated_regions = 0

        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and self.layout[row][col] == ' ':
                    count = self.flood_fill(row, col, visited, self.layout)
                    if count > 0:
                        if self.layout[row][col] == ' ':
                            self.region_starting_points.append((row, col))

        chef_coordinates = self.region_starting_points[:4]

        for cor in chef_coordinates:
            print(cor)
        # chef_coordinates = [
        #     (5, 1),
        #     (4, 1),
        #     (4, 4),
        #     (2, 4)
        # ]
        #raise Exception("Sorry, no numbers below zero")
        for x, y in chef_coordinates:
            self.layout.append([str(x), " ", str(y)])
        # Write the generated layout to the specified file
        with open(self.file_path, 'w') as f:
            for row in self.layout:
                f.write("".join(row) + '\n')
        print("file made")

    def print_map(self, map):
        for row in map:
            print(''.join(row))


class CollabMap(BaseMap):

    def mutate(self, layout):
        rows, cols = len(layout), len(layout[0])
        mutation_rate = 0.001  # Adjust this as needed

        while True:
            #print("in loop")
            # Apply mutation to the layout
            for i in range(rows):
                for j in range(cols):
                    if random.random() < mutation_rate:
                        # Mutate the character at this position
                        new_character = random.choice(self.object_chars)
                        layout[i][j] = new_character

            layout = self.avoid_crowding(layout)
            # Check the number of separated regions
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            separated_regions = 0

            for row in range(rows):
                for col in range(cols):
                    if not visited[row][col] and layout[row][col] == ' ':
                        # If an unvisited empty space is found, perform flood-fill from that position
                        count = self.flood_fill(row, col, visited, layout)
                        if count > 0:
                            separated_regions += 1

            if separated_regions == 2 or separated_regions == 3:
                break  # Exit the loop if the layout has exactly 2 separated regions

        return layout
    
    def avoid_crowding(self, layout, density_threshold=0.5, max_iterations=100):
        rows, cols = len(layout), len(layout[0])
        empty_count = sum(row.count(' ') for row in layout)
        total_cells = rows * cols
        empty_density = empty_count / total_cells
        iterations = 0

        while empty_density < density_threshold and iterations < max_iterations:
            #print("trying to reduce overcrowding")
            # Find a non-empty cell and swap it with an empty cell
            non_empty_cells = [(i, j) for i in range(rows) for j in range(cols) if layout[i][j] != ' ']
            if non_empty_cells:
                i, j = random.choice(non_empty_cells)
                empty_cell = [(x, y) for x in range(rows) for y in range(cols) if layout[x][y] == ' ']
                if empty_cell:
                    x, y = random.choice(empty_cell)
                    layout[i][j], layout[x][y] = layout[x][y], layout[i][j]

            # Recalculate empty density
            empty_count = sum(row.count(' ') for row in layout)
            empty_density = empty_count / total_cells

            iterations += 1

        return layout
    
    def evaluate_fitness(self, map):
        blocked_score = 0
        rows, cols = self.height, self.width
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
                    count = self.flood_fill(row, col, visited, map)
                    if count > 0:
                        separated_regions += 1
        self.print_map(map)
        
        #print("separated regions:", separated_regions)
        if separated_regions == 2:
            separated_regions = distance_score*0.5
        collaboration_score = separated_regions + distance_score
        print("Collaboration Fitness:", collaboration_score)
        # print("separated regions:", separated_regions)
        # print("dist Score:", distance_score)
        return collaboration_score, separated_regions # Su

    # Add any other collaboration-specific methods as needed
    # def calculate_collab_fitness(self, map):
    #     blocked_score = 0
    #     rows, cols = self.height, self.width
    #     visited = [[False for _ in range(cols)] for _ in range(rows)]

    #     def find_object_positions(layout):
    #         pass

    #     def calculate_distance_score(object_positions):
    #         pass

    #     object_positions = find_object_positions(map)
    #     distance_score = calculate_distance_score(object_positions)
    #     separated_regions = 0
    #     collaboration_score = 0

    #     for row in range(rows):
    #         for col in range(cols):
    #             if not visited[row][col] and map[row][col] == ' ':
    #                 count = self.flood_fill(row, col, visited, map)
    #                 if count > 0:
    #                     separated_regions += 1

    #     if separated_regions == 2:
    #         separated_regions = distance_score * 0.5
    #     collaboration_score = separated_regions + distance_score

    #     return collaboration_score, separated_regions












# first random maps
    #evaluate and selects ones for production
    #evolve them, crossover and then mutate
    #that is 1 generation










# import random

# class Map:
#     def __init__(self, file_path, num_objects, arglist):
#         self.width, self.height = map(int, arglist.grid_size.split('x'))
#         self.arglist = arglist
#         self.num_objects = num_objects
#         self.file_path = file_path
#         self.object_chars = "tlp----/* "
#         self.layout = None
#         self.population_size = 10
#         self.num_generations = 3
#         self.population = None
#         self.region_starting_points = []
#         random.seed()

#     def generate_random_layout(self):
#         characters = list(self.object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
#         random.shuffle(characters)
#         layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]
#         return layout

#     def evaluate_fitness(self, map):
#         fitness_score = 0

#         if self.type == 'r':
#             fitness_score = self.calculate_random_fitness()
#         elif self.type == 's':
#             fitness_score = self.calculate_spread_fitness()
#         elif self.type == 'a':
#             fitness_score = self.calculate_collab_optional_fitness()
#         elif self.type == 't':
#             fitness_score = self.calculate_collab_fitness(map)
#         else:
#             print(f"Invalid grid type: {self.type}")

#         return (map,fitness_score)

#     def calculate_collab_fitness(self, map):
#         blocked_score = 0
#         rows, cols = self.height, self.width
#         visited = [[False for _ in range(cols)] for _ in range(rows)]

#         def find_object_positions(layout):
#             object_positions = {obj: [] for obj in ['p', 't', 'l', '/', '*']}
#             for row_idx, row in enumerate(layout):
#                 for col_idx, char in enumerate(row):
#                     if char in object_positions:
#                         object_positions[char].append((row_idx, col_idx))
#             return object_positions

#         def calculate_distance_score(object_positions):
#             distance_score = 0

#             # Define the weights for each distance
#             weight_t_to_slash = 0.3
#             weight_l_to_slash = 0.3
#             weight_slash_to_p = 0.4
#             weight_p_to_star = 0.3

#             # Calculate distances and update the distance score
#             if 't' in object_positions and '/' in object_positions:
#                 for tomato_pos in object_positions['t']:
#                     for slash_pos in object_positions['/']:
#                         distance_t_to_slash = abs(tomato_pos[0] - slash_pos[0]) + abs(tomato_pos[1] - slash_pos[1])
#                         distance_score += weight_t_to_slash * distance_t_to_slash
            
#             if 'l' in object_positions and '/' in object_positions:
#                 for lettuce_pos in object_positions['l']:
#                     for slash_pos in object_positions['/']:
#                         distance_l_to_slash = abs(lettuce_pos[0] - slash_pos[0]) + abs(lettuce_pos[1] - slash_pos[1])
#                         distance_score += weight_l_to_slash * distance_l_to_slash
            
#             if '/' in object_positions and 'p' in object_positions:
#                 for slash_pos in object_positions['/']:
#                     for plate_pos in object_positions['p']:
#                         distance_slash_to_p = abs(slash_pos[0] - plate_pos[0]) + abs(slash_pos[1] - plate_pos[1])
#                         distance_score += weight_slash_to_p * distance_slash_to_p
            
#                         if 't' in object_positions and 'l' in object_positions and '*' in object_positions:
#                             for delivery_pos in object_positions['*']:
#                                 distance_p_to_star = abs(plate_pos[0] - delivery_pos[0]) + abs(plate_pos[1] - delivery_pos[1])
#                                 distance_score += weight_p_to_star * distance_p_to_star

#             return distance_score
        
#         object_positions = find_object_positions(map)
#         distance_score = calculate_distance_score(object_positions)
#         separated_regions = 0
#         collaboration_score = 0

#         for row in range(rows):
#             for col in range(cols):
#                 if not visited[row][col] and map[row][col] == ' ':
#                     # If an unvisited empty space is found, perform flood-fill from that position
#                     count = self.flood_fill(row, col, visited, map)
#                     if count > 0:
#                         separated_regions += 1
#         #self.print_map(map)
        
#         #print("separated regions:", separated_regions)
#         if separated_regions == 2:
#             separated_regions = distance_score*0.5
#         collaboration_score = separated_regions + distance_score
#         #print("Collaboration Fitness:", collaboration_score)
#         # print("separated regions:", separated_regions)
#         # print("dist Score:", distance_score)
#         return collaboration_score, separated_regions # Su
    
#     def flood_fill(self, row, col, visited, layout):
#         """
#         Perform a flood-fill from the given position and mark all directly connected empty spaces.
#         """
#         rows, cols = self.height, self.width
#         if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or layout[row][col] != ' ':
#             return 0  # Return 0 if the current position is out of bounds or not an empty space

#         visited[row][col] = True

#         # Recursively perform flood-fill in all directions
#         count = 1  # Initialize count to 1 for the current empty space
#         count += self.flood_fill(row + 1, col, visited, layout)
#         count += self.flood_fill(row - 1, col, visited, layout)
#         count += self.flood_fill(row, col + 1, visited, layout)
#         count += self.flood_fill(row, col - 1, visited, layout)

#         return count

#     def crossover(self, map1, map2):
#         # Perform crossover between two parents to create a child
#         child_layout = [self.crossover_row(row1, row2) for row1, row2 in zip(map1, map2)]
#         #print(random.randint(1,10))
#         return child_layout

#     def crossover_row(self, row1, row2):
#         # Select a random crossover point along the row length
#         crossover_point = random.randint(1, min(len(row1), len(row2)) - 1)
#         # Combine the left part of row1 and the right part of row2
#         child_row = row1[:crossover_point] + row2[crossover_point:]
#         return child_row

#     def mutate(self, layout):
#         rows, cols = len(layout), len(layout[0])
#         mutation_rate = 0.001  # Adjust this as needed

#         while True:
#             #print("in loop")
#             # Apply mutation to the layout
#             for i in range(rows):
#                 for j in range(cols):
#                     if random.random() < mutation_rate:
#                         # Mutate the character at this position
#                         new_character = random.choice(self.object_chars)
#                         layout[i][j] = new_character

#             layout = self.avoid_crowding(layout)
#             # Check the number of separated regions
#             visited = [[False for _ in range(cols)] for _ in range(rows)]
#             separated_regions = 0

#             for row in range(rows):
#                 for col in range(cols):
#                     if not visited[row][col] and layout[row][col] == ' ':
#                         # If an unvisited empty space is found, perform flood-fill from that position
#                         count = self.flood_fill(row, col, visited, layout)
#                         if count > 0:
#                             separated_regions += 1

#             if separated_regions == 2 or separated_regions == 3:
#                 break  # Exit the loop if the layout has exactly 2 separated regions

#         return layout
    
#     def avoid_crowding(self, layout, density_threshold=0.5, max_iterations=100):
#         rows, cols = len(layout), len(layout[0])
#         empty_count = sum(row.count(' ') for row in layout)
#         total_cells = rows * cols
#         empty_density = empty_count / total_cells
#         iterations = 0

#         while empty_density < density_threshold and iterations < max_iterations:
#             #print("trying to reduce overcrowding")
#             # Find a non-empty cell and swap it with an empty cell
#             non_empty_cells = [(i, j) for i in range(rows) for j in range(cols) if layout[i][j] != ' ']
#             if non_empty_cells:
#                 i, j = random.choice(non_empty_cells)
#                 empty_cell = [(x, y) for x in range(rows) for y in range(cols) if layout[x][y] == ' ']
#                 if empty_cell:
#                     x, y = random.choice(empty_cell)
#                     layout[i][j], layout[x][y] = layout[x][y], layout[i][j]

#             # Recalculate empty density
#             empty_count = sum(row.count(' ') for row in layout)
#             empty_density = empty_count / total_cells

#             iterations += 1

#         return layout

#     def evolve_population(self, selected_maps):
#         next_generation = []
#         crossover_children = []
#         mutated_children = []
#         #make next gen same size as prev pop
#         while len(next_generation) < len(self.population):
#             # Choose parents for crossover
#             parent1 = random.choice(selected_maps)
#             parent2 = random.choice(selected_maps)
#             # Perform crossover to create offspring
#             child = self.crossover(parent1[0], parent2[0])
#             crossover_children.append(child)
#             # Apply mutation to the child
#             mutated_child = self.mutate(child)
#             mutated_children.append(mutated_child)
#             # Add the child to the next generation
#             next_generation.append(mutated_child)

#         print("CROSSOVER CHILDREN:")
#         self.print_horizontal_maps(crossover_children)

#         print("\nMUTATED CHILDREN:")
#         self.print_horizontal_maps(mutated_children)
#         return next_generation
    
#     def print_horizontal_maps(self, maps):
#         # Find the maximum number of rows in the maps
#         max_rows = max(len(map_) for map_ in maps)

#         # Print the maps horizontally with extra spaces between each map
#         for i in range(max_rows):
#             for map_ in maps:
#                 if i < len(map_):
#                     print(''.join(map_[i]), end="     ")
#                 else:
#                     print(" " * len(map_[0]), end="     ")
#             print()


#     def select_maps_for_reproduction(self):
        
#         best_maps = [self.evaluate_fitness(map_instance) for map_instance in self.population]
#         # Sort the list based on fitness in descending order
#         sorted_best_maps = sorted(best_maps, key=lambda x: x[1], reverse=True)

#         # Select the top 1/3 of the best maps
#         top_third = int(len(sorted_best_maps) / 3)
#         selected_maps = sorted_best_maps[:top_third]
#         print("Best map had score: ", selected_maps[-1][1])
#         return selected_maps

#     def genetic_algorithm(self):
#         # Generate an initial population
#         self.population = [self.generate_random_layout() for _ in range(self.population_size)]

#         for generation in range(self.num_generations):
#             # Evaluate fitness for each map in the population
#             # for map_instance in self.population:
#             #     self.evaluate_fitness(map_instance)

#             selected_maps = self.select_maps_for_reproduction()

#             # Create the next generation
#             self.population = self.evolve_population(selected_maps)

#             print("NEW POPULATION: ")
#             self.print_horizontal_maps(self.population)

#             # Optionally, you can keep track of the best map in each generation
#             #best_map = max(self.population, key=lambda x: x.fitness)
#             #print(f"Generation {generation + 1}, Best Fitness: {best_map.fitness}")

#         # Get the optimal map from the final generation
#         self.layout = max(self.population, key=lambda x: self.evaluate_fitness(x))
#         #self.print_map(self.layout)
#         print("FINAL MAP: ")
#         self.print_map(self.layout)
#         # Optionally, return or save the best map from the final generation
#         #return max(population, key=lambda x: x.fitness)

#     def generate_best_map(self):
#         pass

#     def generate_map(self):
#         self.type = self.arglist.grid_type.lower()
#         self.genetic_algorithm()
#         self.place_players_and_objects()

#     def find_empty_coordinates(self, region):
#         rows, cols = len(self.layout)-1, len(self.layout[0])
#         empty_coordinates = []
#         print (rows, cols)

#         for row in range(rows):
#             for col in range(cols):
#                 if 0 <= row < rows and 0 <= col < cols and self.layout[row][col] == ' ':
#                     # Add the empty coordinate if it's within the valid range
#                     empty_coordinates.append((row, col))

#         return empty_coordinates[:4]  # Return the first four empty coordinates

#     def place_players_and_objects(self):
#         self.layout.append(["\n", "SimpleTomato", "\n"])

#         # chef_region1 = self.find_empty_coordinates("Region 1")
#         # chef_region2 = self.find_empty_coordinates("Region 2")
#         # chef2_region1 = self.find_empty_coordinates("Region 1")
#         # chef2_region2 = self.find_empty_coordinates("Region 2")

#         # chef_coordinates = chef_region1 + chef_region2 + chef2_region1 + chef2_region2

#         rows, cols = self.width, self.height
#         visited = [[False for _ in range(cols)] for _ in range(rows)]

#         separated_regions = 0

#         for row in range(rows):
#             for col in range(cols):
#                 if not visited[row][col] and self.layout[row][col] == ' ':
#                     count = self.flood_fill(row, col, visited, self.layout)
#                     if count > 0:
#                         if self.layout[row][col] == ' ':
#                             self.region_starting_points.append((row, col))

#         chef_coordinates = self.region_starting_points[:4]

#         for cor in chef_coordinates:
#             print(cor)
#         # chef_coordinates = [
#         #     (5, 1),
#         #     (4, 1),
#         #     (4, 4),
#         #     (2, 4)
#         # ]
#         #raise Exception("Sorry, no numbers below zero")
#         for x, y in chef_coordinates:
#             self.layout.append([str(x), " ", str(y)])
#         # Write the generated layout to the specified file
#         with open(self.file_path, 'w') as f:
#             for row in self.layout:
#                 f.write("".join(row) + '\n')
#         print("file made")

#     def print_map(self, map):
#         for row in map:
#             print(''.join(row))
