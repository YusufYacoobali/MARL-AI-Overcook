import random
#call start to generate map for each type
#generate map does genetic, processing, final touches like recipe, place chefs

#genetic does

class BaseMap:
    def __init__(self, file_path, num_objects, arglist):
        self.width, self.height = map(int, arglist.grid_size.split('x'))
        self.arglist = arglist
        self.num_objects = num_objects
        self.file_path = file_path
        self.object_chars = "tlp----/*"
        self.layout = None
        self.population_size = 10
        self.num_generations = 3
        self.population = None
        self.region_starting_points = []
        self.regions = []
        random.seed()

    def start(self):
        dish = self.arglist.dish.lower()
        if dish not in ["simpletomato", "simplelettuce", "salad"]:
            raise ValueError("Error: Dish does not exist. Please choose from SimpleTomato, SimpleLettuce, or Salad.")
        
        map_type = self.arglist.grid_type.lower()
        if map_type == 'r':
            map_instance = RandomMap(self.file_path, self.num_objects, self.arglist)
            map_instance.generate_map()
        elif map_type == 's':
            map_instance = GroupedTasksMap(self.file_path, self.num_objects, self.arglist)
            map_instance.generate_map()
        elif map_type == 'o':
            map_instance = OptionalCollabMap(self.file_path, self.num_objects, self.arglist)
            map_instance.generate_map()
        elif map_type == 't':
            map_instance = MandatoryCollabMap(self.file_path, self.num_objects, self.arglist)
            map_instance.generate_map()
        else:
            #print(f"Invalid grid type: {self.type}")
            raise ValueError("Error: Invalid map type. Please choose from t, o, s, r.")


    def generate_map(self):
        self.genetic_algorithm()
        self.post_processing()
        self.place_players_and_objects()

    def genetic_algorithm(self):
        # Generate an initial population
        self.population = [self.generate_random_layout() for _ in range(self.population_size)]
        for generation in range(self.num_generations):
            selected_maps = self.select_maps_for_reproduction()
            print("Maps selected:")
            self.print_horizontal_maps([map_[0] for map_ in selected_maps])
            # Create the next generation
            self.population = self.evolve_population(selected_maps)
            print("NEW POPULATION: ")
            self.print_horizontal_maps(self.population)
        # Get the optimal map from the final generation
        self.layout = max(self.population, key=lambda x: self.evaluate_fitness(x))
        print("FINAL MAP: ")
        self.print_map(self.layout)

    def select_maps_for_reproduction(self):
        best_maps = [(map_instance, self.evaluate_fitness(map_instance)[0]) for map_instance in self.population]
        sorted_best_maps = sorted(best_maps, key=lambda x: x[1], reverse=True)
        # Select the top 1/3 of the best maps
        top_third = int(len(sorted_best_maps) / 3)
        selected_maps = sorted_best_maps[:top_third]
        return selected_maps
    
    def evolve_population(self, selected_maps):
        next_generation = []
        crossover_children = []
        mutated_children = []
        #make next gen same size as prev population
        while len(next_generation) < len(self.population):
            # Choose parents for crossover
            parent1 = random.choice(selected_maps)
            parent2 = random.choice(selected_maps)
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

    def generate_random_layout(self):
        characters = list(self.object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
        random.shuffle(characters)
        layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]
        return layout
    
    def mutate(self, layout):
        pass

    def evaluate_fitness(self, map):
        pass

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
    
    def post_processing(self):
        # Only leave 1 '*', change all others to '-'
        star_positions = [(i, j) for i, row in enumerate(self.layout) for j, char in enumerate(row) if char == '*']
        
        if len(star_positions) > 1:
            for i, j in star_positions[1:]:
                self.layout[i][j] = '-'
        elif len(star_positions) < 1:
            i, j = random.choice([(i, j) for i in range(len(self.layout)) for j in range(len(self.layout[0]))])
            self.layout[i][j] = '*'

        for i, j in star_positions:
            empty_neighbors = 0
            non_empty_neighbor_positions = []
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dir_i, dir_j in directions:
                new_i, new_j = i + dir_i, j + dir_j
                if 0 <= new_i < len(self.layout) and 0 <= new_j < len(self.layout[0]):
                    if self.layout[new_i][new_j] == ' ':
                        empty_neighbors += 1
                    elif self.layout[new_i][new_j] != '*':
                        non_empty_neighbor_positions.append((new_i, new_j))

            if empty_neighbors < 2:
                # If '*' has less than 2 empty space neighbors, change a non-empty neighbor to ' '
                if non_empty_neighbor_positions:
                    new_i, new_j = random.choice(non_empty_neighbor_positions)
                    self.layout[new_i][new_j] = ' '

        dish = self.arglist.dish.lower()

        if dish == "salad":
            # Check if 't' is present in the layout, if not, add it
            if 't' not in {char for row in self.layout for char in row}:
                self.add_ingredient('t')
            # Check if 'l' is present in the layout, if not, add it
            if 'l' not in {char for row in self.layout for char in row}:
                self.add_ingredient('l')

        elif dish == "simpletomato":
            # Check if 't' is present in the layout, if not, add it
            if 't' not in {char for row in self.layout for char in row}:
                self.add_ingredient('t')

        elif dish == "simplelettuce":
            # Check if 'l' is present in the layout, if not, add it
            if 'l' not in {char for row in self.layout for char in row}:
                self.add_ingredient('l')

        # Make sure there is at least 1 'p' and 1 '/' in the map, if not then add one
        if 'p' not in {char for row in self.layout for char in row}:
            # If 'p' doesn't exist, add it in a random position with '-'
            i, j = random.choice([(i, j) for i, row in enumerate(self.layout) for j, char in enumerate(row) if char == '-'])
            self.layout[i][j] = 'p'

        if '/' not in {char for row in self.layout for char in row}:
            # If '/' doesn't exist, add it in a random position with '-'
            i, j = random.choice([(i, j) for i, row in enumerate(self.layout) for j, char in enumerate(row) if char == '-'])
            self.layout[i][j] = '/'

    def place_players_and_objects(self):

        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        regions = []

        for row in range(self.height):
            for col in range(self.width):
                if not visited[row][col] and self.layout[row][col] == ' ':
                    _, region = self.getAllRegionCoordinates(row, col, visited, self.layout, [])
                    if region:  # Ensure the region is not empty
                        regions.append(region)            
        print(regions)
        result = self.get_surrounding_counters(regions, self.layout)
        region_list_no_dash = [[coord for coord in region if coord != '-'] for region in result]
        print("ADJACENT VALUES: ", region_list_no_dash)

        player_object_coordinates = []
        
        # Get coordinates in different regions to get chefs into
        i = 0
        while len(player_object_coordinates) < 4:
            region = regions[i % len(regions)]
            selected_coordinates = random.sample(region, min(2, len(region)))
            flipped_coordinates = [(col, row) for row, col in selected_coordinates]
            player_object_coordinates.extend(flipped_coordinates)
            i += 1

        new_order = [0, 2, 1, 3]  # Change the order as needed
        player_object_coordinates = [player_object_coordinates[i] for i in new_order]
        print("Chef coordinates:", player_object_coordinates)

        #Add dish to file
        self.layout.append(["\n", self.arglist.dish.capitalize(), "\n"])

        #Add chef coordinates to file
        for x, y in player_object_coordinates:
            self.layout.append([str(x), " ", str(y)])

        # Write the layout to the map file
        with open(self.file_path, 'w') as f:
            for row in self.layout:
                f.write("".join(row) + '\n')
        print("file made")

    def add_ingredient(self, ingredient):
        i, j = random.choice([(i, j) for i, row in enumerate(self.layout) for j, char in enumerate(row) if char == '-'])
        self.layout[i][j] = ingredient

    def print_map(self, map):
        for row in map:
            print(''.join(row))

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

    def count_separated_regions(self, layout, visited):
        rows, cols = len(layout), len(layout[0])
        print("rows", rows)
        print("cols", cols)
        separated_regions = 0

        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and layout[row][col] == ' ':
                    # If an unvisited empty space is found, perform flood-fill from that position
                    count = self.flood_fill(row, col, visited, layout)
                    if count > self.width:
                        separated_regions += 1

        return separated_regions

    def flood_fill(self, row, col, visited, layout):
        """
        Perform a flood-fill from the given position and mark all directly connected empty spaces.
        """
        rows, cols = self.width, self.height
        # print(f"Checking: row={row}, col={col}, rows={rows}, cols={cols}")
        # #print(f"Visited: {visited[row][col]}")
        # self.print_map(layout)
        # print(f"Layout: {layout}")
        # print(f"Layout: {layout[row][col]}")

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
    
    #gets all coordinates of the current region
    def getAllRegionCoordinates(self, row, col, visited, layout, current_region):
        rows, cols = self.width, self.height
        if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or layout[row][col] != ' ':
            return 0, current_region  # Return 0 and the current_region list if the current position is out of bounds or not an empty space

        visited[row][col] = True
        current_region.append((row, col))  # Store the current coordinate in the list

        # Recursively perform flood-fill in all directions
        count = 1  # Initialize count to 1 for the current empty space
        count += self.getAllRegionCoordinates(row + 1, col, visited, layout, current_region)[0]
        count += self.getAllRegionCoordinates(row - 1, col, visited, layout, current_region)[0]
        count += self.getAllRegionCoordinates(row, col + 1, visited, layout, current_region)[0]
        count += self.getAllRegionCoordinates(row, col - 1, visited, layout, current_region)[0]

        return count, current_region
    
    def get_surrounding_counters(self, regions, layout):
        surrounding_counters = []

        for region in regions:
            unique_counters = set()

            for coord in region:
                row, col = coord

                # Define the horizontal and vertical directions
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                for direction in directions:
                    new_row, new_col = row + direction[0], col + direction[1]

                    # Check if the new position is within bounds
                    if 0 <= new_row < len(layout) and 0 <= new_col < len(layout[0]):
                        value = layout[new_row][new_col]

                        # Only consider non-empty values
                        if value != ' ':
                            unique_counters.add(value)

            surrounding_counters.append(list(unique_counters))

        return surrounding_counters

class MandatoryCollabMap(BaseMap):

    def mutate(self, layout, max_iterations=100):
        rows, cols = len(layout), len(layout[0])
        mutation_rate = 0.01  # Adjust this as needed
        iterations = 0

        while True:
            # Apply mutation to the layout
            # Choose two random points to create a line of separation
            point1 = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            point2 = (random.randint(0, rows - 1), random.randint(0, cols - 1))

            for i in range(rows):
                for j in range(cols):
                    # Check if the current point is on one side of the line or the other
                    side = (point2[0] - point1[0]) * (j - point1[1]) - (point2[1] - point1[1]) * (i - point1[0])

                    if random.random() < mutation_rate and side > 0:
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
                        if count > self.width:
                            separated_regions += 1

            if separated_regions == 2 or separated_regions == 3 or iterations >= max_iterations:
                break  # Exit the loop if the layout has exactly 2 separated regions or max iterations are reached
            iterations += 1

        return layout
    
    def avoid_crowding(self, layout, density_threshold=0.4, max_iterations=100):
        rows, cols = len(layout), len(layout[0])
        empty_count = sum(row.count(' ') for row in layout)
        total_cells = rows * cols
        empty_density = empty_count / total_cells
        iterations = 0

        while empty_density < density_threshold and iterations < max_iterations:
            # Flatten the layout to a 1D list
            flat_layout = [char for row in layout for char in row]
            # Calculate the number of cells to convert to empty
            cells_to_convert = int((1 - density_threshold) * total_cells)
            # Get indices of non-empty cells
            non_empty_indices = [i for i, char in enumerate(flat_layout) if char != ' ']
            # Randomly select cells to convert to empty
            cells_to_convert_indices = random.sample(non_empty_indices, cells_to_convert)
            # Convert selected cells to empty
            for index in cells_to_convert_indices:
                flat_layout[index] = ' '

            # Reshape the flat list back to a 2D layout
            layout = [flat_layout[i:i + cols] for i in range(0, total_cells, cols)]

            # Recalculate empty density
            empty_count = sum(row.count(' ') for row in layout)
            empty_density = empty_count / total_cells
            iterations += 1

        return layout

    def evaluate_fitness(self, map):
        blocked_score = 0
        rows, cols = self.width, self.height
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
        #self.print_map(map)
        
        #print("separated regions:", separated_regions)
        if separated_regions == 2:
            collaboration_score += distance_score*3
        else:
            collaboration_score == distance_score*0.5
        print("Collaboration Fitness:", collaboration_score)
        print("separated regions:", separated_regions)
        # print("dist Score:", distance_score)
        return collaboration_score, separated_regions # Su
        

#must have all counters on each seperation 
    # still need >1 regions
    # check both regions have p (plate), / (cutting), * (delivery), t or l or both (tomato lettuce)
    # that will have highest fitness value
class OptionalCollabMap(BaseMap):

    def mutate(self, layout, max_iterations=100):
        rows, cols = len(layout), len(layout[0])
        mutation_rate = 0.001  # Adjust this as needed
        iterations = 0

        while True:
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
                        if count > (self.width):
                            separated_regions += 1
            if separated_regions == 2 or separated_regions == 3 or iterations >= max_iterations:
                break  # Exit the loop if the layout has exactly 2 separated regions or max iterations are reached
            iterations += 1

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
        print("Evaluating OPTIONAL")
        blocked_score = 0
        rows, cols = self.width, self.height
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        def find_object_positions(layout):
            object_positions = {obj: [] for obj in ['p', 't', 'l', '/', '*']}
            for row_idx, row in enumerate(layout):
                for col_idx, char in enumerate(row):
                    if char in object_positions:
                        object_positions[char].append((row_idx, col_idx))
            return object_positions

        def get_adjacent_values(row, col, visited, layout):
            rows, cols = self.height, self.width
            adjacent_values = set()

            # Define the horizontal and vertical directions
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            for direction in directions:
                new_row, new_col = row + direction[0], col + direction[1]

                # Check if the new position is within bounds
                if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]:
                    visited[new_row][new_col] = True
                    value = layout[new_row][new_col]

                    # Only consider non-empty values
                    if value != ' ':
                        adjacent_values.add(value)

            return adjacent_values

        def get_region_counters(region, layout):
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            counters = []

            for pos in region:
                row, col = pos
                adjacent_values = get_adjacent_values(row, col, visited, layout)
                counters.extend(adjacent_values)

            return counters

        object_positions = find_object_positions(map)
        separated_regions = 0
        collaboration_score = 0

        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and map[row][col] == ' ':
                    # If an unvisited empty space is found, perform flood-fill from that position
                    count = self.flood_fill(row, col, visited, map)
                    if count > self.width:
                        separated_regions += 1
                        # Get counters for the adjacent values of each position in the region
                        # region = [(r, c) for r in range(rows) for c in range(cols) if visited[r][c]]
                        # counters = get_region_counters(region, map)
                        #print("Region Counters:", counters)
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        regions = []

        for row in range(self.height):
            for col in range(self.width):
                if not visited[row][col] and map[row][col] == ' ':
                    _, region = self.getAllRegionCoordinates(row, col, visited, map, [])
                    if region:  # Ensure the region is not empty
                        regions.append(region)            
        print("REGIONS FOR adjacent: " , regions)

        result = self.get_surrounding_counters(regions, map)
        print("ADJACENT VALUES: ", result)
        self.print_map(map)
        print("--------")

        if separated_regions != 2:
            collaboration_score = 0
        else:
            collaboration_score += 2000
        print("Collaboration Fitness:", collaboration_score)
        print("Separated regions:", separated_regions)

        return collaboration_score, separated_regions


#tasks grouped together
    # random amount of regions
    # check how many of each counter and how close they are
    # check both regions have p (plate), / (cutting), * (delivery), t or l or both (tomato lettuce)
    # that will have highest fitness value
class GroupedTasksMap(BaseMap):

    def mutate(self, layout, max_iterations = 40):
        rows, cols = len(layout), len(layout[0])
        mutation_rate = 0.001  # Adjust this as needed

        for _ in range(max_iterations):
            # Apply mutation to the layout
            for i in range(rows):
                for j in range(cols):
                    if random.random() < mutation_rate and layout[i][j] == ' ':
                        # If it's an empty space, mutate based on neighbors
                        neighbors = self.get_neighbors(i, j, layout)

                        if neighbors:
                            # Count occurrences of each character in the neighborhood
                            char_counts = {char: neighbors.count(char) for char in self.object_chars}
                            # Choose the most common character in the neighborhood
                            new_character = max(char_counts, key=char_counts.get)
                            layout[i][j] = new_character

            if random.random() < mutation_rate:
                # Randomly shuffle the layout to introduce more randomness
                random.shuffle(layout)
        return layout

    def get_neighbors(self, row, col, layout):
        """
        Get the values of neighboring cells.
        """
        rows, cols = len(layout), len(layout[0])
        neighbors = []

        for i in range(max(0, row - 1), min(rows, row + 2)):
            for j in range(max(0, col - 1), min(cols, col + 2)):
                if (i, j) != (row, col):
                    neighbors.append(layout[i][j])

        return neighbors
    
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
        # Calculate the fitness score based on the number of counters and their distances

        # Sample implementation (you can customize this based on your needs)
        fitness_score = 0
        counters_to_score = ['/','l','t','p',]

        # Group counters based on their type
        counter_groups = {char: [] for char in self.object_chars if char != ' '}
        for row_idx, row in enumerate(map):
            for col_idx, char in enumerate(row):
                if char in counter_groups:
                    counter_groups[char].append((row_idx, col_idx))

        # Iterate over counters to score
        for counter_type in counters_to_score:
            positions = counter_groups.get(counter_type, [])
            num_counters = len(positions)

            if num_counters > 1:
                # Calculate total pairwise distances within this counter type
                total_distance = 0
                for pos1 in positions:
                    for pos2 in positions:
                        distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                        total_distance += distance

                # Calculate average distance
                average_distance = total_distance / (num_counters * (num_counters - 1))
                # Normalize the average distance based on the map size
                normalized_distance = average_distance / (self.width + self.height)

                # Calculate score based on distance and number of counters
                score = max(0, 10 - normalized_distance * 10)
                fitness_score += score

        print(f"Fitness Score: {fitness_score}")
        return fitness_score, 0
    
#check for no patterns
    #choose number of counters depending on grid size?
    #mutate by random?
    #evaluate by no pattern somehow?
class RandomMap(BaseMap):

    def mutate(self, layout):
        return layout
    
    def evaluate_fitness(self, map):
        best_score = self.width * self.height * 0.5
        empty_spaces = sum(row.count(' ') for row in map)
        difference = abs(empty_spaces - best_score)
        score = best_score - difference
        return score,0
