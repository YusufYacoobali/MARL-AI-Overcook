import random

class BaseMap:
    def __init__(self, file_path, arglist):
        """
        Initialize the BaseMap instance.

        Parameters:
        file_path: Where to save the file.
        arglist: The arguments list.
        """
        try:
            self.size = int(arglist.grid_size)
            if not 4 <= self.size <= 20:
                raise ValueError("Error: Invalid grid size. Use a number between 4 and 20.")
        except ValueError:
            raise ValueError("Error: Invalid grid size. Please provide a valid number between 4 and 20.")
        self.arglist = arglist
        self.file_path = file_path
        self.object_chars = "tlp----/*"
        self.layout = None
        self.population_size = 100
        self.num_generations = 80
        self.population = None
        random.seed()

    def start(self):
        """
        This is the first function to be called, it checks various input values and makes the correct map type
        """
        dish = self.arglist.dish.lower()
        if dish not in ["simpletomato", "simplelettuce", "salad"]:
            raise ValueError("Error: Dish does not exist. Please choose from SimpleTomato, SimpleLettuce, or Salad.")
                
        map_type = self.arglist.grid_type.lower()
        if map_type == 'r':
            map_instance = RandomMap(self.file_path, self.arglist)
            map_instance.generate_map()
        elif map_type == 's':
            map_instance = GroupedTasksMap(self.file_path, self.arglist)
            map_instance.generate_map()
        elif map_type == 'o':
            map_instance = OptionalCollabMap(self.file_path, self.arglist)
            map_instance.generate_map()
        elif map_type == 't':
            map_instance = MandatoryCollabMap(self.file_path, self.arglist)
            map_instance.generate_map()
        else:
            raise ValueError("Error: Invalid map type. Please choose from t, o, s, r.")

    def generate_map(self):
        """
        Gets the final map which it processes and saves to be able to play the game
        """
        self.genetic_algorithm()
        self.post_processing()
        self.finish_file()

    def genetic_algorithm(self):
        """
        This function is responsible for making the map through various other functions.
        It creates the initial population, picks the best ones and evolves them over many generations.
        Then it returns the best layout made.
        """
        self.population = [self.generate_random_layout() for _ in range(self.population_size)]
        for generation in range(self.num_generations):
            selected_maps = self.select_maps_for_reproduction()
            self.population = self.evolve_population(selected_maps)
        self.layout = max(self.population, key=lambda x: self.evaluate_fitness(x))

    def select_maps_for_reproduction(self):
        """
        Selects the best one third of the population according to their fitness for the next stages of evolution
        Returns: selected_maps - the best maps of the current population
        """
        best_maps = [(map_instance, self.evaluate_fitness(map_instance)[0]) for map_instance in self.population]
        sorted_best_maps = sorted(best_maps, key=lambda x: x[1], reverse=True)
        top_third = int(len(sorted_best_maps) / 3)
        selected_maps = sorted_best_maps[:top_third]
        return selected_maps
    
    def evolve_population(self, selected_maps):
        """
        This method is responsible for evolving the entire population.
        It does this by taking the best of a population, crossing them over and then mutating them,
        until a new population of the same size is created

        Parameters: selected_maps - The best of the current population
        Returns: next_generation - the new population based on the best of the previous
        """
        next_generation = []
        while len(next_generation) < len(self.population):
            parent1 = random.choice(selected_maps)
            parent2 = random.choice(selected_maps)
            child = self.crossover(parent1[0], parent2[0])
            mutated_child = self.mutate(child)
            next_generation.append(mutated_child)
        return next_generation
    
    def crossover(self, map1, map2):
        """
        A crossover between the two maps is done by selecting a random point in each row to make a new child map

        Parameters: 
        map1 - the first map to crossover with
        map2 - the second map to crossover with
        Returns: child_layout - the newly created map
        """
        child_layout = [self.crossover_row(row1, row2) for row1, row2 in zip(map1, map2)]
        return child_layout

    def crossover_row(self, row1, row2):
        """
        Selects a random point in each row and makes the new row

        Parameters: 
        row1 - the first row to crossover with
        row2 - the second row to crossover with
        Returns: child_row - the newly created row
        """
        crossover_point = random.randint(1, min(len(row1), len(row2)) - 1)
        child_row = row1[:crossover_point] + row2[crossover_point:]
        return child_row

    def generate_random_layout(self):
        """
        This method makes a completely random layout.
        This is used to create the initial population and for the Random map type
        Returns: layout - a single randomly created map
        """
        characters = list(self.object_chars + " " * (self.size * self.size-1))
        random.shuffle(characters)
        layout = [characters[i:i + self.size] for i in range(0, self.size * self.size, self.size)]
        return layout
    
    def mutate(self, layout):
        """
        Placeholder for the mutate method
        Parameters: layout - The layout to be mutated
        """
        pass

    def evaluate_fitness(self, map):
        """
        Placeholder for the evaluate_fitness method
        Parameters: map - the map to be evaluated
        """
        pass

    def avoid_crowding(self, layout, density_threshold=0.6, max_iterations=100):
        """
        This method returns a less crowded map depending on the density_threshold

        Parameters: 
        layout - the map to check 
        density_threshold - how empty the map should be. 1 being fully empty.
        max_iterations - maximum number of iterations before breaking the while loop
        Returns: layout - the new layout
        """
        rows, cols = self.size, self.size
        empty_count = sum(row.count(' ') for row in layout)
        total_cells = rows * cols
        empty_density = empty_count / total_cells
        iterations = 0

        while empty_density < density_threshold and iterations < max_iterations:
            # Flatten the layout to a 1D list
            flat_layout = [char for row in layout for char in row]
            cells_to_convert = int((1 - density_threshold) * total_cells)
            non_empty_cells = [i for i, char in enumerate(flat_layout) if char != ' ']
            cells_to_convert_indices = random.sample(non_empty_cells, cells_to_convert)
            for index in cells_to_convert_indices:
                flat_layout[index] = ' '

            # Turn layout back to 2D
            layout = [flat_layout[i:i + cols] for i in range(0, total_cells, cols)]

            # check density again
            empty_count = sum(row.count(' ') for row in layout)
            empty_density = empty_count / total_cells
            iterations += 1
        return layout
    
    def post_processing(self):
        """
        Proccess the final layout that was selected by ensurinh that the level is playable.
        Limits or adds a delivery station to be only one on the map
        Adds a plate and slicing counter if there isn't one and adds missing ingredients for the selected dish
        """
        # Only leave 1 delivery station, change all others to regular counter
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
            self.add_ingredient('t') if 't' not in {char for row in self.layout for char in row} else None
            self.add_ingredient('l') if 'l' not in {char for row in self.layout for char in row} else None

        elif dish == "simpletomato":
            self.add_ingredient('t') if 't' not in {char for row in self.layout for char in row} else None

        elif dish == "simplelettuce":
            self.add_ingredient('l') if 'l' not in {char for row in self.layout for char in row} else None

        self.add_ingredient('p') if 'p' not in {char for row in self.layout for char in row} else None
        self.add_ingredient('/') if '/' not in {char for row in self.layout for char in row} else None

    def finish_file(self):
        """
        This method adds the dish and chef coordinates to the map file and saves it so it can be played.
        It also finds suitable coordinates to place the chefs into
        """
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        regions = []

        for row in range(self.size):
            for col in range(self.size):
                if not visited[row][col] and self.layout[row][col] == ' ':
                    _, region = self.getAllRegionCoordinates(row, col, visited, self.layout, [])
                    if region:  
                        regions.append(region)            

        # If the map is full with no space for the chefs, it tries 10 times before terminating
        max_attempts, attempts = 10, 0
        while not regions and attempts < max_attempts:
            self.start()
            attempts += 1
        if not regions:
            raise ValueError("Error: No valid spaces in the map after multiple attempts.")

        player_object_coordinates = []
        # Get coordinates in different regions to get chefs into
        i = 0
        while len(player_object_coordinates) < 4:
            region = None if len(regions) == 0 else regions[i % len(regions)]
            selected_coordinates = random.sample(region, min(2, len(region)))
            flipped_coordinates = [(col, row) for row, col in selected_coordinates]
            player_object_coordinates.extend(flipped_coordinates)
            i += 1

        new_order = [0, 2, 1, 3]  
        player_object_coordinates = [player_object_coordinates[i] for i in new_order]

       # Check if the dish is "simpletomato" or "simplelettuce"
        if self.arglist.dish.lower() == "simpletomato":
            dish_name = "SimpleTomato"
        elif self.arglist.dish.lower() == "simplelettuce":
            dish_name = "SimpleLettuce"
        else:
            dish_name = self.arglist.dish.capitalize()

        # Add dish to file 
        self.layout.append(["\n", dish_name, "\n"])

        #Add chef coordinates to file
        for x, y in player_object_coordinates:
            self.layout.append([str(x), " ", str(y)])

        # Write the layout to the map file
        with open(self.file_path, 'w') as f:
            for row in self.layout:
                f.write("".join(row) + '\n')
        print("Map file made")

    def add_ingredient(self, ingredient):
        """
        This function adds the ingredient to the final layout by swapping it with a regular counter
        Parameters: ingredient - the ingredient that needs to be added
        """
        valid_positions = [(i, j) for i, row in enumerate(self.layout) for j, char in enumerate(row) if char == '-']

        if not valid_positions:
            valid_positions = [(i, j) for i, row in enumerate(self.layout) for j, char in enumerate(row) if char == ' ']

        if valid_positions:
            i, j = random.choice(valid_positions)
            self.layout[i][j] = ingredient

    def count_separated_regions(self, layout, visited):
        """
        This function gets a map and returns the number of how many seperate regions there are.
        For example, if a kitchen is seperated in 2 by a line of counters, it counts as 2 regions

        Parameters: 
        layout - the map to check 
        visited - which cells in the layout have been visited
        Returns: separated_regions - how many regions there are in the layout
        """
        rows, cols = self.size, self.size
        separated_regions = 0

        for row in range(rows):
            for col in range(cols):
                if not visited[row][col] and layout[row][col] == ' ':
                    count = self.flood_fill(row, col, visited, layout)
                    if count > 0:
                        separated_regions += 1

        return separated_regions

    def flood_fill(self, row, col, visited, layout):
        """
        Perform a flood-fill from the given position and returns how many connected empty spaces there are.
        Helpful to check how many seperated regions there are in the layout.

        Parameters: 
        row - which row to start from
        col - which column to start from
        visited - which cells in the layout have been visited
        layout - the map to check 
        Returns: count - number of connected spaces in that region
        """
        rows, cols = self.size, self.size
        if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or layout[row][col] != ' ':
            return 0  # Return 0 if the current position is out of bounds or not an empty space

        visited[row][col] = True

        # Recursively perform flood-fill in all directions
        count = 1 
        count += self.flood_fill(row + 1, col, visited, layout)
        count += self.flood_fill(row - 1, col, visited, layout)
        count += self.flood_fill(row, col + 1, visited, layout)
        count += self.flood_fill(row, col - 1, visited, layout)
        return count
    
    def getAllRegionCoordinates(self, row, col, visited, layout, current_region):
        """
        A modified flood fill method that also returns all coordinates in that region.
        Used to see what counters are surrounding the region

        Parameters: 
        row - which row to start from
        col - which column to start from
        visited - which cells in the layout have been visited
        layout - the map to check 
        current_region - the region to check

        Returns: 
        count - number of connected spaces in that region
        current_region - the coordinates in the that region
        """
        rows, cols = self.size, self.size
        if row < 0 or row >= rows or col < 0 or col >= cols or visited[row][col] or layout[row][col] != ' ':
            return 0, current_region  

        visited[row][col] = True
        current_region.append((row, col))  

        count = 1 
        count += self.getAllRegionCoordinates(row + 1, col, visited, layout, current_region)[0]
        count += self.getAllRegionCoordinates(row - 1, col, visited, layout, current_region)[0]
        count += self.getAllRegionCoordinates(row, col + 1, visited, layout, current_region)[0]
        count += self.getAllRegionCoordinates(row, col - 1, visited, layout, current_region)[0]

        return count, current_region
    
    def get_surrounding_counters(self, regions, layout):
        """
        This method returns a list of what each region is surrounded by.

        Parameters: 
        regions - which row to start from
        layout - the map to check 
        Returns: surrounding_counters - All unique counter types surrounding each region
        """
        surrounding_counters = []

        for region in regions:
            unique_counters = set()

            for coord in region:
                row, col = coord
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

    def print_map(self, map):
        """
        Used for debugguing purposes
        Parameters: map - the map to be printed
        """
        for row in map:
            print(''.join(row))

    def print_horizontal_maps(self, maps):
        """
        Used for debugguing purposes. It prints the entire population.
        Parameters: maps - the maps to be printed
        """
        max_rows = max(len(map_) for map_ in maps)
        for i in range(max_rows):
            for map_ in maps:
                if i < len(map_):
                    print(''.join(map_[i]), end="     ")
                else:
                    print(" " * len(map_[0]), end="     ")
            print()


class MandatoryCollabMap(BaseMap):
    """
    Subclass of BaseMap representing a map for mandatory collaboration.

    This class inherits from BaseMap and introduces additional functionality specific to maps
    where collaboration between objects is mandatory.
    """
    def mutate(self, layout, max_iterations=100):
        """
        A specific implementation for mandatory collaboration.
        It mutates each cell on a side of a randomly generated line through the map.
        Mutation only occurs if the random value is greater than the mutation rate.

        Parameters: 
        layout - the map to check 
        max_iterations - maximum number of iterations before breaking the while loop
        Returns: layout - the new layout
        """
        rows, cols = self.size, self.size
        mutation_rate = 0.01 
        iterations = 0

        while True:
            # Choose two random points to create a line of separation
            point1 = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            point2 = (random.randint(0, rows - 1), random.randint(0, cols - 1))

            for i in range(rows):
                for j in range(cols):
                    # Check if the current point is on one side of the line or the other
                    side = (point2[0] - point1[0]) * (j - point1[1]) - (point2[1] - point1[1]) * (i - point1[0])
                    if random.random() < mutation_rate and side > 0:
                        new_character = random.choice(self.object_chars)
                        layout[i][j] = new_character

            # Make sure there is enough empty space in the map
            layout = self.avoid_crowding(layout)

            visited = [[False for _ in range(cols)] for _ in range(rows)]
            separated_regions = 0
            # Slightly different to counting seperated regions, it only recognises a region if it has more than self.size connected cells
            for row in range(rows):
                for col in range(cols):
                    if not visited[row][col] and layout[row][col] == ' ':
                        count = self.flood_fill(row, col, visited, layout)
                        if count > self.size:
                            separated_regions += 1

            if separated_regions == 2 or separated_regions == 3 or iterations >= max_iterations:
                break 
            iterations += 1
        return layout

    def evaluate_fitness(self, map):
        """
        This method evaluates the map it is given. 
        It is done by seeing how many regions there are and what surrounding counters are common to all regions

        Parameters: map - the map to evaluate 
        Returns: 
        fitness - total fitness value of the map
        separated_regions - how many seperate regions there are
        """
        fitness = 1
        rows, cols = self.size, self.size

        def calculate_score(surrounding_counters):
            """
            This method gives a score depending on if a region has a counter type the other regions do not have.
            This encourages the maps to have distinct counters on each side so that the chefs must work together.

            Parameters: surrounding_counters - the counters surrounding each region
            Returns: score - the score for counter uniqueness
            """
            unique_elements = set(element for lst in surrounding_counters for element in lst)
            score = 0
            # Check if each unique element exists in any other list
            for unique_element in unique_elements:
                count = sum(unique_element in lst for lst in surrounding_counters)
                if count == 1:
                    score += 3
            return score

        visited = [[False for _ in range(cols)] for _ in range(rows)]
        separated_regions = self.count_separated_regions(map, visited=visited)

        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        regions_cords = []
        for row in range(self.size):
            for col in range(self.size):
                if not visited[row][col] and map[row][col] == ' ':
                    _, region = self.getAllRegionCoordinates(row, col, visited, map, [])
                    if region:  
                        regions_cords.append(region)            

        # Get region coordinates, see whats surrounding it and calculate a score off unqiueness
        surrounding_counters = self.get_surrounding_counters(regions_cords, map)
        surrounding_counters = [[coord for coord in region if coord != '-'] for region in surrounding_counters]
        surrounding_score = calculate_score(surrounding_counters=surrounding_counters)
        
        if separated_regions == 2:
            fitness = fitness + 20
        elif separated_regions == 1 or separated_regions == 3:
            fitness = fitness + 2
        fitness += surrounding_score
        return fitness, separated_regions


class OptionalCollabMap(BaseMap):
    """
    Subclass of BaseMap representing a map for optional collaboration.

    This class inherits from BaseMap and introduces additional functionality specific to maps
    where collaboration between objects is optional.
    """
    def mutate(self, layout, max_iterations=100):
        """
        An implementation for optional collaboration.
        It mutates each cell in the layout at a random rate if it is greater than the mutation rate.

        Parameters: 
        layout - the map to check 
        max_iterations - maximum number of iterations before breaking the while loop
        Returns: layout - the new layout
        """
        rows, cols = self.size, self.size
        mutation_rate = 0.001  
        iterations = 0

        while True:
            for i in range(rows):
                for j in range(cols):
                    if random.random() < mutation_rate:
                        new_character = random.choice(self.object_chars)
                        layout[i][j] = new_character
            # Make sure there is enough empty space in the map
            layout = self.avoid_crowding(layout)
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            separated_regions = 0
            # Slightly different to counting seperated regions, it only recognises a region if it has more than self.size connected cells
            for row in range(rows):
                for col in range(cols):
                    if not visited[row][col] and layout[row][col] == ' ':
                        count = self.flood_fill(row, col, visited, layout)
                        if count > (self.size):
                            separated_regions += 1
            if separated_regions == 2 or separated_regions == 3 or iterations >= max_iterations:
                break  
            iterations += 1
        return layout
    
    def evaluate_fitness(self, map):
        """
        This method evaluates the map it is given. 
        It is done by seeing how many regions there are and what surrounding counters are common to all regions

        Parameters: map - the map to evaluate 
        Returns: 
        fitness - total fitness value of the map
        separated_regions - how many seperate regions there are
        """
        fitness = 1
        rows, cols = self.size, self.size

        def calculate_score(surrounding_counters):
            """
            This method gives a score depending on if a region has a counter type that all other regions have
            This encourages the maps to have the same counters on each side so that the chefs do not have to work together.

            Parameters: surrounding_counters - the counters surrounding each region
            Returns: score - the score for counter uniqueness
            """
            score = 0
            # Check if each element is common to all lists
            for element in surrounding_counters[0]:
                if all(element in lst for lst in surrounding_counters[1:]):
                    score += 10
            return score

        visited = [[False for _ in range(cols)] for _ in range(rows)]
        separated_regions = self.count_separated_regions(map, visited=visited)

        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        regions_cords = []
        for row in range(self.size):
            for col in range(self.size):
                if not visited[row][col] and map[row][col] == ' ':
                    _, region = self.getAllRegionCoordinates(row, col, visited, map, [])
                    if region:  
                        regions_cords.append(region)        

        # Get region coordinates, see whats surrounding it and calculate a score off unqiueness        
        surrounding_counters = self.get_surrounding_counters(regions_cords, map)
        surrounding_counters = [[coord for coord in region if coord != '-'] for region in surrounding_counters]
        surrounding_score = calculate_score(surrounding_counters=surrounding_counters)
        
        if separated_regions == 2:
            fitness = fitness + 30
        elif separated_regions == 1 or separated_regions == 3:
            fitness = fitness + 2
        fitness += surrounding_score
        return fitness, separated_regions


class GroupedTasksMap(BaseMap):
    """
    Subclass of BaseMap representing a map for grouped tasks.

    This class inherits from BaseMap and introduces additional functionality specific to maps
    where tasks are grouped together.
    """
    def mutate(self, layout, max_iterations = 40):
        """
        An implementation for grouping tasks together.
        It mutates each cell in the layout at a random rate by considering its neighbors.

        Parameters: 
        layout - the map to check 
        max_iterations - maximum number of iterations before breaking the while loop
        Returns: layout - the new layout
        """
        rows, cols = self.size, self.size
        mutation_rate = 0.001  

        # Check which of its neighbours is most common and change into that
        for _ in range(max_iterations):
            for i in range(rows):
                for j in range(cols):
                    if random.random() < mutation_rate and layout[i][j] == ' ':
                        neighbors = self.get_neighbors(i, j, layout)
                        if neighbors:
                            char_counts = {char: neighbors.count(char) for char in self.object_chars}
                            new_character = max(char_counts, key=char_counts.get)
                            layout[i][j] = new_character

            layout = self.avoid_crowding(layout)
            # Still apply randomness
            if random.random() < mutation_rate:
                random.shuffle(layout)
        return layout

    def get_neighbors(self, row, col, layout):
        """
        A method to get the neighbours of a cell

        Parameters: 
        row - which row to start from
        col - which column to start from
        layout - the map to check 
        Returns: neighbors - get the neighbours of a cell
        """
        rows, cols = len(layout), len(layout[0])
        neighbors = []

        for i in range(max(0, row - 1), min(rows, row + 2)):
            for j in range(max(0, col - 1), min(cols, col + 2)):
                if (i, j) != (row, col):
                    neighbors.append(layout[i][j])
        return neighbors
    
    def evaluate_fitness(self, map):
        """
        This method evaluates the map it is given. 
        It is done by seeing how close each type of counter is and averging it out.

        Parameters: map - the map to evaluate 
        Returns: 
        fitness - total fitness value of the map
        separated_regions - how many seperate regions there are
        """
        fitness_score = 0
        counters_to_score = ['/','l','t','p']

        # Group the counters and their coordinates
        counter_groups = {char: [] for char in self.object_chars if char != ' '}
        for row_idx, row in enumerate(map):
            for col_idx, char in enumerate(row):
                if char in counter_groups:
                    counter_groups[char].append((row_idx, col_idx))

        # Calculate position between each of them and average them out for each type
        for counter_type in counters_to_score:
            positions = counter_groups.get(counter_type, [])
            num_counters = len(positions)

            if num_counters > 1:
                total_distance = 0
                for pos1 in positions:
                    for pos2 in positions:
                        distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                        total_distance += distance

                average_distance = total_distance / (num_counters * (num_counters - 1))
                normalized_distance = average_distance / (self.size + self.size)

                # Score is from 0 to 10, 10 being the best case and it gets smaller as distance increases
                score = max(0, 10 - normalized_distance * 10)
                fitness_score += score
        return fitness_score, 0


class RandomMap(BaseMap):
    """
    Subclass of BaseMap representing completely random maps.

    This class inherits from BaseMap and introduces additional functionality specific to maps
    where it needs to be random.
    """
    def mutate(self, layout):
        """
        No functionality here, randomness needs no mutation
        Parameters: layout - the map to check 
        Returns: layout - the new layout
        """
        return layout
    
    def evaluate_fitness(self, map):
        """
        This method evaluates the map it is given. 
        It is done by giving the best score to the one which is half full, half empty

        Parameters: map - the map to evaluate 
        Returns: 
        fitness - total fitness value of the map
        separated_regions - how many seperate regions there are
        """
        best_score = self.size * self.size * 0.3
        empty_spaces = sum(row.count(' ') for row in map)
        difference = abs(empty_spaces - best_score * (self.size * 0.35))
        score = best_score - difference
        return score, 0