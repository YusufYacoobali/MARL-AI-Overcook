import random

class Map:
    def __init__(self, file_path, num_objects, arglist):
        self.width, self.height = map(int, arglist.grid_size.split('x'))
        self.arglist = arglist
        self.num_objects = num_objects
        self.file_path = file_path
        self.layout = None
        self.object_chars = "tlop/-/*"
        self.layout = None

    def generate_random_layout(self):
        characters = list(self.object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
        random.shuffle(characters)
        self.layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]

    def generate_spread_out_layout(self):
        # Implement logic for spread-out map generation
        # Workstations are far apart, and cooperation is not required

        # Set a minimum distance between workstations
        min_distance = 3
        # Number of workstations
        num_workstations = self.num_objects
        # Create an empty layout
        self.layout = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Place workstations randomly with a minimum distance
        for _ in range(num_workstations):
            x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
            
            # Check minimum distance from existing workstations
            while any(abs(x - wx) < min_distance and abs(y - wy) < min_distance for wx, wy in self.workstations):
                x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
            
            self.workstations.append((x, y))
            self.layout[y][x] = random.choice(self.object_chars)

    def generate_all_blocked_layout(self):
        # Implement logic for all blocked map generation
        # A wall is put in the middle, and both sides have all workstations
        # Create an empty layout
        self.layout = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Create a vertical wall in the middle
        wall_x = self.width // 2
        for y in range(self.height):
            self.layout[y][wall_x] = "|"

        # Place workstations on both sides of the wall
        for x in range(wall_x):
            for y in range(1, self.height - 1):
                self.layout[y][x] = random.choice(self.object_chars)

        for x in range(wall_x + 1, self.width - 1):
            for y in range(1, self.height - 1):
                self.layout[y][x] = random.choice(self.object_chars)

    def generate_trapped_layout(self):
        # Create an empty layout
        self.layout = [[" " for _ in range(self.width)] for _ in range(self.height)]
        # Determine the location of the blocking wall
        #its the same each time, need to FIX
        wall_x, wall_y = random.randint(1, self.width - 1), random.randint(1, self.height - 1)
        print(wall_x, wall_y)

        random.seed()
        characters = list(self.object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
        random.shuffle(characters)
        self.layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]
               # Place the blocking walls to create a gap in the middle
        for y in range(self.height):
            for x in range(self.width):
                if x == wall_x and y <= wall_y:
                    self.layout[y][x] = "-"
                elif y == wall_y and x < wall_x:
                    self.layout[y][x] = "-"

    def generate_best_map(self):
        pass

    def generate_map(self):
        type = self.arglist.grid_type.lower()
        if type == 'r':
            self.generate_random_layout()
        elif type == 's':
            self.generate_spread_out_layout()
        elif type == 'a':
            self.generate_all_blocked_layout()
        elif type == 't':
            self.generate_trapped_layout()
        else:
            print(f"Invalid grid type: {type}")
        self.place_players_and_objects()

    def place_players_and_objects(self):
        # object_chars = "tlop/-/*"
        # random.seed()
        # characters = list(object_chars * self.num_objects + " " * (self.width * self.height - self.num_objects))
        # random.shuffle(characters)
        # layout = [characters[i:i + self.width] for i in range(0, self.width * self.height, self.width)]
       # player_locations = [(1, 1), (self.width - 2, 1), (self.width - 2, self.height - 2), (1, self.height - 2)]
        # for x, y in player_locations:
        #     self.layout[y][x] = 'p'

        # Place objects ('t', 'l', 'o', 'p') on the map
        # for _ in range(self.num_objects):
        #     x = random.randint(1, self.width - 2)
        #     y = random.randint(1, self.height - 2)
        #     self.layout[y][x] = random.choice(self.object_chars)

        self.layout.append(["\n", "Salad", "\n"])

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
