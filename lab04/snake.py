import pygame
import sys
import random

# Game settings
WINDOW_SIZE = 800
BLOCK_SIZE = 10
SPEED = 10
NUMBER_OF_OBSTACLES = 10

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((WINDOW_SIZE // 2), (WINDOW_SIZE // 2))]
        self.direction = random.choice([(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)])
        self.color = (0, 255, 0)
        self.obstacles = []
        self.generate_obstacles()  # Generowanie przeszkód przy inicjalizacji węża

    def get_head_position(self):
        return self.positions[0]

    def turn(self, direction):
        if self.length > 1 and (direction[0]*-1, direction[1]*-1) == self.direction:
            return
        else:
            self.direction = direction

    def generate_obstacles(self):
        # TODO: generacja bordera
        # TODO: fill obstacles
        def is_valid_position(x, y):
            # Sprawdza, czy pozycja dla przeszkody jest wystarczająco odległa od granicy ekranu i innych przeszkód
            min_distance_from_border = BLOCK_SIZE * 3
            min_distance_from_obstacles = BLOCK_SIZE * 3

            # Sprawdzanie odległości od granicy ekranu
            if x < min_distance_from_border or x > WINDOW_SIZE - min_distance_from_border or \
               y < min_distance_from_border or y > WINDOW_SIZE - min_distance_from_border:
                return False

            # Sprawdzanie odległości od innych przeszkód
            for obstacle_x, obstacle_y in self.obstacles:
                if abs(obstacle_x - x) < min_distance_from_obstacles and abs(obstacle_y - y) < min_distance_from_obstacles:
                    return False

            return True

        def generate_shape(x, y):
            # Losuje jeden z trzech kształtów przeszkód
            shapes = [
                # Krzyż
                [(0, -BLOCK_SIZE), (0,0), (0, BLOCK_SIZE), (-BLOCK_SIZE, 0), (BLOCK_SIZE, 0)], 
                # L
                [(0, -BLOCK_SIZE), (0, 0), (0, BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE)],
                # T
                [(-BLOCK_SIZE, -BLOCK_SIZE), (0, 0), (0, -BLOCK_SIZE), (BLOCK_SIZE, -BLOCK_SIZE), (0, BLOCK_SIZE)]
            ]
            return random.choice(shapes)

        for i in range(NUMBER_OF_OBSTACLES):
            valid = False
            while not valid:
                # Losujemy pozycję dla przeszkody
                x = random.randint(0, WINDOW_SIZE // BLOCK_SIZE - 1) * BLOCK_SIZE
                y = random.randint(0, WINDOW_SIZE // BLOCK_SIZE - 1) * BLOCK_SIZE

                if is_valid_position(x, y):
                    valid = True
                    shape = generate_shape(x, y)

                    # Dodajemy przeszkodę na planszę
                    for dx, dy in shape:
                        self.obstacles.append((x + dx, y + dy))

    def draw_obstacles(self, surface):
        for p in self.obstacles:
            pygame.draw.rect(surface, (0, 0, 255), (p[0], p[1], BLOCK_SIZE, BLOCK_SIZE))

    def move(self, surface):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + x) % WINDOW_SIZE), (cur[1] + y) % WINDOW_SIZE)
        if new in self.positions[2:] or new in self.obstacles:
            # self.reset()
            # TODO: zatrzymaj grę, wyświetl napis "Game Over", zakończ grę po kilku sekundach
            pygame.quit()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [((WINDOW_SIZE // 2), (WINDOW_SIZE // 2))]
        self.direction = random.choice([(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)])
        self.blocks.clear()  # Wyczyszczenie listy przeszkód
        self.generate_block()  # Ponowne wygenerowanie przeszkód przy resecie



    def draw(self, surface):
        for p in self.positions:
            pygame.draw.rect(surface, self.color, (p[0], p[1], BLOCK_SIZE, BLOCK_SIZE))
        for p in self.obstacles:
            pygame.draw.rect(surface, (0, 0, 255), (p[0], p[1], BLOCK_SIZE, BLOCK_SIZE))


class Food:
    def __init__(self, obstacles):
        self.position = (0, 0)
        self.color = (255, 0, 0)
        self.randomize_position()
        while self.position in obstacles:
            self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, WINDOW_SIZE//BLOCK_SIZE - 1) * BLOCK_SIZE, random.randint(0, WINDOW_SIZE//BLOCK_SIZE - 1) * BLOCK_SIZE)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

def play_game():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()

    snake = Snake()
    food = Food(snake.obstacles)

    # add borders to obstacles
    for i in range(0, WINDOW_SIZE, BLOCK_SIZE):
        snake.obstacles.append((i, 0))
        snake.obstacles.append((i, WINDOW_SIZE - BLOCK_SIZE))
        snake.obstacles.append((0, i))
        snake.obstacles.append((WINDOW_SIZE - BLOCK_SIZE, i))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.turn((0, -BLOCK_SIZE))
                elif event.key == pygame.K_DOWN:
                    snake.turn((0, BLOCK_SIZE))
                elif event.key == pygame.K_LEFT:
                    snake.turn((-BLOCK_SIZE, 0))
                elif event.key == pygame.K_RIGHT:
                    snake.turn((BLOCK_SIZE, 0))

        snake.move(surface)

        if snake.get_head_position() == food.position:
            snake.length += 1
            food.randomize_position()

        surface.fill((0, 0, 0))
        snake.draw_obstacles(surface)
        snake.draw(surface)
        food.draw(surface)

        screen.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(SPEED)

if __name__ == "__main__":
    play_game()