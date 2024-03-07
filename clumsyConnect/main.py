import random
import time

try:
    import numpy as np
except ImportError:
    print("Sorry, this example requires Numpy installed !")
    raise

from easyAI import TwoPlayerGame


class ConnectFour(TwoPlayerGame):
    """
    The game of Connect Four, as described here:
    http://en.wikipedia.org/wiki/Connect_Four
    """
    time_since_move = time.time()
    time_sum = 0
    number_of_moves = 0

    def __init__(self, players, board=None):
        self.players = players
        self.board = (
            board
            if (board is not None)
            else (np.array([[0 for i in range(7)] for j in range(6)]))
        )
        self.current_player = 1  # player 1 starts.

    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    def make_move(self, column):
        # flip to change between probabilistic and deterministic
        # True - probabilistic
        # False - deterministic
        if False:
            random.seed(time.time())
            random_move = random.randint(-1, 1)
            new_column = column + random_move
            # stay on the board
            if new_column >= 7:
                new_column -= 2
            if new_column < 0:
                new_column += 2
            # if column is full, don't slip
            if np.argmin(self.board[:, new_column] != 0):
                column = new_column

        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.current_player

    def show(self):
        time_diff = time.time() - self.time_since_move
        self.time_sum += time_diff
        print("Current player", self.current_player)
        print(
            "\n"
            + "\n".join(
                ["0 1 2 3 4 5 6", 13 * "-"]
                + [
                    " ".join([[".", "O", "X"][self.board[5 - j][i]] for i in range(7)])
                    for j in range(6)
                ]
            )
        )
        self.time_since_move = time.time()
        self.number_of_moves += 1

    def lose(self):
        return find_four(self.board, self.opponent_index)

    def is_over(self):
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0


def find_four(board, current_player):
    """
    Returns True iff the player has connected  4 (or more)
    This is much faster if written in C or Cython
    """
    for pos, direction in POS_DIR:
        streak = 0
        while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
            if board[pos[0], pos[1]] == current_player:
                streak += 1
                if streak == 4:
                    return True
            else:
                streak = 0
            pos = pos + direction
    return False


class expecti_minimax:
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, game):
        self.opponent_index = 3 - game.current_player
        self.time_since_move = time.time()
        self.time_sum = 0
        self.number_of_moves = 0
        scores = []
        for move in game.possible_moves():
            game_copy = game.copy()
            game_copy.make_move(move)
            game_copy.switch_player()
            score = self.mini(game_copy, self.depth, float("-inf"), float("inf"))
            scores.append((score, move))
        print(scores)
        print("Time to take decision:", time.time() - self.time_since_move)
        return max(scores)[1]

    def maxi(self, game, depth, alpha, beta):
        if game.is_over() or depth == 0:
            return -game.scoring()

        value = float("-inf")

        for move in game.possible_moves():
            game_copy = game.copy()
            game_copy.make_move(move)
            game_copy.switch_player()
            value = max(value, self.mini(game_copy, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def mini(self, game, depth, alpha, beta):
        if game.is_over() or depth == 0:
            if game.scoring() != 0:
                print("Scoring:", game.scoring())
            return game.scoring()

        value = float("inf")

        for move in game.possible_moves():
            game_copy = game.copy()
            game_copy.make_move(move)
            game_copy.switch_player()
            value = min(value, self.maxi(game_copy, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value



POS_DIR = np.array(
    [[[i, 0], [0, 1]] for i in range(6)]
    + [[[0, i], [1, 0]] for i in range(7)]
    + [[[i, 0], [1, 1]] for i in range(1, 3)]
    + [[[0, i], [1, 1]] for i in range(4)]
    + [[[i, 6], [1, -1]] for i in range(1, 3)]
    + [[[0, i], [1, -1]] for i in range(3, 7)]
)

if __name__ == "__main__":
    # LET'S PLAY !

    from easyAI import AI_Player, Negamax

    ai_algo_neg_one = Negamax(5)
    ai_algo_neg_two = Negamax(5)

    ai_expecti_minimax = expecti_minimax(5)

    player_one_score = 0
    player_two_score = 0
    draw_count = 0

    for i in range(10):
        # choose player to start
        game = ConnectFour([AI_Player(ai_algo_neg_one), AI_Player(ai_expecti_minimax)]) \
            if i % 2 == 0 \
            else ConnectFour([AI_Player(ai_expecti_minimax), AI_Player(ai_algo_neg_one)])
        game.play()

        if game.lose():
            print("Player %d wins." % game.opponent_index)
            avg_time = game.time_sum / game.number_of_moves
            print(f"Average time to take decision: {game.time_sum / game.number_of_moves}")
            # player one started
            print("Oponent index:", game.opponent_index)
            if i % 2 == 0:
                player_one_score += game.opponent_index == 1
                player_two_score += game.opponent_index == 2
            # player two started
            else:
                player_one_score += game.opponent_index == 2
                player_two_score += game.opponent_index == 1
        else:
            draw_count += 1

    print("Player 1 score:", player_one_score)
    print("Player 2 score:", player_two_score)
    print("Draw count:", draw_count)
