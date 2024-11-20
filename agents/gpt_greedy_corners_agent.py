# greedy_corners_agent.py
#
# This file is the direct output of ChatGPT. You can start from 
# a prompt you give to GPT and likely can get even better play than this one.
# You will have to cite your sources including providing the full prompt you used
# as your starting point in the report, but there is no penalty for doing so.
# 
# We have played all GPT agents we could get against code written by real human
# AI designers. We think most of the 424 students can outperform what GPT has to 
# offer, but you are free to use whatever method you find most suitable!
#

from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, count_capture, execute_move, check_endgame
import copy
import random

@register_agent("gpt_greedy_corners_agent")
class StudentAgent(Agent):
    """
    A custom agent for playing Reversi/Othello.
    """

    def __init__(self):
        super().__init__()
        self.name = "gpt_greedy_corners_agent"

    def step(self, board, color,opponent):
        """
        Choose a move based on an improved heuristic logic.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).

        Returns:
        - Tuple (x, y): The coordinates of the chosen move.
        """
        # Get all legal moves for the current player
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        # Advanced heuristic: prioritize corners and maximize flips while minimizing opponent's potential moves
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            _, player_score, opponent_score = check_endgame(simulated_board, color, 3 - color)
            move_score = self.evaluate_board(simulated_board, color, player_score, opponent_score)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        # Return the best move found
        return best_move if best_move else random.choice(legal_moves)

    def evaluate_board(self, board, color, player_score, opponent_score):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -10

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Combine scores
        total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score
        return total_score

# Ensure to test with:
# python simulator.py --player_1 student_agent --player_2 random_agent --display
