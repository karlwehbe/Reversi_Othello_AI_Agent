import numpy as np

from agents.agent import Agent
from helpers import random_move
from store import register_agent


# Important: you should register your agent with a name
@register_agent("random_agent")
class RandomAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True

    def step(self, chess_board, player, opponent):
        """
        Randomly selects a valid position to place a disc.

        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).

        Returns
        -------
        move_pos : tuple of int
            The position (x, y) where the player places the disc.
        """

        return random_move(chess_board, player)
