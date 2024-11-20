# Human Input agent
import sys
from helpers import get_valid_moves
from agents.agent import Agent
from store import register_agent


@register_agent("human_agent")
class HumanAgent(Agent):
    def __init__(self):
        super(HumanAgent, self).__init__()
        self.name = "HumanAgent"

    def step(self, chess_board, player, opponent):
        """
        Get human input for the position to place the disc

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
            The position (r,c) where the player places the disc.
        """
        print(get_valid_moves(chess_board, player))
        text = input("Your move (row,column) or input q to quit: ")
        while len(text.split(",")) != 2 and "q" not in text.lower():
            print("Wrong Input Format! Input should be row,column.")
            text = input("Your move (row,column) or input q to quit: ")

        if "q" in text.lower():
            print("Game ended by user!")
            sys.exit(0)

        x, y = text.split(",")
        x, y = int(x.strip()), int(y.strip())

        while not self.check_valid_input(x, y, chess_board):
            print(
                "Invalid Move! (row,column) should be within the board and the position must be empty."
            )
            text = input("Your move (row,column) or input q to quit: ")
            while len(text.split(",")) != 2 and "q" not in text.lower():
                print("Wrong Input Format! Input should be row,column.")
                text = input("Your move (row,column) or input q to quit: ")
            if "q" in text.lower():
                print("Game ended by user!")
                sys.exit(0)
            x, y = text.split(",")
            x, y = int(x.strip()), int(y.strip())

        return (x, y)

    def check_valid_input(self, x, y, chess_board):
        """
        Check if the input position is valid (within the board and the spot is empty)

        Parameters
        ----------
        x : int
            The x position on the board.
        y : int
            The y position on the board.
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black, and 2 for white.

        Returns
        -------
        bool
            True if the input is valid, False otherwise.
        """
        board_size = chess_board.shape[0]
        return 0 <= x < board_size and 0 <= y < board_size and chess_board[x, y] == 0
