import numpy as np
from copy import deepcopy
import traceback
from agents import *
from ui import UIEngine
from time import sleep, time
import click
import logging
from store import AGENT_REGISTRY
from constants import *
import sys
from helpers import count_capture, execute_move, check_endgame, random_move, get_valid_moves

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)

class World:
    def __init__(
        self,
        player_1="random_agent",
        player_2="random_agent",
        board_size=None,
        display_ui=False,
        display_delay=0.4,
        display_save=False,
        display_save_path=None,
        autoplay=False,
    ):
        """
        Initialize the game world

        Parameters
        ----------
        player_1: str
            The registered class of the first player
        player_2: str
            The registered class of the second player
        board_size: int
            The size of the board. If None, board_size = a number between MIN_BOARD_SIZE and MAX_BOARD_SIZE
        display_ui : bool
            Whether to display the game board
        display_delay : float
            Delay between each step
        display_save : bool
            Whether to save an image of the game board
        display_save_path : str
            The path to save the image
        autoplay : bool
            Whether the game is played in autoplay mode
        """
        # Two players
        logger.info("Initialize the game world")
        # Load agents as defined in decorators
        self.player_1_name = player_1
        self.player_2_name = player_2
        if player_1 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_1}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )
        if player_2 not in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{player_2}' is not registered. {AGENT_NOT_FOUND_MSG}"
            )

        p0_agent = AGENT_REGISTRY[player_1]
        p1_agent = AGENT_REGISTRY[player_2]
        logger.info(f"Registering p0 agent : {player_1}")
        self.p0 = p0_agent()
        logger.info(f"Registering p1 agent : {player_2}")
        self.p1 = p1_agent()

        # check autoplay
        if autoplay:
            if not self.p0.autoplay or not self.p1.autoplay:
                raise ValueError(
                    f"Autoplay mode is not supported by one of the agents ({self.p0} -> {self.p0.autoplay}, {self.p1} -> {self.p1.autoplay}). Please set autoplay=True in the agent class."
                )

        self.player_names = {PLAYER_1_ID: PLAYER_1_NAME, PLAYER_2_ID: PLAYER_2_NAME}

        if board_size is None:
            # Random board size, ensuring even numbers for Reversi Othello
            self.board_size = np.random.choice([6, 8, 10, 12])
            logger.info(
                f"No board size specified. Randomly generating size: {self.board_size}x{self.board_size}"
            )
        else:
            self.board_size = board_size
            logger.info(f"Setting board size to {self.board_size}x{self.board_size}")

        # Initialize the game board (0: empty, 1: black disc, 2: white disc)
        self.chess_board = np.zeros((self.board_size, self.board_size), dtype=int)

        # Initialize the center discs for Reversi Othello
        mid = self.board_size // 2
        self.chess_board[mid - 1][mid - 1] = 2  # White
        self.chess_board[mid - 1][mid] = 1      # Black
        self.chess_board[mid][mid - 1] = 1      # Black
        self.chess_board[mid][mid] = 2          # White

        # Whose turn to step
        self.turn = 0

        # Time taken by each player
        self.p0_time = []
        self.p1_time = []

        # Cache to store and use the data
        self.results_cache = ()
        # UI Engine
        self.display_ui = display_ui
        self.display_delay = display_delay
        self.display_save = display_save
        self.display_save_path = display_save_path

        if display_ui:
            # Initialize UI Engine
            logger.info(
                f"Initializing the UI Engine, with display_delay={display_delay} seconds"
            )
            self.ui_engine = UIEngine(self.board_size, self)
            self.render()

    def get_current_player(self):
        """
        Get the current player (1: Black, 2: White)
        """
        return 1 if self.turn == 0 else 2
    
    def get_current_opponent(self):
        """
        Get the opponent player (1: Black, 2: White)
        """
        return 2 if self.turn == 0 else 1

    def update_player_time(self, time_taken):
        """
        Update the time taken by the player

        Parameters
        ----------
        time_taken : float
            Time taken by the player
        """
        if not self.turn:
            self.p0_time.append(time_taken)
        else:
            self.p1_time.append(time_taken)

    def step(self):
        """
        Take a step in the game world.
        Runs the agents' step function and updates the game board accordingly.
        If the agents' step function raises an exception, the step will be replaced by a Random Move.

        Returns
        -------
        results: tuple
            The results of the step containing (is_endgame, player_1_score, player_2_score)
        """
        cur_player = self.get_current_player()
        opponent = self.get_current_opponent()

        valid_moves = get_valid_moves(self.chess_board,cur_player)

        if not valid_moves:
            logger.info(f"Player {self.player_names[self.turn]} must pass due to having no valid moves.")
        else:
            time_taken = None
            try:
                # Run the agent's step function
                start_time = time()
                move_pos = self.get_current_agent().step(
                    deepcopy(self.chess_board),
                    cur_player,
                    opponent,
                )
                time_taken = time() - start_time
                self.update_player_time(time_taken)

                if count_capture(self.chess_board, move_pos, cur_player) == 0:
                    raise ValueError(f"Invalid move by player {cur_player}: {move_pos}")

            except BaseException as e:
                ex_type = type(e).__name__
                if (
                    "SystemExit" in ex_type and isinstance(self.get_current_agent(), HumanAgent)
                ) or "KeyboardInterrupt" in ex_type:
                    sys.exit(0)
                print(
                    "An exception raised. The traceback is as follows:\n{}".format(
                        traceback.format_exc()
                    )
                )
                print("Executing Random Move!")
                move_pos = random_move(self.chess_board,cur_player)

            # Execute move
            execute_move(self.chess_board,move_pos, cur_player)
            logger.info(
                f"Player {self.player_names[self.turn]} places at {move_pos}. Time taken this turn (in seconds): {time_taken}"
            )

        # Change turn
        self.turn = 1 - self.turn

        results = check_endgame(self.chess_board, self.get_current_player(),self.get_current_opponent())
        self.results_cache = results

        # Render board and show results
        if self.display_ui:
            self.render()
            if results[0]:
                click.echo("Press a button to exit the game.")
                try:
                    _ = click.getchar()
                except:
                    _ = input()

        return results

    def get_current_agent(self):
        """
        Get the current player's agent

        Returns
        -------
        agent : object
            The agent object of the current player
        """
        return self.p0 if self.turn == 0 else self.p1

    def render(self, debug=False):
        """
        Render the game board using the UI Engine
        """
        self.ui_engine.render(self.chess_board, debug=debug)
        sleep(self.display_delay)

if __name__ == "__main__":
    world = World()
    is_end, p0_score, p1_score = world.step()
    while not is_end:
        is_end, p0_score, p1_score = world.step()
    print(p0_score, p1_score)