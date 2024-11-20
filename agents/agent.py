class Agent:
    def __init__(self):
        """
        Initialize the agent, add a name which is used to register the agent
        """
        self.name = "DummyAgent"
        # Flag to indicate whether the agent can be used to autoplay
        self.autoplay = True

    def __str__(self) -> str:
        return self.name

    def step(self, chess_board, player, opponent):        
        """
        Main decision logic of the agent, which is called by the simulator.
        Extend this method to implement your own agent to play the game.

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
        pass
