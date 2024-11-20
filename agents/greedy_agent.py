# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import random
import math

@register_agent("greedy_agent")

class GreedyAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
  
    def __init__(self):
        super(GreedyAgent, self).__init__()
        self.name = "GreedyAgent"
        self.tree = []   
        self.curr_pos_moves = []

    def step(self, board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
        where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
        and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
    
        start_time = time.time()
        self.curr_pos_moves = []
        possible_moves = get_valid_moves(board, player)
        
        if len(possible_moves) == 1 : return possible_moves[0]

        for move in self.tree : 
            row, col = move[0]  # Assuming move[0] is the row and move[1] is the column
            if board[row, col] != 0:  # Check if the cell on the board is not empty
                self.tree.remove(move)

        for move in possible_moves : 
            for tree_move in self.tree:
                if move == tree_move[0]:  # If the move is present in both possible_moves and self.tree
                    self.curr_pos_moves.append([move, tree_move[1], tree_move[2], 0])  # Append with score and visits from tree
            if not any(tree_move[0] == move for tree_move in self.tree):
                self.curr_pos_moves.append([move, 0, 1, 0])  # Default score and visits and uct value
                self.tree.append([move, 0, 1])
        
        best_move = self.mcts(board, player, opponent)
        
        for move in self.tree : 
            if move[0] == best_move : 
                self.tree.remove(move)

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds with", len(possible_moves), "possible moves")
        return best_move
    


    def mcts(self, board, player, opponent):
        start_time = time.time()
        
        time_limit = 1.95
        match board.shape[0] :
            case 6 : 
                if (self.is_opening_game(board)) :
                    time_limit = 1.97
                elif (self.is_midgame(board)) :
                    time_limit = 1.94
                else : 
                    time_limit = 1.982

            case 8 : 
                if (self.is_opening_game(board)) :
                    time_limit = 1.96
                elif (self.is_midgame(board)) :
                    time_limit = 1.93
                else : 
                    time_limit = 1.98

            case 10 : 
                if (self.is_opening_game(board)) :
                    time_limit = 1.93
                elif (self.is_midgame(board)) :
                    time_limit = 1.90
                else : 
                    time_limit = 1.98

            case 12 : 
                if (self.is_opening_game(board)) :
                    time_limit = 1.90
                elif (self.is_midgame(board)) :
                    time_limit = 1.85
                else : 
                    time_limit = 1.98

        while (time.time() - start_time < time_limit) :
            for i in range(10000):  # Limit number of simulations
                if time.time() - start_time >= time_limit: break
                total_visits = sum(m[2] for m in self.tree)
                best_search_move = max(self.curr_pos_moves, key=lambda m: self.uct(board, m, total_visits))
                move = best_search_move[0]
                score = self.simulate_game(board, move, player, opponent)
                self.update_score_and_visits(move, score) 

        best_move = max(self.curr_pos_moves, key=lambda  move: self.heuristic_score(board, move, player, opponent))
        return best_move[0]


    def update_score_and_visits(self, move, score):
        for m in self.curr_pos_moves:
            if m[0] == move:
                m[1] += score
                m[2] += 1
        for m in self.tree:
            if m[0] == move:
                m[1] += score
                m[2] += 1
    

    def simulate_game(self, board, move, player, opponent):     
        board_copy = deepcopy(board)
        execute_move(board_copy, move, player)
        result = 0

        curr_player = opponent  
        while not check_endgame(board_copy, player, opponent)[0]:  
            valid_moves = get_valid_moves(board_copy, curr_player)
            if valid_moves:
                #if curr_player == player and (board.shape == 6 or board.shape == 8) : 
                   # best_move = max(self.curr_pos_moves, key=lambda m: self.heuristic_score(board, m, player, opponent))
                   # execute_move(board_copy, best_move, curr_player)
                #else :
                random_move = random.choice(valid_moves)
                execute_move(board_copy, random_move, curr_player)
            
            curr_player = player if curr_player == opponent else opponent
            
        if player == 1 :
            _, player_score, opponent_score = check_endgame(board_copy, player, opponent)
        else : 
            _, opponent_score, player_score = check_endgame(board_copy, player, opponent)
        
        result = player_score - opponent_score
        if result > 0:
            score = 1  
        elif result < 0:
            score = -1  
        else:
            score = 0 
        return score


    def uct(self, board, move, total_visits):
        score = move[1]
        visits = move[2]
        
        c = 1.414  # Exploration constant
        if self.is_endgame(board) : c = 1
        elif self.is_opening_game(board) : c = 2

        if (visits == 0) : 
            return 0
        else : 
            uct_value = (score / visits) + c * (math.sqrt(math.log(total_visits) / visits))
            move[3] = uct_value
        return uct_value


    def heuristic_score(self, board, move_stats, player, opponent): 
        uct_val = move_stats[3]
        move = move_stats[0]
        
        if board.shape[0] == 6 :
            st_fp = 1
            st_cp = 1
            mid_cp = 1
            mid_cc = 0.5
            mid_omp = 1
            end_c = 0.5
            end_cc = 0.5

        elif board.shape[0] == 8 :
            st_fp= 1.5
            st_cp  = 1
            mid_cp = 1
            mid_cc = 1
            mid_omp = 1.5
            end_c = 2
            end_cc = 1
        else : 
            st_fp = 1.5
            st_cp = 1
            mid_cp = 1.5
            mid_cc = 1
            mid_omp = 1
            end_c = 2
            end_cc = 1
    
        if self.is_opening_game(board):
            center_control = self.evaluate_control_of_center(board, move, player, opponent) * 2
            center_proximity = -self.distance_to_center(board, move)  * st_cp # Lower is better
            corner_control = self.corner_control(board, move, player, opponent)
            flips_penalty = -count_capture(board, move, player) * st_fp
            opponent_moves_penalty = -self.count_opponent_moves(board, move, player, opponent) * mid_omp
            if (opponent_moves_penalty == 0) :
                opponent_moves_penalty = opponent_moves_penalty + 10
            score = flips_penalty + uct_val + corner_control + center_proximity + opponent_moves_penalty + center_control
            #print(f"Opening Game - Move : {move}, center_control: {center_control}, center_proximity: {center_proximity}, corner_control: {corner_control}, flips_penalty: {flips_penalty}, opponent_moves_penalty: {opponent_moves_penalty}, score: {score}")
            return score

        elif self.is_midgame(board):
            center_control = self.evaluate_control_of_center(board, move, player, opponent) * 2
            opp_corner = self.opp_corner_next(board, move, player, opponent)
            wall_for_corner = self.wall_for_corner(board, move, opponent)
            corner_control = self.corner_control(board, move, player, opponent) * mid_cc
            wall_move = self.evaluate_wall_move(board, move, player, opponent) * 0.5
            opponent_moves_penalty = -self.count_opponent_moves(board, move, player, opponent) * mid_omp
            if (opponent_moves_penalty == 0) :
                opponent_moves_penalty = opponent_moves_penalty + 10
            center_proximity = -self.distance_to_center(board, move)  * mid_cp
            flips_penalty = -count_capture(board, move, player) * 0.5
            score = center_proximity + opponent_moves_penalty + corner_control + uct_val + flips_penalty + opp_corner + wall_for_corner + wall_move + center_control
            #print(f"Midgame - Move : {move}, center_control: {center_control}, opp_corner: {opp_corner}, wall_for_corner: {wall_for_corner}, corner_control: {corner_control}, wall_move: {wall_move}, opponent_moves_penalty: {opponent_moves_penalty}, center_proximity: {center_proximity}, flips_penalty: {flips_penalty}, score: {score}")
            return score
    
        elif self.is_endgame(board):
            center_control = self.evaluate_control_of_center(board, move, player, opponent) * 2
            opp_corner = self.opp_corner_next(board, move, player, opponent)
            wall_for_corner = self.wall_for_corner(board, move, opponent)
            wall_move = self.evaluate_wall_move(board, move, player, opponent)
            corner_control = self.corner_control(board, move, player, opponent) * end_cc
            opponent_moves_penalty = -self.count_opponent_moves(board, move, player, opponent)
            if (opponent_moves_penalty == 0) :
                opponent_moves_penalty = opponent_moves_penalty + 10
            captures = count_capture(board, move, player) * end_c
            score = opponent_moves_penalty + (captures * end_c) + corner_control + uct_val + wall_move + wall_for_corner + opp_corner + center_control
            #print(f"Endgame - Move : {move}, center_control: {center_control}, opp_corner: {opp_corner}, wall_for_corner: {wall_for_corner}, wall_move: {wall_move}, corner_control: {corner_control}, opponent_moves_penalty: {opponent_moves_penalty}, captures: {captures}, score: {score}")
            return score
        
        else :
            opp_corner = self.opp_corner_next(board, move, player, opponent)
            wall_for_corner = self.wall_for_corner(board, move, opponent)
            opponent_moves_penalty = -self.count_opponent_moves(board, move, player, opponent)
            if (opponent_moves_penalty == 0) :
                opponent_moves_penalty = opponent_moves_penalty + 10
            captures = self.best_score(board, move, player, opponent) * end_c
            corner_control = self.corner_control(board, move, player, opponent) * 0.25
            score = opponent_moves_penalty + captures + uct_val + corner_control + opp_corner
            #print(f"Other Game - Move : {move}, opp_corner: {opp_corner}, wall_for_corner: {wall_for_corner}, opponent_moves_penalty: {opponent_moves_penalty}, captures: {captures}, corner_control: {corner_control}, score: {score}")
            return score


    def is_opening_game(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        # Threshold for startgame based on board size
        board_size = board.shape[1]
        if board_size == 6:
            return captured < 8  # ~15-20% of 36
        elif board_size == 8:
            return captured < 16  # ~15-20% of 64
        elif board_size == 10:
            return captured < 27  # ~15-20% of 100
        elif board_size == 12:
            return captured < 37  # ~15-20% of 144
        return False
  

    def is_midgame(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        # Threshold for midgame based on board size
        board_size = board.shape[1]
        if board_size == 6:
            return 8 <= captured < 24  # ~20-80% of 36
        elif board_size == 8:
            return 16 <= captured < 50  # ~20-80% of 64
        elif board_size == 10:
            return 27 <= captured < 80  # ~20-80% of 100
        elif board_size == 12:
            return 37 <= captured < 115  # ~20-80% of 144
        return False
    

    def is_endgame(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        # Threshold for endgame based on board size
        board_size = board.shape[1]
        if board_size == 6:
            return 24 <= captured < 32 # ~80-100% of 36
        elif board_size == 8:
            return 50 <= captured < 61 # ~80-100% of 64
        elif board_size == 10:
            return 80 <= captured < 97  # ~80-100% of 100
        elif board_size == 12:
            return 115 <= captured < 140  # ~80-100% of 144
        return False


    def is_finalmoves(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        # Threshold for endgame based on board size
        board_size = board.shape[1]
        if board_size == 6:
            return captured >= 33  # ~80-100% of 36
        elif board_size == 8:
            return captured >= 61  # ~80-100% of 64
        elif board_size == 10:
            return captured >= 97  # ~80-100% of 100
        elif board_size == 12:
            return captured >= 140  # ~80-100% of 144
        return False
    

    def is_danger_zone(self, board, move) :
        danger_zones = [
            (0, 1), (1, 0), (1, 1),  # Top-left corner
            (0, board.shape[0] - 2), (1, board.shape[0] - 1), (1, board.shape[0] - 2),  # Top-right corner
            (board.shape[0] - 2, 0), (board.shape[0] - 1, 1), (board.shape[0] - 2, 1),  # Bottom-left corner
            (board.shape[0] - 2, board.shape[0] - 2), (board.shape[0] - 1, board.shape[0] - 2), (board.shape[0] - 2, board.shape[0] - 1)  # Bottom-right corner
        ]
        if move in danger_zones:
            return True
        return False


    def distance_to_center(self, board, move):
        center = (board.shape[0] // 2, board.shape[1] // 2)
        return abs(move[0] - center[0]) + abs(move[1] - center[1])


    def count_opponent_moves(self, board, move, player, opponent):
        board_copy = deepcopy(board)
        execute_move(board_copy, move, player)
        return len(get_valid_moves(board_copy, opponent))
  

    def corner_control(self, board, move, player, opponent):
        corners = [(0, 0), (0, board.shape[0] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        result = 0
        board_size = board.shape[0]

        
        # C-squares (adjacent to corners along edges)
        c_squares = [
            (0, 1), (1, 0),
            (0, board_size - 2), (1, board_size - 1),
            (board_size - 1, 1), (board_size - 2, 0),
            (board_size - 2, board_size - 1), (board_size - 1, board_size - 2)
        ]

        if move in corners :
            if (self.is_endgame(board)):
                result = 20
            elif board.shape[0] == 6:
                result = 20
            else : 
                result = 10
            return result

        closest_corner = None
        min_distance = float('inf')  # Initialize with a very large value
        
        for corner in corners:
            distance = abs(corner[0] - move[0]) + abs(corner[1] - move[1])  # Manhattan distance
            if distance < min_distance:
                min_distance = distance
                closest_corner = corner

        if self.is_danger_zone(board, move) :
            if board[closest_corner] == 0 or board[closest_corner] == opponent:
                result -= 5  
            else : 
                result += 10

        board_copy = deepcopy(board)
        execute_move(board_copy, move, player)
        
        if board_copy[closest_corner] == 0:  
            valid_moves = get_valid_moves(board_copy, opponent)
            if closest_corner in valid_moves: 
               result -= 5  

        return result
  

    def evaluate_wall_move(self, board, move, player, opponent):
        board_size = board.shape[0]
        board_copy = deepcopy(board)
        stability_score = 0
        row, col = move

        # Check if the move is on the wall (first or last row or column)
        if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:
            # Check if the move is connected to a corner
            if (row == 0 and col == 0) or (row == 0 and col == board_size - 1) or (row == board_size - 1 and col == 0) or (row == board_size - 1 and col == board_size - 1):
                return stability_score  # Skip corners (handled separately below)
            
            # Check connection to the corners
            corner_connection = 0
            if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:  # If it's on top or bottom row
                if col == 0 and (board[board_size - 1, 0] == player or board[0, board_size - 1]):
                    corner_connection = 1 
                elif col == board_size - 1 and (board[board_size - 1, col] == player or board[0, board_size - 1] == player):
                    corner_connection = 1 
                elif row == 0 and (board[0, board_size - 1] == player or board[0,0] == player):
                    corner_connection = 1 
                elif row == board_size - 1 and (board[board_size - 1, 0] == player or board[0, board_size - 1] == player) :
                    corner_connection = 1  
                    
            if corner_connection == 1:
                stability_score += 5  # Very good move if connected to a corner you control
            elif corner_connection == 0:
                # Check for moves connected to an opponent's corner
                if col == 0 and (board[0, 0] == opponent or board[board_size - 1, 0] == opponent):
                    stability_score -= 5  # Bad move if connected to an opponent's corner
                elif col == board_size - 1 and (board[0, board_size - 1] == opponent or board[board_size - 1, board_size - 1] == opponent):
                    stability_score -= 5  # Bad move if connected to an opponent's corner
            
            # Check for horizontal or vertical captures (captures pieces on the wall)
            capture_bonus = 0
            
            if row == 0 or row == board_size - 1:
                capture = count_capture(board_copy, move, player) 
                if capture > 1:
                    if self.is_danger_zone(board_copy, move) : 
                        capture_bonus = capture
                    else : capture_bonus = capture
            
            if col == 0 or col == board_size - 1:
                capture = count_capture(board_copy, move, player)
                if capture > 1:
                    if self.is_danger_zone(board_copy, move) : 
                        capture_bonus = capture
                    else : capture_bonus = capture
                    
            stability_score += capture_bonus
        
        return stability_score



    def wall_for_corner(self, board, move, opponent):
        board_size = board.shape[0]

        # Define the corners and their neighboring cells
        corners = {
            (0, 0): [(0, 1), (1, 0)],  # Top-left corner
            (0, board_size - 1): [(0, board_size - 2), (1, board_size - 1)],  # Top-right corner
            (board_size - 1, 0): [(board_size - 2, 0), (board_size - 1, 1)],  # Bottom-left corner
            (board_size - 1, board_size - 1): [(board_size - 2, board_size - 1), (board_size - 1, board_size - 2)]  # Bottom-right corner
        }

        # Check if the move is adjacent to an opponent's corner
        for corner, adjacent_cells in corners.items():
            if board[corner] == opponent and move in adjacent_cells:
                # Determine the wall direction
                if move[0] == 0 or move[0] == (board.shape[0] - 1):  # Top or bottom wall
                    wall = [(corner[0], c) for c in range(board_size)]
                else:  # Left or right wall
                    wall = [(r, corner[1]) for r in range(board_size)]
                # Find the other corner on the same wall
                other_corner = wall[0] if wall[-1] == corner else wall[-1]

                for cell in adjacent_cells :
                    if cell in wall :
                        wall.remove(cell)
                if other_corner in wall :
                    wall.remove(other_corner)

                # Check if the other corner is empty and the wall has opponent pieces
                if board[other_corner] == 0 and all(board[r, c] == opponent for r, c in wall):
                    return 10 # Capturing this cell allows you to potentially take the other corner

        return 0
    

    def best_score(self, board, move, player, opponent):
        board_copy = deepcopy(board)
        execute_move(board_copy, move, player)
        result = 0
        max_result = float('-inf')

        curr_player = opponent  
        for m in get_valid_moves(board_copy, opponent):
            while not check_endgame(board_copy, player, opponent)[0]:  
                valid_moves = get_valid_moves(board_copy, curr_player)
                if valid_moves:
                    rand_move = random.choice(valid_moves)  
                    execute_move(board_copy, rand_move, curr_player)
            
                curr_player = player if curr_player == opponent else opponent

        if player == 1 :
            _, player_score, opponent_score = check_endgame(board_copy, player, opponent)
        else : 
            _, opponent_score, player_score = check_endgame(board_copy, player, opponent)
        
        result = player_score - opponent_score
        
        max_result = max(max_result, result)

        if (max_result == float('-inf')) :
            max_result = 0
        
        if max_result > 0 : return 5
        if max_result < 0 : return -5
        else : return 0

    

    def opp_corner_next(self, board, move, player, opponent) :
        corners = [(0, 0), (0, board.shape[0] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        
        reward = 0
        n_corners = 0
        board_copy = deepcopy(board)
        for m in get_valid_moves(board, opponent) :
            if m in corners :
                corners.remove(m)
        
        execute_move(board_copy, move, player)
        
        for m in get_valid_moves(board_copy, opponent) :
            if m in corners : 
                reward += -5

        return reward
    

    def evaluate_control_of_center(self, board, move, player, opponent):
        # Get the board size
        board_size = board.shape[0]
        
        corners = [(0, 0), (0, board.shape[0] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        center_cells = [
            (board_size//2 - 1, board_size//2 - 1),  # Top-left of center
            (board_size//2 - 1, board_size//2),      # Top-right of center
            (board_size//2, board_size//2 - 1),      # Bottom-left of center
            (board_size//2, board_size//2)           # Bottom-right of center
        ]
        
        for m in get_valid_moves(board, opponent) :
            if m in corners :
                corners.remove(m)
        
        board_copy = deepcopy(board)
        execute_move(board_copy, move, player)
        
        # Check if all the center cells are controlled by the player
        result = 0
        for cell in center_cells:
            r, c = cell
            if board_copy[r, c] != player:  # If any of the center cells are not controlled by the player
                result -= 1
            else : 
                result += 1
        
        for m in get_valid_moves(board_copy, opponent) :
            if m in corners : 
                result += -5
        
        return result