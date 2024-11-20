# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import math

@register_agent("mcts_agent")


class MCTSAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    
    class Node():
        def __init__(self, state, parent=None):
            self.state = state  # The current state of the game or problem
            self.parent = parent  # The parent node (None for the root node)
            self.children = []  # List of child nodes
            self.visits = 0  # Number of times this node has been visited
            self.score = 0
            self.uct = 0  
            self.move = None
            self.depth = np.sum(self.state != 0) 
    
        def add_child(self, child_node):
            for child in self.children:
                if np.array_equal(child.state, child_node.state):
                    return True  # Prevent adding duplicate child
            self.children.append(child_node)
            return False

        def get_og_node(self, child_node):
            for child in self.children:
                if np.array_equal(child.state, child_node.state):
                    return child # Prevent adding duplicate child 
            

    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"
        self.root = None

    
    def step(self, board, player, opponent):
        start_time = time.time()
        self.curr_moves = []
        
        if np.sum(board != 0) == 5 and player == 2 :
           self.root = self.Node(deepcopy(board), None) 
        elif np.sum(board != 0) == 4 and player == 1 :
            self.root = self.Node(deepcopy(board), None)
        else : 
            self.prune_tree(board)  #prune branches removed because of opponent move
        
        possible_moves = get_valid_moves(board, player)
        if len(possible_moves) == 1 : 
            return possible_moves[0]
        
        best_move = self.mcts(board, player, opponent, start_time)

        board_copy = deepcopy(board)
        execute_move(board_copy, best_move, player)   
        self.prune_tree(board_copy)

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds choosing move:", best_move)
        return best_move


    def prune_tree(self, board):
        matching_child = next((child for child in self.root.children if np.array_equal(child.state, board)), None,)

        if matching_child is not None:
            self.root = matching_child
            self.root.parent = None
        else:
            self.root = self.Node(state=board) 


    def mcts(self, board, player, opponent, start_time):
        time_limit = 1
        match board.shape[0] :
            case 6 : 
                if (self.is_opening_game(board)) :
                    time_limit = 1.98
                elif (self.is_midgame(board)) :
                    time_limit = 1.96
                else : 
                    time_limit = 1.98

            case 8 : 
                if (self.is_opening_game(board)) :
                    time_limit = 1.96
                elif (self.is_midgame(board)) :
                    time_limit = 1.95
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
                    time_limit = 1.88
                elif (self.is_midgame(board)) :
                    time_limit = 1.88
                else : 
                    time_limit = 1.98
        
        for move in get_valid_moves(board, player):
            board_copy = deepcopy(board)
            execute_move(board_copy, move, player)
            child_node = self.Node(state=board_copy, parent=self.root)
            child_node.move = move
            self.root.add_child(child_node)
         
        while (time.time() - start_time < time_limit) :
            for _ in range(10000):  
                if time.time() - start_time >= time_limit: break
                best_search_move = max(self.root.children, key=lambda children: self.uct(board, children, player, opponent))
                move = best_search_move.move
                self.simulate_game(board, move, player, opponent)

        if (self.is_opening_game(board) or self.is_midgame(board)) :
            best_move = max(self.root.children, key=lambda  children: (self.evaluate_board(children.state, player, opponent)))

        else:
            best_move = max(self.root.children, key=lambda  children: (children.visits) )
        
        return best_move.move


    def back_propagate(self, current_node, score):
        while current_node is not None:
            current_node.visits += 1
            current_node.score += score
            current_node = current_node.parent


    def simulate_game(self, board, move, player, opponent):     
        board_copy = deepcopy(board)
        execute_move(board_copy, move, player)
        
        child_node = self.Node(state=deepcopy(board_copy), parent=self.root)  
        child_node.move = move
        present = self.root.add_child(child_node)
        if present : 
            child_node = self.root.get_og_node(child_node)
        
        current_node = child_node
        
        result = 0
        curr_player = opponent  
        while not check_endgame(board_copy, player, opponent)[0]:  
            valid_moves = get_valid_moves(board_copy, curr_player)
            if len(valid_moves) > 0:
                r_move = random_move(board_copy, curr_player)
                execute_move(board_copy, r_move, curr_player) 
                
                new_child_node = self.Node(deepcopy(board_copy), parent=current_node)
                new_child_node.move = r_move
                present = current_node.add_child(new_child_node)
                if present : 
                    new_child_node = current_node.get_og_node(new_child_node)
                
                current_node = new_child_node

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

        self.back_propagate(current_node, score)


    def uct(self, board, children, player, opponent):
        score = children.score
        visits = children.visits
        
        if self.root.visits == 0 : return 0
        if visits == 0 : return float('inf')
        
        board_size = board.shape[0]  
        if board_size == 6:
            if self.is_opening_game(board): c = 2
            elif self.is_midgame(board): c = 1.414
            else : c = 1

        elif board_size == 8:
            if self.is_opening_game(board): c = 2.5
            elif self.is_midgame(board): c = 1.414
            else : c = 1

        elif board_size == 10:
            if self.is_opening_game(board): c = 2.5
            elif self.is_midgame(board): c = 2
            else : c = 1.2

        elif board_size == 12:
            if self.is_opening_game(board): c = 3.0
            elif self.is_midgame(board): c = 2.5
            else : c = 1.414

        total_visits = self.root.visits
        uct_value = (score / visits) + c * (math.sqrt(math.log(total_visits) / visits)) 

        h_score = self.evaluate_board(children.state, player, opponent)
        normalized_h = self.normalized_heurisitic(h_score)
        scaled_h = self.scaled_heuristic(normalized_h)
        
        final_uct = uct_value + scaled_h 
        #final_uct = uct_value + normalized_h
        children.uct = final_uct
        
        return final_uct


    def get_minmax_uct(self) :
        max_move = max(self.root.children, key=lambda children: children.uct)
        max_uct = max_move.uct
        min_move = min(self.root.children, key=lambda children: children.uct)
        min_uct = min_move.uct
        return max_uct, min_uct


    def normalized_heurisitic(self, h_score) :
        max_uct, min_uct = self.get_minmax_uct()
        if max_uct != min_uct : 
            normalized_h = (h_score - min_uct) / (max_uct - min_uct)
        else : 
            normalized_h = h_score
        return normalized_h


    def scaled_heuristic(self, normalized_h) :
        max_uct, min_uct = self.get_minmax_uct()
        scaled_h = min_uct + normalized_h * (max_uct - min_uct)
        return scaled_h


    def evaluate_board(self, board, player, opponent):
        move = self.get_move(board)
        score = 0

        # -------- Corners Heuristic --------
        score += self.corner_control(board, player, opponent)
       
        # -------- Inner Corners Heuristic --------
        score += self.inner_corners(board, player, opponent)
        
        # -------- Opp Moves Heuristic --------
        score += self.count_opponent_moves(board, opponent)
        
        # -------- Adjacent to Corners Heuristic (X-squares and C-squares) --------
        score +=  self.danger_zones(board, player, opponent)

        # -------- Wall Moves Heuristic --------
        score += self.evaluate_wall_move(board, move, player, opponent)
        
        # -------- Wall for Corner Heuristic --------
        score += self.wall_for_corner(board, player, opponent)

        # -------- Center Control Heuristic --------
        score += self.center_control(board, player, opponent)

        # -------- Mobility Heuristic --------
        score += self.mobility(board, player, opponent)

        # -------- Potential Mobility Heuristic --------
        score += self.potential_mobility(board, player, opponent)

        # -------- Disc Difference Heuristic --------
        score += self.captures(board, player, opponent)

        return score


    def get_move(self, board) :
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != self.root.state[i, j] :
                    return (i, j)
                
    

    def is_opening_game(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        board_size = board.shape[1]
        if board_size == 6:
            return captured < 8  
        elif board_size == 8:
            return captured < 16  
        elif board_size == 10:
            return captured < 27  
        elif board_size == 12:
            return captured < 37  
        return False

    def is_midgame(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        board_size = board.shape[1]
        if board_size == 6:
            return 8 <= captured < 24  
        elif board_size == 8:
            return 16 <= captured < 50 
        elif board_size == 10:
            return 27 <= captured < 80  
        elif board_size == 12:
            return 37 <= captured < 115  
        return False
    
    def is_endgame(self, board):
        captured = sum(cell != 0 for row in board for cell in row)
        board_size = board.shape[1]
        if board_size == 6:
            return 24 <= captured
        elif board_size == 8:
            return 50 <= captured 
        elif board_size == 10:
            return 80 <= captured  
        elif board_size == 12:
            return 115 <= captured  
        return False
    
    
    def corner_control(self, board, player, opponent):
        corners = [(0, 0), (0, board.shape[0] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        score = 0

        for corner in corners:
            if board[corner] == player:
                score += 15 if not self.is_endgame(board) else 20
            elif board[corner] == opponent:
                score -= 15
        return score
    
    def inner_corners(self, board, player, opponent):
        board_size = board.shape[0]

        inner_corners = [
            (0, 2), (0, board_size - 3),
            (2, 0), (board_size - 3, 0),
            (board_size - 1, 2), (board_size - 1, board_size - 3),
            (2, board_size - 1), (board_size - 3, board_size - 1)
        ]
        score = 0
        for pos in inner_corners:
            if board[pos] == player:
                score += 10
            elif board[pos] == opponent:
                score -= 10

        return score
        

    def danger_zones(self, board, player, opponent):
        board_size = board.shape[0]

        danger_zones = {
            (0, 0): [(0, 1), (1, 0)],  
            (0, board_size - 1): [(0, board_size - 2), (1, board_size - 1)],  
            (board_size - 1, 0): [(board_size - 2, 0), (board_size - 1, 1)], 
            (board_size - 1, board_size - 1): [(board_size - 1, board_size - 2), (board_size - 2, board_size - 1)]  
        }

        opposite_corner = {
            (0, 0): (0, board_size - 1),  
            (0, board_size - 1): (0, 0),  
            (board_size - 1, 0): (board_size - 1, board_size - 1), 
            (board_size - 1, board_size - 1): (board_size - 1, 0)
        }

        diagonal_cells = {
            (0,0): (1, 1),
            (0, board_size - 1): (1, board_size - 2),
            (board_size - 1, 0): (board_size - 2, 1),
            (board_size - 1, board_size - 1): (board_size - 2, board_size - 2)
        }
        
        result = 0
        for corner, dz in danger_zones.items(): 
            for cell in dz:
                opposite_corner_pos = opposite_corner.get(corner)  
                
                if board[cell] == player and board[corner] == 0:
                    result += -3
                if board[cell] == player and board[opposite_corner_pos] == opponent:
                    result += 3  
                if board[cell] == player and board[corner] == player:
                    result += 5  
                if board[cell] == opponent and board[opposite_corner_pos] == opponent:
                    result += -3  
                if board[cell] == opponent and board[opposite_corner_pos] == 0:
                    result += 3  
        
        for corner, diagonal in diagonal_cells.items() :
            if board[diagonal] == player and board[corner] == 0 :
                result += -10
            if board[diagonal] == player and board[corner] == player :
                result += 5
            if board[diagonal] == player and board[corner] == opponent :
                result += -5 
            
            if board[diagonal] == opponent and board[corner] == 0 :
                result += 10
            if board[diagonal] == opponent and board[corner] == opponent :
                result += -5
            if board[diagonal] == opponent and board[corner] == player :
                result += +5 
                    
        return result


    def count_opponent_moves(self, board, opponent):
        score = 0
        opp_moves = len(get_valid_moves(board, opponent)) 
        if opp_moves == 0 :
            score += 10
        else : 
            score -= opp_moves
        return score
  

    def evaluate_wall_move(self, board, move, player, opponent):
        board_size = board.shape[0]
        board_copy = deepcopy(board)
        stability_score = 0
        row, col = move

        if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:
            if (row == 0 and col == 0) or (row == 0 and col == board_size - 1) or (row == board_size - 1 and col == 0) or (row == board_size - 1 and col == board_size - 1):
                return stability_score 
            
            corner_connection = 0
            if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:  
                if col == 0 and board[0, 0] == player:
                    corner_connection = 1  
                elif col == board_size - 1 and board[0, board_size - 1] == player:
                    corner_connection = 1  
                elif col == 0 and board[board_size - 1, 0] == player:
                    corner_connection = 1  
                elif col == board_size - 1 and board[board_size - 1, board_size - 1] == player:
                    corner_connection = 1  
            
            if corner_connection == 1:
                stability_score += 5  
            elif corner_connection == 0:
                if col == 0 and (board[0, 0] == opponent or board[board_size - 1, 0] == opponent):
                    stability_score -= 5  
                elif col == board_size - 1 and (board[0, board_size - 1] == opponent or board[board_size - 1, board_size - 1] == opponent):
                    stability_score -= 5  
            
            capture_bonus = 0
            
            if row == 0 or row == board_size - 1:
                capture = count_capture(board_copy, move, player) 
                if capture > 1:
                    if self.is_danger_zone(board_copy, move) : 
                        capture_bonus = 1
                    else : capture_bonus = capture
            
            if col == 0 or col == board_size - 1:
                capture = count_capture(board_copy, move, player)
                if capture > 1:
                    if self.is_danger_zone(board_copy, move) : 
                        capture_bonus = 1
                    else : capture_bonus = capture
                    
            stability_score += capture_bonus
        
        return stability_score


    def wall_for_corner(self, board, player, opponent):
        board_size = board.shape[0]
        corners = {
            (0, 0): [(0, 1), (1, 0)],
            (0, board_size - 1): [(0, board_size - 2), (1, board_size - 1)],
            (board_size - 1, 0): [(board_size - 2, 0), (board_size - 1, 1)],
            (board_size - 1, board_size - 1): [(board_size - 2, board_size - 1), (board_size - 1, board_size - 2)]
        }

        opposite_corner = {
            (0,0) : (0, board_size -1),  
            (0, board_size - 1): (0, 0),  
            (board_size - 1, 0): (board_size - 1, board_size - 1), 
            (board_size - 1, board_size - 1): (board_size - 1, 0)
        }

        score = 0

        wall = []
        for corner, adjacent_cells in corners.items():
            if board[corner] == opponent :
                for cell in adjacent_cells : 
                    if board[cell] == player : 
                        if cell[0] == 1 or cell[0] == board_size - 2: 
                            wall = [(corner[0], r) for r in range(board_size)]
                        else : 
                            wall = [(c, corner[0]) for c in range(board_size)]     

                other_corner = opposite_corner.get(corner)

                if wall : 
                    for cell in adjacent_cells :
                        if cell in wall :
                            wall.remove(cell)
                    if other_corner in wall :
                        wall.remove(other_corner)

                    if all(board[w] == opponent for w in wall):
                        score += 10
        
        return score
    

    def center_control(self, board, player, opponent):
        board_size = board.shape[0]
        center_cells = [
            (board_size // 2 - 1, board_size // 2 - 1),
            (board_size // 2 - 1, board_size // 2),
            (board_size // 2, board_size // 2 - 1),
            (board_size // 2, board_size // 2)
        ]
        score = 0

        for cell in center_cells:
            if board[cell] == player:
                score += 1
            elif board[cell] == opponent:
                score -= 1
        return score    
    

    def mobility(self, board, player, opponent) :
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        score = 0
        if player_moves + opponent_moves != 0:
            score = 100 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        
        return score


    def potential_mobility(self, board, player, opponent):
        opponent_positions = np.argwhere(board == opponent)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        board_size = board.shape[0]
        empty_neighbors = set()

        # Use opponent positions to find adjacent empty squares
        for x, y in opponent_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == 0:
                        empty_neighbors.add((nx, ny))

        # Potential mobility is the number of empty squares adjacent to opponent discs
        score = len(empty_neighbors)

        # Repeat for opponent's potential mobility
        player_positions = np.argwhere(board == player)
        opponent_empty_neighbors = set()
        for x, y in player_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == 0:
                        opponent_empty_neighbors.add((nx, ny))

        score -= len(opponent_empty_neighbors)
        return score
    
    def captures(self, board, player, opponent) :
        score = 0
        if player == 1 :
            _, player_score, opponent_score = check_endgame(board, player, opponent)
        else : 
            _, opponent_score, player_score = check_endgame(board, player, opponent)
        
        score = player_score - opponent_score
        
        if (self.is_opening_game(board)) : score * 5
        if (self.is_endgame(board)) : score * -5 

        return score
