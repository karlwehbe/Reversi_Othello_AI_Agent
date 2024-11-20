# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "Student_Agent"
        self.cache = {}

    def step(self, board, player, opponent): 
        start_time = time.time()
        
        match board.shape[0]:
            case 6 : time_limit = 1.93
            case 8 : time_limit = 1.9
            case 10 : time_limit = 1.88
            case 12 : time_limit = 1.85

        depth = 1 
        best_move = None

        try:
            while True:
                current_time = time.time()
                remaining_time = current_time - start_time
                if remaining_time + 0.1 > time_limit:
                    break
                
                max_player = True
                alpha = float('-inf')
                beta = float('inf')
                _, move = self.minimax(board, depth, player, opponent, max_player, alpha, beta, start_time, time_limit)
                
                if move is not None:
                    best_move = move

                depth += 1
        except TimeoutError:
            pass  
        
        if best_move is None:
            return random_move(board, player)
        
        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds")
        return best_move


    def minimax(self, board, depth, player, opponent, max_player, alpha, beta, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            raise TimeoutError
        
        board_key = (board.tostring(), depth, max_player)
        if board_key in self.cache:
            return self.cache[board_key]

        is_endgame, _,_ = check_endgame(board, player, opponent)
        if depth == 0 or is_endgame:
            score = self.evaluate_board(board, player, opponent)
            self.cache[board_key] = (score, None)
            return score, None

        if max_player:
            max_eval = float('-inf')
            best_move = None
            valid_moves = get_valid_moves(board, player)
            
            if not valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError
                eval_score, _ = self.minimax(board, depth - 1, player, opponent, False, alpha, beta, start_time, time_limit)
                self.cache[board_key] = (eval_score, None)
                return eval_score, None

            valid_moves = self.order_moves(board, valid_moves, player, opponent, True, start_time, time_limit)
            if valid_moves == None :
                raise TimeoutError
            
            for move in valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError  
                board_copy = deepcopy(board)
                execute_move(board_copy, move, player)
                eval_score, _ = self.minimax(board_copy, depth - 1, player, opponent, False, alpha, beta, start_time, time_limit)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break 
            
            self.cache[board_key] = (max_eval, best_move)
            return max_eval, best_move
    
        else:
            min_eval = float('inf')
            best_move = None
            valid_moves = get_valid_moves(board, opponent)
            
            if not valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError  
                eval_score, _ = self.minimax(board, depth - 1, player, opponent, True, alpha, beta, start_time, time_limit)
                self.cache[board_key] = (eval_score, None)
                return eval_score, None
            
            valid_moves = self.order_moves(board, valid_moves, opponent, player, False, start_time, time_limit) 
            if valid_moves == None :
                raise TimeoutError
            
            for move in valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError  
                board_copy = deepcopy(board)
                execute_move(board_copy, move, opponent)
                eval_score, _ = self.minimax(board_copy, depth - 1, player, opponent, True, alpha, beta, start_time, time_limit)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break 
            
            self.cache[board_key] = (min_eval, best_move)
            return min_eval, best_move


    def order_moves(self, board, possible_moves, player, opponent, max_player, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            return None
          
        move_order = []
        for move in possible_moves:
            board_copy = deepcopy(board)
            execute_move(board_copy, move, player)
            score = self.evaluate_board(board_copy, player, opponent)
            move_order.append((score, move))
        
        move_order.sort(reverse=max_player)
        ordered_moves = [move for _, move in move_order]
        return ordered_moves

    def evaluate_board(self, board, player, opponent):
        score = 0
        # Corner-Based heuristic
        score += (self.corner_control(board, player, opponent) + self.inner_corners(board, player, opponent) + self.danger_zones(board, player, opponent) + self.wall_for_corner(board, player, opponent)) + self.prevent_corner_take(board, opponent)
        # Score-Based Heuristic
        score +=  (self.finish_him(board, player, opponent) + self.captures(board, player, opponent))
        # Center Control Heuristic
        score += self.center_control(board, player, opponent)
        # Move-Based heuristic
        score += (self.count_opponent_moves(board, opponent) + self.mobility(board, player, opponent) + self.potential_mobility(board, player, opponent))

        return score


    def finish_him(self, board, player, opponent) :
        if player == 1 :
            end, player_score, opponent_score = check_endgame(board, player, opponent)
        else : 
            end, opponent_score, player_score = check_endgame(board, player, opponent)
        result = player_score - opponent_score

        if result > 0 and end : 
            return 100
        if result < 0 and end : 
            return -100
        return 0


    def is_opening(self, board):
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
                score -= 15 if not self.is_endgame(board) else 20
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
        for cell in inner_corners:
            if board[cell] == player:
                score += 10
            elif board[cell] == opponent:
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
                    result -= 3
                if board[cell] == player and board[opposite_corner_pos] == player:
                    result += 3  
                if board[cell] == player and board[corner] == player:
                    result += 10  
                if board[cell] == opponent and board[opposite_corner_pos] == opponent:
                    result -= 3  
                if board[cell] == opponent and board[opposite_corner_pos] == 0:
                    result += 3  
        
        for corner, diagonal in diagonal_cells.items() :
            if board[diagonal] == player and board[corner] == 0 :
                result -= 10
            if board[diagonal] == player and board[corner] == player :
                result += 5
            if board[diagonal] == player and board[corner] == opponent :
                result += -5 
            
            if board[diagonal] == opponent and board[corner] == 0 :
                result += 10
            if board[diagonal] == opponent and board[corner] == opponent :
                result -= 5
            if board[diagonal] == opponent and board[corner] == player :
                result += +5 
                    
        return result


    def count_opponent_moves(self, board, opponent):
        score = 0
        opp_moves = len(get_valid_moves(board, opponent)) 
        if opp_moves == 0 :
            score += 10
        elif opp_moves < 3 : 
            score += 5
        else : 
            score -= opp_moves
        return score
  

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

                    if all(board[w] == opponent for w in wall) and other_corner == 0:
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

        for x, y in opponent_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == 0:
                        empty_neighbors.add((nx, ny))

        score = len(empty_neighbors)

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
        
        if (self.is_opening(board)) : score * 5
        if (self.is_endgame(board)) : score * -5 

        return score

    
    def prevent_corner_take(self, board, opponent):
        board_size = board.shape[0]
        corners = [
            (0, 0), 
            (0, board_size - 1), 
            (board_size - 1, 0), 
            (board_size - 1, board_size - 1)
        ]

        score = 0
        opp_moves = get_valid_moves(board, opponent)
        if any(move in corners for move in opp_moves):
            score -= 40
        else : 
            score += 40
        return score
        

