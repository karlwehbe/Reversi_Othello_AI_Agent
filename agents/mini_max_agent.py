# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

# Transposition table entry types.
# EXACT  = the stored value is the true minimax value for that depth
# LOWER  = search failed high, stored value is a lower bound
# UPPER  = search failed low, stored value is an upper bound
EXACT, LOWER, UPPER = 0, 1, 2


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Iterative deepening minimax with alpha-beta pruning, a bounded
    transposition table, heuristic move ordering, and position heuristics.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "Student_Agent"
        self.cache = {}
        self.max_cache_entries = 400000
        self.phase_endgame = False

    def step(self, board, player, opponent):
        start_time = time.time()

        match board.shape[0]:
            case 6:
                time_limit = 1.93
            case 8:
                time_limit = 1.9
            case 10:
                time_limit = 1.88
            case 12:
                time_limit = 1.85
            case _:
                time_limit = 1.85

        # FIX: the cache was never cleared, so it grew across every move of
        # every game in a batch run. Entries from a finished board are dead
        # weight and eventually push the process into swap.
        if len(self.cache) > self.max_cache_entries:
            self.cache.clear()

        # FIX: is_endgame() scanned the whole board and was called once per
        # corner inside corner_control. Compute the phase once per move.
        self.phase_endgame = self.is_endgame(board)

        depth = 1
        best_move = None

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed + 0.1 > time_limit:
                    break

                alpha = float('-inf')
                beta = float('inf')
                _, move = self.minimax(board, depth, player, opponent, True,
                                       alpha, beta, start_time, time_limit)

                if move is not None:
                    best_move = move

                depth += 1
        except TimeoutError:
            pass

        if best_move is None:
            return random_move(board, player)

        return best_move

    def minimax(self, board, depth, player, opponent, max_player,
                alpha, beta, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            raise TimeoutError

        alpha_orig = alpha
        beta_orig = beta

        # FIX: tostring() is removed in newer numpy. tobytes() is the
        # supported equivalent and is what tostring aliased to anyway.
        board_key = (board.tobytes(), max_player)

        # FIX: the old code cached the value returned after a beta cutoff and
        # replayed it as if it were exact. A value produced by a cutoff is only
        # a BOUND on the true score, so reusing it under a different window
        # returned wrong scores and could pick a losing move. Store the bound
        # type alongside the value and only trust it when the window allows.
        entry = self.cache.get(board_key)
        if entry is not None:
            e_depth, e_flag, e_value, e_move = entry
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_value, e_move
                if e_flag == LOWER:
                    alpha = max(alpha, e_value)
                elif e_flag == UPPER:
                    beta = min(beta, e_value)
                if alpha >= beta:
                    return e_value, e_move

        is_over, _, _ = check_endgame(board, player, opponent)
        if depth == 0 or is_over:
            score = self.evaluate_board(board, player, opponent)
            self.cache[board_key] = (depth, EXACT, score, None)
            return score, None

        if max_player:
            side, other = player, opponent
        else:
            side, other = opponent, player

        valid_moves = get_valid_moves(board, side)

        if not valid_moves:
            # Pass the turn. check_endgame above catches the double-pass
            # terminal case, so this cannot loop forever.
            eval_score, _ = self.minimax(board, depth - 1, player, opponent,
                                         not max_player, alpha, beta,
                                         start_time, time_limit)
            self.cache[board_key] = (depth, EXACT, eval_score, None)
            return eval_score, None

        valid_moves = self.order_moves(board, valid_moves, side, other,
                                       max_player, start_time, time_limit)
        if valid_moves is None:
            raise TimeoutError

        best_move = None

        if max_player:
            best_eval = float('-inf')
            for move in valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError
                # FIX: deepcopy on a numpy array goes through the generic
                # pickle path and is roughly an order of magnitude slower than
                # .copy(). This ran once per move per node, so it was the
                # single biggest drag on reachable search depth.
                board_copy = board.copy()
                execute_move(board_copy, move, side)
                eval_score, _ = self.minimax(board_copy, depth - 1, player,
                                             opponent, False, alpha, beta,
                                             start_time, time_limit)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
        else:
            best_eval = float('inf')
            for move in valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError
                board_copy = board.copy()
                execute_move(board_copy, move, side)
                eval_score, _ = self.minimax(board_copy, depth - 1, player,
                                             opponent, True, alpha, beta,
                                             start_time, time_limit)
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

        if best_eval <= alpha_orig:
            flag = UPPER
        elif best_eval >= beta_orig:
            flag = LOWER
        else:
            flag = EXACT

        self.cache[board_key] = (depth, flag, best_eval, best_move)
        return best_eval, best_move

    def order_moves(self, board, possible_moves, side, other, max_player,
                    start_time, time_limit):
        if time.time() - start_time >= time_limit:
            return None

        move_order = []
        for move in possible_moves:
            board_copy = board.copy()
            execute_move(board_copy, move, side)
            score = self.evaluate_board(board_copy, side, other)
            move_order.append((score, move))

        # FIX: sorting raw (score, move) tuples fell back to comparing the move
        # coordinates whenever two boards scored the same, so ordering depended
        # on position rather than quality. Sort on the score only.
        move_order.sort(key=lambda pair: pair[0], reverse=max_player)
        return [move for _, move in move_order]

    def evaluate_board(self, board, player, opponent):
        score = 0
        # Corner-based
        score += (self.corner_control(board, player, opponent)
                  + self.inner_corners(board, player, opponent)
                  + self.danger_zones(board, player, opponent)
                  + self.wall_for_corner(board, player, opponent)
                  + self.prevent_corner_take(board, opponent))
        # Score-based
        score += (self.finish_him(board, player, opponent)
                  + self.captures(board, player, opponent))
        # Center control
        score += self.center_control(board, player, opponent)
        # Move-based
        score += (self.count_opponent_moves(board, opponent)
                  + self.mobility(board, player, opponent)
                  + self.potential_mobility(board, player, opponent))
        return score

    def finish_him(self, board, player, opponent):
        if player == 1:
            end, player_score, opponent_score = check_endgame(board, player, opponent)
        else:
            end, opponent_score, player_score = check_endgame(board, player, opponent)
        if not end:
            return 0
        result = player_score - opponent_score
        if result > 0:
            return 100
        if result < 0:
            return -100
        return 0

    def _phase_thresholds(self, board_size):
        return {6: (8, 24), 8: (16, 50), 10: (27, 80), 12: (37, 115)}.get(board_size)

    def is_opening(self, board):
        t = self._phase_thresholds(board.shape[1])
        if t is None:
            return False
        return int(np.count_nonzero(board)) < t[0]

    def is_midgame(self, board):
        t = self._phase_thresholds(board.shape[1])
        if t is None:
            return False
        return t[0] <= int(np.count_nonzero(board)) < t[1]

    def is_endgame(self, board):
        t = self._phase_thresholds(board.shape[1])
        if t is None:
            return False
        # FIX: the old version summed a Python generator over every cell.
        # np.count_nonzero does the same work in C.
        return int(np.count_nonzero(board)) >= t[1]

    def corner_control(self, board, player, opponent):
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        weight = 20 if self.phase_endgame else 15
        score = 0
        for corner in corners:
            if board[corner] == player:
                score += weight
            elif board[corner] == opponent:
                score -= weight
        return score

    def inner_corners(self, board, player, opponent):
        n = board.shape[0]
        inner_corners = [
            (0, 2), (0, n - 3),
            (2, 0), (n - 3, 0),
            (n - 1, 2), (n - 1, n - 3),
            (2, n - 1), (n - 3, n - 1)
        ]
        score = 0
        for cell in inner_corners:
            if board[cell] == player:
                score += 10
            elif board[cell] == opponent:
                score -= 10
        return score

    def danger_zones(self, board, player, opponent):
        n = board.shape[0]

        danger_zones = {
            (0, 0): [(0, 1), (1, 0)],
            (0, n - 1): [(0, n - 2), (1, n - 1)],
            (n - 1, 0): [(n - 2, 0), (n - 1, 1)],
            (n - 1, n - 1): [(n - 1, n - 2), (n - 2, n - 1)]
        }

        opposite_corner = {
            (0, 0): (0, n - 1),
            (0, n - 1): (0, 0),
            (n - 1, 0): (n - 1, n - 1),
            (n - 1, n - 1): (n - 1, 0)
        }

        diagonal_cells = {
            (0, 0): (1, 1),
            (0, n - 1): (1, n - 2),
            (n - 1, 0): (n - 2, 1),
            (n - 1, n - 1): (n - 2, n - 2)
        }

        result = 0
        for corner, dz in danger_zones.items():
            opp_corner = opposite_corner[corner]
            for cell in dz:
                if board[cell] == player and board[corner] == 0:
                    result -= 3
                if board[cell] == player and board[opp_corner] == player:
                    result += 3
                if board[cell] == player and board[corner] == player:
                    result += 10
                if board[cell] == opponent and board[opp_corner] == opponent:
                    result -= 3
                if board[cell] == opponent and board[opp_corner] == 0:
                    result += 3

        for corner, diagonal in diagonal_cells.items():
            if board[diagonal] == player and board[corner] == 0:
                result -= 10
            if board[diagonal] == player and board[corner] == player:
                result += 5
            if board[diagonal] == player and board[corner] == opponent:
                result -= 5
            if board[diagonal] == opponent and board[corner] == 0:
                result += 10
            if board[diagonal] == opponent and board[corner] == opponent:
                result -= 5
            if board[diagonal] == opponent and board[corner] == player:
                result += 5

        return result

    def count_opponent_moves(self, board, opponent):
        opp_moves = len(get_valid_moves(board, opponent))
        if opp_moves == 0:
            return 10
        if opp_moves < 3:
            return 5
        return -opp_moves

    def wall_for_corner(self, board, player, opponent):
        """
        Detects the pattern where the opponent holds a corner, the player holds
        an adjacent edge cell, and the rest of that edge is opponent discs,
        which sets up a capture of the opposite corner.
        """
        n = board.shape[0]
        corners = {
            (0, 0): [(0, 1), (1, 0)],
            (0, n - 1): [(0, n - 2), (1, n - 1)],
            (n - 1, 0): [(n - 2, 0), (n - 1, 1)],
            (n - 1, n - 1): [(n - 2, n - 1), (n - 1, n - 2)]
        }

        opposite_corner = {
            (0, 0): (0, n - 1),
            (0, n - 1): (0, 0),
            (n - 1, 0): (n - 1, n - 1),
            (n - 1, n - 1): (n - 1, 0)
        }

        score = 0

        for corner, adjacent_cells in corners.items():
            if board[corner] != opponent:
                continue

            # FIX: `wall` used to be declared once outside this loop, so a wall
            # built for one corner survived into the next iteration and got
            # scored against the wrong edge.
            wall = []
            for cell in adjacent_cells:
                if board[cell] != player:
                    continue
                if cell[0] == corner[0]:
                    # Adjacent cell shares a row with the corner, so the
                    # relevant wall is that row.
                    wall = [(corner[0], c) for c in range(n)]
                else:
                    # FIX: the old column branch built [(c, corner[0]) ...],
                    # using the corner's ROW index as a column index. For the
                    # two right-hand corners that pointed at column 0, so the
                    # check ran against the wrong edge entirely.
                    wall = [(r, corner[1]) for r in range(n)]

            if not wall:
                continue

            other_corner = opposite_corner[corner]
            wall = [w for w in wall
                    if w not in adjacent_cells and w != other_corner and w != corner]

            # FIX: this read `other_corner == 0`, comparing a coordinate tuple
            # to an integer. A tuple never equals 0, so the condition was always
            # False and this heuristic never contributed a single point in any
            # game. It has to test the CONTENTS of that square.
            if wall and all(board[w] == opponent for w in wall) and board[other_corner] == 0:
                score += 10

        return score

    def center_control(self, board, player, opponent):
        n = board.shape[0]
        center_cells = [
            (n // 2 - 1, n // 2 - 1),
            (n // 2 - 1, n // 2),
            (n // 2, n // 2 - 1),
            (n // 2, n // 2)
        ]
        score = 0
        for cell in center_cells:
            if board[cell] == player:
                score += 1
            elif board[cell] == opponent:
                score -= 1
        return score

    def mobility(self, board, player, opponent):
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        total = player_moves + opponent_moves
        if total == 0:
            return 0
        return 100 * (player_moves - opponent_moves) / total

    def potential_mobility(self, board, player, opponent):
        n = board.shape[0]
        empty = (board == 0)

        # FIX: the original built two Python sets by looping over every disc and
        # all eight neighbours. Shifting a boolean mask does the same work in
        # numpy, which matters because this runs at every node.
        def frontier(mask):
            out = np.zeros((n, n), dtype=bool)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    shifted = np.zeros((n, n), dtype=bool)
                    xs = slice(max(0, dx), n + min(0, dx))
                    ys = slice(max(0, dy), n + min(0, dy))
                    xd = slice(max(0, -dx), n + min(0, -dx))
                    yd = slice(max(0, -dy), n + min(0, -dy))
                    shifted[xd, yd] = mask[xs, ys]
                    out |= shifted
            return int(np.count_nonzero(out & empty))

        return frontier(board == opponent) - frontier(board == player)

    def captures(self, board, player, opponent):
        if player == 1:
            _, player_score, opponent_score = check_endgame(board, player, opponent)
        else:
            _, opponent_score, player_score = check_endgame(board, player, opponent)

        score = player_score - opponent_score

        # FIX: this was `score * 5` and `score * -5` with no assignment, so both
        # lines computed a value and threw it away. The phase weighting never
        # applied and the raw disc differential was returned at every stage of
        # the game, which is the classic Othello mistake of greedily maximising
        # disc count during the midgame.
        if self.is_opening(board):
            score = score * 5
        elif self.phase_endgame:
            score = score * -5

        return score

    def prevent_corner_take(self, board, opponent):
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]

        # FIX: this returned +40 or -40, which swamped every other term except
        # mobility and made the evaluation nearly binary on one condition.
        # Scaled to sit in the same range as the other corner heuristics.
        opp_moves = get_valid_moves(board, opponent)
        if any(move in corners for move in opp_moves):
            return -20
        return 20
