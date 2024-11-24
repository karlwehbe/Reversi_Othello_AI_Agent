# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
import math
import random
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


@register_agent("mcts_agent")
class MCTSAgent(Agent):
    """
    Monte Carlo tree search with UCT selection and a progressive-bias term
    driven by a position heuristic.

    The four MCTS phases are now actually separate: descend the tree by UCT,
    expand one node, run a light random playout from there, then back the
    result up with alternating sign.
    """

    class Node:
        __slots__ = ("state", "parent", "children", "visits", "score",
                     "move", "to_move", "untried", "heuristic")

        def __init__(self, state, to_move, parent=None, move=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visits = 0
            # Reward accumulated from the perspective of the player who moved
            # INTO this node, i.e. parent.to_move.
            self.score = 0.0
            self.move = move
            self.to_move = to_move
            self.untried = None
            self.heuristic = None

        def is_fully_expanded(self):
            return self.untried is not None and len(self.untried) == 0

    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"
        self.root = None
        self.bias_weight = 1.5

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def step(self, board, player, opponent):
        start_time = time.time()
        self.player = player
        self.opponent = opponent

        time_limit = self.time_budget(board)

        # Reuse the subtree that matches the current board. After the opponent
        # moves, the statistics gathered under that branch last turn are still
        # valid, so throwing the whole tree away wastes them.
        root = self.find_subtree(board, player)
        if root is None:
            root = self.Node(board.copy(), to_move=player)
            root.untried = self.legal_or_pass(root)

        legal = get_valid_moves(board, player)
        if len(legal) == 1:
            self.root = root
            return legal[0]
        if not legal:
            return random_move(board, player)

        self.root = root

        while time.time() - start_time < time_limit:
            # Run a small batch between clock checks so time.time() is not the
            # dominant cost at high simulation counts.
            for _ in range(8):
                node = self.select(root)
                node = self.expand(node)
                result = self.rollout(node)
                self.backpropagate(node, result)
            if time.time() - start_time >= time_limit:
                break

        # FIX: the old code switched between "best by heuristic" in the opening
        # and midgame and "best by visits" in the endgame. Picking by heuristic
        # discards every simulation that was just run. Robust child (most
        # visits) is the standard choice and is what the search actually
        # produced evidence for.
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def find_subtree(self, board, player):
        """
        Look two plies down from the previous root for a node whose board and
        side to move match the current position, and promote it to root.
        Two plies because between our turns both we and the opponent moved,
        though a pass can make it one.
        """
        if self.root is None:
            return None

        candidates = list(self.root.children)
        candidates.extend(g for c in self.root.children for g in c.children)

        matches = [n for n in candidates
                   if n.to_move == player and np.array_equal(n.state, board)]
        if not matches:
            return None

        # Transpositions mean the same position can appear as more than one
        # node. Promote the copy that accumulated the most evidence.
        best = max(matches, key=lambda n: n.visits)
        best.parent = None
        return best

    def time_budget(self, board):
        n = board.shape[0]
        if n == 6:
            return 1.90
        if n == 8:
            return 1.88
        if n == 10:
            return 1.85
        return 1.82

    # ------------------------------------------------------------------
    # Phase 1: selection
    # ------------------------------------------------------------------

    def select(self, node):
        """
        Descend the tree by UCT until reaching a node that still has untried
        moves or is terminal.

        FIX: the old selection only ever ranked self.root.children. It never
        descended, so the tree below depth 1 was built but never used. The
        agent was effectively a flat bandit over root moves, not MCTS.
        """
        while True:
            if node.untried is None:
                node.untried = self.legal_or_pass(node)
            if node.untried:
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda c: self.uct_value(node, c))

    def uct_value(self, parent, child):
        if child.visits == 0:
            return float('inf')

        exploit = child.score / child.visits
        explore = self.c_param(parent.state) * math.sqrt(
            math.log(parent.visits + 1) / child.visits)

        # FIX: the original computed normalized_h then scaled_h, but the two
        # were algebraic inverses:
        #     norm  = (h - min_uct) / (max_uct - min_uct)
        #     scale = min_uct + norm * (max_uct - min_uct)   ->  == h
        # so the "normalization" was an identity function and the raw heuristic
        # (magnitude in the hundreds) was added straight onto a UCT term of
        # magnitude ~1. The heuristic swamped the search and turned this into a
        # greedy evaluator.
        #
        # Progressive bias is the correct construction: bound the heuristic to
        # [-1, 1] so it lives on the same scale as the exploit term, then decay
        # it as real simulation evidence accumulates.
        if child.heuristic is None:
            raw = self.evaluate_board(child.state, parent.to_move,
                                      self.other(parent.to_move))
            child.heuristic = math.tanh(raw / 100.0)

        bias = self.bias_weight * child.heuristic / (1 + child.visits)

        return exploit + explore + bias

    def c_param(self, board):
        n = board.shape[0]
        if self.is_opening_game(board):
            return {6: 1.6, 8: 1.8, 10: 1.8, 12: 2.0}.get(n, 1.8)
        if self.is_midgame(board):
            return 1.414
        return 1.0

    # ------------------------------------------------------------------
    # Phase 2: expansion
    # ------------------------------------------------------------------

    def expand(self, node):
        if not node.untried:
            return node

        move = node.untried.pop()

        if move is None:
            # Pass: same board, turn flips.
            child_state = node.state
            child = self.Node(child_state, to_move=self.other(node.to_move),
                              parent=node, move=None)
        else:
            child_state = node.state.copy()
            execute_move(child_state, move, node.to_move)
            child = self.Node(child_state, to_move=self.other(node.to_move),
                              parent=node, move=move)

        node.children.append(child)
        return child

    def legal_or_pass(self, node):
        moves = get_valid_moves(node.state, node.to_move)
        if moves:
            moves = list(moves)
            # expand() pops from the end, so without a shuffle the first move
            # tried at every node is fixed by board coordinates. At low
            # simulation counts on large boards that bias never washes out.
            random.shuffle(moves)
            return moves
        over, _, _ = check_endgame(node.state, self.player, self.opponent)
        if over:
            return []
        # No legal move but game continues: the only action is a pass.
        return [None]

    # ------------------------------------------------------------------
    # Phase 3: simulation
    # ------------------------------------------------------------------

    def rollout(self, node):
        """
        Random playout to a terminal state.

        FIX: the old simulate_game allocated a Node with a full deepcopy of the
        board at EVERY ply of the rollout and attached it to a tree that
        selection never read. On a 12x12 board that is well over a hundred
        allocations and deep-copies per simulation, which is the actual reason
        the agent could not get enough simulations inside two seconds. It was
        an implementation problem, not a limitation of MCTS.

        This version mutates one board in place and allocates nothing.
        """
        board = node.state.copy()
        to_move = node.to_move
        passes = 0

        # The two-consecutive-passes counter is already the terminal condition
        # for Othello, so calling check_endgame every ply just recomputes the
        # same thing with a full board scan. Measured at roughly 2x the total
        # rollout cost on 8x8 and 12x12.
        while passes < 2:
            moves = get_valid_moves(board, to_move)
            if moves:
                passes = 0
                move = moves[random.randrange(len(moves))]
                execute_move(board, move, to_move)
            else:
                passes += 1
            to_move = self.other(to_move)

        if self.player == 1:
            _, p_score, o_score = check_endgame(board, self.player, self.opponent)
        else:
            _, o_score, p_score = check_endgame(board, self.player, self.opponent)

        if p_score > o_score:
            return 1.0
        if p_score < o_score:
            return -1.0
        return 0.0

    # ------------------------------------------------------------------
    # Phase 4: backpropagation
    # ------------------------------------------------------------------

    def backpropagate(self, node, result):
        """
        `result` is from the root player's perspective. Each node stores reward
        for the player who moved into it, so the sign flips depending on which
        side made that move.

        FIX: the original added the same signed score at every level. In a
        two-player game that credits the opponent's moves with the root
        player's wins, so the values were meaningless above depth 1.
        """
        while node is not None:
            node.visits += 1
            if node.parent is not None:
                mover = node.parent.to_move
                node.score += result if mover == self.player else -result
            node = node.parent

    def other(self, side):
        return self.opponent if side == self.player else self.player

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_board(self, board, player, opponent):
        score = 0
        score += self.corner_control(board, player, opponent)
        score += self.inner_corners(board, player, opponent)
        score += self.count_opponent_moves(board, opponent)
        score += self.danger_zones(board, player, opponent)
        score += self.wall_for_corner(board, player, opponent)
        score += self.center_control(board, player, opponent)
        score += self.mobility(board, player, opponent)
        score += self.potential_mobility(board, player, opponent)
        score += self.captures(board, player, opponent)
        return score

    # NOTE: evaluate_wall_move was dropped from evaluate_board. It depended on
    # get_move(), which diffed a board against self.root.state to guess which
    # move had just been played. Once selection descends more than one ply that
    # diff is meaningless, and it returned None on an unchanged board, which
    # would crash on `row, col = move`.

    def _phase_thresholds(self, board_size):
        return {6: (8, 24), 8: (16, 50), 10: (27, 80), 12: (37, 115)}.get(board_size)

    def is_opening_game(self, board):
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
        return int(np.count_nonzero(board)) >= t[1]

    def corner_control(self, board, player, opponent):
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        weight = 20 if self.is_endgame(board) else 15
        score = 0
        for corner in corners:
            if board[corner] == player:
                score += weight
            elif board[corner] == opponent:
                # FIX: the penalty was hardcoded to 15 while the reward used a
                # phase-dependent weight, so in the endgame the agent valued
                # taking a corner more than denying one. Made symmetric.
                score -= weight
        return score

    def inner_corners(self, board, player, opponent):
        n = board.shape[0]
        cells = [
            (0, 2), (0, n - 3),
            (2, 0), (n - 3, 0),
            (n - 1, 2), (n - 1, n - 3),
            (2, n - 1), (n - 3, n - 1)
        ]
        score = 0
        for cell in cells:
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

            # FIX: `wall` was declared once outside this loop in the original,
            # so a wall built for one corner leaked into the next iteration.
            wall = []
            for cell in adjacent_cells:
                if board[cell] != player:
                    continue
                if cell[0] == corner[0]:
                    wall = [(corner[0], c) for c in range(n)]
                else:
                    # FIX: the original used corner[0] as a column index here,
                    # which pointed at column 0 for both right-hand corners.
                    wall = [(r, corner[1]) for r in range(n)]

            if not wall:
                continue

            other_corner = opposite_corner[corner]
            wall = [w for w in wall
                    if w not in adjacent_cells and w != other_corner and w != corner]

            # FIX: the minimax copy of this function tested `other_corner == 0`,
            # comparing a coordinate tuple to an int, so it never fired. The
            # MCTS copy dropped the check entirely, which is the opposite error:
            # it scored the pattern even when the opposite corner was already
            # taken and there was nothing left to win.
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
            _, p_score, o_score = check_endgame(board, player, opponent)
        else:
            _, o_score, p_score = check_endgame(board, player, opponent)

        score = p_score - o_score

        # FIX: `score * 5` and `score * -5` with no assignment. Both lines
        # computed a value and discarded it, so the raw disc differential was
        # returned at every phase.
        if self.is_opening_game(board):
            score = score * 5
        elif self.is_endgame(board):
            score = score * -5

        return score
