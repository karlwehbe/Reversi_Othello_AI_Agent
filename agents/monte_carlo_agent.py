# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import time
import math
import random
import numpy as np

from agents import bitboard as bb
from helpers import random_move

# Rollouts stop after this many plies and hand off to the heuristic. A full
# playout on 12x12 is roughly 140 plies, so truncating buys most of the
# simulation count back at the cost of a noisier estimate.
ROLLOUT_PLIES = 18

# RAVE blending constant. Larger values keep the AMAF estimate influential for
# longer, which is what you want when the simulation budget is small.
RAVE_BIAS = 300.0


@register_agent("mcts_agent")
class MCTSAgent(Agent):
    """
    Monte Carlo tree search on bitboards with UCT, RAVE, progressive bias,
    truncated weighted rollouts, and subtree reuse between moves.
    """

    class Node:
        __slots__ = ("own", "opp", "parent", "children", "visits", "score",
                     "move", "untried", "heuristic",
                     "rave_visits", "rave_score")

        def __init__(self, own, opp, parent=None, move=0):
            self.own = own          # side to move at this node
            self.opp = opp
            self.parent = parent
            self.children = []
            self.visits = 0
            self.score = 0.0        # reward for the player who moved in here
            self.move = move
            self.untried = None
            self.heuristic = None
            # AMAF statistics, keyed by square index.
            self.rave_visits = None
            self.rave_score = None

    def __init__(self):
        super(MCTSAgent, self).__init__()
        self.name = "MCTSAgent"
        self.root = None
        self.bias_weight = 1.2

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def step(self, board, player, opponent):
        start = time.time()
        n = board.shape[0]
        self.g = bb.geometry(n)
        self.n = n
        limit = {6: 1.90, 8: 1.88, 10: 1.85, 12: 1.82}.get(n, 1.82)
        deadline = start + limit

        own, opp = bb.from_numpy(board, player)
        legal = bb.moves(own, opp, self.g)
        if not legal:
            return random_move(board, player)
        move_bits = list(bb.bits_of(legal))
        if len(move_bits) == 1:
            self.root = None
            return bb.bit_to_move(move_bits[0], n)

        root = self.find_subtree(own, opp)
        if root is None:
            root = self.Node(own, opp)
            root.untried = move_bits[:]
            random.shuffle(root.untried)
        self.root = root

        while time.time() < deadline:
            for _ in range(8):
                node, path, played = self.select(root)
                node = self.expand(node, path, played)
                result = self.rollout(node, path, played)
                self.backpropagate(path, played, result)
            if time.time() >= deadline:
                break

        best = max(root.children, key=lambda c: c.visits)
        return bb.bit_to_move(best.move, n)

    def find_subtree(self, own, opp):
        """Promote the node matching the current position, keeping its stats."""
        if self.root is None:
            return None
        pool = list(self.root.children)
        pool.extend(g for c in self.root.children for g in c.children)
        matches = [x for x in pool if x.own == own and x.opp == opp]
        if not matches:
            return None
        best = max(matches, key=lambda x: x.visits)
        best.parent = None
        return best

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, node):
        """
        Descend by RAVE-blended UCT.

        Parity convention, used identically everywhere below: path[i] has the
        player of parity (i % 2) to move, where parity 0 is the root player.
        A move made FROM path[i] therefore carries parity i % 2. Getting this
        off by one ply inverts the sign of every stored value, so it is worth
        stating once and following exactly.
        """
        path = [node]
        played = []
        while True:
            if node.untried is None:
                legal = bb.moves(node.own, node.opp, self.g)
                node.untried = list(bb.bits_of(legal))
                random.shuffle(node.untried)
            if node.untried or not node.children:
                return node, path, played
            mover_parity = (len(path) - 1) % 2
            node = max(node.children, key=lambda c: self.value(node, c))
            played.append((mover_parity, node.move.bit_length() - 1))
            path.append(node)

    def value(self, parent, child):
        if child.visits == 0:
            return float('inf')

        q = child.score / child.visits
        explore = self.c_param(parent) * math.sqrt(
            math.log(parent.visits + 1) / child.visits)

        # RAVE / AMAF. The all-moves-as-first estimate reuses every playout that
        # ever contained this square, so a single simulation informs many moves
        # instead of one. That is exactly the right trade when the budget only
        # allows a few dozen simulations, which is the case on 12x12.
        idx = child.move.bit_length() - 1
        rv = parent.rave_visits[idx] if parent.rave_visits else 0
        if rv > 0:
            amaf = parent.rave_score[idx] / rv
            beta = rv / (rv + child.visits + child.visits * rv / RAVE_BIAS)
            q = (1 - beta) * q + beta * amaf

        if child.heuristic is None:
            # Child stores the position from the opponent's side to move, so
            # flip it back to score from the mover's perspective.
            raw = self.evaluate(child.opp, child.own)
            child.heuristic = math.tanh(raw / 60.0)
        bias = self.bias_weight * child.heuristic / (1 + child.visits)

        return q + explore + bias

    def c_param(self, node):
        empties = self.n * self.n - bb.popcount(node.own | node.opp)
        total = self.n * self.n
        progress = 1.0 - empties / total
        return 1.9 - 0.9 * progress

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self, node, path, played):
        if not node.untried:
            return node
        bit = node.untried.pop()
        mover_parity = (len(path) - 1) % 2
        n_own, n_opp = bb.apply_move(node.own, node.opp, bit, self.g)
        child = self.Node(n_own, n_opp, parent=node, move=bit)
        node.children.append(child)
        played.append((mover_parity, bit.bit_length() - 1))
        path.append(child)
        return child

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def rollout(self, node, path, played):
        """
        Weighted, truncated playout. Returns a reward in [-1, 1] from the ROOT
        player's perspective, so backpropagation has one fixed reference.

        Two changes from a plain random rollout. Moves are sampled with weights
        from the static square table, so the simulation does not casually hand
        over a corner. And the playout stops after a fixed number of plies and
        scores with the heuristic instead of playing to the end, which buys
        several times more simulations per second.
        """
        own, opp = node.own, node.opp
        g = self.g
        weights = g.square_weight
        parity = (len(path) - 1) % 2   # parity of the side to move at `node`
        passes = 0

        for _ in range(ROLLOUT_PLIES):
            legal = bb.moves(own, opp, g)
            if not legal:
                passes += 1
                if passes >= 2:
                    break
                own, opp = opp, own
                parity ^= 1
                continue
            passes = 0

            bits = list(bb.bits_of(legal))
            if len(bits) == 1:
                bit = bits[0]
            else:
                ws = [max(1.0, weights[b.bit_length() - 1] + 45.0) for b in bits]
                bit = random.choices(bits, weights=ws, k=1)[0]

            played.append((parity, bit.bit_length() - 1))
            own, opp = bb.apply_move(own, opp, bit, g)
            parity ^= 1

        # `own` belongs to the side of the current parity. Convert the score to
        # the root player's frame.
        raw = self.evaluate(own, opp)
        if parity == 1:
            raw = -raw
        return math.tanh(raw / 60.0)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backpropagate(self, path, played, result):
        """
        `result` is from the root player's perspective.

        path[i].score holds reward for whoever moved INTO path[i], which is the
        player of parity (i - 1) % 2. RAVE at path[i] instead credits the player
        of parity i % 2, since those are the moves selectable from that node.
        """
        size = self.n * self.n
        for i in range(len(path)):
            n = path[i]
            n.visits += 1

            if i > 0:
                mover_parity = (i - 1) % 2
                n.score += result if mover_parity == 0 else -result

            node_parity = i % 2
            rave_result = result if node_parity == 0 else -result
            for parity, idx in played:
                if parity == node_parity:
                    if n.rave_visits is None:
                        n.rave_visits = [0] * size
                        n.rave_score = [0.0] * size
                    n.rave_visits[idx] += 1
                    n.rave_score[idx] += rave_result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, own, opp):
        g = self.g
        n = self.n
        occupied = own | opp
        empties = n * n - bb.popcount(occupied)
        total = n * n

        own_moves = bb.popcount(bb.moves(own, opp, g))
        opp_moves = bb.popcount(bb.moves(opp, own, g))

        d_own, d_opp = bb.popcount(own), bb.popcount(opp)
        if own_moves == 0 and opp_moves == 0:
            return 1000.0 if d_own > d_opp else (-1000.0 if d_own < d_opp else 0.0)

        if own_moves + opp_moves:
            mobility = 100.0 * (own_moves - opp_moves) / (own_moves + opp_moves)
        else:
            mobility = 0.0

        corners_own = sum(1 for c in g.corners if own >> c & 1)
        corners_opp = sum(1 for c in g.corners if opp >> c & 1)
        corners = 25.0 * (corners_own - corners_opp)

        stab_own = bb.stable_discs(own, opp, g)
        stab_opp = bb.stable_discs(opp, own, g)
        if stab_own + stab_opp:
            stability = 100.0 * (stab_own - stab_opp) / (stab_own + stab_opp)
        else:
            stability = 0.0

        discs = 100.0 * (d_own - d_opp) / (d_own + d_opp)
        progress = 1.0 - empties / total
        w_disc = 2.0 + 18.0 * progress ** 3
        w_mob = 22.0 * (1.0 - progress) + 4.0

        return (corners * 2.0
                + stability * 8.0
                + mobility * w_mob / 10.0
                + discs * w_disc / 10.0)
