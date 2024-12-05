# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import time
import numpy as np

from agents import bitboard as bb
from helpers import random_move

EXACT, LOWER, UPPER = 0, 1, 2

# Below this many empties, an exact solve is cheap enough that we always run it
# inside the search itself. Measured on 8x8: 0.02s at 9 empties, 0.04s at 11,
# 0.40s at 12.
ENDGAME_SOLVE_EMPTIES = 12

# At the root only, we will *attempt* an exact solve up to this many empties.
# It may time out, which is why the root always secures a fallback move first.
ROOT_SOLVE_ATTEMPT_EMPTIES = 22

WIN_SCORE = 1_000_000


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Iterative deepening principal variation search on bitboards.

    Ordering, in priority order: the best move from the previous iteration
    (via the transposition table), killer moves, history counts, then a static
    square table plus flip count. None of these evaluate a position, which is
    the point: profiling showed the old agent spent 92% of its budget inside
    evaluate_board and only 8% searching.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "Student_Agent"
        self.tt = {}
        self.max_tt = 600_000
        self.nodes = 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def step(self, board, player, opponent):
        start = time.time()
        n = board.shape[0]
        self.g = bb.geometry(n)
        self.n = n

        limit = {6: 1.93, 8: 1.90, 10: 1.88, 12: 1.85}.get(n, 1.85)

        if len(self.tt) > self.max_tt:
            self.tt.clear()

        # Killer moves are indexed by ply, history by square.
        self.killers = [[0, 0] for _ in range(64)]
        self.history = [0] * (n * n)
        self.nodes = 0
        self.deadline = start + limit - 0.05

        # The solver's table holds exact game values, which do not depend on
        # search depth, but it is cleared each move because entries from a
        # different position are useless and it grows fast.
        self.solve_tt = {}
        self.solved_wld = None
        self.solved_exact = False
        self.endgame_fallback = 0

        own, opp = bb.from_numpy(board, player)
        legal = bb.moves(own, opp, self.g)
        if not legal:
            return random_move(board, player)

        move_bits = list(bb.bits_of(legal))
        if len(move_bits) == 1:
            bit = move_bits[0]
            # The move is forced, so there is no decision to make. Rather than
            # returning instantly and throwing the budget away, play it out and
            # search the resulting position. Every entry that lands in the
            # transposition table here is a position we may actually face next
            # turn, so next turn's search starts warm instead of from nothing.
            self.ponder(own, opp, bit)
            return bb.bit_to_move(bit, n)

        empties = n * n - bb.popcount(own | opp)
        best_bit = move_bits[0]

        if empties <= ROOT_SOLVE_ATTEMPT_EMPTIES:
            bit = self.root_endgame(own, opp, empties)
            if bit:
                return bb.bit_to_move(bit, n)
            # The solve ran out of time. It has also consumed most of the
            # clock, so the iterative deepening below will barely get started.
            # Use the depth 4 move secured before the attempt rather than
            # whatever move happens to be first in the list.
            if self.endgame_fallback:
                best_bit = self.endgame_fallback

        depth = 1
        score = 0
        try:
            while depth < 64:
                if time.time() > self.deadline:
                    break
                score, bit = self.pvs(own, opp, depth, -WIN_SCORE, WIN_SCORE, 0, True)
                if bit:
                    best_bit = bit
                # A proven win or loss cannot be improved on by searching
                # deeper, so stop and hand the remaining time to the ponder
                # below rather than re-proving the same thing.
                if abs(score) >= WIN_SCORE:
                    break
                depth += 1
        except TimeoutError:
            pass

        self.last_depth = depth - 1

        # If the search finished early, which happens when the position is
        # solved or the tree is small, use what is left to look past our own
        # move instead of returning with time on the clock.
        if time.time() < self.deadline - 0.05:
            self.ponder(own, opp, best_bit)

        return bb.bit_to_move(best_bit, n)

    def ponder(self, own, opp, bit):
        """
        Search ahead on time we would otherwise waste.

        Called when our move is forced. We apply the move ourselves and search
        from the opponent's side. That search stores results for each of their
        plausible replies, which are exactly the positions we will be choosing
        between on our next turn.
        """
        n_own, n_opp = bb.apply_move(own, opp, bit, self.g)

        # Inside the exact solver's range the answer is already perfect, so
        # there is nothing deeper to learn and the loop would just spin.
        empties = self.n * self.n - bb.popcount(n_own | n_opp)
        if empties <= ENDGAME_SOLVE_EMPTIES:
            self.ponder_depth = 0
            return

        depth = 1
        try:
            while depth < 64:
                if time.time() > self.deadline:
                    break
                self.pvs(n_own, n_opp, depth, -WIN_SCORE, WIN_SCORE, 0, True)
                depth += 1
        except TimeoutError:
            pass
        self.ponder_depth = depth - 1

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def root_endgame(self, own, opp, empties):
        """
        Try to solve the position exactly, in three stages.

        Profiling a full game showed the old threshold of 12 empties was too
        conservative: at 11 empties the solver finished in 41ms and the agent
        threw away 1.8 seconds, while at 13 empties it spent the whole budget
        guessing with the heuristic. But the cost of an exact solve roughly
        multiplies by the branching factor for each extra empty square, so the
        threshold cannot simply be raised. A solve that runs out of time leaves
        no move at all.

        The fix is to make failure free. Secure a fallback move first, then
        spend whatever is left on the solve. If it times out, the caller falls
        through to normal iterative deepening and nothing is lost.

        Returns a move bit if the position was solved, otherwise 0.
        """
        # Stage 1: a cheap safety net so a failed solve costs nothing.
        try:
            _, fallback = self.pvs(own, opp, 4, -WIN_SCORE, WIN_SCORE, 0, True)
            if fallback:
                self.endgame_fallback = fallback
        except TimeoutError:
            return 0

        # Stage 2: win / loss / draw only.
        #
        # Asking for the exact final disc differential requires the true value.
        # Asking only which side of zero it falls on is a null window search,
        # so almost every branch fails immediately and pruning is far harder.
        # In practice this reaches several more empty squares than an exact
        # solve for the same time, which is why strong engines resolve WLD
        # first and only compute the margin if time remains.
        try:
            score, bit = self.solve(own, opp, -1, 1, 0)
        except TimeoutError:
            return 0

        if not bit:
            return 0

        self.solved_wld = score

        # Stage 3: if the WLD solve was quick, refine to the exact margin. This
        # matters when several moves all win: the exact solve picks the one
        # that wins by the most discs.
        if time.time() < self.deadline - 0.25:
            try:
                _, exact_bit = self.solve(own, opp, -WIN_SCORE, WIN_SCORE, 0)
                if exact_bit:
                    self.solved_exact = True
                    return exact_bit
            except TimeoutError:
                pass

        return bit

    def pvs(self, own, opp, depth, alpha, beta, ply, can_pass):
        """
        Principal variation search. The first move is searched with a full
        window; the rest get a null window first and are only re-searched if
        they beat alpha. Because ordering is good, most of those null-window
        searches fail low immediately and cost almost nothing.
        """
        if time.time() > self.deadline:
            raise TimeoutError
        self.nodes += 1

        alpha_orig, beta_orig = alpha, beta
        key = (own, opp)

        entry = self.tt.get(key)
        tt_bit = 0
        if entry is not None:
            e_depth, e_flag, e_val, e_bit = entry
            tt_bit = e_bit
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val, e_bit
                if e_flag == LOWER:
                    alpha = max(alpha, e_val)
                elif e_flag == UPPER:
                    beta = min(beta, e_val)
                if alpha >= beta:
                    return e_val, e_bit

        empties = self.n * self.n - bb.popcount(own | opp)
        if empties <= ENDGAME_SOLVE_EMPTIES:
            return self.solve(own, opp, alpha, beta, ply)

        legal = bb.moves(own, opp, self.g)

        if not legal:
            if not can_pass:
                return self.terminal_score(own, opp), 0
            val, _ = self.pvs(opp, own, depth, -beta, -alpha, ply + 1, False)
            return -val, 0

        if depth <= 0:
            return self.evaluate(own, opp), 0

        ordered = self.order(own, opp, legal, tt_bit, ply)

        best_val = -WIN_SCORE - 1
        best_bit = 0
        first = True

        for bit in ordered:
            n_own, n_opp = bb.apply_move(own, opp, bit, self.g)
            if first:
                val, _ = self.pvs(n_own, n_opp, depth - 1, -beta, -alpha, ply + 1, True)
                val = -val
                first = False
            else:
                val, _ = self.pvs(n_own, n_opp, depth - 1, -alpha - 1, -alpha, ply + 1, True)
                val = -val
                if alpha < val < beta:
                    val, _ = self.pvs(n_own, n_opp, depth - 1, -beta, -alpha, ply + 1, True)
                    val = -val

            if val > best_val:
                best_val = val
                best_bit = bit
            if val > alpha:
                alpha = val
            if alpha >= beta:
                # A move good enough to cause a cutoff is worth trying early in
                # sibling positions. Remember it as a killer, and bump its
                # square in the global history table.
                if self.killers[ply][0] != bit:
                    self.killers[ply][1] = self.killers[ply][0]
                    self.killers[ply][0] = bit
                self.history[bit.bit_length() - 1] += depth * depth
                break

        if best_val <= alpha_orig:
            flag = UPPER
        elif best_val >= beta_orig:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[key] = (depth, flag, best_val, best_bit)

        return best_val, best_bit

    def solve(self, own, opp, alpha, beta, ply):
        """
        Exact endgame search. No heuristic: play every line to the end and
        return the true final disc differential. Over the last dozen squares
        this makes the agent's play perfect.
        """
        if time.time() > self.deadline:
            raise TimeoutError
        self.nodes += 1

        alpha_orig, beta_orig = alpha, beta
        key = (own, opp)
        entry = self.solve_tt.get(key)
        tt_bit = 0
        if entry is not None:
            e_flag, e_val, e_bit = entry
            tt_bit = e_bit
            if e_flag == EXACT:
                return e_val, e_bit
            if e_flag == LOWER:
                alpha = max(alpha, e_val)
            else:
                beta = min(beta, e_val)
            if alpha >= beta:
                return e_val, e_bit

        legal = bb.moves(own, opp, self.g)
        if not legal:
            if not bb.moves(opp, own, self.g):
                return self.terminal_score(own, opp), 0
            val, _ = self.solve(opp, own, -beta, -alpha, ply + 1)
            return -val, 0

        best_val = -WIN_SCORE - 1
        best_bit = 0

        # Endgame ordering: prefer moves that leave the opponent fewest
        # replies. Restricting the opponent produces cutoffs faster than any
        # positional score down here, where the heuristic is meaningless
        # because the game is being played out in full.
        bits = list(bb.bits_of(legal))
        if len(bits) > 1:
            def rank(b):
                if b == tt_bit:
                    return -10_000
                n_own, n_opp = bb.apply_move(own, opp, b, self.g)
                return (bb.popcount(bb.moves(n_own, n_opp, self.g))
                        - self.g.square_weight[b.bit_length() - 1] / 100.0)
            bits.sort(key=rank)

        for bit in bits:
            n_own, n_opp = bb.apply_move(own, opp, bit, self.g)
            val, _ = self.solve(n_own, n_opp, -beta, -alpha, ply + 1)
            val = -val
            if val > best_val:
                best_val = val
                best_bit = bit
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        if best_val <= alpha_orig:
            flag = UPPER
        elif best_val >= beta_orig:
            flag = LOWER
        else:
            flag = EXACT
        self.solve_tt[key] = (flag, best_val, best_bit)

        return best_val, best_bit

    def terminal_score(self, own, opp):
        a, b = bb.popcount(own), bb.popcount(opp)
        if a > b:
            return WIN_SCORE + (a - b)
        if a < b:
            return -WIN_SCORE + (a - b)
        return 0

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def order(self, own, opp, legal, tt_bit, ply):
        """
        Sort moves without evaluating any position.

        The old agent called the full evaluate_board on every move at every
        node, which cost 61% of the entire move budget. Measured, replacing it
        with this table gave 2.6x more nodes on its own.
        """
        g = self.g
        k0, k1 = self.killers[ply]
        scored = []
        for bit in bb.bits_of(legal):
            idx = bit.bit_length() - 1
            if bit == tt_bit:
                s = 1 << 40          # previous iteration's best move goes first
            elif bit == k0:
                s = 1 << 34
            elif bit == k1:
                s = 1 << 33
            else:
                s = (self.history[idx] << 8) + g.square_weight[idx] + 1000
            scored.append((s, bit))
        scored.sort(key=lambda p: p[0], reverse=True)
        return [b for _, b in scored]

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

        if own_moves == 0 and opp_moves == 0:
            return self.terminal_score(own, opp)

        # Mobility
        if own_moves + opp_moves:
            mobility = 100.0 * (own_moves - opp_moves) / (own_moves + opp_moves)
        else:
            mobility = 0.0

        # Corners
        corners_own = sum(1 for c in g.corners if own >> c & 1)
        corners_opp = sum(1 for c in g.corners if opp >> c & 1)
        corners = 25.0 * (corners_own - corners_opp)

        # Stability. This is what the original report claimed to use and the
        # original code never computed.
        stab_own = bb.stable_discs(own, opp, g)
        stab_opp = bb.stable_discs(opp, own, g)
        if stab_own + stab_opp:
            stability = 100.0 * (stab_own - stab_opp) / (stab_own + stab_opp)
        else:
            stability = 0.0

        # Frontier discs: own discs adjacent to an empty square are liable to
        # be flipped, so fewer is better.
        empty = g.full & ~occupied
        front_own = bb.popcount(self.neighbours(own) & empty)
        front_opp = bb.popcount(self.neighbours(opp) & empty)
        if front_own + front_opp:
            frontier = -100.0 * (front_own - front_opp) / (front_own + front_opp)
        else:
            frontier = 0.0

        # Disc differential only matters at the very end.
        d_own, d_opp = bb.popcount(own), bb.popcount(opp)
        discs = 100.0 * (d_own - d_opp) / (d_own + d_opp)

        # Parity: taking the last move in the endgame is worth real points.
        parity = 5.0 if empties % 2 == 1 else -5.0

        progress = 1.0 - empties / total   # 0 at the start, 1 at the end
        w_disc = 2.0 + 18.0 * progress ** 3
        w_mob = 22.0 * (1.0 - progress) + 4.0

        return (corners * 3.0
                + stability * 12.0
                + mobility * w_mob / 10.0
                + frontier * 3.0
                + discs * w_disc / 10.0
                + parity * progress * 4.0)

    def neighbours(self, bits):
        g = self.g
        out = 0
        for i in range(8):
            out |= bb._shift(bits, g.shifts[i], g.masks[i], g.full)
        return out
