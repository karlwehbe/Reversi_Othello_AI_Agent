"""
Bitboard representation for Othello / Reversi on n x n boards.

A position is a pair of Python ints (own, opp). Bit (r * n + c) is set when
that square holds a disc for the corresponding side. Python ints are arbitrary
precision, so this works for 6x6 through 12x12 and beyond without special
casing 64-bit words.

Move generation uses the standard directional-fill technique: shift the mover's
discs toward a direction, keep only the squares that land on opponent discs,
repeat, then a final shift onto an empty square yields a legal move. All of the
per-square Python looping in the original helpers is replaced by a fixed number
of integer shifts and masks, which is where the speedup comes from.
"""

# Direction deltas as (row shift, col shift).
DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)]

_CACHE = {}


class Geometry:
    """Precomputed masks and shift amounts for one board size."""

    __slots__ = ("n", "full", "shifts", "masks", "corners", "square_weight",
                 "edge_masks", "bit_of")

    def __init__(self, n):
        self.n = n
        self.full = (1 << (n * n)) - 1

        # A shift by (dr, dc) moves a bit by dr * n + dc positions. To stop
        # bits wrapping around a row edge, mask off the source columns that
        # would cross the boundary before shifting.
        not_left = 0
        not_right = 0
        for r in range(n):
            for c in range(n):
                b = 1 << (r * n + c)
                if c != 0:
                    not_left |= b
                if c != n - 1:
                    not_right |= b

        self.shifts = []
        self.masks = []
        for dr, dc in DIRS:
            self.shifts.append(dr * n + dc)
            if dc == -1:
                self.masks.append(not_left)
            elif dc == 1:
                self.masks.append(not_right)
            else:
                self.masks.append(self.full)

        self.corners = [0, n - 1, (n - 1) * n, n * n - 1]

        # Static square values used only for move ordering. Cheap by design:
        # the point is a fast rough sort, not an accurate score.
        self.square_weight = [0] * (n * n)
        for r in range(n):
            for c in range(n):
                er = min(r, n - 1 - r)
                ec = min(c, n - 1 - c)
                idx = r * n + c
                if er == 0 and ec == 0:
                    w = 100          # corner
                elif er <= 1 and ec <= 1:
                    w = -40          # X square, diagonally next to a corner
                elif (er == 0 and ec == 1) or (er == 1 and ec == 0):
                    w = -30          # C square, edge next to a corner
                elif er == 0 or ec == 0:
                    w = 20           # rest of the edge
                elif er == 1 or ec == 1:
                    w = -5           # ring just inside the edge
                else:
                    w = 5
                self.square_weight[idx] = w

        # Full row / column / diagonal masks through each corner, used by the
        # stability calculation.
        self.edge_masks = []
        top = sum(1 << c for c in range(n))
        bottom = sum(1 << ((n - 1) * n + c) for c in range(n))
        left = sum(1 << (r * n) for r in range(n))
        right = sum(1 << (r * n + n - 1) for r in range(n))
        self.edge_masks = [top, bottom, left, right]


def geometry(n):
    g = _CACHE.get(n)
    if g is None:
        g = Geometry(n)
        _CACHE[n] = g
    return g


def _shift(bits, amount, mask, full):
    bits &= mask
    if amount > 0:
        return (bits << amount) & full
    return bits >> (-amount)


def moves(own, opp, g):
    """Bitmask of all legal destination squares for `own`."""
    empty = g.full & ~(own | opp)
    result = 0
    for i in range(8):
        s = g.shifts[i]
        m = g.masks[i]
        # Walk outward from own discs across a run of opponent discs.
        x = _shift(own, s, m, g.full) & opp
        for _ in range(g.n - 3):
            x |= _shift(x, s, m, g.full) & opp
        result |= _shift(x, s, m, g.full)
    return result & empty


def flips(own, opp, move_bit, g):
    """Bitmask of opponent discs captured by playing `move_bit`."""
    captured = 0
    for i in range(8):
        s = g.shifts[i]
        m = g.masks[i]
        run = 0
        x = _shift(move_bit, s, m, g.full) & opp
        while x:
            run |= x
            nxt = _shift(x, s, m, g.full)
            if nxt & own:
                captured |= run
                break
            x = nxt & opp
        # If the run never terminates on an own disc, nothing is captured.
    return captured


def apply_move(own, opp, move_bit, g):
    """Return (new_opp, new_own) so the caller can swap sides directly."""
    f = flips(own, opp, move_bit, g)
    new_own = own | move_bit | f
    new_opp = opp & ~f
    return new_opp, new_own


def popcount(x):
    return bin(x).count("1")


def bits_of(x):
    """Yield each set bit as a standalone mask."""
    while x:
        low = x & -x
        yield low
        x ^= low


def from_numpy(board, player):
    """Convert an (n, n) numpy board to (own, opp) ints."""
    n = board.shape[0]
    own = 0
    opp = 0
    flat = board.reshape(-1)
    for idx in range(n * n):
        v = flat[idx]
        if v == player:
            own |= 1 << idx
        elif v != 0:
            opp |= 1 << idx
    return own, opp


def bit_to_move(bit, n):
    idx = bit.bit_length() - 1
    return (idx // n, idx % n)


def move_to_bit(move, n):
    return 1 << (move[0] * n + move[1])


def stable_discs(own, opp, g):
    """
    Count discs that can never be flipped.

    Seeded from occupied corners and grown along rows, columns, and both
    diagonals: a disc is stable if in each of the four directions it is either
    against a wall, or adjacent to an already-stable disc of the same colour.

    This is a lower bound on true stability, which is the standard practical
    approximation and is far stronger than the danger-zone proxy.
    """
    n = g.n
    stable = 0
    for c in g.corners:
        b = 1 << c
        if own & b:
            stable |= b

    if not stable:
        return 0

    changed = True
    while changed:
        changed = False
        candidates = own & ~stable
        for b in bits_of(candidates):
            idx = b.bit_length() - 1
            r, c = idx // n, idx % n
            ok = True
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                safe = False
                for sign in (1, -1):
                    rr, cc = r + dr * sign, c + dc * sign
                    if not (0 <= rr < n and 0 <= cc < n):
                        safe = True
                        break
                    nb = 1 << (rr * n + cc)
                    if stable & nb:
                        safe = True
                        break
                if not safe:
                    ok = False
                    break
            if ok:
                stable |= b
                changed = True

    return popcount(stable)
