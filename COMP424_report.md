# COMP424 Final Project Report

## I. Overall Motivation for the Approach

For this project, I built an AI agent that plays **Reversi Othello**. My main approach uses **alpha-beta pruning**, **iterative deepening**, and a board evaluation function based on several Othello heuristics.

Alpha-beta pruning is a search algorithm that looks ahead at possible moves and responses. Instead of exploring every possible sequence of moves, it skips entire branches once it knows they cannot lead to a better result than something it has already found.

I combined this with **iterative deepening**. Instead of immediately searching as deeply as possible, the agent first searches 1 move ahead, then 2 moves ahead, then 3, and continues increasing the depth until the time limit is reached.

Othello is a good game for this type of search because it is **deterministic** and has **perfect information**. There is no randomness, and both players can see the entire board. Because of this, it makes sense to search while assuming that the opponent will also choose strong moves.

I also built a **Monte Carlo Tree Search (MCTS)** agent so I could compare the two approaches. MCTS works differently from alpha-beta. It does not require a traditional evaluation function. Instead, it simulates many games from the current position and prefers moves that tend to produce better results.

My first MCTS implementation performed very poorly. At first, I concluded that MCTS simply could not run enough simulations, or **playouts**, within the 2-second move limit. That conclusion turned out to be only partly correct. Finding the real reason for the poor performance changed how I approached the rest of the project.

## II. Performance Bottleneck and Change in Direction

To understand why the agents were performing poorly, I profiled my alpha-beta implementation.

I discovered that it was only searching **744 positions in 2 seconds**, which is extremely low for a search-based agent.

The profiler showed that:

* `evaluate_board` used **92% of the entire move budget**
* `order_moves` alone used **61%**

This meant the agent was spending almost all of its available time evaluating and sorting positions instead of actually searching through the game tree.

The main problem was **move ordering**, which is the process of deciding which legal move should be searched first.

My original implementation called the complete board evaluation function for every legal move at every position. That evaluation function itself called `get_valid_moves` five times and `check_endgame` twice.

For example, if a position had 12 legal moves, simply sorting those moves required about **60 move-generation calls** before the search even explored the first one.

In other words, **sorting the moves was more expensive than searching them**.

This measurement changed the direction of the project. I had originally treated the main problem as improving the quality of the heuristic evaluation. The profiler showed that the real problem was speed.

A slightly weaker heuristic that can be evaluated **30,000 times** is much more useful than a slightly better heuristic that can only be evaluated **744 times**.

From that point, I divided the work into two goals:

1. Make the search much faster.
2. Improve the evaluation after the search was fast enough.

## III. Agent Design

### A. Board Representation

One of the largest performance improvements came from changing the board representation from a NumPy grid to **bitboards**.

The basic idea is simple: instead of storing the board as a grid of individual squares, the board can be stored as an integer. This allows many squares to be processed at the same time using bit operations.

For example, imagine a row of four squares:

```text
. X . X
```

This can be represented using binary as:

```text
0101
```

That binary value is also just a number, so the board can be represented as a number.

If we multiply the number by 2, all of the bits shift:

```text
. X . X     multiply by 2     . . X .
```

Instead of checking and moving each square individually, one operation changes the entire row.

The same idea is used to find legal Othello moves. I shift all of my discs one square in a direction, keep only the shifted bits that overlap an opponent disc, and continue shifting until I reach an empty square. That empty square is then a possible legal move.

On a real board representation:

* a shift of 1 moves horizontally
* a shift of one complete row moves vertically
* other shifts represent the remaining directions

The same process is repeated for all eight possible directions. This is much faster than my original NumPy implementation.

The old code checked square 1, 2, 3... and continued one square at a time. Depending on the board size, this meant checking anywhere from 36 to 144 squares.

The bitboard implementation instead performs a small number of shift operations that process many squares at once.

Each player has a separate integer representing their discs, meaning an entire game position can be represented using only **two integers**.

Python integers do not have a fixed size limit, so the same implementation works for every board size used in the project, from **6x6 to 12x12**.

Before changing the search algorithm, I tested the bitboard implementation against my original NumPy implementation.

I ran **7,235 tests across all four board sizes** and found **zero mismatches**. The tests checked:

* legal move sets
* number of discs flipped
* the resulting board after a move

I did this before changing the search because a small bug in move generation could otherwise produce incorrect search results and be very difficult to find later.

The performance improvement was large:

| Board |                      NumPy |    Bitboard | Speedup |
| ----- | -------------------------: | ----------: | ------: |
| 8x8   | 9,572 move generations/sec | 105,339/sec |     11x |
| 12x12 |                  4,190/sec |  64,060/sec |     15x |

The bitboard representation was therefore around **11 times faster on 8x8** and **15 times faster on 12x12**.

### B. Search

#### Alpha-Beta Pruning and PVS

The main search algorithm is **alpha-beta pruning**. It keeps a range of acceptable scores, called a **search window**, and skips branches that cannot affect the final decision. As the window becomes smaller, more branches can usually be removed.

I also implemented **Principal Variation Search (PVS)**. After fully searching the first move, PVS tests the remaining moves using a very small **null window**. For example, if the current best score is 50, the next move can first be searched with `(50, 51)`. The search still explores the game tree normally, but it only tries to determine whether the move can beat 50. If it cannot, the search stops early. If it can, the move is searched again with a full window to find its actual score.

This works best when move ordering is good, since the first move is likely to already be the best one.

I compared PVS with normal alpha-beta at fixed depths:

| Board | Depth | Plain Alpha-Beta | With PVS | Saved |
| ----- | ----: | ---------------: | -------: | ----: |
| 8x8   |     5 |            3,181 |    2,887 |    9% |
| 8x8   |     6 |            9,423 |    9,708 |   -3% |
| 8x8   |     7 |           36,111 |   37,494 |   -4% |
| 12x12 |     5 |           11,151 |   11,722 |   -5% |
| 12x12 |     6 |           38,396 |   38,127 |    1% |
| 12x12 |     7 |          195,794 |  185,934 |    5% |

Overall, **PVS had almost no effect**. The results ranged from about 5% worse to 9% better. Regular alpha-beta was already narrowing its search window as better moves were found, and my move ordering usually placed strong moves first. I kept PVS because it did not significantly hurt performance, but the main improvements came from **move ordering, bitboards, and the endgame solver**.

#### Iterative Deepening

The agent also uses **iterative deepening**, meaning it searches depth 1, then depth 2, then depth 3, and continues until the time limit is reached.

The game allows **2 seconds per move**, so I set an internal deadline between **1.85 and 1.93 seconds**, depending on the board size. If the search reaches this deadline, it stops and returns the best move from the last completed depth.

Although searching the same position several times may seem wasteful, the deeper searches are much more expensive than the earlier ones. The earlier searches therefore use relatively little time and provide a major benefit: the best move found at one depth can be searched first at the next depth, improving move ordering and allowing more alpha-beta pruning.

### C. Move Ordering

Move ordering produced the **largest single improvement** in the search.

The key lesson was that move ordering does not need to perfectly predict the best move. It only needs to be approximately correct while being extremely cheap to calculate.

My move-ordering priorities are:

1. **Best move from the previous iterative-deepening search**

   If the best move at depth 5 was move A, move A is usually still a strong candidate at depth 6.

   I therefore search it first at the next depth.

   This is one of the main reasons iterative deepening is useful, but my original agent was not taking advantage of this information.

2. **Killer moves**

   Sometimes a move is so strong that it causes alpha-beta to stop searching the remaining moves in that part of the tree.

   When this happens, I save that move as a **killer move** for that search depth.

   Similar positions often have similar strong responses, so trying the killer move earlier can cause more cutoffs.

3. **History heuristic**

   The agent keeps a running count of how often each board square has caused an early alpha-beta cutoff.

   Moves involving squares that frequently caused cutoffs are searched earlier.

4. **Static square table**

   I assign simple values to board locations.

   For example:

   * corners are worth 100
   * diagonal squares next to corners are worth -40
   * edges are worth 20

   I also include the number of discs that the move flips.

   This calculation is intentionally simple. It does not create new board copies and does not call the full evaluation function.

I tested only the last change first, before adding the other move-ordering techniques.

The results were:

| Ordering Method | Positions Searched in 1.9s |
| --------------- | -------------------------: |
| Full evaluation |                      1,094 |
| Static table    |                      2,862 |

The simpler static ordering allowed the agent to search **2.6 times more positions**.

This result was interesting because I intentionally made the ordering less accurate, but the agent became stronger because the new ordering was much cheaper.

### D. Transposition Table

A **transposition table** is a cache of positions that have already been searched.

The same Othello position can sometimes be reached through different sequences of moves. Without a cache, the agent may search the same position several times.

The transposition table allows the agent to reuse previous results instead.

However, my original implementation had an important bug caused by the way alpha-beta pruning stops early.

The bug came from storing a value as exact even when alpha-beta had stopped early.

Suppose a MAX position has these moves:

```text
Position X
├── A → 40
├── B → 55
├── C → 80   ← cutoff
└── D → ?    ← not searched
```

Assume the parent search has a **beta value of 60**. Once Move C returns 80, alpha-beta can stop because MAX has already found something better than 60, so the parent would never choose this line.

However, the true value of Position X is not necessarily 80. Move D could have been worth 120. All we know is:

```text
Position X >= 80
```

My original transposition table incorrectly stored this as:

```text
Position X = 80
```

If the same board appeared later, the agent could treat 80 as the exact score even though the position was never fully searched.

The fix is to store:

```text
Position X → 80, lower bound
```

If the position appears again, this lower bound can raise alpha to at least 80. If that already makes `alpha >= beta`, the position can be pruned immediately. Otherwise, the agent continues searching because it still does not know the exact score.

The table therefore stores whether a value is **exact**, a **lower bound**, or an **upper bound**, so incomplete searches are reused correctly.

The table now distinguishes between exact values, lower bounds, and upper bounds, so cached results are only reused when appropriate.

Because of this, every transposition-table entry must be marked as one of three types:

* **Exact:** the position was fully searched and the stored number is its actual value.
* **Lower bound:** the search stopped early and proved that the position is worth at least this value.
* **Upper bound:** nothing in the search beat the current limit, so the position is worth at most this value.

The corrected transposition table stores:

* the value
* the type of value
* the depth at which the position was searched

An exact value can be returned immediately. A lower bound can only increase the lower limit of the current search. An upper bound can only decrease the upper limit.
If the lower and upper limits meet, the position can still be skipped, but the algorithm never treats a bound as if it were an exact score.

I also measured how common repeated positions actually are, because caching is only useful if repeated positions happen often enough.

For the 6x6 MCTS tree, there were:

* **1,305 total positions**
* **1,274 unique positions**

This means approximately **2.4% of the positions were repeats**.

That is enough for a transposition table to be useful in alpha-beta, which searches far more positions. However, I concluded that the extra bookkeeping was not worthwhile for MCTS.

### E. Evaluation Function

The evaluation function estimates how good a board position is when the search cannot reach the end of the game.

I rebuilt the evaluation around **five main terms**.

I also scaled the terms so that they produce values in similar ranges. This prevents one heuristic from completely overpowering the others.

My original evaluation had a problem where one heuristic returned approximately **+40 or -40** based on a yes/no condition, while other heuristics returned values closer to **+10 or -10**.

As a result, that one feature could dominate the entire evaluation. The new evaluation uses the following five terms.

#### 1. Stability

**Stable discs** are discs that can no longer be flipped for the rest of the game.

I calculate stability by starting from occupied corners and expanding outward through rows, columns, and diagonals. Stability is one of the strongest positional ideas in Othello after mobility because a stable disc is a permanent gain that the opponent cannot remove.

#### 2. Mobility

Mobility measures how many legal moves I have compared with the opponent. In Othello, limiting the opponent's future choices is often more valuable than simply owning more discs.

#### 3. Corners

Corners receive a large positive weight. A corner can never be flipped after it is captured. It also helps create stable discs around it, making corners strategically very important.

#### 4. Frontier Discs

A **frontier disc** is one of my discs that is next to an empty square.

These discs are exposed and can often be flipped by the opponent. Because of this, having fewer frontier discs is generally better.

#### 5. Disc Count and Parity

Disc count measures how many discs each player currently owns. **Parity** refers to which player is forced to make the final move.

This matters because the player who moves last often gets a free capture near the end of the game. The importance of each heuristic changes depending on how full the board is.

For example:

* mobility weight decreases from **26 to 4**
* disc-count weight increases from **2 to 20**

This is necessary because different features matter at different stages of Othello.

Having many discs in the middle of the game can actually be bad. Controlling too many squares often reduces your future mobility and gives the opponent more opportunities.

### F. Endgame Solver

Near the end of the game, the agent no longer needs to estimate positions with a heuristic. Instead, it can search every remaining possibility until the game ends.

This is called an **exact solve**.

An exact solver returns the true final result rather than an estimated evaluation. Once the solver can finish within the time limit, the agent can therefore play perfectly for the remainder of the game.

I profiled a complete game to see how much time the solver was actually using. I found that the solver was starting too late.

With **11 empty squares**, the exact solve finished in only **41 milliseconds**.
The agent therefore solved the position almost immediately and then wasted around **1.8 seconds** of its allowed time. However, with **13 empty squares**, the agent still spent the entire move budget searching using the heuristic.

This showed that my threshold of 12 empty squares was too conservative. I could not simply increase the threshold, though. Each additional empty square increases the search cost by roughly the branching factor. If an exact solve starts but does not finish before the timeout, it may leave the agent without a completed move.
To make attempting an early solve safe, I divided the endgame search into three stages.

#### Stage 1: Quick depth-4 search

The agent first performs a depth-4 search. This is only used as a safety net.
It guarantees that the agent has a valid move available even if the later solving stages run out of time.

#### Stage 2: Win/Loss/Draw solve

Instead of immediately calculating the exact final disc difference, the agent first asks a simpler question:

```text
Do I win, lose, or draw?
```

This can again be searched using a null window.

Because the agent only needs to determine which side of zero the final result falls on, the search can prune much more aggressively.

This allows the solver to handle positions with several more empty squares in the same amount of time. This is also why strong Othello engines normally determine win/loss/draw first and only calculate the exact final margin if enough time remains.

#### Stage 3: Exact solve

If the win/loss/draw search finishes early enough, the agent then performs the exact solve. This calculates the exact final disc margin.

The exact margin still matters when several legal moves all lead to a win. In that situation, the exact solver allows the agent to choose the move that wins by the largest number of discs.

I also made two additional changes to the endgame solver:

* I gave the solver its own transposition table.
* I changed its move ordering so that it prefers moves that leave the opponent with fewer legal replies.

The results were:

| Empty Squares | Win/Loss/Draw Solve | Exact Solve |
| ------------- | ------------------: | ----------: |
| 12            |              0.038s |      0.050s |
| 14            |              0.771s |      0.885s |
| 18            |              1.392s |    over 10s |

Before these improvements, solving a position with **14 empty squares took more than 12 seconds**. It now takes **less than one second**.

## IV. Quantitative Performance

### A. Search Depth and Number of Positions

I compared the old and new agents using the same midgame positions and the same **2-second move constraint**.

The results were:

| Board |       Old Agent |                 New Agent | Improvement |
| ----- | --------------: | ------------------------: | ----------: |
| 8x8   | 1,073 positions | 38,741 positions, depth 7 |         36x |
| 12x12 |   414 positions | 29,945 positions, depth 6 |         72x |

On an 8x8 board, my earlier report said the agent usually searched to around depth **3 to 4** in the midgame. The new agent reaches **depth 7**.

On 12x12, the old agent reached around depth **2 to 4**. The new agent reaches **depth 6**.

The improvement is larger on bigger boards because that is where the expensive evaluation and move-ordering code hurt the old implementation the most.

During a full game, the agent can now begin solving positions at around **15 to 17 empty squares** instead of waiting until 12. This gives the agent approximately **4 to 5 additional moves of perfect play per game**.

One example from a complete game was:

```text
ply 44   17 empties   1.722s   solved win/loss/draw
ply 46   15 empties   0.401s   solved exactly
ply 48   13 empties   0.029s   solved exactly
ply 50   11 empties   0.009s   solved exactly
```

### B. Breadth of the Search

The **branching factor** is the average number of legal moves available from a position. A basic minimax search has a cost of approximately:

```text
b^d
```

where:

* `b` is the branching factor
* `d` is the search depth

With perfect move ordering, alpha-beta pruning can reduce this to approximately:

```text
b^(d/2)
```

This means that, in the ideal case, alpha-beta can effectively search about twice as deep within the same amount of work. With random move ordering, the expected cost is closer to:

```text
b^(3d/4)
```

This explains why move ordering had such a large impact in this project.

Move ordering is not just a small optimization added on top of the search. It changes the effective exponent of the search cost.

My original ordering was more accurate because it used the complete evaluation function, but it was so expensive that the agent could not search enough positions for the improved ordering to be useful.
The cheaper ordering allowed the search to explore far more of the tree, which mattered much more.

### C. Impact of Board Size

Larger Othello boards usually have more legal moves in each position. This creates a wider search tree, which means the agent cannot search as deeply within the same 2-second time limit.

For example, a 12x12 midgame position can have **more than 20 legal moves**, while a 6x6 position may have around **8**. Larger boards are also where bitboards provide the largest advantage.

The original NumPy implementation checked eight directions from every square. Because of this, its cost increased with the number of squares on the board. The bitboard implementation instead performs a fixed set of eight shift operations, regardless of the board size.

As a result, the measured speedup increased from:

* **11x on 8x8**
* to **15x on 12x12**

### D. Comparison Against MCTS

I also rewrote the MCTS agent because my first implementation was too inefficient to provide a fair comparison with alpha-beta.

The new MCTS implementation uses:

* bitboards
* RAVE
* truncated playouts
* weighted playouts

One of the most important improvements is **RAVE**, which stands for **Rapid Action Value Estimation**.

On a large board such as 12x12, the MCTS agent may only complete a small number of simulations.

If there are many possible moves, each individual move receives very little data. RAVE helps by sharing information between moves. Instead of only updating the value of the move that began a playout, RAVE uses the result to learn about other moves that appeared later in the same playout. This means one simulated game can provide information about several possible moves.

I also added **truncated playouts**. Instead of simulating all the way to the end of the game, a truncated playout stops after **18 moves** and evaluates the resulting board using the heuristic. A full playout on a 12x12 board may last around **140 moves**. Stopping after 18 moves therefore allows the agent to run several times more simulations, although each simulation gives a less accurate estimate of the final game result.

I also found a redundant `check_endgame` call inside the playout loop.

Simply removing this extra call approximately doubled the playout rate.

| Board | With Redundant Check | Without |
| ----- | -------------------: | ------: |
| 8x8   |    195 playouts / 2s |     368 |
| 12x12 |     39 playouts / 2s |      74 |

After all of the MCTS improvements, the number of playouts from the opening position was approximately:

* **1,000 on 6x6**
* **380 on 8x8**
* **72 on 12x12**

The last result explains the original problem.

Even after fixing the implementation and making it approximately **15 times faster**, the 12x12 MCTS agent still completes only around **72 playouts** from the opening. That is not enough simulations to make reliable decisions.

My original conclusion that MCTS was poorly suited to this setting was therefore correct. However, before improving the implementation, I could not properly support that conclusion because I was comparing alpha-beta against an inefficient MCTS implementation.

## V. Advantages and Disadvantages

Alpha-beta is a strong fit for Othello for several reasons.

First, Othello is deterministic. There is no randomness in the game.
Second, the full state of the game is visible to both players, so the search does not need to deal with hidden information.
Third, there is already a large amount of useful strategic knowledge available for evaluating Othello positions.

Features such as, corners, mobility, stability are well understood. Because of this, an algorithm that relies on a board evaluation function has useful strategic information available. However, alpha-beta also has weaknesses.

* **It depends heavily on the evaluation function.**
  If the evaluation is wrong, the search can confidently choose a bad line. Because it searched deeply, the result can look very certain even though the original evaluation assumption was incorrect.

* **Weight tuning is difficult.**
  Unless the heuristic weights are measured or learned from data, choosing them is mostly guesswork. My current weights are still manually selected.

* **The search is still limited on large boards.**
  On a 12x12 board, the midgame search reaches only around depth 6. This is a real improvement over the previous version, but it is still not very deep.

* **Pruning depends on good ordering.**
  If the move ordering is misleading, the search may fail to prune efficiently and can spend time exploring less useful branches.

MCTS has almost the opposite advantages and disadvantages.

It does not require the same type of evaluation function, so it cannot be misled by an incorrect hand-written evaluation in the same way. However, MCTS depends heavily on running a large number of simulations.

In this project, **72 playouts on a 12x12 board is simply not enough volume**.

## VI. Possible Improvements

There are several ways the agent could be improved further.

### 1. Automated Weight Tuning

The heuristic weights are currently selected by hand. A better approach would be to collect several thousand board positions and record which player eventually wins from each one.

For each position, I could calculate the heuristic features such as mobility, stability, corners, frontier discs, and disc count. I could then train a **logistic regression** model. In this case, the model could learn how strongly each heuristic feature relates to the probability of winning.

The learned coefficients could then replace my hand-selected heuristic weights. This would replace guessing with measurement and is probably the most natural place to introduce machine learning into this project.

### 2. Pattern-Based Evaluation

Strong Othello engines often use a more advanced evaluation method based on board patterns. Instead of manually combining features such as mobility and corners, the agent could identify common patterns around edges and corner regions.

These patterns could then be looked up in a table trained using **millions of games**. This would likely provide the largest possible improvement to the quality of the evaluation function.

### 3. Deeper Endgame Solver

The current solver can usually begin solving the game at around **15 to 17 empty squares**. With stronger move ordering inside the endgame solver, it may be possible to reach **20 or more empty squares**. If this were possible, roughly the final third of each game could potentially be played perfectly.

### 4. Opening Book

The first several moves of Othello repeat frequently.

Instead of searching these positions from scratch during every game, the best early moves could be calculated beforehand and stored in an **opening book**. The agent could immediately play one of these precomputed moves during the opening. This would save search time for the more complicated midgame positions.

### 5. Better MCTS Playouts

My current MCTS simulations are guided using a static square-value table.

A stronger approach would be to use either, a shallow search or a trained policy to choose moves during the simulated games.

This would make the simulations more realistic and strategically stronger. The disadvantage is that each playout would become more expensive, which means the agent would complete fewer simulations within the 2-second limit.

### 6. Profile Before Optimizing

The most important lesson I would take from this project is to **profile the program before deciding what to optimize**.

Profiling showed that heuristic quality was not the main bottleneck. The real problem was that the agent spent so much time evaluating and ordering moves that it barely searched the game tree at all. Without profiling, I would not have guessed this.

Almost every major performance improvement in the final agent came from something I first measured rather than something I assumed would be important. That was one of the most useful results of the project: better measurements led to better design decisions.
