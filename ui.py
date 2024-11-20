## UI Placeholder
import matplotlib.pyplot as plt
from constants import *
from pathlib import Path


class UIEngine:
    def __init__(self, grid_width=5, world=None) -> None:
        self.grid_size = (grid_width, grid_width)
        self.world = world
        self.step_number = 0
        plt.figure()
        plt.ion()

    def plot_box(
        self,
        x,
        y,
        w,
        text="",
        disc_color=None,
        color="silver",
    ):
        """
        Plot a box with optional disc (black/white)

        Parameters
        ----------
        x : int
            x position of the box
        y : int
            y position of the box
        w : int
            width of the box
        text : str
            text to display in the box
        disc_color : str
            color of the disc (either black or white)
        color : str
            color of the box border
        """
        # Draw the grid box
        plt.plot([x, x], [y, y + w], "-", lw=2, color=color)  # left wall
        plt.plot([x + w, x], [y + w, y + w], "-", lw=2, color=color)  # top wall
        plt.plot([x + w, x + w], [y, y + w], "-", lw=2, color=color)  # right wall
        plt.plot([x, x + w], [y, y], "-", lw=2, color=color)  # bottom wall

        # Place disc if applicable
        if disc_color:
            plt.gca().add_patch(
                plt.Circle((x + w / 2, y + w / 2), w / 2.5, color=disc_color)
            )

        # Display text in the box (optional)
        if len(text) > 0:
            plt.text(
                x + w / 2,
                y + w / 2,
                text,
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            )

    def plot_grid(self):
        """
        Plot the grid of the game
        """
        for x in range(1, self.grid_size[0] * 2 + 1, 2):
            for y in range(1, self.grid_size[1] * 2 + 1, 2):
                self.plot_box(x, y, 2)

    def plot_grid_with_board(
        self, chess_board, debug=False
    ):
        """
        Main function to plot the grid of the game

        Parameters
        ----------
        chess_board : np.array of size (grid_size[0], grid_size[1])
            chess board containing disc information (0 for empty, 1 for black, 2 for white)
        debug : bool
            if True, plot the grid coordinates for debugging
        """
        x_pos = 0
        for y in range(self.grid_size[1] * 2 + 1, 1, -2):
            y_pos = 0
            for x in range(1, self.grid_size[0] * 2 + 1, 2):
                disc_color = None
                if chess_board[x_pos, y_pos] == 1:
                    disc_color = PLAYER_1_COLOR  # Black disc
                elif chess_board[x_pos, y_pos] == 2:
                    disc_color = PLAYER_2_COLOR  # White disc

                # Optional debug text
                text = ""
                if debug:
                    text += " " + str(x_pos) + "," + str(y_pos)

                self.plot_box(x, y, 2, disc_color=disc_color, text=text)
                y_pos += 1
            x_pos += 1

    def fix_axis(self):
        """
        Fix the axis of the plot and set labels
        """
        # Set X labels
        ticks = list(range(0, self.grid_size[0] * 2))
        labels = [x // 2 for x in ticks]
        ticks = [x + 2 for i, x in enumerate(ticks) if i % 2 == 0]
        labels = [x for i, x in enumerate(labels) if i % 2 == 0]
        plt.xticks(ticks, labels)
        # Set Y labels
        ticks = list(range(0, self.grid_size[1] * 2))
        labels = [x // 2 for x in ticks]
        ticks = [x + 3 for i, x in enumerate(ticks) if i % 2 == 1]
        labels = [x for i, x in enumerate(reversed(labels)) if i % 2 == 1]
        plt.yticks(ticks, labels)
        # Move x-axis to the top
        plt.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        plt.xlabel("Column")
        plt.ylabel("Row", position="top")

    def plot_text_info(self):
        """
        Plot game textual information at the bottom
        """
        turn = 1 - self.world.turn
        agent_0 = f"{PLAYER_1_NAME}: {self.world.p0}"
        agent_1 = f"{PLAYER_2_NAME}: {self.world.p1}"
        plt.figtext(
            0.15,
            0.1,
            agent_0,
            wrap=True,
            horizontalalignment="left",
            color=PLAYER_1_COLOR,
            fontweight="bold" if turn == 0 else "normal",
        )
        plt.figtext(
            0.15,
            0.05,
            agent_1,
            wrap=True,
            horizontalalignment="left",
            color=PLAYER_2_COLOR,
            fontweight="bold" if turn == 1 else "normal",
        )

        if len(self.world.results_cache) > 0:
            plt.figtext(
                0.5,
                0.1,
                f"Scores: Blue: [{self.world.results_cache[1]}], Brown: [{self.world.results_cache[2]}]",
                horizontalalignment="left",
            )
            if self.world.results_cache[0]:
                if self.world.results_cache[1] > self.world.results_cache[2]:
                    win_player = "Blue wins!"
                elif self.world.results_cache[1] < self.world.results_cache[2]:
                    win_player = "Brown wins!"
                else:
                    win_player = "It is a Tie!"

                plt.figtext(
                    0.5,
                    0.05,
                    win_player,
                    horizontalalignment="left",
                    fontweight="bold",
                    color="green",
                )

    def render(self, chess_board, debug=False):
        """
        Render the board along with current game state

        Parameters
        ----------
        chess_board : np.array of size (grid_size[0], grid_size[1])
            2D array of board positions (0 for empty, 1 for black, 2 for white)
        debug : bool
            if True, display the position of each piece for debugging
        """
        plt.clf()
        self.plot_grid_with_board(chess_board, debug=debug)
        self.fix_axis()
        self.plot_text_info()
        plt.subplots_adjust(bottom=0.2)
        plt.pause(0.1)
        if self.world.display_save:
            Path(self.world.display_save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"{self.world.display_save_path}/{self.world.player_1_name}_{self.world.player_2_name}_{self.step_number}.pdf"
            )
        self.step_number += 1


if __name__ == "__main__":
    engine = UIEngine((5, 5))
    engine.render()
    plt.show()
