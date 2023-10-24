import Settings
import pandas as pd
import numpy as np
import helper_methods
import re
import copy
import random
import typing
import chess
import logging
import log_config

logger = logging.getLogger(__name__)

class Agent:
    """The `Agent` class is responsible for deciding what chess move to play based on the current state.

    The state is passed to the agent by the `Environ` class. The `Agent` class does not make any changes to the chessboard.

    Args:
        color (str): A string indicating the color of the agent, either 'W' or 'B'.
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data used for training the agent.

    Attributes:
        color (str): A string indicating the color of the agent, either 'W' or 'B'.
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data used for training the agent.
        settings (Settings.Settings): An instance of the `Settings` class containing the settings for the agent.
        is_trained (bool): A boolean indicating whether the agent has been trained.
        Q_table (pd.DataFrame): A Pandas DataFrame containing the Q-values for the agent.

    """

    MOVE_VALUES: dict[str, int] = {
        'check': 10,
        'capture': 5,
        'promotion': 25,
        'promotion_to_queen': 50
    }

    def __init__(self, color: str, chess_data: pd.DataFrame):        
        self.color = color
        self.chess_data = chess_data
        self.settings: Settings.Settings = Settings.Settings()
        self.is_trained: bool = False
        self.Q_table: pd.DataFrame = self.init_Q_table(self.chess_data)

    @log_config.log_execution_time_every_N
    def choose_action(self, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> dict[str]:
        """Chooses the next chess move for the agent based on the current state.

        This method does two things. First, it helps to train the agent. Once the agent is trained, 
        it helps to pick the appropriate move based on the highest value in the Q table for a given turn. 
        Each agent will play through the database games exactly as shown during training.

        Preconditions:
            The `environ_state` dictionary must contain the following keys:
                'turn_index': A string representing the current turn number.
                'curr_turn': A string representing the current turn, e.g. 'W1'.
                'legal_moves': A list of strings representing the legal moves for the current turn.
            curr_game must be a string that is a key in the chess_data dataframe.
        
        Invariants:
            environ_state and curr_game are not modified in this method.
        
        Side Effects:
            The Q table may be updated.

        Args:
            environ_state (dict): A dictionary containing the current state of the environment.
                Make sure environ_state is not modified in this method.
            curr_game (str): A string indicating the current game being played. 
                Relevant when initially training the agents. Defaults to 'Game 1'.

        Returns:
            dict[str]: A dictionary containing the chosen chess move.

        """
        environ_state_copy = copy.deepcopy(environ_state)
        self.legal_moves: List[str] = environ_state_copy['legal_moves']
        self.curr_turn: str = environ_state_copy['curr_turn']     
        self.curr_game: str = curr_game

        # check if any of the legal moves is not already in the Q table
        moves_not_in_Q_table: List[str] = [move for move in self.legal_moves if move not in self.Q_table[self.curr_turn].index]
        if moves_not_in_Q_table:
            self.update_Q_table(moves_not_in_Q_table)
                
        if self.is_trained: 
            return self.policy_game_mode() # this function call returns a dict that contains a chess move.
        else:
            return self.policy_training_mode() # this function call returns a dict that contains a chess move.
    ### end of choose_action ###
    
    @log_config.log_execution_time_every_N
    def policy_training_mode(self) -> dict[str]:
        """Determines how the agents choose a move at each turn during training.

        In this implementation, the agents will play out the games in the database exactly as shown.
        
        Args:
            None

        Returns:
            dict[str]: A dictionary containing the selected chess move as a string.

        """
        return {'chess_move_str': self.chess_data.at[self.curr_game, self.curr_turn]}
    ### end of policy_training_mode ###

    @log_config.log_execution_time_every_N        
    def policy_game_mode(self) -> dict[str]:
        """Determines how the agent chooses a move during a game between a human player and the agent.

        The agent searches its Q table to find the moves with the highest Q values at each turn. 
        However, sometimes the agent will pick a random move. 
        This method is also used when the two agents continue to be trained (when the play agains each other).

        Args:
            None

        Returns:
            dict[str]: A dictionary containing the selected chess move as a string.

        """
        # dice roll will be 0 or 1
        dice_roll: int = helper_methods.get_number_with_probability(self.settings.chance_for_random)
        
        # get the list of chess moves in the q table, then filter that so that 
        # only the legal moves for this turn remain.
        q_values: pd.Series = self.get_Q_values()
        legal_moves_in_q_table: pd.DataFrame = q_values.loc[q_values.index.intersection(self.legal_moves)]
        
        if dice_roll == 1:
            # pick random move, that would be an index value of the pandas series (legal_moves_in_q_table)
            chess_move_str: str = legal_moves_in_q_table.sample().index[0]
        else:
            # pick existing move in the q table that has the highest q value
            chess_move_str = legal_moves_in_q_table.idxmax()
        
        # if the q table val at that index has almost no points, add some points.
        # compare the abs difference with a small tolerance.
        # this should be small enough to account for any rounding errors
        if abs(self.Q_table.at[chess_move_str, self.curr_turn] - 0) < 1e-6:
            self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)

        return {'chess_move_str': chess_move_str}
    ### end of policy_game_mode ###

    @log_config.log_execution_time_every_N
    def choose_high_val_move(self) -> dict[str]:
        """ Selects the best chess move from a list of legal moves during training mode.
        
        The method is used during training mode to select the 
        best chess move from a list of legal moves. The method assigns a value to each 
        move based on whether it results in a check, capture, or promotion, and selects the 
        move with the highest value. If there are no moves with a value greater than 0, 
        the method selects a random move from the list of legal moves. 
        
        Args:
                None

        Returns:
            dict[str]: A dictionary containing the selected chess move as a string.

        """
        highest_move_value: int = 0
        best_move: dict[str] = None
        
        # loop through legal moves list to find the best move
        for chess_move_str in self.legal_moves:
            move: chess.Move = chess.Move.from_uci(chess_move_str)
            move_value: int = 0

            conditions: list[bool] = [
                (move.promotion == chess.QUEEN,"promotion_to_queen"),
                (move.promotion, "promotion"),
                (move.capture, "capture"),
                (self.board.is_check(move), "check")
            ]

            for condition, value_key in conditions:
                if condition:
                    move_value = self.MOVE_VALUES[value_key]
                    break

            if move_value > highest_move_value:
                highest_move_value = move_value
                best_move = {'chess_move_str': chess_move_str}

        if best_move is None:
            chess_move_str: str = random.sample(self.legal_moves, 1)
            best_move = {'chess_move_str': chess_move_str[0]}
        
        return best_move
    ### end of choose_high_val_move ###

    @log_config.log_execution_time_every_N
    def init_Q_table(self, chess_data: pd.DataFrame) -> pd.DataFrame:
        """Creates the Q table so the agent can be trained.

        The Q table index represents unique moves across all games in the database for all turns.
        Columns are the turns, 'W1' to 'BN' where N is determined by max number of turns per player, see Settings class.

        Args:
            chess_data: A pandas dataframe containing chess data.

        Returns:
            A pandas dataframe representing the Q table.
        """
        unique_moves: pd.Index = self.get_unique_moves(chess_data, self.color)
        turns_list: pd.Index =  self.get_turns_list(chess_data, self.color)
        q_table: pd.DataFrame = pd.DataFrame(0, columns = turns_list, index = unique_moves, dtype = np.int32)
        return q_table
    ### end of init_Q_table ###

    @log_config.log_execution_time_every_N
    def get_unique_moves(chess_data: pd.DataFrame, color: str) -> pd.Index:
        return pd.concat([chess_data.loc[:, f"{color}{i}"].value_counts() for i in range(1, self.settings.max_num_turns_per_player + 1)]).index.unique()

    @log_config.log_execution_time_every_N
    def get_turns_list(chess_data: pd.DataFrame, color: str) -> pd.Index: 
        return chess_data.loc[:, f"{color}1": f"{color}{self.settings.max_num_turns_per_player}": 2].columns

    @log_config.log_execution_time_every_N
    def change_Q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """Adds points to a cell in the Q table.

        This method adds the specified number of points to the cell in the Q table corresponding to the given chess move and turn number.

        Args:
            chess_move (str): A string representing the chess move, e.g. 'e4'.
            curr_turn (str): A string representing the turn number, e.g. 'W10'.
            pts (int): An integer representing the number of points to add to the Q table cell.

        Returns:
            None

        """
        self.Q_table.at[chess_move, curr_turn] += pts
    ### end of change_Q_table_pts ###

    @log_config.log_execution_time_every_N
    def update_Q_table(self, new_chess_moves: list[str]) -> Union[None, List[str]]:
        """Updates the Q table with new chess moves.

        This method filters out moves that are already in the Q table, creates a new DataFrame 
        with the new chess moves, and appends it to the Q table. If the list of new chess moves 
        is empty, a warning is logged and the method returns None.

        Args:
            new_chess_moves (list[str]): A list of chess moves (strings) that are not already in the Q table.

        Returns:
            None of list of str

        """
        # Filter out moves that are already in the Q_table
        # copying the list to avoid in-place modifications
        filtered_moves = [move for move in new_chess_moves if move not in self.Q_table.index]
        
        if not filtered_moves:
            logger.warning(f'new_chess_moves list was empty')
            return ["new_chess_moves list is empty"]

        q_table_new_values: pd.DataFrame = pd.DataFrame(0, index = filtered_moves, columns = self.Q_table.columns, dtype = np.int32)
        self.Q_table = self.Q_table.append(q_table_new_values)

        return None
    ### update_Q_table ###

    @log_config.log_execution_time_every_N        
    def reset_Q_table(self) -> None:
        """Resets the Q table to all zeros.

        This method sets all cells in the Q table to zero. Call this method when you want to retrain the agent.

        Args:
            None

        Returns:
            None

        """
        self.Q_table.iloc[:, :] = 0
    ### end of reset_Q_table ###

    @log_config.log_execution_time_every_N
    def get_Q_values(self) -> pd.Series:
        """Returns a Pandas series of Q values for the current turn.

        The series index represents the unique moves that have been found in the chess data for the current turn.

        Args:
            None

        Returns:
            pd.Series: A Pandas series where the index represents the chess moves, and the column is the current turn in the game.

        """
        return self.Q_table[self.curr_turn]
    ### end of get_Q_values