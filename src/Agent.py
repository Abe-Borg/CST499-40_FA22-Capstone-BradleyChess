import game_settings
import pandas as pd
import numpy as np
import helper_methods

# import logging
# import log_config
# logger = logging.getLogger(__name__)

class Agent:
    """The `Agent` class is responsible for deciding what chess move to play 
    based on the current state. The state is passed to the agent by 
    the `Environ` class.

    Args:
        - color (str): A string indicating the color of the agent, either 'W' or 'B'.
        - chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data 
          used for training the agent.
        - learn_rate (float): A float between 0 and 1 that represents the learning rate.
        - discount_factor (float): A float between 0 and 1 that represents the discount factor.
    Attributes:
        - color (str): A string indicating the color of the agent, either 'W' or 'B'.
        - chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data 
        used for training the agent.
        - learn_rate (float): A float between 0 and 1 that represents the learning rate.
        - discount_factor (float): A float between 0 and 1 that represents the discount factor.
        - is_trained (bool): A boolean indicating whether the agent has been trained.
        - Q_table (pd.DataFrame): A Pandas DataFrame containing the Q-values for the agent.
    """
    def __init__(self, color: str, chess_data: pd.DataFrame, learn_rate = 0.6, discount_factor = 0.35):
        self.debug_file = open(game_settings.agent_debug_filepath, 'a')
        self.errors_file = open(game_settings.agent_errors_filepath, 'a')

        # too high num here means too focused on recent knowledge, 
        self.learn_rate = learn_rate
        # lower discount_factor number means more opportunistic, but not good long term planning
        self.discount_factor = discount_factor
        self.color = color
        self.chess_data = chess_data
        self.is_trained: bool = False
        self.Q_table: pd.DataFrame = self.init_Q_table(self.chess_data)
    ### end of __init__ ###

    def __del__(self):
        self.errors_file.close()
        self.debug_file.close()
    ### end of __del__ ###

    def choose_action(self, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> str:
        """Chooses the next chess move for the agent based on the current state.
        Args:
            environ_state (dict): A dictionary containing the current state.
            curr_game (str): A string indicating the current game being played. 
        Returns:
            str: A string representing the chess move chosen by the agent.
        """
        # check if any of the legal moves is not already in the Q table
        moves_not_in_Q_table: list[str] = [move for move in environ_state['legal_moves'] if move not in self.Q_table.index]

        if moves_not_in_Q_table:
            self.update_Q_table(moves_not_in_Q_table)

        if self.is_trained:
            return self.policy_game_mode(environ_state['legal_moves'])
        else:
            return self.policy_training_mode(curr_game, environ_state["curr_turn"])
    ### end of choose_action ###
    
    def policy_training_mode(self, curr_game: str, curr_turn: str) -> str:
        """Determines how the agents choose a move at each turn during training.
        Args:
            curr_game: A string representing the current game being played.
            curr_turn: A string representing the current turn, e.g. 'W1'.
        Returns:
            str: A string representing the chess move chosen by the agent.
        """
        return self.chess_data.at[curr_game, curr_turn]
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: list[str]) -> str:
        """Determines how the agent chooses a move during a game between a human player and the agent.
        The agent searches its Q table to find the moves with the highest Q values at each turn. 
        sometimes the agent will pick a random move. 

        Args:
            legal_moves: A list of strings representing the legal moves for the current turn.
        Returns:
            str: A string representing the chess move chosen by the agent.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent policy_game_mode ==========\n')
            self.debug_file.write(f'{self.color} Agent is choosing an action\n')
            self.debug_file.write("going to helper_methods get_number_with_probability\n\n")

        # dice roll will be 0 or 1
        dice_roll: int = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'back to Agent.policy_game_mode, arrived from helper_methods get_number_with_probability\n')
            self.debug_file.write(f'dice roll val is: {dice_roll}\n')
        
        # get the list of chess moves in the q table, then filter that so that 
        # only the legal moves for this turn remain.
        legal_moves_in_q_table = self.Q_table[curr_turn].loc[self.Q_table[curr_turn].index.intersection(legal_moves)]

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'legal_moves_in_q_table: {legal_moves_in_q_table}\n')
        
        if dice_roll == 1:
            # pick random move, that would be an index value of the pandas series (legal_moves_in_q_table)
            chess_move: str = legal_moves_in_q_table.sample().index[0]

            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Dice Roll was a 1, random move will be selected.\n')
                self.debug_file.write(f'chess_move_str: {chess_move}\n')
        else:
            # pick existing move in the q table that has the highest q value
            chess_move = legal_moves_in_q_table.idxmax()

            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Dice roll was not a 1\n')
                self.debug_file.write(f'chess_move_str: {chess_move}\n')

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'========== bye from Agent policy_game_mode ==========\n\n\n')

        return chess_move
    ### end of policy_game_mode ###

    def init_Q_table(self, chess_data: pd.DataFrame) -> pd.DataFrame:
        """Creates the Q table so the agent can be trained.
        The Q table index represents unique moves across all games in the database for all turns.
        Columns are the turns, 'W1' to 'BN' where N is determined by max number of turns per player.

        Args:
            chess_data: A pandas dataframe containing chess data.
        Returns:
            A pandas dataframe representing the Q table.
        """
        move_columns = [col for col in chess_data.columns if col.startswith(self.color)]

        # Flatten all the values in these columns and find unique values for the specified color
        unique_moves = chess_data[move_columns].values.flatten()
        unique_moves = pd.Series(unique_moves).unique()

        turns_list: pd.Index = chess_data.loc[:, f"{self.color}1": f"{self.color}{game_settings.max_num_turns_per_player}": 2].columns
        q_table: pd.DataFrame = pd.DataFrame(0, columns = turns_list, index = unique_moves, dtype = np.int32)
        return q_table
    ### end of init_Q_table ###

    def change_Q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """Adds points to a cell in the Q table.
        Args:
            chess_move (str): A string representing the chess move, e.g. 'e4'.
            curr_turn (str): A string representing the turn number, e.g. 'W10'.
            pts (int): An integer representing the number of points to add to the Q table cell.
        """
        self.Q_table.at[chess_move, curr_turn] += pts
    ### end of change_Q_table_pts ###

    def update_Q_table(self, new_chess_moves: list[str]) -> None:
        """Updates the Q table with new chess moves.
        This method creates a new DataFrame with the new chess moves, and appends it to the Q table. 

        Args:
            new_chess_moves (list[str]): A list of chess moves (strings) that are not already in the Q table.
        """
        q_table_new_values: pd.DataFrame = pd.DataFrame(0, index = new_chess_moves, columns = self.Q_table.columns, dtype = np.int32)
        self.Q_table = pd.concat([self.Q_table, q_table_new_values])
    ### update_Q_table ###

    # @log_config.log_execution_time_every_N()        
    def reset_Q_table(self) -> None:
        """Resets the Q table to all zeros.
        """
        self.Q_table.iloc[:, :] = 0        
    ### end of reset_Q_table ###
