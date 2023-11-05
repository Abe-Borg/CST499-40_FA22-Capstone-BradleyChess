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
    the `Environ` class. The `Agent` class does not make any changes to the chessboard.

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

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent __init__ ==========\n')
            self.debug_file.write(f'Agent color is: {self.color}\n')
            self.debug_file.write("going to Agent.init_Q_table\n\n")

        self.Q_table: pd.DataFrame = self.init_Q_table(self.chess_data)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back to Agent __init__ just arrived from Agent.init_Q_table\n")
            self.debug_file.write(f'Q_table:\n{self.Q_table}\n\n')
            self.debug_file.write(f'is_trained: {self.is_trained}\n')
            self.debug_file.write(f'{self.color} Agent has been initialized\n')
            self.debug_file.write(f'========== Bye from Agent __init__ ==========\n\n\n')
    ### end of __init__ ###

    def __del__(self):
        self.debug_file.write(f'========== Hello & bye from Agent __del__ ==========\n')
        self.errors_file.close()
        self.debug_file.close()
    ### end of __del__ ###

    # @log_config.log_execution_time_every_N()
    def choose_action(self, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> str:
        """Chooses the next chess move for the agent based on the current state.
        Args:
            environ_state (dict): A dictionary containing the current state of the environment.
                Make sure environ_state is not modified in this method.
            curr_game (str): A string indicating the current game being played. 
                Relevant when initially training the agents. Defaults to 'Game 1'.
        Returns:
            str: A string representing the chess move chosen by the agent.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent choose_action ========== \n')
            self.debug_file.write(f'{self.color} Agent is choosing an action\n')
            self.debug_file.write(f'environ_state: {environ_state}\n')
            self.debug_file.write(f'legal_moves: {environ_state["legal_moves"]}\n')

        # check if any of the legal moves is not already in the Q table
        moves_not_in_Q_table: list[str] = [move for move in environ_state['legal_moves'] if move not in self.Q_table.index]

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'moves_not_in_Q_table: {moves_not_in_Q_table}\n')

        if moves_not_in_Q_table:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'========== going to Agent update_Q_table =========== \n\n')

            self.update_Q_table(moves_not_in_Q_table)

            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'========== back to Agent choose_action, arrived from Agen update_Q_table ===========\n\n')
                
        if self.is_trained:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'========== going to Agent policy_game_mode ========== \n\n')

            return self.policy_game_mode(environ_state['legal_moves'])
        else:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'========== going to Agent policy_training_mode ==========\n\n')

            return self.policy_training_mode(curr_game, environ_state["curr_turn"])
    ### end of choose_action ###
    
    def policy_training_mode(self, curr_game: str, curr_turn: str) -> str:
        """Determines how the agents choose a move at each turn during training.
        In this implementation, the agents will play out the games in the database exactly as shown.
        Args:
            curr_game: A string representing the current game being played.
            curr_turn: A string representing the current turn, e.g. 'W1'.
        Returns:
            str: A string representing the chess move chosen by the agent.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent policy_training_mode ========== \n')
            self.debug_file.write(f'{self.color} Agent is choosing an action\n')
            self.debug_file.write(f"the selected chess move from db is: {self.chess_data.at[curr_game, curr_turn]}\n")
            self.debug_file.write(f'========== bye from Agent policy_training_mode ========== \n\n\n')

        return self.chess_data.at[curr_game, curr_turn]
    ### end of policy_training_mode ###

    # @log_config.log_execution_time_every_N()        
    def policy_game_mode(self, legal_moves: list[str]) -> str:
        """Determines how the agent chooses a move during a game between a human player and the agent.
        The agent searches its Q table to find the moves with the highest Q values at each turn. 
        However, sometimes the agent will pick a random move. 

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

    # @log_config.log_execution_time_every_N()
    def init_Q_table(self, chess_data: pd.DataFrame) -> pd.DataFrame:
        """Creates the Q table so the agent can be trained.
        The Q table index represents unique moves across all games in the database for all turns.
        Columns are the turns, 'W1' to 'BN' where N is determined by max number of turns per player.

        Args:
            chess_data: A pandas dataframe containing chess data.
        Returns:
            A pandas dataframe representing the Q table.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent init_Q_table ==========\n')
            
        move_columns = [col for col in chess_data.columns if col.startswith(self.color)]

        # Flatten all the values in these columns and find unique values for
        # the specified color
        unique_moves = chess_data[move_columns].values.flatten()
        unique_moves = pd.Series(unique_moves).unique()

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'unique_moves: {unique_moves}\n')

        turns_list: pd.Index = chess_data.loc[:, f"{self.color}1": f"{self.color}{game_settings.max_num_turns_per_player}": 2].columns
        q_table: pd.DataFrame = pd.DataFrame(0, columns = turns_list, index = unique_moves, dtype = np.int32)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'turns_list: {turns_list}\n')
            self.debug_file.write(f'chess_data: {chess_data}\n')
            self.debug_file.write(f'chess_data shape: {chess_data.shape}\n')
            self.debug_file.write(f'q_table:\n{q_table}\n\n')
            self.debug_file.write(f'q_table shape: {q_table.shape}\n')
            self.debug_file.write(f'========== bye from Agent init_Q_table ==========\n\n\n')

        return q_table
    ### end of init_Q_table ###

    # @log_config.log_execution_time_every_N()
    def change_Q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """Adds points to a cell in the Q table.
        Args:
            chess_move (str): A string representing the chess move, e.g. 'e4'.
            curr_turn (str): A string representing the turn number, e.g. 'W10'.
            pts (int): An integer representing the number of points to add to the Q table cell.
        Returns:
            None
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent change_Q_table_pts ==========\n')
            self.debug_file.write(f'chess_move: {chess_move}\n')
            self.debug_file.write(f'curr_turn: {curr_turn}\n')
            self.debug_file.write(f'pts: {pts}\n')
            self.debug_file.write(f'========== bye from Agent change_Q_table_pts ==========\n\n\n')

        try:
            self.Q_table.at[chess_move, curr_turn] += pts
        except KeyError as e:
            self.errors_file.write(f'\n========== something went wrong at Agent change_Q_table_pts ==========\n')
            self.errors_file.write(f'KeyError: {e}\n')
            self.errors_file.write(f'chess_move: {chess_move}\n')
            self.errors_file.write(f'curr_turn: {curr_turn}\n')
            self.errors_file.write(f'pts: {pts}\n')
            raise KeyError from e
    ### end of change_Q_table_pts ###

    # @log_config.log_execution_time_every_N()
    def update_Q_table(self, new_chess_moves: list[str]) -> None:
        """Updates the Q table with new chess moves.
        This method creates a new DataFrame with the new chess moves, and 
        appends it to the Q table. 

        Args:
            new_chess_moves (list[str]): A list of chess moves (strings) that are not already in the Q table.
        Returns:
            None
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent update_Q_table ==========\n')
            self.debug_file.write(f'new_chess_moves: {new_chess_moves}\n')
    
        q_table_new_values: pd.DataFrame = pd.DataFrame(0, index = new_chess_moves, columns = self.Q_table.columns, dtype = np.int32)
        self.Q_table = pd.concat([self.Q_table, q_table_new_values])

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'q_table_new_values: {q_table_new_values}\n')
            self.debug_file.write(f'Q_table:\n{self.Q_table}\n\n')
            self.debug_file.write(f'========== bye from Agent update_Q_table ==========\n\n\n')
    ### update_Q_table ###

    # @log_config.log_execution_time_every_N()        
    def reset_Q_table(self) -> None:
        """Resets the Q table to all zeros.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'\n========== Hello from Agent reset_Q_table ==========\n')
            self.debug_file.write(f'Q_table:\n{self.Q_table.head()}\n\n')

        self.Q_table.iloc[:, :] = 0
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'reset Q_table is:\n{self.Q_table.head()}\n\n')
            self.debug_file.write(f'bye from Agent reset_Q_table\n\n\n')
    ### end of reset_Q_table ###
