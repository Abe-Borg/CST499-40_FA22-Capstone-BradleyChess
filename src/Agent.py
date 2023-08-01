import Settings
import pandas as pd
import numpy as np
import helper_methods
import re
import copy
import random
import typing
import chess

class Agent:
    """ 
        The agent class is responsible for deciding what chess move to play 
        based on the current state. The state is passed the the agent by the environ class.
        The agent class does not make any changes to the chessboard. 
    """
    def __init__(self, color: str, chess_data: pd.DataFrame):        
        self.color = color
        self.chess_data = chess_data
        self.settings = Settings.Settings()
        self.is_trained = False
        self.Q_table = self.init_Q_table(self.chess_data)


    def choose_action(self, environ_state: dict, curr_game: str = 'Game 1') -> dict[str]:
        """ 
            method does two things. First, this method helps to train the agent. Once the agent is trained, 
            this method helps to pick the appropriate move based on highest val in the Q table for a given turn.
            Each agent will play through the database games exactly as shown during training
            :param environ_state is a dictionary 
            :param curr_game. This is relevant when initally training the agents.
            :return a dict containing chess move
        """
        self.legal_moves = environ_state['legal_moves'] # this is a list of strings
        self.curr_turn = environ_state['curr_turn'] # this is a string, like 'W1'       
        self.curr_game = curr_game

        # check if any of the legal moves is not already in the Q table
        moves_not_in_Q_table = [move for move in self.legal_moves if move not in self.Q_table[self.curr_turn].index]
        if moves_not_in_Q_table:
            self.update_Q_table(moves_not_in_Q_table)
                
        
        if self.is_trained: 
            return self.policy_game_mode() # this function call returns a dict that contains a chess move.
        else:
            return self.policy_training_mode() # this function call returns a dict that contains a chess move.
    ### end of choose_action ###
    
    def policy_training_mode(self) -> dict[str]:
        """ 
            this policy determines how the agents choose a move at each 
            turn during training. In this implementation, the agents
            will play out the games in the database exactly as shown.
            :param none
            :return dictionary with selected chess move as a string
        """
        return {'chess_move_str': self.chess_data.at[self.curr_game, self.curr_turn]}
    ### end of policy_training_mode ###
        
    def policy_game_mode(self) -> dict[str]:
        """ 
            policy to use during game between human player and agent 
            the agent searches its q table to find the moves with the 
            highest q values at each turn. However, sometimes
            the agent will pick a random move.

            This method is also used when the agents continue to be trained.
            :param none
            :return a dictionary with chess_move as a string 
        """
        # dice roll will be 0 or 1
        dice_roll = helper_methods.get_number_with_probability(self.settings.chance_for_random)
        
        # get the list of chess moves in the q table, then filter that so that 
        # only the legal moves for this turn remain.
        q_values = self.get_Q_values()  # this is a pandas series
        legal_moves_in_q_table = q_values.loc[q_values.index.intersection(self.legal_moves)]
        
        if dice_roll == 1:
            # pick random move, that would be an index value of the pandas series (legal_moves_in_q_table)
            chess_move_str = legal_moves_in_q_table.sample().index[0]
        else:
            # pick existing move in the q table that has the highest q value
            chess_move_str = legal_moves_in_q_table.idxmax()
        
        # if the q table val at that index has no points, add some points.
        # compare the abs difference with a small tolerance.
        # this should be small enough to account for any rounding errors
        if abs(self.Q_table.at[chess_move_str, self.curr_turn] - 0) < 1e-6:
            self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)

        return {'chess_move_str': chess_move_str}
    ### end of policy_game_mode ###

    def choose_high_val_move(self) -> dict[str]:
        """ 
            The choose_high_val_move method is used during training mode to select the 
            best chess move from a list of legal moves. The method assigns a value to each 
            move based on whether it results in a check, capture, or promotion, and selects the 
            move with the highest value. If there are no moves with a value greater than 0, 
            the method selects a random move from the list of legal moves. 
            The method returns a dictionary containing the selected chess move as a string.
            :param none
            :return selected chess move
        """ 
        move_values = {
            'check': 10,
            'capture': 5,
            'promotion': 25,
            'promotion_to_queen': 50,
        }

        max_val = 0
        best_move = None
        
        for chess_move_str in self.legal_moves:
            move = chess.Move.from_uci(chess_move_str)
            move_val = 0

            if move.promotion:
                if move.promotion == chess.QUEEN:
                    move_val = move_values['promotion_to_queen']
                else:
                    move_val = move_values['promotion']
            elif move.capture:
                move_val = move_values['capture']
            elif self.board.is_check(move):
                move_val = move_values['check']

            if move_val > max_val:
                max_val = move_val
                best_move = {'chess_move_str': chess_move_str}

        if best_move is None:
            chess_move_str = random.sample(self.legal_moves, 1)
            best_move = {'chess_move_str': chess_move_str[0]}
        
        return best_move
    ### end of choose_high_val_move ###

    def init_Q_table(self, chess_data: pd.DataFrame) -> pd.DataFrame:
        """Creates the Q table so the agent can be trained.

        The Q table index represents unique moves across all games in the database for all turns.
        Columns are the turns, 'W1' to 'BN' where N is determined by max number of turns per player, see Settings class.

        Args:
            chess_data: A pandas dataframe containing chess data.

        Returns:
            A pandas dataframe representing the Q table.
        """
        unique_moves = pd.concat([chess_data.loc[:, f"{self.color}{i}"].value_counts() for i in range(1, self.settings.num_turns_per_player + 1)])
        unique_moves = unique_moves.index.unique()
        turns_list = chess_data.loc[:, self.color + '1': self.color + str(self.settings.num_turns_per_player): 2].columns
        q_table = pd.DataFrame(0, columns = turns_list, index = unique_moves, dtype = np.int32)
        return q_table
    ### end of init_Q_table ###

    def change_Q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """ 
            :param chess_move is a string, 'e4' for example
            :param curr_turn is a string representing the turn num, for example, 'W10'
            :param pts is an int for the number of points to add to a q table cell
            :return none
        """
        self.Q_table.at[chess_move, curr_turn] += pts
    ### end of change_Q_table_pts ###

    def update_Q_table(self, new_chess_moves: list[str]) -> None:
        """ 
        Update the Q table with new chess moves.

        :pre: The list parameter represents moves that are not already in the q table.
        :param new_chess_moves: List of chess moves (strings).
        :return: None
        """
        # Filter out moves that are already in the Q_table
        new_chess_moves = [move for move in new_chess_moves if move not in self.Q_table.index]
        
        if new_chess_moves:
            q_table_new_values = pd.DataFrame(index = new_chess_moves, columns = self.Q_table.columns, dtype = np.int32)
            q_table_new_values.values[:] = 0
            self.Q_table = self.Q_table.append(q_table_new_values)
    ### update_Q_table ###
        
    def reset_Q_table(self) -> None:
        """ Zeroes out the Q-table. Call this method when you want to retrain the agent. """
        self.Q_table.iloc[:, :] = 0
    ### end of reset_Q_table ###

    def get_Q_values(self) -> pd.Series:
        """ 
        Returns the series for the given turn. The series index represents the 
        unique moves that have been found in the chess data for that turn.

        :return: A pandas series, the index represents the chess moves, and the column is the current turn in the game.
        """
        return self.Q_table[self.curr_turn]
    ### end of get_Q_values