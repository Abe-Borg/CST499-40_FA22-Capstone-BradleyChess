import Settings
import pandas as pd
import numpy as np
from helper_methods import *
import re
import copy
import math
import random

class Agent:
    """ The agent class is responsible for deciding what chess move to play 
        based on the current state. The state is passed the the agent by the environ class.
        The agent class does not make any changes to the chessboard. 
    """
    def __init__(self, color, chess_data, is_opp_agent = False):        
        self.color = color
        self.chess_data = chess_data
        self.settings = Settings.Settings()
        self.is_trained = False
        self.is_opp_agent = is_opp_agent
        self.Q_table = self.init_Q_table(self.chess_data)


    def choose_action(self, environ_state, curr_game = 'Game 1', is_rl_agent_trained = False):
        """ method does two things. First, this method helps to train the agent. Once the agent is trained, 
            this method helps to pick the appropriate move based on highest val in the Q table for a given turn
            each agent will play through the database games exactly as shown during training
            :param environ_state is a dictionary 
            :return a dict containing chess move
        """
        self.legal_moves = environ_state['legal_moves']
        self.curr_turn = environ_state['curr_turn']        
        self.curr_game = curr_game
        self.is_rl_agent_trained = is_rl_agent_trained
        
        if self.is_trained: 
            # it's possible that there will be a completely new set of legal_moves during 
            # game play with humna that is not represented in the Q table
            # when that happens, update the Q_table
            matching_moves_in_q_table = self.Q_table[self.curr_turn].filter(items = self.legal_moves).index 
            if (len(matching_moves_in_q_table) == 0): 
                self.update_Q_table(self.legal_moves)
                return {'chess_move_str': self.legal_moves[0], 'chess_move_val': self.settings.new_move_pts}
            
            # there aren't any unique moves, go to game mode
            else: 
                return self.policy_game_mode() 

        # the agent is not trained OR this agent instance is the opposing agent
        else:
            # when the agent is being trained again, we want the opposing rl agent to behave differently 
            # during training
            return self.policy_training_mode()
    

    def policy_training_mode(self):
        """ this policy determines how the agents choose a move at each turn during training
            :param none
            :return dictionary with selected chess move information
        """
        if self.is_rl_agent_trained:
            ########## prioritize important moves ##########
            # if important move is found, return to caller
            valuable_move = self.choose_high_val_move()
            return valuable_move
        
        # the agent is not trained
        else:
            chess_move_str = self.chess_data.at[self.curr_game, self.curr_turn]
            return {'chess_move_str': chess_move_str}
        
    
    def policy_game_mode(self):
        """ policy to use during game between human player and agent 
            the agent searches its q table to find the moves with the highest q values at each turn.
            
            :param none
            :return a dictionary with chess_move as a string 
        """
        dice_roll = math.floor(random.uniform(0, 1 / (1 - self.settings.chance_for_random)))
        q_table_move_list = self.get_Q_values()
        matching_moves_in_q_table = q_table_move_list.filter(items = self.legal_moves)
        
        if self.curr_turn == 'W1':
            chess_move_str = matching_moves_in_q_table.sort_values(ascending = False).sample().index[0]
        else:
            if dice_roll == 1:
                chess_move_str = matching_moves_in_q_table.sort_values(ascending = False).sample().index[0]
                if self.Q_table.at[chess_move_str, self.curr_turn] == 0:
                    self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)
            else:
                chess_move_str = matching_moves_in_q_table.sort_values(ascending = False).index[0]
                if self.Q_table.at[chess_move_str, self.curr_turn] == 0:
                    self.change_Q_table_pts(chess_move_str, self.curr_turn, self.settings.new_move_pts)
        
        return {'chess_move_str': chess_move_str}


    def choose_high_val_move(self):
        """ method is used during training mode. during training, the dataframe of games with legal moves
            at current turn will be scanned for any strings that match the following regex options.
            the game at which the move occurs will be noted, and the chess move infor will be returned to caller. 
            :param none
            :return selected chess move and other information 
        """ 
        max_val = 0
        for chess_move_str in self.legal_moves:
            # checkmate will always be the best move, so return immediately
            if re.search(r'\#', chess_move_str):
                return {'chess_move_str': chess_move_str}

            if re.search(r'=Q', chess_move_str): 
                Q_move_val = 50
                if Q_move_val > max_val:
                    max_val = Q_move_val
                    promo_Q_chess_move_str = copy.copy(chess_move_str)
                    promotion_to_queen = {'chess_move_str': promo_Q_chess_move_str}

            if re.search(r'=', chess_move_str): 
                promotion_move_val = 25   
                if promotion_move_val > max_val:
                    max_val = promotion_move_val
                    promo_chess_move_str = copy.copy(chess_move_str)
                    promotion = {'chess_move_str': promo_chess_move_str}
                
            if re.search(r'\+', chess_move_str):
                check_move_val = 10
                if check_move_val > max_val:
                    max_val = check_move_val
                    check_chess_move_str = copy.copy(chess_move_str)
                    check_move = {'chess_move_str': check_chess_move_str}

            if re.search(r'x', chess_move_str):
                capture_move_val = 5 
                if capture_move_val > max_val:
                    max_val = capture_move_val
                    capture_move_str = copy.copy(chess_move_str)
                    capture_move = {'chess_move_str': capture_move_str}
            
        if max_val == 50:
            return promotion_to_queen
        elif max_val == 25:
            return promotion
        elif max_val == 10:
            return check_move
        elif max_val == 5:
            return capture_move
        else: 
            chess_move_str = random.sample(self.legal_moves, 1)
            return {'chess_move_str': chess_move_str[0]}
    # end of choose_high_val_move


    def init_Q_table(self, chess_data):
        """ creates the q table so the agent can be trained 
            the q table index represents unique moves across all games in the database for all turns.
            columns are the turns, 'W1' to 'B75'
            :param none
            :return q_table a pandas dataframe
        """
        # initialize array that will be used to build a list of pd Series.
        # each Series represents the unique moves for the turn for a player color, W1 for example.
        uniq_mov_list = []
        # this loop will make an array of pandas series, add 1 to make it 1 through total columns (inclusive)
        for i in range(1, self.settings.num_turns_per_player + 1):
            uniq_mov_list.append(chess_data.loc[:, self.color + str(i)].value_counts())

        uniq_mov_list = pd.concat(uniq_mov_list)        
        uniq_mov_list = uniq_mov_list.index.drop_duplicates(keep = 'first')
        turns_list = chess_data.loc[:, self.color + '1': self.color + str(self.settings.num_turns_per_player): 2].columns
        q_table = pd.DataFrame(columns = turns_list, index = uniq_mov_list)
        for col in q_table.columns:
            q_table[col].values[:] = 0

        q_table = q_table.astype(np.int32)
        return q_table # returns a pd dataframe


    def change_Q_table_pts(self, chess_move, curr_turn, pts):
        """ 
            :param chess_move is a string, 'e4' for example
            :param curr_turn is a string representing the turn num, for example, 'W10'
            :param pts is an int for the number of points to add to a q table cell
            :return none
        """
        self.Q_table.at[chess_move, curr_turn] += pts


    def update_Q_table(self, new_chess_moves):
        """ method will accept a list of strings
            the strings represents chess moves
            :pre the list parameter represents moves that are not already in the q table.
            :param new_chess_moves, a list of strings
            :return none
        """
        q_table_new_values = pd.DataFrame(index = new_chess_moves, columns = self.Q_table.columns, dtype = np.int32)
        for col in q_table_new_values.columns: 
            q_table_new_values[col].values[:] = 0

        self.Q_table = pd.concat([self.Q_table, q_table_new_values])
        # protect against duplicate indices
        if any(self.Q_table.index.duplicated()):
            self.Q_table = self.Q_table[~self.Q_table.index.duplicated()]
        
        
    def reset_Q_table(self):
        """ zeroes out the q table, call this method when you want to retrain the agent """
        for col in self.Q_table.columns:
            self.Q_table[col].values[:] = 0


    def get_Q_values(self):
        """ 
            Returns the series for the given turn. the series index represents the moves for that turn.
            :param curr_turn is a string, like 'W10'
            :return a pandas series, the index represents the chess moves, and the col is the curr turn in the game.
        """
        return self.Q_table[self.curr_turn]