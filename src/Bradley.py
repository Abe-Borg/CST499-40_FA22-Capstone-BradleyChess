import Environ
import Agent
import Settings
import re
from helper_methods import *
import chess
import chess.engine
import pandas as pd
import copy
import logging
import log_config

logger = logging.getLogger(__name__)

## note to reader: throughout this code you will see dictionaries for things that
## don't necessily need a dictionary. chess_move is a good example.
## I did this to make this implementation flexible. There is more
## than one way to indicate the chess move, along with other info related to the chess move.

class Bradley:
    """ 
        Bradley class acts as the single point of communication between the RL agent and the player.
        this class trains the agent and helps to manage the chessboard during play between the comp and the
        user. This is a composite class with members of the Environ class and the Agent class
    """
    def __init__(self, chess_data: pd.DataFrame):
        self.chess_data = chess_data
        self.settings = Settings.Settings()
        self.environ = Environ.Environ(self.chess_data)   
        self.W_rl_agent = Agent.Agent('W', self.chess_data)             
        self.B_rl_agent = Agent.Agent('B', self.chess_data)

        # stockfish is used to analyze positions during training
        # this is how we estimate the q value at each position, and also for anticipated next position
        self.engine = chess.engine.SimpleEngine.popen_uci(self.settings.stockfish_filepath)

        # we only want to look ahead one move, that's the anticipated q value at next state, and next action
        # this number has a massive inpact on how long it takes to analyze a position and it really doesn't help to go beyond 8.
        self.num_moves_to_return = 1
        self.depth_limit = 8
        self.time_limit = None
        self.search_limit = chess.engine.Limit(depth = self.depth_limit, time = self.time_limit)


    def recv_opp_move(self, chess_move: str) -> bool:                                                                                 
        """ 
            receives human's chess move and sends it to the environ so 
            that the chessboard can be loaded w/ opponent move.
            this method assumes that the incoming chess move is valid and playable.
            :param chess_move this is the str input, which comes from the human player
            :return bool, True for successful move input
        """
        # load_chessboard returns False if failure to add move to board,
        if self.environ.load_chessboard(chess_move):
            # loading the chessboard was a success, now just update the curr state
            self.environ.update_curr_state()
            return True
        else:
            return False
    ### end of recv_opp_move ###

    def rl_agent_chess_move(self, rl_agent_color: str) -> dict[str]:
        """ 
            this method will do two things, load up the environ.chessboard with the agent's action (chess_move)
            and return the string of the agent's chess move
            :param str that indicates the players as W or B
            :return a dictionary with the chess move str as one of the items
        """
        if rl_agent_color == 'W':
            chess_move = self.W_rl_agent.choose_action(self.environ.get_curr_state()) # choose_action returns a dictionary
        else:
            chess_move = self.B_rl_agent.choose_action(self.environ.get_curr_state()) # choose_action returns a dictionary
        
        self.environ.load_chessboard(chess_move['chess_move_str'])
        self.environ.update_curr_state()
        return chess_move
    ### end of rl_agent_chess_move

    def get_fen_str(self) -> str:
        """ 
            call this at each point in the chess game for the fen 
            string. The FEN string can be used to reconstruct a chessboard position
            The FEN string will change each time a move is made.
            :param none
            :return a string that represents a board state, something like this,
            'rnbqkbnr/pppp1ppp/8/8/4p1P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3'
        """
        return copy.copy(self.environ.board.fen())
    ### end of get_gen_str ###

    def get_opp_agent_color(self, rl_agent_color: str) -> str:
        """ 
            method determines what color the opposing rl agent should be assigned to
            :param string that represents the rl_agent color 
            :return string that represents the opposing rl agent color
        """
        if rl_agent_color == 'W':
            return 'B'
        else:
            return 'W'
    ### end of get_opp_agent_color
            
    def get_curr_turn(self) -> str:
        """ 
            find curr turn which would be a string like 'W1' or 'B5'
            :param none
            :return a string representing the current turn
        """
        return self.environ.get_curr_turn()
    ### end of get_curr_turn

    def game_on(self) -> bool:
        """ 
            method determines when the game is over 
            game can end if python chess board determines the game is over, 
            or if the game is at num_turns_per_player * 2 - 1 moves per player (-1 b/c index starts at 0)
            :param none
            :return bool, False means the game is over
        """
        if self.environ.board.is_game_over() or self.environ.turn_index >= self.settings.num_turns_per_player * 2 - 1:
            return False
        else:
            return True
    ### end of game_on
    
    def get_legal_moves(self) -> list[str]:
        """ 
            returns a list of strings that represents the legal moves for the 
            current turn and state of the chessboard
            :param none
            :return a list of strings
        """
        return self.environ.get_legal_moves()
    ### end of get_legal_moves
    
    def get_rl_agent_color(self) -> str: 
        """ 
            simple getter. returns the string for the color of the agent. 
            :param none
            :return string, 'W' for example
        """
        return copy.copy(self.rl_agent.color)
    ### end of get_rl_agent_color
    
    def get_game_outcome(self) -> chess.Outcome or str:   
        """ method will return the winner, loser, or draw for a chess game
            :param none
            :return chess.Outcome class, this class has a result() method that returns 1-0, 0-1 or 1/2-1/2 
            or false if failure
        """
        try:
            return self.environ.board.outcome().result()
        except AttributeError:
            return 'outcome not available, most likely game ended because turn_index was too high or player resigned'
    ### end of get_game_outcome
    
    def get_game_termination_reason(self) -> str:
        """ the method determines why the game ended, for example
            if the game ended due to a checkmate, a string termination.CHECKMATE will be returned
            this method will return an exception if the game ended by too many moves or a player resigning
            :param none
            :return a single string that describes the reason for the game ending
        """
        try:
            return str(self.environ.board.outcome().termination)
        except AttributeError:
            return 'termination reason not available, most likely game ended because turn_index was too high or player resigned'
    ### end of get_game_termination_reason
    
    def get_chessboard(self) -> chess.Board:
        """ returns the chessboard object
            the chessboard object can be printed and the output will be an ASCII representation of the chessboard
            and current state of the game.
            :param none
            :return chessboard object
        """
        return self.environ.board
    ### end of get_chess_board

    def train_rl_agents(self, training_results_filepath: str) -> None:
        """ trains the agents and then sets their is_trained flag to True.
            the algorithm used for training is SARSA. Two rl agents train each other
            A chess game can end at multiple places during training, so we need to 
            check for end-game conditions throughout this method.

            The agents are trained by playing games from a database exactly as
            shown, and learning from that. Then the agents are trained again (in another method), 
            but this time they makes their own decisions. A White or Black agent can be trained.

            This training regimen first trains the agents to play a good positional game.
            Then when the agents are retrained, the agents behaviour can be fine-tuned
            by adjusting hyperparameters.

            In the current implementation, training takes a very long time. To train both
            agents on 100k games, it takes 1 week. That can probably be faster, but I 
            just don't know how to speed it up. It could be a limitation of Python.

            :param: results doc filepath
            :return none
        """ 
        training_results = open(training_results_filepath, 'a')

        # init Qval to get things started.
        W_curr_Qval = self.settings.initial_q_val
        B_curr_Qval = self.settings.initial_q_val

        # for each game in the training data set.
        for game_num in self.chess_data.index:
            num_chess_moves = self.chess_data.at[game_num, 'Num Moves']
            training_results.write(f'\n\n Start of {game_num} training\n\n')

            #initialize environment to provide a state, s
            curr_state = self.environ.get_curr_state()

            # this loop plays through one game in the database, exactly as shown.
            while curr_state['turn_index'] < num_chess_moves:
                ##### WHITE AGENT PICKS MOVE, DONT PLAY IT YET THOUGH #####
                # choose action a from state s, using policy
                W_curr_action = self.W_rl_agent.choose_action(curr_state, game_num)
                W_chess_move = W_curr_action['chess_move_str']
                curr_turn = curr_state['curr_turn']

                ### ASSIGN POINTS TO Q_TABLE FOR WHITE
                # on the first turn for white, this would assign to W1 col at chess_move row.
                # on W's second turn, this would be Q_next which is calculated on the first loop.
                try:
                    self.W_rl_agent.change_Q_table_pts(W_chess_move, curr_turn, W_curr_Qval)
                except KeyError: 
                    # chess move is not represented in the Q table, update Q table and try again.
                    self.W_rl_agent.update_Q_table(W_chess_move)
                    self.W_rl_agent.change_Q_table_pts(W_chess_move, curr_turn, W_curr_Qval)

                ##### WHITE AGENT PLAYS SELECTED MOVE #####
                # take action a, observe r, s', and load chessboard
                self.environ.load_chessboard(W_chess_move)
                self.environ.update_curr_state()

                # the state changes each time a move is made, so get curr state again.                
                curr_state = self.environ.get_curr_state()

                # false means we don't care about the anticipated next move
                analysis_results = self.analyze_board_state(self.get_chessboard(), False)
                if analysis_results['mate_score'] is None:
                    W_reward = analysis_results['centipawn_score']
                else: # there is an impending checkmate
                    W_reward = analysis_results['mate_score'] * self.settings.mate_score_factor
                
                # check if game ended after W's move
                if curr_state['turn_index'] >= num_chess_moves:
                    break
                else: # game continues
                    ##### WHITE AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT !!! #####
                    # observe next_state, s' (this would be after the player picks a move
                    # and choose action a'

                    # W just played a move. the board has changed, if stockfish analyzes the board, 
                    # it will give points for White, based on White's latest move.
                    # We also need the points for the ANTICIPATED next state, 
                    # given the ACTICIPATED next action. In this case, the anticipated response from Black.

                    # analysis returns an array of dicts. in our analysis, we only consider the first dict returned by 
                    # the stockfish analysis.                    
                    analysis_results = self.analyze_board_state(self.get_chessboard())               
                    self.environ.load_chessboard_for_Q_est(analysis_results) # anticipated next action is a str like, 'e6f2'
                    
                    W_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False) # we want the points, and the anticipated next move
                    
                    # get pts for est_Qval 
                    if W_est_Qval_analysis['mate_score'] is None:
                        W_est_Qval = W_est_Qval_analysis['centipawn_score']
                    else: # there is an impending checkmate
                        W_est_Qval = W_est_Qval_analysis['mate_score'] * self.settings.mate_score_factor
    
                    # IMPORTANT STEP!!! pop the chessboard of last move, we are estimating board states, not
                    # playing a move.
                    self.environ.pop_chessboard()

                ##### BLACK AGENT PICKS MOVE, DONT PLAY IT YET THOUGH #####
                B_curr_action = self.B_rl_agent.choose_action(curr_state, game_num)
                B_chess_move = B_curr_action['chess_move_str']
                curr_turn = curr_state['curr_turn']

                ### ASSIGN POINTS TO Q_TABLE FOR BLACK
                try:
                    self.B_rl_agent.change_Q_table_pts(B_chess_move, curr_turn, B_curr_Qval)
                except KeyError: 
                    # chess move is not represented in the Q table, update Q table and try again.
                    self.B_rl_agent.update_Q_table(B_chess_move)
                    self.B_rl_agent.change_Q_table_pts(B_chess_move, curr_turn, B_curr_Qval)

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                # take action a, observe r, s', and load chessboard
                self.environ.load_chessboard(B_chess_move)
                self.environ.update_curr_state()

                # the state changes each time a move is made, so get curr state again.                
                curr_state = self.environ.get_curr_state()

                # false means we don't care about the anticipated next move
                analysis_results = self.analyze_board_state(self.get_chessboard(), False)
                if analysis_results['mate_score'] is None:
                    B_reward = analysis_results['centipawn_score']
                else: # there is an impending checkmate
                    B_reward = analysis_results['mate_score'] * self.settings.mate_score_factor

                if self.environ.turn_index >= self.settings.num_turns_per_player * 2:
                    # index has reached max value, this will only happen for Black at Black's final max turn, 
                    # White won't ever have this problem.
                    break

                # check if game ended after B's move
                if curr_state['turn_index'] >= num_chess_moves:
                    break
                else: # game continues
                    ##### AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT !!! #####
                    analysis_results = self.analyze_board_state(self.get_chessboard())               
                    self.environ.load_chessboard_for_Q_est(analysis_results) # anticipated next action is a str like, 'e6f2'
                    
                    B_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False) # we want the points, and the anticipated next move
                    
                    # get pts for est_Qval 
                    if B_est_Qval_analysis['mate_score'] is None:
                        B_est_Qval = B_est_Qval_analysis['centipawn_score']
                    else: # there is an impending checkmate
                        B_est_Qval = B_est_Qval_analysis['mate_score'] * self.settings.mate_score_factor
                    self.environ.pop_chessboard()

                # CRITICAL STEP, this is the main part of the SARSA algorithm. Do this for both agents
                W_next_Qval = W_curr_Qval + self.W_rl_agent.settings.learn_rate * (W_reward + ((self.W_rl_agent.settings.discount_factor * W_est_Qval) - W_curr_Qval))
                B_next_Qval = B_curr_Qval + self.B_rl_agent.settings.learn_rate * (B_reward + ((self.B_rl_agent.settings.discount_factor * B_est_Qval) - B_curr_Qval))

                # on the next turn, this Q value will be added to the Q table. so if this is the end of the first round,
                # next round it will be W2 and then we assign the q value at W2 col
                W_curr_Qval = W_next_Qval
                B_curr_Qval = B_next_Qval

                # this is the next state, s'  the next action, a' is handled at the beginning of the while loop
                curr_state = self.environ.get_curr_state()

            # reset environ to prepare for the next game
            training_results.write(f'Game {game_num} is over.\n')
            training_results.write(f'\nchessboard looks like this:\n\n')
            training_results.write(f'\n {self.environ.board}\n\n')
            training_results.write(f'Game result is: {self.get_game_outcome()}\n')
            training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')
            self.reset_environ()
        
        # training is complete
        self.W_rl_agent.is_trained = True
        self.B_rl_agent.is_trained = True
        training_results.close()   
        self.reset_environ()
    ### end of train_rl_agents

    def continue_training_rl_agents(self, training_results_filepath: str, num_games: int) -> None:
        """ continues to train the agent, this time the agents make their own decisions instead 
            of playing through the database.
            I KNOW I KNOW .... TERRIBLE TO DUPLICATE CODE LIKE THIS, I'm a lazy programmer.
            :param num_games, how long to train the agent
            :return none
        """ 
        training_results = open(training_results_filepath, 'a')

        # init Qval to get things started.
        W_curr_Qval = self.settings.initial_q_val
        B_curr_Qval = self.settings.initial_q_val

        # for each game in the training data set.
        for curr_training_game in range(num_games):
            training_results.write(f'\n\n Start of game {curr_training_game} training\n\n') 
            
            curr_state = self.environ.get_curr_state()
            while self.game_on():
                ##### WHITE AGENT PICKS MOVE, DONT PLAY IT YET THOUGH #####
                W_curr_action = self.W_rl_agent.choose_action(curr_state)
                W_chess_move = W_curr_action['chess_move_str']
                curr_turn = curr_state['curr_turn']

                ### ASSIGN POINTS TO Q_TABLE FOR WHITE
                try:
                    self.W_rl_agent.change_Q_table_pts(W_chess_move, curr_turn, W_curr_Qval)
                except KeyError: 
                    self.W_rl_agent.update_Q_table(W_chess_move)
                    self.W_rl_agent.change_Q_table_pts(W_chess_move, curr_turn, W_curr_Qval)

                ##### WHITE AGENT PLAYS SELECTED MOVE #####
                self.environ.load_chessboard(W_chess_move)
                self.environ.update_curr_state()              
                curr_state = self.environ.get_curr_state()
                W_reward = self.get_reward(W_chess_move)
                
                if self.game_on():
                    ##### WHITE AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT !!! #####                
                    analysis_results = self.analyze_board_state(self.get_chessboard())               
                    self.environ.load_chessboard_for_Q_est(analysis_results)
                    W_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False)
                    
                    if W_est_Qval_analysis['mate_score'] is None:
                        W_est_Qval = W_est_Qval_analysis['centipawn_score']
                    else:
                        W_est_Qval = W_est_Qval_analysis['mate_score'] * self.settings.mate_score_factor
                    
                    self.environ.pop_chessboard()
                else:
                    break

                ##### BLACK AGENT PICKS MOVE, DONT PLAY IT YET THOUGH #####
                B_curr_action = self.B_rl_agent.choose_action(curr_state)
                B_chess_move = B_curr_action['chess_move_str']
                curr_turn = curr_state['curr_turn']

                ### ASSIGN POINTS TO Q_TABLE FOR BLACK
                try:
                    self.B_rl_agent.change_Q_table_pts(B_chess_move, curr_turn, B_curr_Qval)
                except KeyError: 
                    self.B_rl_agent.update_Q_table(B_chess_move)
                    self.B_rl_agent.change_Q_table_pts(B_chess_move, curr_turn, B_curr_Qval)

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                self.environ.load_chessboard(B_chess_move)
                self.environ.update_curr_state()
                curr_state = self.environ.get_curr_state()
                B_reward = self.get_reward(B_chess_move)

                if self.environ.turn_index >= self.settings.num_turns_per_player * 2:
                    # index has reached max value, this will only happen for Black at Black's final max turn, 
                    # White won't ever have this problem.
                    break

                if self.game_on():
                    ##### BLACK AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT !!! #####                
                    analysis_results = self.analyze_board_state(self.get_chessboard())               
                    self.environ.load_chessboard_for_Q_est(analysis_results)
                    B_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False)
                    
                    if B_est_Qval_analysis['mate_score'] is None:
                        B_est_Qval = B_est_Qval_analysis['centipawn_score']
                    else:
                        B_est_Qval = B_est_Qval_analysis['mate_score'] * self.settings.mate_score_factor

                    self.environ.pop_chessboard()
                else:
                    break

                # CRITICAL STEP, this is the main part of the SARSA algorithm. Do this for both agents
                W_next_Qval = W_curr_Qval + self.W_rl_agent.settings.learn_rate * (W_reward + ((self.W_rl_agent.settings.discount_factor * W_est_Qval) - W_curr_Qval))
                B_next_Qval = B_curr_Qval + self.B_rl_agent.settings.learn_rate * (B_reward + ((self.B_rl_agent.settings.discount_factor * B_est_Qval) - B_curr_Qval))

                W_curr_Qval = W_next_Qval
                B_curr_Qval = B_next_Qval

                # this is the next state, s'  the next action, a' is handled at the beginning of the while loop
                curr_state = self.environ.get_curr_state()

            # reset environ to prepare for the next game
            training_results.write(f'Game {curr_training_game} is over.\n')
            training_results.write(f'\nchessboard looks like this:\n\n')
            training_results.write(f'\n {self.environ.board}\n\n')
            training_results.write(f'Game result is: {self.get_game_outcome()}\n')
            training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')
            self.reset_environ()
        
        training_results.close()   
        self.reset_environ()
    ### end of continue_training_rl_agents

    def analyze_board_state(self, board: chess.Board, is_for_est_Qval_analysis: bool = True) -> dict:
        """ this function will return a move score based on the analysis results from stockfish 
            :param board, this is the chessboard
            :param is_for_est_Qval_analysis. False means... I don't even remember what is happening in this method anymore.
            :return a dictionary with analysis results.
        """
        analysis_result = self.engine.analyse(board, self.search_limit, multipv = self.num_moves_to_return)
        
        # normalize by looking at it from White's perspective
        score = analysis_result[0]['score'].white()

        mate_score = score.mate()
        centipawn_score = score.score()

        if is_for_est_Qval_analysis:
            anticipated_next_move = analysis_result[0]['pv'][0] # this would be the anticipated response from opposing agent
            return {'mate_score': mate_score, 'centipawn_score': centipawn_score, 'anticipated_next_move': anticipated_next_move}
        else:
            return {'mate_score': mate_score, 'centipawn_score': centipawn_score} # we don't need the anticipated next move
    ### end of analyze_board_state
 
    def get_reward(self, chess_move_str: str) -> int:                                     
        """
            returns the number of points for a special chess action
            :param chess_move, string representing selected chess move
            :return reward based on type of move as an integer
        """
        total_reward = 0
        if re.search(r'N.', chess_move_str): # encourage development of pieces
            total_reward += self.settings.piece_dev_pts
        if re.search(r'R.', chess_move_str):
            total_reward += self.settings.piece_dev_pts
        if re.search(r'B.', chess_move_str):
            total_reward += self.settings.piece_dev_pts
        if re.search(r'Q.', chess_move_str):
            total_reward += self.settings.piece_dev_pts
        if re.search(r'x', chess_move_str):    # capture
            total_reward += self.settings.capture_pts
        if re.search(r'=Q', chess_move_str):    # a promotion to Q
            total_reward += self.settings.promotion_Queen_pts
        if re.search(r'#', chess_move_str): # checkmate
            total_reward += self.settings.checkmate_pts
        return total_reward
    ## end of get_reward

    def reset_environ(self) -> None:
        """ method is useful when training and also when finding
            the value of each move. the board needs to be cleared each time a
            game is played.
            return: None
        """
        self.environ.reset_environ()
    ### end of reset_environ