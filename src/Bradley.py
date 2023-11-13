import Environ
import Agent
import game_settings
import chess
import pandas as pd
import re
# import logging
# import log_config
# logger = logging.getLogger(__name__)

class Bradley:
    """Acts as the single point of communication between the RL agent and the player.
    This class trains the agent and helps to manage the chessboard during play between the computer and the user.

    Args:
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data.
    Attributes:
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data.
        environ (Environ.Environ): An Environ object representing the chessboard environment.
        W_rl_agent (Agent.Agent): A white RL Agent object.
        B_rl_agent (Agent.Agent): A black RL Agent object.
        engine (chess.engine.SimpleEngine): A Stockfish engine used to analyze positions during training.
    """
    def __init__(self, chess_data: pd.DataFrame):
        self.q_est_log = open(game_settings.q_est_log_filepath, 'a')
        self.debug_file = open(game_settings.bradley_debug_filepath, 'a')
        self.errors_file = open(game_settings.bradley_errors_filepath, 'a')
        self.initial_training_results = open(game_settings.initial_training_results_filepath, 'a')
        self.additional_training_results = open(game_settings.additional_training_results_filepath, 'a')
        
        self.chess_data = chess_data
        self.environ = Environ.Environ()
        self.W_rl_agent = Agent.Agent('W', self.chess_data)
        self.B_rl_agent = Agent.Agent('B', self.chess_data)

        # stockfish is used to analyze positions during training this is how we estimate the q value 
        # at each position, and also for anticipated next position
        self.engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    ### end of Bradley constructor ###

    def __del__(self):
        self.debug_file.close()
        self.errors_file.close()
        self.initial_training_results.close()
        self.additional_training_results.close()
        self.q_est_log.close()
    ### end of Bradley destructor ###

    def receive_opp_move(self, chess_move: str) -> bool:                                                                                 
        """Receives the opponent's chess move and loads it onto the chessboard.
        Args:
            chess_move (str): A string representing the opponent's chess move, such as 'Nf3'.
        Returns:
            bool: A boolean value indicating whether the move was successfully loaded.
        """
        try:
            self.environ.load_chessboard(chess_move)
        except Exception as e:
            self.errors_file.write("hello from Bradley.receive_opp_move, an error occurred\n")
            self.errors_file.write(f'Error: {e}, failed to load chessboard with move: {chess_move}\n')
            return False

        try:
            self.environ.update_curr_state()
            return True
        except Exception as e:
            self.errors.file.write(f'hello from Bradley.receive_opp_move, an error occurrd\n')
            self.errors_file.write(f'Error: {e}, failed to update_curr_state\n') 
            return False
    ### end of receive_opp_move ###

    def rl_agent_selects_chess_move(self, rl_agent_color: str) -> str:
        """The Agent selects a chess move and loads it onto the chessboard.
        This method assumes that the agents have already been trained. This 
        method is used during the actual game play between the computer and the user. 
        It is not used during training.
    
        Args:
            rl_agent_color (str): A string indicating the color of the RL agent, either 'W' or 'B'.
        Returns:
            dict[str]: A dictionary containing the selected chess move string.
        """
        try:
            curr_state = self.environ.get_curr_state()
        except Exception as e:
            self.errors_file.write("hello from Bradley.rl_agent_selects_chess_move, an error occurred\n")
            self.errors_file.write(f'Error: {e}, failed to get_curr_state\n')
            raise Exception from e
        
        if rl_agent_color == 'W':    
            # W agent selects action
            chess_move: str= self.W_rl_agent.choose_action(curr_state)
        else:
            # B agent selects action
            chess_move = self.B_rl_agent.choose_action(curr_state)        

        try:
            self.environ.load_chessboard(chess_move) 
        except Exception as e:
            self.errors_file.write(f'Error {e}: failed to load chessboard with move: {chess_move}\n')
            raise Exception from e

        try:
            self.environ.update_curr_state()            
            return chess_move
        except Exception as e:
            self.errors_file.write(f'Error: {e}, failed to update_curr_state\n')
            raise Exception from e
    ### end of rl_agent_selects_chess_move

    def is_game_over(self) -> bool:
        """Determines whether the game is still ongoing. this is used only
        during phase 2 of training and also during human vs agent play.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.is_game_over ==========\n\n")

        if self.environ.board.is_game_over() or (self.environ.turn_index >= game_settings.max_turn_index):
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game over\n')
                self.debug_file.write(f'curr turn index is: {self.environ.turn_index}\n')
                self.debug_file.write("Bye from Bradley.is_game_over\n\n\n")
            return True
        else:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game is still ongoing\n')
                self.debug_file.write(f'curr turn index is: {self.environ.turn_index}\n')
                self.debug_file.write("========== Bye from Bradley.is_game_over ==========\n\n\n")
            return False
    ### end of is_game_over
        
    def get_game_outcome(self) -> str:
        """ Returns the outcome of the chess game.
        Returns:
            chess.Outcome or str: An instance of the `chess.Outcome` class with a `result()` 
            method that returns the outcome of the game
        """
        try:
            game_outcome = self.environ.board.outcome().result()
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game outcome is: {game_outcome}\n')
                self.debug_file.write("========== Bye from Bradley.get_game_outcome ===========\n\n\n")
            return game_outcome
        except AttributeError as e:
            return f'error at get_game_outcome: {e}'
    ### end of get_game_outcome
    
    def get_game_termination_reason(self) -> str:
        """returns a string that describes the reason for the game ending.
        """
        try:
            termination_reason = str(self.environ.board.outcome().termination)
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Termination reason is: {termination_reason}\n')
            return termination_reason
        except AttributeError as e:
            return 'error at get_game_termination_reason: {e}'
    ### end of get_game_termination_reason
    
    # @log_config.log_execution_time_every_N()
    def train_rl_agents(self) -> None:
        """Trains the RL agents using the SARSA algorithm and sets their `is_trained` flag to True.
        Two rl agents train each other by playing games from a database exactly as shown, and learning from that.
        """ 
        W_curr_Qval: int = game_settings.initial_q_val
        B_curr_Qval: int = game_settings.initial_q_val

        ### FOR EACH GAME IN THE TRAINING SET ###
        for game_num_str in self.chess_data.index:
            num_chess_moves_curr_training_game: int = self.chess_data.at[game_num_str, 'Num Moves']
            
            if game_settings.PRINT_TRAINING_RESULTS:
                self.initial_training_results.write(f'\nStart of {game_num_str} training\n\n')
            
            try:
                curr_state = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write(f'An error occurred at self.environ.get_curr_state: {e}\n')
                self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                raise Exception from e

            ### LOOP PLAYS THROUGH ONE GAME ###
            while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                ##################### WHITE'S TURN ####################
                # choose action a from state s, using policy
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'White agent will pick a move given the current state: {curr_state}\n')

                W_chess_move: self.W_rl_agent.choose_action(curr_state, game_num_str)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'White agent picked move: {W_chess_move}\n')
                    self.debug_file.write(f'on turn: {curr_state["turn_index"]}\n')

                ### ASSIGN POINTS TO Q TABLE FOR WHITE AGENT ###
                # on the first turn for white, this would assign to W1 col at chess_move row.
                # on W's second turn, this would be Q_next which is calculated on the first loop.                
                self.assign_points_to_Q_table(W_chess_move, curr_state['curr_turn'], W_curr_Qval, self.W_rl_agent.color)

                ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_plays_move(W_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    raise Exception from e

                W_reward = self.get_reward(W_chess_move)

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    raise Exception from e
                
                # find the estimated Q value for White, but first check if game ended
                if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game):
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(f'Game ended on White turn\n')
                        self.debug_file.write(f'curr_state is: {curr_state}\n')
                        self.debug_file.write(f'last chess move was: {W_chess_move}\n')
                        self.debug_file.write(f'board looks like this:\n{self.environ.board}\n\n')
                    
                    self.environ.reset_environ()
                    
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(f'envrion was reset\n')
                        self.debug_file.write(f'board looks like this:\n{self.environ.board}\n\n')
                        self.debug_file.write(f'curr state is: {curr_state}\n')
                    break # and go to next game

                else: # current game continues
                    try:
                        W_est_Qval: int = self.find_estimated_Q_value()
                        if game_settings.PRINT_Q_EST:
                            self.q_est_log.write(f'W_est_Qval: {W_est_Qval}\n')
                    except Exception as e:
                        self.errors_file.write(f'An error occurred while retrieving W_est_Qval: {e}\n')
                        self.errors_file.write(f"at White turn {curr_state['curr_turn']}, failed to find_estimated_Q_value\n")
                        self.errors_file.write(f'curr state is:{curr_state}\n')
                        self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                        raise Exception from e

                ##################### BLACK'S TURN ####################
                # choose action a from state s, using policy
                
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("\nIt's black's turn now:\n")
                    self.debug_file.write(f'curr state is: {curr_state}')

                B_chess_move: self.B_rl_agent.choose_action(curr_state, game_num_str)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f"Black chess move is: {B_chess_move}\n")
                
                # assign points to Q table
                self.assign_points_to_Q_table(B_chess_move, curr_state['curr_turn'], B_curr_Qval, self.B_rl_agent.color)

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_plays_move(B_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    raise Exception from e

                B_reward = self.get_reward(B_chess_move)

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at environ.get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e

                # find the estimated Q value for Black, but first check if game ended
                if self.environ.board.is_game_over():
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(f'Game ended on Blacks turn\n')
                        self.debug_file.write(f'curr_state is: {curr_state}\n')
                        self.debug_file.write(f'last chess move was: {B_chess_move}\n')
                        self.debug_file.write(f'board looks like this:\n{self.environ.board}\n\n')
                    
                    self.environ.reset_environ()
                    
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(f'envrion was reset\n')
                        self.debug_file.write(f'board looks like this:\n{self.environ.board}\n\n')
                        self.debug_file.write(f'curr state is: {curr_state}\n')
                    break # and go to next game

                else: # current game continues
                    try:
                        B_est_Qval: int = self.find_estimated_Q_value()
                        if game_settings.PRINT_Q_EST:
                            self.q_est_log.write(f'B_est_Qval: {B_est_Qval}\n') 
                    except Exception as e:
                        self.errors_file.write(f"at Black turn, failed to find_estimated_Qvalue because error: {e}\n")
                        self.errors_file.write(f'curr turn is:{curr_state["curr_turn"]}\n')
                        self.errors_file.write(f'turn index is: {curr_state["turn_index"]}\n')
                        self.errors_file.write(f'curr game is: {game_num_str}\n')
                        raise Exception from e

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from find_estimated_Q_value")
                    self.debug_file.write("going to SARSA calculations\n")
                    self.debug_file.write("going to self.find_next_Qval\n")

                # ***CRITICAL STEP***, this is the main part of the SARSA algorithm.
                W_next_Qval: int = self.find_next_Qval(W_curr_Qval, self.W_rl_agent.learn_rate, W_reward, self.W_rl_agent.discount_factor, W_est_Qval)
                B_next_Qval: int = self.find_next_Qval(B_curr_Qval, self.B_rl_agent.learn_rate, B_reward, self.B_rl_agent.discount_factor, B_est_Qval)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("SARSA calc was successful\n")
                    self.debug_file.write(f'W next Q val is: {W_next_Qval}\n')
                    self.debug_file.write(f'B next Q val is: {B_next_Qval}\n')
            
                # on the next turn, this Q value will be added to the Q table. so if this is the end of the first round,
                # next round it will be W2 and then we assign the q value at W2 col
                W_curr_Qval = W_next_Qval
                B_curr_Qval = B_next_Qval

                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n")
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    raise Exception from e
            ### END OF CURRENT GAME LOOP ###

            # this curr game is done, reset environ to prepare for the next game
            if game_settings.PRINT_TRAINING_RESULTS:
                self.initial_training_results.write(f'{game_num_str} is over.\n')
                self.initial_training_results.write(f'\nThe Chessboard looks like this:\n')
                self.initial_training_results.write(f'\n {self.environ.board}\n\n')
                self.initial_training_results.write(f'Game result is: {self.get_game_outcome()}\n')    
                self.initial_training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')

            self.environ.reset_environ()
        # end of training, all games in database have been processed
        
        # training is complete
        self.W_rl_agent.is_trained = True
        self.B_rl_agent.is_trained = True
    ### end of train_rl_agents

    # @log_config.log_execution_time_every_N()
    def continue_training_rl_agents(self, num_games_to_play: int) -> None:
        """ continues to train the agent, this time the agents make their own decisions instead 
            of playing through the database.
        """ 
    ### end of continue_training_rl_agents
    
    def assign_points_to_Q_table(self, chess_move: str, curr_turn: str, curr_Qval: int, rl_agent_color: str) -> None:
        """ Assigns points to the Q table for the given chess move, current turn, current Q value, and RL agent color.
        Args:
            chess_move (str): The chess move to assign points to in the Q table.
            curr_turn (str): The current turn of the game.
            curr_Qval (int): The current Q value for the given chess move.
            rl_agent_color (str): The color of the RL agent making the move.
        """
        if rl_agent_color == 'W':
            try:
                self.W_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
            except KeyError as e: 
                # chess move is not represented in the Q table, update Q table and try again.
                if game_settings.PRINT_DEBUG:
                    self.errors_file.write(f'caught exception: {e} at assign_points_to_Q_table\n')
                    self.debug_file.write(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')

                self.W_rl_agent.update_Q_table([chess_move])
                self.W_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
        else:
            try:
                self.B_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'Black agent changed Q table points for move: {chess_move}\n')
            except KeyError as e: 
                # chess move is not represented in the Q table, update Q table and try again. 
                if game_settings.PRINT_DEBUG:
                    self.errors_file.write(f'caught exception: {e} at assign_points_to_Q_table\n')
                    self.debug_file.write(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')

                self.B_rl_agent.update_Q_table([chess_move])
                self.B_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
    # enf of assign_points_to_Q_table

    def rl_agent_plays_move(self, chess_move: str, curr_game) -> None:
        """ This method is used during training and is responsible for:
                1. Loading the chessboard with the given move.
                2. Updating the current state of the environment.
        Args:
            chess_move (str): A string representing the chess move in standard algebraic notation.
        """
        try:
            self.environ.load_chessboard(chess_move, curr_game)
        except Exception as e:
            self.errors_file.write(f'@ Bradley.rl_agent_plays_move. An error occurred at {curr_game}: {e}\n')
            self.errors_file.write(f"failed to load_chessboard with move {chess_move}\n")
            raise Exception from e

        try:
            self.environ.update_curr_state()
        except Exception as e:
            self.errors_file.write(f'@ Bradley.rl_agent_plays_move. update_curr_state() failed to increment turn_index, Caught exception: {e}\n')
            self.errors_file.write(f'current state is: {curr_state}\n')
    # end of rl_agent_plays_move

    def find_estimated_Q_value(self) -> int:
        """ Estimates the Q-value for the RL agent's next action without actually playing the move.
        This method simulates the agent's next action and the anticipated response from the opposing agent 
        to estimate the Q-value.
        The method:
        1. Observes the next state of the chessboard after the agent's move.
        2. Analyzes the current state of the board to predict the opposing agent's response.
        3. Loads the board with the anticipated move of the opposing agent.
        4. Estimates the Q-value based on the anticipated state of the board.
    
        The estimation of the Q-value is derived from analyzing the board state with the help of a chess engine 
        (like Stockfish). If there's no impending checkmate, the estimated Q-value is the centipawn score of 
        the board state. Otherwise, it's computed based on the impending checkmate turns multiplied by a predefined 
        mate score reward.
        After estimating the Q-value, the method reverts the board state to its original state before the simulation.
        
        Returns:
            int: The estimated Q-value for the agent's next action.
        """
        # RL agent just played a move. the board has changed, if stockfish analyzes the board, 
        # it will give points for the agent, based on the agent's latest move.
        # We also need the points for the ANTICIPATED next state, 
        # given the ACTICIPATED next action. In this case, the anticipated response from opposing agent.

        try:
            analysis_results = self.analyze_board_state(self.environ.board)
        except Exception as e:
            self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write(f'failed to analyze_board_state\n')
            raise Exception from e
        
        # load up the chess board with opponent's anticipated chess move 
        try:
            self.environ.load_chessboard_for_Q_est(analysis_results)
        except Exception as e:
            self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write(f'failed to load_chessboard_for_Q_est\n')
            raise Exception from e
        
        # check if the game would be over with the anticipated next move
        if self.environ.board.is_game_over():
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game would be over with the anticipated next move\n')
                self.debug_file.write(f'board looks like this:\n{self.environ.board}\n\n')
            try:
                self.environ.pop_chessboard()
            except Exception as e:
                self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
                self.errors_file.write(f'failed at self.environ.pop_chessboard\n')
                raise Exception from e
            return 1

        # this is the Q estimated value due to what the opposing agent is likely to play in response to our move.
        try:
            est_Qval_analysis = self.analyze_board_state(self.environ.board)
        except Exception as e:
            self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write(f'failed at self.analyze_board_state\n')
            raise Exception from e

        # get pts for est_Qval 
        if est_Qval_analysis['mate_score'] is None:
            est_Qval = est_Qval_analysis['centipawn_score']
        else: # there is an impending checkmate
            est_Qval = game_settings.CHESS_MOVE_VALUES['mate_score']

        # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
        # playing a move.
        try:
            self.environ.pop_chessboard()
        except Exception as e:
            self.errors_file.write(f'@ Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write("failed to pop_chessboard\n")
            raise Exception from e

        return est_Qval
    # end of find_estimated_Q_value

    def find_next_Qval(self, curr_Qval: int, learn_rate: float, reward: int, discount_factor: float, est_Qval: int) -> int:
        """
        Calculates the next Q-value
        Args:
            curr_Qval (int)
            learn_rate (float): The learning rate, a value between 0 and 1.
            reward (int): The reward obtained from the current action.
            discount_factor (float): The discount factor to consider future rewards, a value between 0 and 1.
            est_Qval (int): The estimated Q-value for the next state-action pair.
        Returns:
            int: The next Q-value.
        """
        next_Qval = int(curr_Qval + learn_rate * (reward + ((discount_factor * est_Qval) - curr_Qval)))
        return next_Qval
    # end of find_next_Qval
    
    def analyze_board_state(self, board: chess.Board) -> dict:
        """
        Analyzes the current state of the chessboard using the Stockfish engine.
        The analysis results include the mate score and centipawn score.
        Args:
            board (chess.Board): The current state of the chessboard to analyze.
        Returns:
            dict: Analysis results, including the mate score, centipawn score, and the anticipated next move. 
        """
        try: 
            analysis_result = self.engine.analyse(board, game_settings.search_limit, multipv=game_settings.num_moves_to_return)
        except Exception as e:
            self.errors_file.write(f'@ Bradley_analyze_board_state. An error occurred during analysis: {e}\n')
            self.errors_file.write(f"Chessboard is:\n{board}\n")
            raise Exception from e

        mate_score = None
        centipawn_score = None
        anticipated_next_move = None

        try:
            # Get score from analysis_result and normalize for player perspective
            pov_score = analysis_result[0]['score'].white() if board.turn == chess.WHITE else analysis_result[0]['score'].black()

            # Check if the score is a mate score and get the mate score, otherwise get the centipawn score
            if pov_score.is_mate():
                mate_score = pov_score.mate()
            else:
                centipawn_score = pov_score.score()
        except Exception as e:
            self.errors_file.write(f'An error occurred while extracting scores: {e}\n')
            raise Exception from e

        try:
            # Extract the anticipated next move from the analysis
            anticipated_next_move = analysis_result[0]['pv'][0]
        except Exception as e:
            self.errors_file.write(f'An error occurred while extracting the anticipated next move: {e}\n')
            raise Exception from e
        
        if PRINT_DEBUG:
            self.debug_file.write(f'analysis results are: {analysis_result}\n')
            self.debug_file.write(f'mate score is: {mate_score}\n')
            self.debug_file.write(f'centipawn score is: {centipawn_score}\n')
            self.debug_file.write(f'anticipated next move is: {anticipated_next_move}\n')

        return {
            'mate_score': mate_score,
            'centipawn_score': centipawn_score,
            'anticipated_next_move': anticipated_next_move
        }
    ### end of analyze_board_state
 
    def get_reward(self, chess_move: str) -> int:
        """Calculates the reward for a given chess move.
        Args:
            chess_move (str): A string representing the selected chess move.
        Returns:
            int: The reward based on the type of move as an integer.
        """
        total_reward = 0
        # Check for piece development (N, R, B, Q)
        if re.search(r'[NRBQ]', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['piece_development']
        # Check for capture
        if 'x' in chess_move:
            total_reward += game_settings.CHESS_MOVE_VALUES['capture']
        # Check for promotion (with additional reward for queen promotion)
        if '=' in chess_move:
            total_reward += game_settings.CHESS_MOVE_VALUES['promotion']
            if '=Q' in chess_move:
                total_reward += game_settings.CHESS_MOVE_VALUES['promotion_queen']
        return total_reward
    ## end of get_reward