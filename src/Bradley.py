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
        self.debug_file = open(game_settings.bradley_debug_filepath, 'a')
        self.errors_file = open(game_settings.bradley_errors_filepath, 'a')
        self.initial_training_results = open(game_settings.initial_training_results_filepath, 'a')
        self.additional_training_results = open(game_settings.additional_training_results_filepath, 'a')

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley Constructor ==========\n")

        self.chess_data = chess_data
        self.environ = Environ.Environ(self.chess_data)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("Initializing White & Black RL Agents\n")

        self.W_rl_agent = Agent.Agent('W', self.chess_data)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("White agent initialized\n")

        self.B_rl_agent = Agent.Agent('B', self.chess_data)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("Black agent initialized\n")

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'White agent learn rate is: {self.W_rl_agent.learn_rate}\n')
            self.debug_file.write(f'White agent discount factor is: {self.W_rl_agent.discount_factor}\n') 
            self.debug_file.write(f'Black agent learn rate is: {self.B_rl_agent.learn_rate}\n')
            self.debug_file.write(f'Black agent discount factor is: {self.B_rl_agent.discount_factor}\n')

        # stockfish is used to analyze positions during training
        # this is how we estimate the q value at each position, 
        # and also for anticipated next position
        self.engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"========== End of Bradley Constructor ==========\n\n\n")
    ### end of Bradley constructor ###

    def __del__(self):
        self.debug_file.close()
        self.errors_file.close()
        self.initial_training_results.close()
        self.additional_training_results.close()
    ### end of Bradley destructor ###

    def set_agent_learn_rate(self, rl_agent_color: str, learn_rate: float) -> None:
        """Sets the learn rate for the RL agent.
            pre: 0 < learn_rate < 1 & rl_agent_color == 'W' or rl_agent_color == 'B'
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.set_agent_learn_rate ==========\n")
            self.debug_file.write(f'Agent color is: {rl_agent_color}\n')
            self.debug_file.write(f'Learn rate is: {learn_rate}\n')
        
        if rl_agent_color == 'W':
            self.W_rl_agent.learn_rate = learn_rate
        else:
            self.B_rl_agent.learn_rate = learn_rate

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"========== End of Bradley.set_agent_learn_rate ==========\n\n\n")
    # end of set_agent_learn_rate

    def set_agent_discount_factor(self, rl_agent_color: str, discount_factor: float) -> None:
        """Sets the discount factor for the RL agent.
            pre: 0 < discount_factor < 1 & rl_agent_color == 'W' or rl_agent_color == 'B'
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.set_agent_discount_factor ==========\n")
            self.debug_file.write(f'Agent color is: {rl_agent_color}\n')
            self.debug_file.write(f'Discount factor is: {discount_factor}\n')
        
        if rl_agent_color == 'W':
            self.W_rl_agent.discount_factor = discount_factor
        else:
            self.B_rl_agent.discount_factor = discount_factor
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"========== End of Bradley.set_agent_discount_factor ==========\n\n\n")

    # @log_config.log_execution_time_every_N()
    def recv_opp_move(self, chess_move: str) -> bool:                                                                                 
        """Receives the opponent's chess move and loads it onto the chessboard.
        Args:
            chess_move (str): A string representing the opponent's chess move, such as 'Nf3'.
        Returns:
            bool: A boolean value indicating whether the move was successfully loaded.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.recv_opp_move ==========\n\n")
            self.debug_file.write("going to environ.load_chessboard\n")

        # load_chessboard returns False if failure to add move to board,
        try:
            self.environ.load_chessboard(chess_move)
            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley recv_opp_move, arrived from environ.load_chessboard\n")
                self.debug_file.write("going to environ.update_curr_state\n")

            # loading the chessboard was a success, now just update the curr state
            try:
                self.environ.update_curr_state()

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back to Bradley recv_opp_move, arrived from environ.update_curr_state\n")
                    self.debug_file.write(f'Chessboard was successfully loaded with move: {chess_move}\n')
                    self.debug_file.write(f'Chessboard looks like this:\n')
                    self.debug_file.write(f'\n{self.environ.board}\n\n')
                    self.debug_file.write(f'Current turn index is: {self.environ.turn_index}\n')
                    self.debug_file.write("Bye from Bradley.recv_opp_move\n\n\n")

                return True
            except Exception as e:
                self.errors.file.write(f'hello from Bradley.recv_opp_move, an error occurrd\n')
                self.errors_file.write(f'Error: {e}, failed to update_curr_state\n')
                self.errors_file.write("========== Bye from Bradley.recv_opp_move ==========\n\n\n")
                return False
        except Exception as e:
            self.errors_file.write("hello from Bradley.recv_opp_move, an error occurred\n")
            self.errors_file.write(f'Error: {e}, failed to load chessboard with move: {chess_move}\n')
            self.errors_file.write("========== Bye from Bradley.recv_opp_move ==========\n\n\n")
            return False
    ### end of recv_opp_move ###

    # @log_config.log_execution_time_every_N()
    def rl_agent_selects_chess_move(self, rl_agent_color: str) -> str:
        """The Agent selects a chess move and loads it onto the chessboard.
        This method assumes that the agents have already been trained.
        
        Args:
            rl_agent_color (str): A string indicating the color of the RL agent, either 'W' or 'B'.
        Returns:
            dict[str]: A dictionary containing the selected chess move string.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.rl_agent_selects_chess_move ==========\n\n")
            self.debug_file.write("going to Environ.get_curr_state\n")

        if rl_agent_color == 'W':
            if game_settings.PRINT_DEBUG:
                self.debug_file.write('White agent will attempt to get_curr_state and then pick a move\n')
            try:
                curr_state = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write("hello from Bradley.rl_agent_selects_chess_move, an error occurred\n")
                self.errors_file.write(f'Error: {e}, failed to get_curr_state\n')
                self.errors_file.write("========== Bye from Bradley.rl_agent_selects_chess_move ==========\n\n\n")
                raise Exception(f'Error: {e}, failed to choose_action\n')
            
            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley rl_agent_selects_chess_move, arrived from Environ.get_curr_state\n")
                self.debug_file.write("going to self.W_rl_agent.choose_action\n")
            
            chess_move: str= self.W_rl_agent.choose_action(curr_state)
        else:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write('Black agent will attempt to get_curr_state and then pick a move\n')
            
            try:
                curr_state = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write("hello from Bradley.rl_agent_selects_chess_move, an error occurred\n")
                self.errors_file.write(f'Error: {e}, failed to get_curr_state\n')
                self.errors_file.write("========== Bye from Bradley.rl_agent_selects_chess_move ==========\n\n\n")
                raise Exception(f'Error: {e}, failed to get_curr_state\n')
            
            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley rl_agent_selects_chess_move, arrived from Environ.get_curr_state\n")
                self.debug_file.write("going to self.B_rl_agent.choose_action\n") 
            chess_move = self.B_rl_agent.choose_action(curr_state)
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back to Bradley.rl_agent_selects_chess_move, arrived from Agent.choose_action\n")
            self.debug_file.write(f'Chess move is: {chess_move}\n')
            self.debug_file.write("we MAY be going to environ.load_chessboard\n")

        try:
            self.environ.load_chessboard(chess_move)
            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back from environ.load_chessboard\n")
                self.debug_file.write("going to environ.update_curr_state\n")
            
            try:
                self.environ.update_curr_state()
            
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from environ.update_curr_state\n")
                    self.debug_file.write(f'Chessboard was successfully loaded with move: {chess_move}\n')
                    self.debug_file.write(f'Chessboard looks like this:\n\n')
                    self.debug_file.write(f'\n {self.environ.board}\n\n')
                    self.debug_file.write(f'Current turn index is: {self.environ.turn_index}\n')
                    self.debug_file.write("Bye from Bradley.rl_agent_selects_chess_move\n\n\n")

                return chess_move
            except Exception as e:
                self.errors_file.write(f'Error: {e}, failed to update_curr_state\n')
                self.errors_file.write("========== Bye from Bradley.rl_agent_selects_chess_move ==========\n\n\n")
                raise Exception from e
            
        except Exception as e:
            self.errors_file.write(f'Error: failed to load chessboard with move: {chess_move}\n')
            self.errors_file.write("========== Bye from Bradley.rl_agent_selects_chess_move ==========\n\n\n")
            raise Exception(f'Error: failed to load chessboard with move: {chess_move}')
    ### end of rl_agent_selects_chess_move
    
    def get_fen_str(self) -> str:
        """Returns the FEN string representing the current board state.
        Args:
            None
        Returns:
            str: A string representing the current board state in FEN format, 
            such as 'rnbqkbnr/pppp1ppp/8/8/4p1P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3'.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.get_fen_str ==========\n")
            self.debug_file.write("going to environ.board.fen()\n")

        try:
            fen: str = self.environ.board.fen()

            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley get_fen_str, arrived from environ.board.fen()\n")
                self.debug_file.write(f'FEN string is: {fen}\n')
                self.debug_file.write("Bye from Bradley.get_fen_str\n\n\n")
            
            return fen
        except Exception as e:
            self.errors_file.write(f'An error occurred: {e}\n')
            self.errors_file.write("invalid board state or fen str was invalid\n")
            self.errors_file.write(f"chessboard looks like this:\n{self.environ.board}\n\n")
            self.errors_file.write("========== Bye from Bradley.get_fen_str ==========\n\n\n")
            raise Exception(f'An error occurred: {e}\n')
    ### end of get_gen_str ###

    def get_opp_agent_color(self, rl_agent_color: str) -> str:
        """Determines the color of the opposing RL agent.
        Args:
            rl_agent_color (str): A string indicating the color of the current RL agent, either 'W' or 'B'.
        Returns:
            str: A string indicating the color of the opposing RL agent, either 'W' or 'B'.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"========== Hello from Bradley.get_opp_agent_color ==========\n\n")

        if rl_agent_color == 'W':
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Opposing agent color is: B\n')
                self.debug_file.write("Bye from Bradley.get_opp_agent_color\n\n\n")
            return 'B'
        else:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Opposing agent color is: W\n')
                self.debug_file.write("Bye from Bradley.get_opp_agent_color\n\n\n")
            return 'W'
    ### end of get_opp_agent_color
            
    # @log_config.log_execution_time_every_N()        
    def get_curr_turn(self) -> str:
        """Returns the current turn as a string.
        Args:
            None
        Returns:
            str: A string representing the current turn. eg "W1"
        """
        try: 
            return self.environ.get_curr_turn()
        except Exception as e:
            self.errors_file.write(f'An error occurred: {e}, unable to get curr_turn\n')
            raise Exception(f'An error occurred: {e}, unable to get curr_turn\n')
    ### end of get_curr_turn

    # @log_config.log_execution_time_every_N()
    def is_game_on(self) -> bool:
        """Determines whether the game is still ongoing.
        The game can end if the Python chess board determines the game is over,
        or if the game is at `max_num_turns_per_player * 2 - 1` moves per player (minus 1 because the index starts at 0).

        Args:
            None
        Returns:
            bool: A boolean value indicating whether the game is still ongoing (`True`) or not (`False`).
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.is_game_on ==========\n\n")
            self.debug_file.write("going to self.environ.board.is_game_over()\n")

        if self.environ.board.is_game_over() or self.environ.turn_index >= game_settings.max_num_turns_per_player * 2 - 1:
            
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game over, is_game_on is: False\n')
                self.debug_file.write("Bye from Bradley.is_game_on\n\n\n")
            
            return False
        else:
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game is still ongoing, is_game_on is: True\n')
                self.debug_file.write("========== Bye from Bradley.is_game_on ==========\n\n\n")
            return True
    ### end of is_game_on

    # @log_config.log_execution_time_every_N()
    def get_legal_moves(self) -> list[str]:
        """Returns a list of legal moves for the current turn and state of the chessboard.
        Args:
            None
        Returns:
            list[str]: A list of strings representing the legal moves for the current turn and state of the chessboard.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.get_legal_moves ==========\n\n")
            self.debug_file.write("going to environ.get_legal_moves\n")

        legal_moves = self.environ.get_legal_moves()

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back from environ.get_legal_moves\n")
            self.debug_file.write(f'Legal moves are: {legal_moves}\n')
            self.debug_file.write("========== Bye from Bradley.get_legal_moves ===========\n\n\n")

        if len(legal_moves) == 0:
            return ['no legal moves']
        else:
            return legal_moves
    ### end of get_legal_moves
        
    # @log_config.log_execution_time_every_N()
    def get_game_outcome(self) -> str:
        """ Returns the outcome of the chess game.
        Call this method to get the outcome of the chess game, either '1-0', '0-1', '1/2-1/2', 
        or 'False' if the outcome is not available.

        Args:
            None
        Returns:
            chess.Outcome or str: An instance of the `chess.Outcome` class with a `result()` 
            method that returns the outcome of the game, or a string indicating that the outcome is not available.
        Raises:
        AttributeError: If the outcome is not available due to an invalid board state.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.get_game_outcome ==========\n\n")
            self.debug_file.write("going to self.environ.board.outcome().result()\n")

        try:
            game_outcome = self.environ.board.outcome().result()

            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back from self.environ.board.outcome().result()\n")
                self.debug_file.write(f'Game outcome is: {game_outcome}\n')
                self.debug_file.write("========== Bye from Bradley.get_game_outcome ===========\n\n\n")

            return game_outcome
        except AttributeError as e:
            self.errors_file.write(f'An error occurred: {e}\n')
            self.errors_file.write("outcome not available, most likely game ended because turn_index was too high or player resigned\n")
            self.errors_file.write("========== Bye from Bradley.get_game_outcome ===========\n\n\n")
            return 'game outcome unavailable, game ended because turn_index was too high or player resigned'
    ### end of get_game_outcome
    
    # @log_config.log_execution_time_every_N()
    def get_game_termination_reason(self) -> str:
        """Determines why the game ended.
        Args:
            None
        Returns:
            str: A single string that describes the reason for the game ending.
        Raises:
            AttributeError: If the outcome is not available due to an invalid board state.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.get_game_termination_reason ==========\n\n")
            self.debug_file.write("going to self.environ.board.outcome().termination\n")

        try:
            termination_reason = str(self.environ.board.outcome().termination)

            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley get_game_termination_reason, arrived from self.environ.board.outcome().termination\n")
                self.debug_file.write(f'Termination reason is: {termination_reason}\n')
                self.debug_file.write("========== Bye from Bradley.get_game_termination_reason ===========\n\n\n")

            return termination_reason
        except AttributeError as e:
            self.errors_file.write(f'An error occurred: {e}\n')
            self.errors_file.write("termination reason not available, most likely game ended because turn_index was too high or player resigned")
            self.errors_file.write("========== Bye from Bradley.get_game_termination_reason ===========\n\n\n")
            return 'game termination reason unavailable, game ended because turn_index was too high or player resigned'
    ### end of get_game_termination_reason
    
    def get_chessboard(self) -> chess.Board:
        """Returns the current state of the chessboard.
        Args:
            None
        Returns:
            chess.Board: A `chess.Board` object representing the current state of the chessboard.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.get_chessboard ==========\n")
            self.debug_file.write("going to self.environ.board\n")
        return self.environ.board
    ### end of get_chess_board

    # @log_config.log_execution_time_every_N()
    def train_rl_agents(self) -> None:
        """Trains the RL agents using the SARSA algorithm and sets their `is_trained` flag to True.
        Two rl agents train each other by playing games from a database exactly as shown, and learning from that.
        """ 
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.train_rl_agents ==========\n")

        W_curr_Qval: int = game_settings.initial_q_val
        B_curr_Qval: int = game_settings.initial_q_val

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'White agent initial Q value is: {W_curr_Qval}\n')
            self.debug_file.write(f'Black agent initial Q value is: {B_curr_Qval}\n')
            self.debug_file.write("entering main for loop for training for all games\n")

        # for each game in the training data set.
        for game_num_str in self.chess_data.index:
            num_chess_moves_curr_training_game: int = self.chess_data.at[game_num_str, 'Num Moves']
            
            if game_settings.PRINT_TRAINING_RESULTS:
                self.initial_training_results.write(f'\nStart of {game_num_str} training\n\n')
                self.initial_training_results.write(f'Number of chess moves in this game is: {num_chess_moves_curr_training_game}\n')

            # initialize environment to provide a state, s
            # setting things up for the first game
            if game_settings.PRINT_DEBUG:
                self.debug_file.write("entering self.environ.get_curr_state()\n")

            try:
                curr_state: dict[str, str, list[str]] = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write(f'An error occurred: {e}\n')
                self.errors_file.write("failed to get_curr_state\n")
                self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                raise Exception from e

            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back from self.environ.get_curr_state()\n")
                self.debug_file.write(f'Current state is: {curr_state}\n')
                self.debug_file.write("now entering loop for ONE game\n")

            # loop plays through one game in the database, exactly as shown.
            while curr_state['turn_index'] < num_chess_moves_curr_training_game:
                ##################### WHITE'S TURN ####################
                # choose action a from state s, using policy
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'White agent will pick a move given the current state: {curr_state}\n')
                    self.debug_file.write("going to self.rl_agent_PICKS_move\n")

                W_chess_move: str = self.rl_agent_PICKS_move(curr_state, self.W_rl_agent.color, game_num_str)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.rl_agent_PICKS_move\n")
                    self.debug_file.write(f'White agent picked move: {W_chess_move}\n')
                    self.debug_file.write(f'on turn: {curr_state["turn_index"]}\n')

                # assign points to Q table
                # on the first turn for white, this would assign to W1 col at chess_move row.
                # on W's second turn, this would be Q_next which is calculated on the first loop.                
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'White agent will assign points to its Q table for move: {W_chess_move}\n')
                    self.debug_file.write("going to self.assign_points_to_Q_table\n")

                self.assign_points_to_Q_table(W_chess_move, curr_state['curr_turn'], W_curr_Qval, self.W_rl_agent.color)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.assign_points_to_Q_table\n")
                    self.debug_file.write(f'White agent assigned points to Q table for move: {W_chess_move}\n')
                    self.debug_file.write("now going to rl_agent_PLAYS_move\n")
                
                # Next, agent will play the selected move and get a reward for that move #####
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_PLAYS_move(W_chess_move)
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to rl_agent_PLAYS_move\n")
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e

                W_reward = self.get_reward(W_chess_move)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from rl_agent_PLAYS_move\n")
                    self.debug_file.write(f'White agent played move: {W_chess_move}\n')
                    self.debug_file.write(f'White agent got reward: {W_reward}\n')
                    self.debug_file.write("going to self.environ.get_curr_state\n")
                
                # the state changes each time a move is made, so get curr state again.
                try:
                    curr_state: dict[str, str, list[str]] = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n")
                    self.errors_file.write(f"chessboard looks like this:\n{self.environ.board}\n\n")
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e
                
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.environ.get_curr_state\n")
                    self.debug_file.write(f'Current state is: {curr_state}\n')

                # find the estimated Q value, but first check if game ended
                if curr_state['turn_index'] >= num_chess_moves_curr_training_game:
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(f'Game ended on White turn\n')
                        self.debug_file.write(f'curr_state["turn_index"] is: {curr_state["turn_index"]}\n')
                        self.debug_file.write(f'num_chess_moves_curr_training_game is: {num_chess_moves_curr_training_game}\n')
                    break # and go to next game
                else:
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write("going to self.find_estimated_Q_value\n")

                    try:
                        W_est_Qval: int = self.find_estimated_Q_value()
                    except Exception as e:
                        self.errors_file.write(f'An error occurred: {e}\n')
                        self.errors_file.write("failed to find_estimated_Q_value\n")
                        self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                        raise Exception from e

                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write("and we're back from self.find_estimated_Q_value\n")
                        self.debug_file.write(f'Estimated Q value for White is: {W_est_Qval}\n')

                ##################### BLACK'S TURN ####################
                # choose action a from state s, using policy
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("\nIt's black's turn now:\n")
                    self.debug_file.write("going to rl.agent_PICKS_MOVE_training_mode\n")

                B_chess_move: str = self.rl_agent_PICKS_move(curr_state, self.B_rl_agent.color, game_num_str)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back to Bradley.train_agents, arrived from rl.agent_PICKS_move_training_mode\n")
                    self.debug_file.write(f"Black chess move is: {B_chess_move}\n")
                
                # assign points to Q table
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("going to assign_points_to_Q_table\n")

                self.assign_points_to_Q_table(B_chess_move, curr_state['curr_turn'], B_curr_Qval, self.B_rl_agent.color)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back to Bradley.train_agents, arrived from assign_points_to_Q_table\n")
                    self.debug_file.write(f'Black agent assigned points to Q table for move: {B_chess_move}\n')

                ##### BLACK AGENT PLAYS SELECTED MOVE and GET REWARD FOR THAT MOVE #####
                # take action a, observe r, s', and load chessboard
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("going to rl_agent_PLAYS_move\n")

                try:
                    self.rl_agent_PLAYS_move(B_chess_move)
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to rl_agent_PLAYS_move\n")
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e

                B_reward = self.get_reward(B_chess_move)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from rl_agent_PLAYS_move\n")
                    self.debug_file.write(f'B reward is: {B_reward} for playing move: {B_chess_move}\n')
                    self.debug_file.write("going to self.environ.get_curr_state\n")

                # the state changes each time a move is made, so get curr state again.                
                try:
                    curr_state: dict[str, str, list[str]] = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n")
                    self.errors_file.write(f"chessboard looks like this:\n{self.environ.board}\n\n")
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.environ.get_curr_state\n")
                    self.debug_file.write(f'Current state is: {curr_state}\n')

                # check if index has reached max value, this will only happen for Black at Black's final max turn, 
                # White won't ever have this problem.
                if self.environ.turn_index >= game_settings.max_num_turns_per_player * 2:
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(
                            f"game is over, max number of turns has been reached: "
                            f"{self.environ.turn_index} >= {game_settings.max_num_turns_per_player}\n")
                    
                    break

                # find the estimated Q value
                if curr_state['turn_index'] >= num_chess_moves_curr_training_game:
                    
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write(f'Game ended on Blacks turn\n')
                        self.debug_file.write(f'curr_state["turn_index"] is: {curr_state["turn_index"]}\n')
                        self.debug_file.write(f'num_chess_moves_curr_training_game is: {num_chess_moves_curr_training_game}\n')
                    
                    break
                else:
                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write("going to find_estimated_Q_value")

                    try:
                        B_est_Qval: int = self.find_estimated_Q_value()
                    except Exception as e:
                        self.errors_file.write(f"failed to find_estimated_Qvalue because error: {e}")
                        raise Exception from e

                    if game_settings.PRINT_DEBUG:
                        self.debug_file.write("and we're back from find_estimated_Q_value")
                        self.debug_file.write(f'estimated b q val is: {B_est_Qval}\n')
                        self.debug_file.write("going to SARSA calculations\n")
                        self.debug_file.write("going to self.find_next_Qval\n")

                # ***CRITICAL STEP***, this is the main part of the SARSA algorithm.
                W_next_Qval: int = self.find_next_Qval(W_curr_Qval, self.W_rl_agent.learn_rate, W_reward, self.W_rl_agent.discount_factor, W_est_Qval)
                B_next_Qval: int = self.find_next_Qval(B_curr_Qval, self.B_rl_agent.learn_rate, B_reward, self.B_rl_agent.discount_factor, B_est_Qval)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from SARSA calculations methods, find_next_Qval\n")
                    self.debug_file.write("SARSA calc was successful\n")
                    self.debug_file.write(f'W next Q val is: {W_next_Qval}\n')
                    self.debug_file.write(f'B next Q val is: {B_next_Qval}\n')
                    self.debug_file.write("going to self.environ.get_curr_state\n")
            
                # on the next turn, this Q value will be added to the Q table. so if this is the end of the first round,
                # next round it will be W2 and then we assign the q value at W2 col
                W_curr_Qval = W_next_Qval
                B_curr_Qval = B_next_Qval

                # this is the next state, s'  the next action, a' is handled at the beginning of the while loop
                try:
                    curr_state: dict[str, str, list[str]] = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n")
                    self.errors_file.write(f"chessboard looks like this:\n{self.environ.board}\n\n")
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e
                
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.environ.get_curr_state\n")
                    self.debug_file.write(f'Current state is: {curr_state}\n')
            # end curr game while loop

            # this curr game is done, reset environ to prepare for the next game
            if game_settings.PRINT_TRAINING_RESULTS:
                self.initial_training_results.write(f'Game {game_num_str} is over.\n')
                self.initial_training_results.write(f'\nThe Chessboard looks like this:\n')
                self.initial_training_results.write(f'\n {self.environ.board}\n\n')
                self.initial_training_results.write(f'Game result is: {self.get_game_outcome()}\n')    
                self.initial_training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')

            
            if game_settings.PRINT_DEBUG:
                self.debug_file.write(f'Game {game_num_str} is over.\n')
                self.debug_file.write("going to self.reset_environ\n")
            
            self.environ.reset_environ()

            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back from self.reset_environ\n")
                self.debug_file.write("going to next game in training session\n")
        # end of training, all games in database have been processed
        
        # training is complete
        self.W_rl_agent.is_trained = True
        self.B_rl_agent.is_trained = True
        self.environ.reset_environ()

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'White agent is trained: {self.W_rl_agent.is_trained}\n')
            self.debug_file.write(f'Black agent is trained: {self.B_rl_agent.is_trained}\n')
            self.debug_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
    ### end of train_rl_agents

    # @log_config.log_execution_time_every_N()
    def continue_training_rl_agents(self, num_games_to_play: int) -> None:
        """ continues to train the agent, this time the agents make their own decisions instead 
            of playing through the database.
        """ 
        
    ### end of continue_training_rl_agents


    ########## TRAINING HELPER METHODS ####################
    
    # @log_config.log_execution_time_every_N()
    def rl_agent_PICKS_move(self, curr_state: dict[str, str, list[str]], rl_agent_color: str, game_num_str: str = 'Game 1') -> str:
        """ The RL agent picks a move to play during training mode
        Parameters:
            curr_state (dict[str, str, list[str]]): A dictionary containing the current state of the chessboard.
            rl_agent_color (str): A string representing the color of the RL agent ('W' for white, 'B' for black).
            game_num_str (str): A string representing the game number (default is 'Game 1').
        Returns:
            str: A string representing the chess move that the RL agent has chosen to play.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.rl_agent_PICKS_move ==========\n\n")
            self.debug_file.write(f'Current state is: {curr_state}\n')
            self.debug_file.write(f'RL agent color is: {rl_agent_color}\n')
            self.debug_file.write(f'Game number is: {game_num_str}\n')
            self.debug_file.write("going to Agent.choose_action\n")

        if rl_agent_color == 'W':
            curr_action: str = self.W_rl_agent.choose_action(curr_state, game_num_str)
            
            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley.rl_agent_PICKS_move, arrived from Agent.choose_action\n")
                self.debug_file.write(f'White agent picked move: {curr_action}\n')
        else:
            curr_action: str = self.B_rl_agent.choose_action(curr_state, game_num_str)

            if game_settings.PRINT_DEBUG:
                self.debug_file.write("and we're back to Bradley.rl_agent_PICKS_move, arrived from Agent.choose_action\n")
                self.debug_file.write(f'Black agent picked move: {curr_action}\n')
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write("========== Bye from Bradley.rl_agent_PICKS_move ===========\n\n\n")

        return curr_action
    # end of rl_agent_PICKS_move
    
    # @log_config.log_execution_time_every_N()
    def assign_points_to_Q_table(self, chess_move: str, curr_turn: str, curr_Qval: int, rl_agent_color: str) -> None:
        """ Assigns points to the Q table for the given chess move, current turn, current Q value, and RL agent color.
        Args:
            chess_move (str): The chess move to assign points to in the Q table.
            curr_turn (str): The current turn of the game.
            curr_Qval (int): The current Q value for the given chess move.
            rl_agent_color (str): The color of the RL agent making the move.
        Returns:
            None
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.assign_points_to_Q_table ==========\n\n")
            self.debug_file.write(f'Chess move is: {chess_move}\n')
            self.debug_file.write(f'Current turn is: {curr_turn}\n')
            self.debug_file.write(f'Current Q value is: {curr_Qval}\n')
            self.debug_file.write(f'RL agent color is: {rl_agent_color}\n')
            self.debug_file.write("and we're going to Agent.change_Q_table_pts\n")

        if rl_agent_color == 'W':
            try:
                self.W_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
                
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back to Bradley.assign_points_to_Q_table, arrived from rl_agent.change_Q_table_pts\n")
                    self.debug_file.write(f'White agent changed Q table points for move: {chess_move}\n')

            except KeyError as e: 
                # chess move is not represented in the Q table, update Q table and try again.
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'caught exception: {e}\n')
                    self.debug_file.write(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')
                    self.debug_file.write("going to Agent.update_Q_table\n")

                self.W_rl_agent.update_Q_table([chess_move])

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back to Bradley.assign_points_to_Q_table from self.W_rl_agent.update_Q_table\n")
                    self.debug_file.write("going to Agent.change_Q_table_pts\n")

                self.W_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back to Bradley.assign_points_to_Q_table, arrived from Agent.change_Q_table_pts\n")
                    self.debug_file.write(f'White agent changed Q table points for move: {chess_move}\n')
        else:
            try:
                self.B_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from rl_agent.change_Q_table_pts\n")
                    self.debug_file.write(f'White agent changed Q table points for move: {chess_move}\n')

            except KeyError as e: 
                # chess move is not represented in the Q table, update Q table and try again. 
                if game_settings.PRINT_DEBUG:
                    self.debug_file.write(f'caught exception: {e}\n')
                    self.debug_file.write(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')
                    self.debug_file.write("going to self.W_rl_agent.update_Q_table\n")

                self.B_rl_agent.update_Q_table([chess_move])

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.W_rl_agent.update_Q_table\n")
                    self.debug_file.write("going to self.W_rl_agent.change_Q_table_pts\n")

                self.B_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)

                if game_settings.PRINT_DEBUG:
                    self.debug_file.write("and we're back from self.W_rl_agent.change_Q_table_pts\n")
                    self.debug_file.write(f'White agent changed Q table points for move: {chess_move}\n')
    
        if game_settings.PRINT_DEBUG:
            self.debug_file.write("========== Bye from Bradley.assign_points_to_Q_table ===========\n\n\n")
    # enf of assign_points_to_Q_table

    # @log_config.log_execution_time_every_N()
    def rl_agent_PLAYS_move(self, chess_move: str) -> None:
        """
            This method is responsible for:
                1. Loading the chessboard with the given move.
                2. Updating the current state of the environment.
        Args:
            chess_move (str): A string representing the chess move in standard algebraic notation.
        Returns:
            None
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.rl_agent_PLAYS_move ==========\n\n")
            self.debug_file.write(f'Chess move is: {chess_move}\n')
            self.debug_file.write("going to self.environ.load_chessboard\n")

        try:
            self.environ.load_chessboard(chess_move)
        except Exception as e:
            self.errors_file.write(f'An error occurred: {e}\n')
            self.errors_file.write("failed to load_chessboard\n")
            self.errors_file.write(f"chessboard looks like this:\n{self.environ.board}\n\n")
            self.errors_file.write("========== Bye from Bradley.rl_agent_PLAYS_move ===========\n\n\n")
            raise Exception from e

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back to Bradley.rl_agent_PLAYS_move, arrived from self.environ.load_chessboard\n")
            self.debug_file.write("going to self.environ.update_curr_state\n")

        try:
            self.environ.update_curr_state()
        except Exception as e:
            self.errors_file.write(f'failed to update_curr_state, Caught exception: {e}\n')
            self.errors_file.write(f'Chess move is: {chess_move}\n')
            self.errors_file.write(f'Chessboard is: {self.environ.board}\n')
            self.errors_file.write(f'Chessboard stack is: {self.environ.board.chessboard_stack}\n')
            self.errors_file.write(f'Chessboard stack length is: {len(self.environ.board.chessboard_stack)}\n')
            self.errors_file.write(f'Chessboard stack top is: {self.environ.board.chessboard_stack[-1]}\n')
            self.errors_file.write(f'Chessboard stack top type is: {type(self.environ.board.chessboard_stack[-1])}\n')
            raise Exception from e

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back to Bradley.rl_agent_PLAYS_move arrived from self.environ.update_curr_state\n")
            self.debug_file.write("========== Bye from Bradley.rl_agent_PLAYS_move ===========\n\n\n")    
    # end of rl_agent_PLAYS_move

    # @log_config.log_execution_time_every_N()
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
        
        Args:
            None            
        Returns:
            int: The estimated Q-value for the agent's next action.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.find_estimated_Q_value ==========\n")
            self.debug_file.write("going to self.analyze_board_state\n")
       
        # RL agent just played a move. the board has changed, if stockfish analyzes the board, 
        # it will give points for the agent, based on the agent's latest move.
        # We also need the points for the ANTICIPATED next state, 
        # given the ACTICIPATED next action. In this case, the anticipated response from opposing agent.

        # the analysis returns an array of dicts. in our analysis, we only consider the first dict returned by 
        # the stockfish analysis. We also care about the opponents likely chess move in response to our own.                    
        analysis_results = self.analyze_board_state(self.environ.board)
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back from self.analyze_board_state\n")
            self.debug_file.write(f'Analysis results are: {analysis_results}\n')
            self.debug_file.write("going to self.environ.load_chessboard_for_Q_est\n")

        # load up the chess board with opponent's anticipated chess move 
        # anticipated next action is a str like, 'e6f2'              
        self.environ.load_chessboard_for_Q_est(analysis_results) 

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back from environ.load_chessboard_for_Q_est\n")
            self.debug_file.write("going to analyze_board_state")
    
        # this is the Q estimated value due to what the opposing agent is likely to play in response to our move.
        est_Qval_analysis = self.analyze_board_state(self.environ.board) 

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back from analyze_board_state")
    
        # get pts for est_Qval 
        if est_Qval_analysis['mate_score'] is None:
            est_Qval = est_Qval_analysis['centipawn_score']
        else: # there is an impending checkmate
            est_Qval = game_settings.CHESS_MOVE_VALUES['mate_score']

        # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
        # playing a move.
        if game_settings.PRINT_DEBUG:
            self.debug_file.write("going to self.environ.pop_chessboard\n")

        try:
            self.environ.pop_chessboard()
        except Exception as e:
            self.errors_file.write(f'An error occurred: {e}\n')
            self.errors_file.write("failed to pop_chessboard\n")
            self.errors_file.write(f"chessboard looks like this:\n{self.environ.board}\n\n")
            self.errors_file.write("========== Bye from Bradley.find_estimated_Q_value ===========\n\n\n")
            raise Exception from e

        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back from self.environ.pop_chessboard\n")
            self.debug_file.write(f'estimated q val is: {est_Qval}\n')
            self.debug_file.write("========== Bye from Bradley.find_estimated_Q_value ===========\n\n\n")

        return est_Qval
    # end of find_estimated_Q_value

    # @log_config.log_execution_time_every_N()
    def find_next_Qval(self, curr_Qval: int, learn_rate: float, reward: int, discount_factor: float, est_Qval: int) -> int:
        """
        Calculates the next Q-value based on the current Q-value, learning rate, reward, discount factor, and estimated Q-value.
        This method uses the Q-learning update formula to compute the next Q-value. 

        Args:
            curr_Qval (int)
            learn_rate (float): The learning rate, a value between 0 and 1.
            reward (int): The reward obtained from the current action.
            discount_factor (float): The discount factor to consider future rewards, a value between 0 and 1.
            est_Qval (int): The estimated Q-value for the next state-action pair.
        Returns:
            int: The next Q-value.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.find_next_Qval ==========\n")
            self.debug_file.write(f'Current Q value is: {curr_Qval}\n')
            self.debug_file.write(f'Learning rate is: {learn_rate}\n')
            self.debug_file.write(f'Reward is: {reward}\n')
            self.debug_file.write(f'Discount factor is: {discount_factor}\n')
            self.debug_file.write(f'Estimated Q value is: {est_Qval}\n')

        next_Qval = curr_Qval + learn_rate * (reward + ((discount_factor * est_Qval) - curr_Qval))

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'Next Q value is: {next_Qval}\n')
            self.debug_file.write("========== Bye from Bradley.find_next_Qval ===========\n\n\n")
        return int(next_Qval)
    # end of find_next_Qval
    
    ########## END OF TRAINING HELPER METHODS ####################


    # @log_config.log_execution_time_every_N()
    def analyze_board_state(self, board: chess.Board) -> dict:
        """Analyzes the current state of the chessboard using the Stockfish engine.
        This method returns a dictionary with the analysis results. The analysis results include the mate 
        score and centipawn score, which are normalized by looking at the board from White's perspective. 

        Args:
            board (chess.Board): The current state of the chessboard to analyze.
        Returns:
            dict: analysis results, including the mate score and centipawn score. 
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.analyze_board_state ==========\n")
            self.debug_file.write(f'Board is: {board}\n')

        # analysis_result is an InfoDict (see python-chess documentation)
        analysis_result = self.engine.analyse(board, game_settings.search_limit, multipv = game_settings.num_moves_to_return)
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'Analysis result is: {analysis_result}\n')

        # normalize by looking at it from White's perspective
        # score datatype is Cp (centipawn) or Mate
        score = analysis_result[0]['score'].white()
        mate_score = score.mate()
        centipawn_score = score.score()

        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'Normalized mate score is: {mate_score}\n')
            self.debug_file.write(f'Normalized centipawn score is: {centipawn_score}\n')

        anticipated_next_move = analysis_result[0]['pv'][0] # this would be the anticipated response from opposing agent
            
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'Anticipated next move is: {anticipated_next_move}\n')
            self.debug_file.write("========== Bye from Bradley.analyze_board_state ===========\n\n\n")
            
        return {'mate_score': mate_score, 'centipawn_score': centipawn_score, 'anticipated_next_move': anticipated_next_move}
    ### end of analyze_board_state
 
    # @log_config.log_execution_time_every_N()
    def get_reward(self, chess_move: str) -> int:                                     
        """Calculates the reward for a given chess move.
        Args:
            chess_move (str): A string representing the selected chess move.
        Returns:
            int: The reward based on the type of move as an integer.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.get_reward ==========\n")
            self.debug_file.write(f'Chess move is: {chess_move}\n')

        total_reward = 0
        if re.search(r'N.', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['piece_development']
        if re.search(r'R.', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['piece_development']
        if re.search(r'B.', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['piece_development']
        if re.search(r'Q.', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['piece_development']
        if re.search(r'x', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['capture']
        if re.search(r'=', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['promotion']
        if re.search(r'=Q', chess_move):
            total_reward += game_settings.CHESS_MOVE_VALUES['promotion_queen']
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f'Total reward is: {total_reward}\n')
            self.debug_file.write("========== Bye from Bradley.get_reward ===========\n\n\n")

        return total_reward
    ## end of get_reward

    # @log_config.log_execution_time_every_N()
    def reset_environ(self) -> None:
        """Resets the environment for a new game.
        """
        if game_settings.PRINT_DEBUG:
            self.debug_file.write(f"\n========== Hello from Bradley.reset_environ ==========\n")
            self.debug_file.write("going to self.environ.reset_environ\n")
        
        self.environ.reset_environ()
        
        if game_settings.PRINT_DEBUG:
            self.debug_file.write("and we're back to Bradley reset_environ from self.environ.reset_environ\n")
            self.debug_file.write("Environment has been reset\n")
            self.debug_file.write("========== Bye from Bradley.reset_environ ===========\n\n\n")
    ### end of reset_environ