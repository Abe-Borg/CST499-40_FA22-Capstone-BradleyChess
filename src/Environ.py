import chess
import game_settings
import pandas as pd
import copy
from typing import IO
# import logging
# import log_config
# logger = logging.getLogger(__name__)


class Environ:
    """Manages the chessboard and determines its state.

        This class passes the state to the agent. It is the only thing that 
        should make changes to the chessboard.
    """
    def __init__(self, chess_data: pd.DataFrame):
        """
        Args:
            chess_data (pd.DataFrame): A pandas DataFrame representing the chess games database.
            The format of the chess_data is extremely important. See this link for an explanation:
            https://github.com/abecsumb/DataScienceProject/blob/main/Chess_Data_Preparation.ipynb

        Attributes:
            chess_data (pd.DataFrame): A pandas DataFrame representing the chess games database.
            board (chess.Board): An object representing the chessboard.
            settings (Settings.Settings): An object representing the settings for the chess game.
            turn_list (list[str]): A list that is utilized to keep track of the current turn (W1, B1 ... W50, B50).
            turn_index (int): An integer that is incremented as each player makes a move.
        """
        self.debug_file = open(game_settings.environ_debug_filepath, 'a')
        self.errors_file = open(game_settings.environ_errors_filepath, 'a')

        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ constructor ==========\n\n')

        self.chess_data: pd.DataFrame = chess_data 
        self.board: chess.Board = chess.Board()

        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'chess_data:\n{self.chess_data.head()}\n')
            self.print_statements_debug.write(f'board:\n{self.board}\n\n')

        # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
        # max_num_turns_per_player, so multiply by 2, then to make sure we get all moves, add 1.
        # column 1 in the chess data pd dataframe corresponds to 'W1'
        self.turn_list: list[str] = self.chess_data.columns[1 : self.settings.max_num_turns_per_player * 2 + 1].tolist()
        self.turn_index: int = 0

        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'turn_list: {self.turn_list}\n')
            self.print_statements_debug.write(f'turn_index: {self.turn_index}\n')
            self.print_statements_debug.write(f'========== End of Environ constructor ==========\n\n')
    ### end of constructor

    def __del__(self):
        self.errorsfile.close()
        self.debug_file.close()
    ### end of Bradley destructor ###

    # @log_config.log_execution_time_every_N()
    def get_curr_state(self) -> dict[str, str, list[str]]:
        """Returns a dictionary that describes the current state of the chessboard and the current turn.
        Returns:
            dict[str, str, list[str]]: A dictionary that defines the current state that an agent will act on.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.get_curr_state ==========\n\n')
            self.print_statements_debug.write("going to get curr turn and get_legal_moves\n")
        
        state = {'turn_index': self.turn_index, 'curr_turn': self.get_curr_turn(), 'legal_moves': self.get_legal_moves()}

        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write("back from get_curr_turn and get_legal_moves\n")
            self.print_statements_debug.write(f'state: {state}\n')
            self.print_statements_debug.write(f'========== End of Environ.get_curr_state ==========\n\n')

        return state
    ### end of get_curr_state
    
    # @log_config.log_execution_time_every_N()
    def update_curr_state(self) -> None:
        """Updates the current state of the chessboard.
        The current state is the current turn and the legal moves at that turn. The state is updated each time a
        chess_move str is loaded to the chessboard. Each time a move is made, the current state needs to be updated.
        Only the index needs to be updated here. The board is updated by other methods.

        Raises:
            IndexError: If the maximum turn index is reached.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.update_curr_state ==========\n\n')

        # in this case, subtract 1, not add 1
        max_turn_index: int = self.settings.max_num_turns_per_player * 2 - 1

        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'max_turn_index: {max_turn_index}\n')
            self.print_statements_debug.write(f'currtent turn_index: {self.turn_index}\n')
        
        if self.turn_index < max_turn_index:
            self.turn_index += 1

            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'updated turn index is: {self.turn_index}\n')

        else:
            self.error_log.write(f'ERROR: max_turn_index reached: {self.turn_index} >= {max_turn_index}\n')
            raise IndexError(f"Maximum turn index ({max_turn_index}) reached!")

        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== End of Environ.update_curr_state ==========\n\n')
    ### end of update_curr_state
    
    # @log_config.log_execution_time_every_N()
    def get_curr_turn(self) -> str:                        
        """Returns the string of the current turn.
        Returns:
            str: A string that corresponds to the current turn, such as 'W2' for index 2.
        Raises:
            IndexError: If the turn index is out of range.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.get_curr_turn ==========\n\n')
            self.print_statements_debug.write(f'turn_index: {self.turn_index}\n')
        try: 
            curr_turn = self.turn_list[self.turn_index]

            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'curr_turn: {curr_turn}\n')
                self.print_statements_debug.write(f'========== End of Environ.get_curr_turn ==========\n\n')

            return curr_turn
        except IndexError as e:
            self.error_log.write(f'list index out of range, turn index is {self.turn_index}, error desc is: {e}')
            self.error_log.write(f'========== End of Environ.get_curr_turn ==========\n\n')
    ### end of get_curr_turn
    
    # @log_config.log_execution_time_every_N()
    def load_chessboard(self, chess_move_str: str) -> bool:
        """Loads a chess move on the chessboard.
        Call this method when you want to commit a chess move. The agent class chooses the move, but the Environ class
        must load up the chessboard with that move.

        Args:
            chess_move_str (str): A string representing the chess move, such as 'Nf3'.
        Returns:
            bool: A boolean value indicating whether the move was successfully loaded.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.load_chessboard ==========\n\n')
            self.print_statements_debug.write(f'chess_move_str: {chess_move_str}\n')

        try:
            self.board.push_san(chess_move_str)

            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'chess move loaded successfully\n')
                self.print_statements_debug.write(f'board: {self.board}\n')
                self.print_statements_debug.write(f'========== End of Environ.load_chessboard ==========\n\n\n')

            return True
        except ValueError as e:
            self.error_log.write(f'An error occurred: {e}, unable to load chessboard with {chess_move_str}')
            self.error_log.write(f'========== End of Environ.load_chessboard ==========\n\n\n')          
            return False
    ### end of load_chessboard    

    # @log_config.log_execution_time_every_N()
    def pop_chessboard(self) -> None:
        """Pops the most recent move applied to the chessboard.
        This method is used during agent training ONLY

        Raises:
            IndexError: If the move stack is empty.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.pop_chessboard ==========\n\n')

        try:
            self.board.pop()
            
            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'chess move popped successfully\n')
                self.print_statements_debug.write(f'board: {self.board}\n')
                self.print_statements_debug.write(f'========== End of Environ.pop_chessboard ==========\n\n\n')
        
        except IndexError as e:
            self.error_log.write(f'An error occurred: {e}, unable to pop chessboard')
            self.error_log.write(f'========== End of Environ.pop_chessboard ==========\n\n\n')
            raise IndexError(f"An error occurred: {e}, unable to pop chessboard'")
    ### end of pop_chessboard

    # @log_config.log_execution_time_every_N()
    def undo_move(self) -> None:
        """Undoes the most recent move applied to the chessboard.

        Raises:
            IndexError: If the move stack is empty.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.undo_move ==========\n\n')

        try:
            self.board.pop()
            self.turn_index -= 1

            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'chess move popped successfully\n')
                self.print_statements_debug.write(f'board: {self.board}\n')
                self.print_statements_debug.write(f'turn_index: {self.turn_index}\n')
                self.print_statements_debug.write(f'========== End of Environ.undo_move ==========\n\n\n')

        except IndexError as e:
            self.error_log.write(f'An error occurred: {e}, unable to undo move')
            self.error_log.write(f'turn index: {self.turn_index}\n')
            self.error_log.write(f'========== End of Environ.undo_move ==========\n\n\n')
    ### end of undo_move

    # @log_config.log_execution_time_every_N()
    def load_chessboard_for_Q_est(self, analysis_results: list[dict]) -> None:
        """Loads the chessboard using a Move.uci string during training.
        This method works in tandem with the Stockfish analysis during training.
        the method loads the anticipated next move from the analysis results.

        Args:
            analysis_results (list[dict]): A list of dictionaries containing the analysis results.
                Each dictionary has the form {'mate_score': <some score>, 'centipawn_score': <some score>,
                'anticipated_next_move': <move>}.
        Returns:
            None
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.load_chessboard_for_Q_est ==========\n\n')
            self.print_statements_debug.write(f'analysis_results: {analysis_results}\n')

        # this is the anticipated chess move due to opponent's previous chess move. so if White plays Ne4, what is Black like to play?
        anticipated_chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        
        try:
            self.board.push(anticipated_chess_move)

            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'chess move loaded successfully\n')
                self.print_statements_debug.write(f'board: {self.board}\n')
                self.print_statements_debug.write(f'========== End of Environ.load_chessboard_for_Q_est ==========\n\n\n')
        
        except ValueError as e:
            self.error_log.write(f'An error occurred: {e}, unable to load chessboard with {anticipated_chess_move}')
            self.error_log.write(f'========== End of Environ.load_chessboard_for_Q_est ==========\n\n\n')
    ### end of load_chessboard_for_Q_est

    # @log_config.log_execution_time_every_N()
    def reset_environ(self) -> None:
        """Resets the Environ object.
        Call this method each time a game ends.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.reset_environ ==========\n\n')
        
        self.board.reset()
        self.turn_index = 0
        
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'board: {self.board}\n')
            self.print_statements_debug.write(f'turn_index: {self.turn_index}\n')
            self.print_statements_debug.write(f'========== End of Environ.reset_environ ==========\n\n\n')
    ### end of reset_environ
    
    # @log_config.log_execution_time_every_N()
    def get_legal_moves(self) -> list[str]:   
        """Returns a list of legal moves at the current turn.
        Args:
            None
        Returns:
            list[str]: A list of strings representing the legal moves at the current turn, given the board state.
        """
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'========== Hello from Environ.get_legal_moves ==========\n\n')
            self.print_statements_debug.write(f'board:\n{self.board}\n\n')

        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        
        if game_settings.PRINT_DEBUG:
            self.print_statements_debug.write(f'legal_moves: {legal_moves}\n')
    
        if legal_moves:
            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'legal_moves is NOT empty\n')
                self.print_statements_debug.write(f'========== End of Environ.get_legal_moves ==========\n\n\n')
                
            return legal_moves
        else:
            if game_settings.PRINT_DEBUG:
                self.print_statements_debug.write(f'legal_moves is empty\n')
                self.print_statements_debug.write(f'========== End of Environ.get_legal_moves ==========\n\n\n')
                
            return ["legal moves list is empty"]
    ### end of get_legal_moves
    