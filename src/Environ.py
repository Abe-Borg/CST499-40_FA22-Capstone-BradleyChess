import chess
import Settings
import pandas as pd
import copy
# import logging
# import log_config

# logger = logging.getLogger(__name__)

class Environ:
    """Manages the chessboard and determines its state.

        This class passes the state to the agent. It is the only thing that 
        should make changes to the chessboard.
    """
    def __init__(self, chess_data: pd.DataFrame):
        """Initializes the Environ object.

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
        self.chess_data: pd.DataFrame = chess_data 
        self.board: chess.Board = chess.Board()
        self.settings: Settings.Settings = Settings.Settings()

        # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
        # max_num_turns_per_player, so multiply by 2, then to make sure we get all moves, add 1.
        # column 1 in the chess data pd dataframe corresponds to 'W1'
        self.turn_list: list[str] = self.chess_data.columns[1 : self.settings.max_num_turns_per_player * 2 + 1].tolist()
        self.turn_index: int = 0
    ### end of constructor

    # @log_config.log_execution_time_every_N()
    def get_curr_state(self) -> dict[str, str, list[str]]:
        """Returns a dictionary that describes the current state of the chessboard and the current turn.

        Returns:
            dict[str, str, list[str]]: A dictionary that defines the current state that an agent will act on.

        """
        return {'turn_index': self.turn_index, 'curr_turn': self.get_curr_turn(), 'legal_moves': self.get_legal_moves()}
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
        # in this case, subtract 1, not add 1 (like we did in the constructor)
        max_turn_index: int = self.settings.max_num_turns_per_player * 2 - 1
        
        if self.turn_index < max_turn_index:
            self.turn_index += 1
        else:
            raise IndexError(f"Maximum turn index ({max_turn_index}) reached!")
    ### end of update_curr_state
    
    # @log_config.log_execution_time_every_N()
    def get_curr_turn(self) -> str:                        
        """Returns the string of the current turn.

        Returns:
            str: A string that corresponds to the current turn, such as 'W2' for index 2.

        Raises:
            IndexError: If the turn index is out of range.

        """
        try: 
            return self.turn_list[self.turn_index]
        except IndexError as e:
            print(f'An error occurred: {e}')
            # logger.error(f'list index out of range, turn index is {self.turn_index}: {e}')
            return 'ERROR: list index out of range'
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
        try:
            self.board.push_san(chess_move_str)
            return True
        except ValueError as e:
            print(f'An error occurred: {e}')
            # logger.error(f'unable to load chessboard with {chess_move_str}: {e}')            
            return False
    ### end of load_chessboard    

    # @log_config.log_execution_time_every_N()
    def pop_chessboard(self) -> None:
        """Pops the most recent move applied to the chessboard.

        This method is used during agent training ONLY

        Raises:
            IndexError: If the move stack is empty.

        """
        try:
            self.board.pop()
        except IndexError as e:
            print(f'An error occurred: {e}')
            # logger.error(f'unable to pop last move as the move stack is empty: {e}')
    ### end of pop_chessboard

    # @log_config.log_execution_time_every_N()
    def undo_move(self) -> None:
        """Undoes the most recent move applied to the chessboard.

        Raises:
            IndexError: If the move stack is empty.

        """
        try:
            self.board.pop()
            self.turn_index -= 1
        except IndexError as e:
            print(f'An error occurred: {e}')
            # logger.error(f'unable to pop last move as the move stack is empty: {e}')
    ### end of undo_move

    # @log_config.log_execution_time_every_N()
    def load_chessboard_for_Q_est(self, analysis_results: list[dict]) -> None:
        """Loads the chessboard using a Move.uci string during training.

        This method works in tandem with the Stockfish analysis during training.

        Args:
            analysis_results (list[dict]): A list of dictionaries containing the analysis results.
                Each dictionary has the form {'mate_score': <some score>, 'centipawn_score': <some score>,
                'anticipated_next_move': <move>}.

        Returns:
            None

        """
        # this is the anticipated chess move due to opponent's previous chess move. so if White plays Ne4, what is Black like to play?
        anticipated_chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        try:
            self.board.push(anticipated_chess_move)
        except ValueError as e:
            print(f'An error occurred: {e}')
            # logger.warning(f'unable to push move: {e}')
    ### end of load_chessboard_for_Q_est

    # @log_config.log_execution_time_every_N()
    def reset_environ(self) -> None:
        """Resets the Environ object.

        Call this method each time a game ends.

        """
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    # @log_config.log_execution_time_every_N()
    def get_legal_moves(self) -> list[str]:   
        """Returns a list of legal moves at the current turn.

        Args:
            None

        Returns:
            list[str]: A list of strings representing the legal moves at the current turn, given the board state.

        """
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        if legal_moves:
            return legal_moves
        else:
            # logger.warning(f'legal_moves list was empty')
            return ["legal moves list is empty"]
    ### end of get_legal_moves
    