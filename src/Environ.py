import chess
import Settings
import pandas as pd
import copy
import logging
import log_config

logger = logging.getLogger(__name__)

class Environ:
    """ 
        This class manages the chessboard and determines what the state is.
        The class passes the state to the agent.
        This class is the only thing that should ake changes to the chessboard. 
    """
    def __init__(self, chess_data: pd.DataFrame):
        """
            :param chess_data, this is a pandas dataframe. It is the chess games database.
                   The format of the chess_data is extremely important.
                   see this link for an explanation: 
                   https://github.com/abecsumb/DataScienceProject/blob/main/Chess_Data_Preparation.ipynb
            
            board, This is an object. It comes from the python chess module
            turn_list, this is a list that is utilized to keep track of the current turn (W1, B1 ... W50, B50) 
            turn_index, increment this as each player makes a move. 
        """
        self.chess_data: pd.DataFrame = chess_data 
        self.board: chess.Board = chess.Board()
        self.settings: Settings.Settings = Settings.Settings()

        # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
        # num_turns_per_player, so multiply by 2, then to make sure we get all moves, add 1.
        # column 1 in the chess data pd dataframe corresponds to 'W1'
        self.turn_list: List[str] = self.chess_data.columns[1 : self.settings.num_turns_per_player * 2 + 1].tolist()
        self.turn_index: int = 0
    ### end of constructor

    def get_curr_state(self) -> dict[str, str, list[str]]:
        """ 
            returns the dictionary that describes the curr state of the board, and the curr turn
            :param none
            :return a dictionary that defines the current state that an agent will act on
        """
        return {'turn_index': self.turn_index, 'curr_turn': self.get_curr_turn(), 'legal_moves': self.get_legal_moves()}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:
        """ current state is the current turn and the legal moves at that turn 
            the state is updated each time a chess_move str is loaded to the chessboard.
            each time a move is made, the curr state needs to be updated
            only the index needs to be updated here. The board is updated by other methods.
        """
        max_turn_index: int = self.settings.num_turns_per_player * 2 - 1
        
        if self.turn_index < max_turn_index:
            self.turn_index += 1
        else:
            raise IndexError("Maximum turn index ({max_turn_index}) reached!")
    ### end of update_curr_state
        
    def get_curr_turn(self) -> str or bool:                        
        """ returns the string of the current turn, 'W2' for example
            which would correspond to index = 2
            :param none
            :return, a string that corresponds to current turn or false
        """
        try: 
            return self.turn_list[self.turn_index]
        except IndexError as e:
            logging.error(f'list index out of range, turn index is {self.turn_index}: {e}')
            return False
    ### end of get_curr_turn
    
    def load_chessboard(self, chess_move_str: str) -> bool:
        """ 
            method to play move on chessboard. call this method when you want to commit a chess move.
            the agent class chooses the move, but the environ class must load up the chessboard with that move
            :param chess_move as string like this, 'Nf3'
            :return bool for success or failure.
        """
        try:
            self.board.push_san(chess_move_str)
            return True
        except ValueError as e:
            logging.error(f'unable to load chessboard with {chess_move_str}: {e}')            
            return False
    ### end of load_chessboard    

    def pop_chessboard(self) -> None:
        """ pops the most recent move applied to the chessboard 
            this method is used during agent training
        """
        try:
            self.board.pop()
        except IndexError as e:
            logging.error(f'unable to pop last move as the move stack is empty: {e}')
    ### end of pop_chessboard

    def undo_move(self) -> None:
        """ this method is used during game mode, human vs rl agent """
        try:
            self.board.pop()
            self.turn_index -= 1
        except IndexError as e:
            logging.error(f'unable to pop last move as the move stack is empty: {e}')
    ### end of undo_move

    def load_chessboard_for_Q_est(self, analysis_results) -> bool:
        """ method should only be called during training. this will load the 
            chessboard using a Move.uci string. This method works in tandem
            with the Stockfish analysis during training
            :pre analysis of proposed board condition has already been done
            :post board is loaded with black's anticipated move
            :param analysis_results, I forgot what this is exactly a list or a dict? This is what happens 
             when you code late at night I guess
            it has this form, [{'mate_score': <some score>, 'centipawn_score': <some score>, 'anticipated_next_move': <move>}]
            :return bool for success or failure
        """
        chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        try:
            self.board.push(chess_move)
            return True
        except ValueError as e:
            logging.warning(f'unable to push move: {e}')
            return False
    ### end of load_chessboard_for_Q_est

    def reset_environ(self) -> None:
        """ resets environ, call each time a game ends.
            :param none
            :return none
        """
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    def get_legal_moves(self) -> list[str]:   
        """ 
            method will return a list of strings that represents the legal moves at that turn
            :param none
            :return list of strings that represents the legal moves at a turn, given the board state
        """
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        if legal_moves:
            return legal_moves
        else:
            logging.warning(f'legal_moves list was empty')

    ### end of get_legal_moves
    