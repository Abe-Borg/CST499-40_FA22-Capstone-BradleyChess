import os
import chess.pgn

def split_pgn_file(file_path, number_of_splits = 10):
    # Extract the base name of the file without the extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Count total number of games
    total_games = 0
    with open(file_path, 'r') as pgn:
        while chess.pgn.read_game(pgn):
            total_games += 1

    games_per_file = total_games // number_of_splits

    current_game = 0
    for i in range(number_of_splits):
        split_filename = f'{base_name}_Part_{i+1}.pgn'
        with open(split_filename, 'w') as split_file:
            with open(file_path, 'r') as pgn:
                for j in range(games_per_file):
                    if current_game < total_games:
                        game = chess.pgn.read_game(pgn)
                        if game is not None:
                            split_file.write(str(game) + "\n\n")
                            current_game += 1

# Use the function
split_pgn_file('Chess_Games_DB_Part_12.pgn')
