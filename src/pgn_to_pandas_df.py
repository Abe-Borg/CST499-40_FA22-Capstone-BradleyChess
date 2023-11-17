import pandas as pd
import chess.pgn
import game_settings
import time

def pgn_to_dataframe(input_file_path):
    games_list = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            game_dict = {
                "PlyCount": game.end().board().ply(),
                "Result": game.headers["Result"]
            }

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                col_prefix = 'W' if i % 2 == 0 else 'B'
                move_num = i // 2 + 1
                col_name = f"{col_prefix}{move_num}"
                game_dict[col_name] = board.san(move)
                board.push(move)

            games_list.append(game_dict)

    chess_df = pd.DataFrame(games_list)
    chess_df.index = ['Game ' + str(i+1) for i in chess_df.index]
    chess_df = chess_df.fillna('')
    chess_df = chess_df[[c for c in chess_df if c not in ['Result']] + ['Result']]
    chess_df = chess_df[(chess_df['PlyCount'] > 0) & (chess_df['PlyCount'] <= game_settings.max_num_turns_per_player * 2)]

    return chess_df

if __name__ == '__main__':
    start_time = time.time()
    
    df = pgn_to_dataframe(game_settings.chess_pgn_file_path_1)


    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_1, compression='zip')
    df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_2, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_3, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_4, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_5, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_6, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_7, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_8, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_9, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_10, compression='zip')
    # df.to_pickle(game_settings.chess_pd_dataframe_file_path_part_11, compression='zip')

    
    
    end_time = time.time()
    print('PGN to DataFrame conversion is complete\n')
    print(f'It took: {end_time - start_time} seconds')
