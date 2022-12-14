import pandas as pd
from helper_methods import *
import time

chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"
bradley_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\bradley_agent_q_table.pkl"

bradley_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\training_results\bradley_training_results.txt'

chess_data = pd.read_pickle(chess_data_path, compression = 'zip')
training_chess_data = chess_data.sample(1_000)
# bootstrap_chess_data = chess_data.head(5) # this is just to init the agent after it's already trained

if __name__ == '__main__':

    # ============= train a new agent =============== #    
    bradley = init_agent(training_chess_data, 'W')
    
    # start_time = time.time() 
    # bradley.train_rl_agent(bradley_training_results_filepath)
    # end_time = time.time()

    # pikl_q_table(bradley, bradley_agent_q_table_path)

    # total_time = end_time - start_time
    # print('training is complete')
    # print(f'it took: {total_time}')

    # ========== bootstrap and continue training agent ========== #
    bootstrap_agent(bradley, bradley_agent_q_table_path)

    start_time = time.time()
    bradley.continue_training_rl_agent(bradley_training_results_filepath, 20_000)
    pikl_q_table(bradley, bradley_agent_q_table_path)
    end_time = time.time()
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time}')
   

    # # ========== bootstrap and play ==========  #
    # bootstrap_agent(bradley, bradley_agent_q_table_path)
    play_game(bradley)

