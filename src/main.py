import pandas as pd
from helper_methods import *
import time
import game_settings

# import logging
# import log_config
# logger = logging.getLogger(__name__)

chess_data = pd.read_pickle(game_settings.chess_data_path, compression = 'zip') 
training_chess_data = chess_data.sample(game_settings.training_sample_size) 

if __name__ == '__main__':
    # ========================= train new agents ========================= # 
    bradley = init_bradley(training_chess_data)    
    start_time = time.time() 
    bradley.train_rl_agents(game_settings.learn_rate)
    end_time = time.time()
    pikl_q_table(bradley, 'W',settings.bradley_agent_q_table_path)
    pikl_q_table(bradley, 'B', settings.imman_agent_q_table_path)
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time}')
    quit()

    # # # # ========================= bootstrap and continue training agents ========================= #
    # bradley = init_bradley(training_chess_data)    # the size of the training set in this step doesnt matter. It's just for initializing the object.
    # bootstrap_agent(bradley, 'W', settings.bradley_agent_q_table_path)
    # bootstrap_agent(bradley, 'B', settings.imman_agent_q_table_path)

    # start_time = time.time()
    # bradley.continue_training_rl_agents(settings.agent_vs_agent_num_games)
    # pikl_q_table(bradley, 'W', settings.bradley_agent_q_table_path)
    # pikl_q_table(bradley, 'B', settings.imman_agent_q_table_path)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print('training is complete')
    # print(f'it took: {total_time}')
    # quit()


    # # # ========================= bootstrap and play against human =========================  #
    # bradley = init_bradley(training_chess_data)
    # bootstrap_agent(bradley, 'W', settings.bradley_agent_q_table_path)
    # bootstrap_agent(bradley, 'B', settings.imman_agent_q_table_path)
    
    # rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
    # if rl_agent_color == 'W':
    #     play_game(bradley, rl_agent_color)
    # else: 
    #     play_game(bradley, 'B')
    

    # # # ========================= bootstrap agents and have them play each other =========================  #
    # bradley = init_bradley(training_chess_data)
    # bootstrap_agent(bradley, 'W', settings.bradley_agent_q_table_path)
    # bootstrap_agent(bradley, 'B', settings.imman_agent_q_table_path)
    # agent_vs_agent(bradley)