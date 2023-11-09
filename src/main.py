import helper_methods
import game_settings
import pandas as pd
import time
import Bradley


# import logging
# import log_config
# logger = logging.getLogger(__name__)

# kaggle_chess_data = pd.read_pickle(game_settings.kaggle_chess_data_path, compression = 'zip') 

chess_data = pd.read_pickle(game_settings.chess_data_cleaned_filepath, compression = 'zip')
training_chess_data = chess_data.sample(game_settings.training_sample_size) 

if __name__ == '__main__':
    # ========================= train new agents ========================= # 
    bradley = Bradley.Bradley(training_chess_data)
    start_time = time.time()

    try:
        bradley.train_rl_agents()
    except Exception as e:
        print(f'training interrupted because of:  {e}')
        quit()
        
    end_time = time.time()
    helper_methods.pikl_q_table(bradley, 'W',game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {game_settings.training_sample_size} games')
    quit()

    # # # # # ========================= bootstrap and continue training agents ========================= #
    # bradley = helper_methods.init_bradley(training_chess_data)    # the size of the training set in this step doesnt matter. It's just for initializing the object.
    # helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)

    # start_time = time.time()
    # try:
    #     bradley.continue_training_rl_agents(game_settings.agent_vs_agent_num_games)
    # except Exception as e:
    #     print(f'training interrupted because of:  {e}')
    #     quit()
        
    # helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print('training is complete')
    # print(f'it took: {total_time}')
    # quit()


    # # # ========================= bootstrap and play against human =========================  #
    # bradley = helper_methods.init_bradley(training_chess_data)
    # helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    # rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
    # if rl_agent_color == 'W':
    #     play_game(bradley, rl_agent_color)
    # else: 
    #     play_game(bradley, 'B')
    

    # # # ========================= bootstrap agents and have them play each other =========================  #
    # bradley = helper_methods.init_bradley(training_chess_data)
    # helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
    # helper_methods.agent_vs_agent(bradley)