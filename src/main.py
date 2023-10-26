import pandas as pd
from helper_methods import *
import time
# import logging
# import log_config
import Settings
settings = Settings.Settings()

# logger = logging.getLogger(__name__)

print_debug_statements_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\print_statements.txt'
print_statements_debug = open(print_debug_statements_filepath, 'a')
PRINT_RESULTS_DEBUG: bool = True
# print_statements_debug.write(f'\n\n Start of {game_num_str} training\n\n')

chess_data = pd.read_pickle(settings.chess_data_path, compression = 'zip') 
training_chess_data = chess_data.sample(settings.training_sample_size) 

if __name__ == '__main__':
    """ 
        You can train agents, or play against an agent or have two agents play against each other.
        Simply comment or uncomment certain sections of the code.

        When first starting out, you need to train new agents. Then you need to continue training
        the agents. The intial training session teaches the agents good positional chess.
        Then the additional training allows you to fine tune agent behaviour 
        (making it more aggressive for example)

        The numbe of moves per player was capped at 50 each. That can be changed, but you'll need
        to also change the number of moves in your dataframe
    """
    # ========================= Hyper parameters ========================= #
    # you can adjust the hyperparameters (see Settings.py) as you wish
    # at initial training or at additional training.

    # ========================= train new agents ========================= # 
    bradley = init_bradley(training_chess_data)    
    start_time = time.time() 
    bradley.train_rl_agents(settings.initial_training_results_filepath)
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
    # bradley.continue_training_rl_agents(settings.additional_training_results_filepath, settings.agent_vs_agent_num_games)
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

    # # undo not working...
    # print('At any time, you can enter \'q\' to end the game.\n')
    # print('Also, once the game starts, to undo the computer\'s last move, type \'pop\'\n')
    # print('And if you want to undo your last move, type \'pop 2x\'\n')
    
    # rl_agent_color = input('Enter color to play as, \'W\' or \'B\': ')
    # if rl_agent_color == 'q':
    #     quit()
    # if rl_agent_color == 'W':
    #     play_game(bradley, rl_agent_color)
    # else: 
    #     play_game(bradley, 'B')
    

    # # # ========================= bootstrap agents and have them play each other =========================  #
    # bradley = init_bradley(training_chess_data)
    # bootstrap_agent(bradley, 'W', settings.bradley_agent_q_table_path)
    # bootstrap_agent(bradley, 'B', settings.imman_agent_q_table_path)
    # agent_vs_agent(bradley)