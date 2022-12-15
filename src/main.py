import pandas as pd
from helper_methods import *
import time

chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"

bradley_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\bradley_agent_q_table.pkl"
imman_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\imman_agent_q_table.pkl"

initial_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\training_results\initial_training_results.txt'
additional_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\training_results\additional_training_results.txt'

training_sample_size = 1_000 # change this to whatever you want
agent_vs_agent_num_games = 20_000 # change this to whatever you want

chess_data = pd.read_pickle(chess_data_path, compression = 'zip')
training_chess_data = chess_data.sample(training_sample_size)


if __name__ == '__main__':
    """ You can train agents, or play against an agent or have two agents play against each other.
        Simply comment or uncomment certain sections of the code.

        When first starting out, you need to train new agents. Then you need to continue training
        the agents.


    """
    # ========================= Hyper parameters ========================= #
    # you can adjust the hyperparameters (see Settings.py) as you wish
    # at initial training or at additional training. Each agent

    # ========================= train new agents ========================= # 
    bradley = init_bradley(training_chess_data)    
    start_time = time.time() 
    bradley.train_rl_agents(initial_training_results_filepath)
    end_time = time.time()
    pikl_q_table(bradley, 'W',bradley_agent_q_table_path)
    pikl_q_table(bradley, 'B', imman_agent_q_table_path)
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time}')

    # ========================= bootstrap and continue training agents ========================= #
    bootstrap_agent(bradley, 'W', bradley_agent_q_table_path)
    bootstrap_agent(bradley, 'B', imman_agent_q_table_path)

    start_time = time.time()
    bradley.continue_training_rl_agents(additional_training_results_filepath, agent_vs_agent_num_games)
    pikl_q_table(bradley, 'W',bradley_agent_q_table_path)
    pikl_q_table(bradley, 'B', imman_agent_q_table_path)
    end_time = time.time()
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time}')

    # ========================= bootstrap and play against human =========================  #
    bootstrap_agent(bradley, 'W', bradley_agent_q_table_path)
    bootstrap_agent(bradley, 'B', imman_agent_q_table_path)

    rl_agent_color = 'W' # pick the agent you wish to play against
    play_game(bradley, rl_agent_color)

    # ========================= bootstrap and play against human =========================  #
    bootstrap_agent(bradley, 'W', bradley_agent_q_table_path)
    bootstrap_agent(bradley, 'B', imman_agent_q_table_path)

    rl_agent_color = 'W' # pick the agent you wish to play against
    play_game(bradley, rl_agent_color)