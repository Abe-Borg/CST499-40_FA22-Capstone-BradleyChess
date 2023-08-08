def test_assign_points_to_Q_table_training_mode():
    # Test case 1: White RL agent updates Q table with existing chess move
    chess_move = 'e2e4'
    curr_turn = 'W'
    curr_Qval = 0.5
    rl_agent_color = 'W'
    W_rl_agent = RLAgent()
    W_rl_agent.Q_table = {'e2e4': {'W': 0.2, 'B': 0.3}}
    assign_points_to_Q_table_training_mode(chess_move, curr_turn, curr_Qval, rl_agent_color)
    assert W_rl_agent.Q_table == {'e2e4': {'W': 0.5, 'B': 0.3}}
    
    # Test case 2: Black RL agent updates Q table with new chess move
    chess_move = 'e7e5'
    curr_turn = 'B'
    curr_Qval = 0.8
    rl_agent_color = 'B'
    B_rl_agent = RLAgent()
    B_rl_agent.Q_table = {'e2e4': {'W': 0.2, 'B': 0.3}}
    assign_points_to_Q_table_training_mode(chess_move, curr_turn, curr_Qval, rl_agent_color)
    assert B_rl_agent.Q_table == {'e2e4': {'W': 0.2, 'B': 0.3}, 'e7e5': {'W': 0, 'B': 0.8}}