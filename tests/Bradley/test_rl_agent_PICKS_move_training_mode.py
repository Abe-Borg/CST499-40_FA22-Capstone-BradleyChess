def test_rl_agent_PICKS_move_training_mode():
    # Test case 1: White RL agent chooses a move
    curr_state = {'board': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'legal_moves': ['e2e3', 'e2e4', 'g1f3', 'b1c3']}
    rl_agent_color = 'W'
    expected_move = 'e2e4'
    assert rl_agent_PICKS_move_training_mode(curr_state, rl_agent_color) == expected_move
    
    # Test case 2: Black RL agent chooses a move
    curr_state = {'board': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1', 'legal_moves': ['e7e6', 'e7e5', 'g8f6', 'b8c6']}
    rl_agent_color = 'B'
    expected_move = 'e7e5'
    assert rl_agent_PICKS_move_training_mode(curr_state, rl_agent_color) == expected_move