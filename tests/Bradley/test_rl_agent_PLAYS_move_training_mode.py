def test_rl_agent_PLAYS_move_training_mode():
    # Test case 1: Move results in positive centipawn score
    chess_move = 'e2e4'
    expected_reward = 20
    environ = ChessEnvironment()
    environ.load_chessboard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    Bradley = BradleyRLAgent()
    Bradley.environ = environ
    Bradley.settings = RLSettings()
    Bradley.settings.mate_score_reward = 100
    Bradley.analyze_board_state = lambda x, y: {'centipawn_score': 20, 'mate_score': None}
    assert rl_agent_PLAYS_move_training_mode(chess_move) == expected_reward
    
    # Test case 2: Move results in checkmate
    chess_move = 'f7f5'
    expected_reward = -500
    environ = ChessEnvironment()
    environ.load_chessboard('rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1')
    Bradley = BradleyRLAgent()
    Bradley.environ = environ
    Bradley.settings = RLSettings()
    Bradley.settings.mate_score_reward = 100
    Bradley.analyze_board_state = lambda x, y: {'centipawn_score': None, 'mate_score': -5}
    assert rl_agent_PLAYS_move_training_mode(chess_move) == expected_reward