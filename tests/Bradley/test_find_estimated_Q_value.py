def test_find_estimated_Q_value():
    # Test case 1: Anticipated move results in positive centipawn score
    environ = ChessEnvironment()
    environ.load_chessboard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    Bradley = BradleyRLAgent()
    Bradley.environ = environ
    Bradley.settings = RLSettings()
    Bradley.settings.mate_score_reward = 100
    Bradley.analyze_board_state = lambda x: {'centipawn_score': 20, 'mate_score': None}
    assert Bradley.find_estimated_Q_value() == 20
    
    # Test case 2: Anticipated move results in checkmate
    environ = ChessEnvironment()
    environ.load_chessboard('rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1')
    Bradley = BradleyRLAgent()
    Bradley.environ = environ
    Bradley.settings = RLSettings()
    Bradley.settings.mate_score_reward = 100
    Bradley.analyze_board_state = lambda x: {'centipawn_score': None, 'mate_score': -5}
    assert Bradley.find_estimated_Q_value() == -500