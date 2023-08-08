def test_find_next_Qval():
    # Test case 1: Positive reward, positive estimated Q value
    curr_Qval = 0.5
    learn_rate = 0.1
    reward = 20
    discount_factor = 0.9
    est_Qval = 10
    expected_next_Qval = 2.05
    assert find_next_Qval(curr_Qval, learn_rate, reward, discount_factor, est_Qval) == expected_next_Qval
    
    # Test case 2: Negative reward, negative estimated Q value
    curr_Qval = -0.2
    learn_rate = 0.2
    reward = -50
    discount_factor = 0.8
    est_Qval = -30
    expected_next_Qval = -0.44
    assert find_next_Qval(curr_Qval, learn_rate, reward, discount_factor, est_Qval) == expected_next_Qval