# our function train_rl_agents is indeed quite long and complex. 
# Breaking it into smaller pieces can make it more readable and easier to maintain. Here are a few suggestions:

    # Separate Agent Movements: You have code for the white and black agents' movements. 
    # These can be separated into their own respective functions, such as white_agent_move and black_agent_move.

    # Handle Rewards and Q-values: The handling of rewards and Q-values (or "value estimation") 
    # could also be separated into their own respective functions, e.g., handle_rewards_and_Q_values.

    # Move Results Logging to a Function: The logging or writing of results to a file could be 
    # extracted into a function of its own, say log_training_results.

    # Separate Analysis-related Operations: You are analyzing the board state several times 
    # in the function. Consider creating a separate function, such as analyze_and_update_state for this.

    # Move Game Termination Checks: Checking for the end of the game is done in several places. 
    # Consider creating a separate function, check_game_end, that can be called when needed.

    # After applying these changes, your train_rl_agents function would be simplified to a sequence of high-level operations:


def train_rl_agents(self, training_results_filepath: str) -> None:
    # Opening of training_results file

    for game_num in self.chess_data.index:
        # Initialize environment and states

        while not self.check_game_end(curr_state, num_chess_moves):
            white_agent_move(game_num)
            if not self.check_game_end(curr_state, num_chess_moves):
                black_agent_move(game_num)
        
        # Log game results
        self.reset_environ()

    # Mark agents as trained and close training_results file

# This leaves your train_rl_agents function as a high-level summary of the steps involved in the training process, 
# with the detailed implementation of each step hidden away in helper functions. This approach to structuring 
# code is often easier to read and maintain. Plus, it allows you to test individual parts of your function independently, 
# which can make debugging easier.




# User
# show me what the white_agent_move method would look like

# ChatGPT
# Sure, based on the given code, the white_agent_move method might look something like this:


def white_agent_move(self, game_num):
    # WHITE AGENT PICKS MOVE
    W_curr_action: dict[str] = self.W_rl_agent.choose_action(curr_state, game_num)
    W_chess_move: str = W_curr_action['chess_move_str']
    curr_turn: str = curr_state['curr_turn']

    # ASSIGN POINTS TO Q_TABLE FOR WHITE
    try:
        self.W_rl_agent.change_Q_table_pts(W_chess_move, curr_turn, W_curr_Qval)
    except KeyError: 
        # chess move is not represented in the Q table, update Q table and try again.
        self.W_rl_agent.update_Q_table([W_chess_move])
        self.W_rl_agent.change_Q_table_pts(W_chess_move, curr_turn, W_curr_Qval)

    # WHITE AGENT PLAYS SELECTED MOVE
    self.environ.load_chessboard(W_chess_move)
    self.environ.update_curr_state()

    # Update the state each time a move is made               
    curr_state = self.environ.get_curr_state()

    # Calculate reward for white
    analysis_results = self.analyze_board_state(self.get_chessboard(), False)
    if analysis_results['mate_score'] is None:
        W_reward = analysis_results['centipawn_score']
    else: # there is an impending checkmate
        W_reward = analysis_results['mate_score'] * self.settings.mate_score_factor

    # WHITE AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT 
    analysis_results = self.analyze_board_state(self.get_chessboard())               
    self.environ.load_chessboard_for_Q_est(analysis_results) 
                    
    W_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False) 
                    
    # get pts for est_Qval 
    if W_est_Qval_analysis['mate_score'] is None:
        W_est_Qval = W_est_Qval_analysis['centipawn_score']
    else: # there is an impending checkmate
        W_est_Qval = W_est_Qval_analysis['mate_score'] * self.settings.mate_score_factor

    self.environ.pop_chessboard()

    return W_reward, W_est_Qval

# In this method, we're handling all the steps that the white agent needs to take in a single round of play, 
# including choosing and executing an action, updating the Q-table, and preparing for the next round.

# Note that this method returns the reward and estimated Q-value, which we need to use in the 
# original train_rl_agents function for the SARSA algorithm update rule. You would need to do something 
# similar for the black_agent_move method.

# Also, this assumes that curr_state, W_curr_Qval and other variables used in this function are either 
# class attributes or passed in as parameters. You would need to make sure that these variables are 
# accessible within the scope of this function.




# User
# show me what the black_agent_move method looks like

# ChatGPT
# Sure, based on the given code, the black_agent_move method might look something like this:


def black_agent_move(self, game_num):
    # BLACK AGENT PICKS MOVE
    B_curr_action = self.B_rl_agent.choose_action(curr_state, game_num)
    B_chess_move = B_curr_action['chess_move_str']
    curr_turn = curr_state['curr_turn']

    # ASSIGN POINTS TO Q_TABLE FOR BLACK
    try:
        self.B_rl_agent.change_Q_table_pts(B_chess_move, curr_turn, B_curr_Qval)
    except KeyError: 
        # chess move is not represented in the Q table, update Q table and try again.
        self.B_rl_agent.update_Q_table(B_chess_move)
        self.B_rl_agent.change_Q_table_pts(B_chess_move, curr_turn, B_curr_Qval)

    # BLACK AGENT PLAYS SELECTED MOVE
    self.environ.load_chessboard(B_chess_move)
    self.environ.update_curr_state()

    # Update the state each time a move is made               
    curr_state = self.environ.get_curr_state()

    # Calculate reward for black
    analysis_results = self.analyze_board_state(self.get_chessboard(), False)
    if analysis_results['mate_score'] is None:
        B_reward = analysis_results['centipawn_score']
    else: # there is an impending checkmate
        B_reward = analysis_results['mate_score'] * self.settings.mate_score_factor

    # BLACK AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT
    analysis_results = self.analyze_board_state(self.get_chessboard())               
    self.environ.load_chessboard_for_Q_est(analysis_results) 

    B_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False) 

    # Get points for estimated Q-value
    if B_est_Qval_analysis['mate_score'] is None:
        B_est_Qval = B_est_Qval_analysis['centipawn_score']
    else: # there is an impending checkmate
        B_est_Qval = B_est_Qval_analysis['mate_score'] * self.settings.mate_score_factor

    self.environ.pop_chessboard()

    return B_reward, B_est_Qval


# This method is quite similar to white_agent_move, but it corresponds to the actions of the black agent. 
# As with the white agent method, we're choosing and executing an action, updating the Q-table, and preparing for the next round.

# Just like in the white_agent_move method, this function returns the reward and estimated Q-value,
# which we will use in the original train_rl_agents function for the SARSA algorithm update rule.

# Again, this assumes that curr_state, B_curr_Qval and other variables used in this function are either 
# class attributes or passed in as parameters. You would need to make sure that these variables are accessible within the scope of this function.



# User
# how would I separate the handling of rewards and Q-values into their own respective functions?

# ChatGPT
# Sure, you could create a separate method for the reward and Q-value computations. Here's how you can separate these into their own methods:


def compute_reward(self, analysis_results):
    if analysis_results['mate_score'] is None:
        reward = analysis_results['centipawn_score']
    else: # there is an impending checkmate
        reward = analysis_results['mate_score'] * self.settings.mate_score_factor
    return reward

def compute_estimated_q_value(self, analysis_results):
    if analysis_results['mate_score'] is None:
        estimated_q_value = analysis_results['centipawn_score']
    else: # there is an impending checkmate
        estimated_q_value = analysis_results['mate_score'] * self.settings.mate_score_factor
    return estimated_q_value


# You can use these functions inside white_agent_move and black_agent_move functions to compute the reward 
# and estimated Q value respectively. Here is how you can use these in white_agent_move:


def white_agent_move(self, game_num):
    ...
    # Calculate reward for white
    analysis_results = self.analyze_board_state(self.get_chessboard(), False)
    W_reward = self.compute_reward(analysis_results)

    ...

    # WHITE AGENT CHOOSES NEXT ACTION, BUT DOES NOT PLAY IT
    analysis_results = self.analyze_board_state(self.get_chessboard())               
    self.environ.load_chessboard_for_Q_est(analysis_results)

    W_est_Qval_analysis = self.analyze_board_state(self.get_chessboard(), False)
    W_est_Qval = self.compute_estimated_q_value(W_est_Qval_analysis)
    ...


# You will replace similar parts in black_agent_move with these function calls as well.

# Remember to pass any needed arguments if the methods require more information to do 
# their computations. Also, remember to consider class vs instance methods based on whether 
# you're manipulating the state of the class itself or just using its properties.



# User
# what does the function analyze_and_update_state look like?

# ChatGPT
# The analyze_and_update_state function performs analysis of the board state and calculates the reward. The function could look like this:


def analyze_and_update_state(self):
    # perform analysis and calculate reward
    analysis_results = self.analyze_board_state(self.get_chessboard(), False)
    
    # assign reward based on the analysis results
    if analysis_results['mate_score'] is None:
        reward = analysis_results['centipawn_score']
    else: # there is an impending checkmate
        reward = analysis_results['mate_score'] * self.settings.mate_score_factor
    
    # update state
    self.environ.update_curr_state()

    return reward


# In your main training function, you would call this function to update the state 
# and obtain the reward after the agent makes a move:

...
##### WHITE AGENT PLAYS SELECTED MOVE #####
self.environ.load_chessboard(W_chess_move)
W_reward = self.analyze_and_update_state()

...
##### BLACK AGENT PLAYS SELECTED MOVE #####
self.environ.load_chessboard(B_chess_move)
B_reward = self.analyze_and_update_state()
...

# This refactoring reduces the repetition in your main training function and isolates the state 
# updating and reward calculation into a separate function, making your code easier to read and maintain.




# User
# what does the function check_game_end look like?

# ChatGPT
# The check_game_end function checks if the game has ended and returns a Boolean value accordingly. Here's how you could write it:


def check_game_end(self, num_chess_moves: int, turn_index: int) -> bool:
    """Checks if the game has ended.

    Args:
        num_chess_moves (int): Total number of moves in the game.
        turn_index (int): The current turn index.

    Returns:
        bool: True if game has ended, False otherwise.
    """

    # check if game ended after the move
    if turn_index >= num_chess_moves:
        return True

    # check if maximum number of turns is reached
    if turn_index >= self.settings.num_turns_per_player * 2:
        return True

    # if none of the conditions are met, the game continues
    return False


# You can then use this function in your main training function like this:

...
##### WHITE AGENT PLAYS SELECTED MOVE #####
self.environ.load_chessboard(W_chess_move)
W_reward = self.analyze_and_update_state()

if self.check_game_end(num_chess_moves, curr_state['turn_index']):
    break

...

##### BLACK AGENT PLAYS SELECTED MOVE #####
self.environ.load_chessboard(B_chess_move)
B_reward = self.analyze_and_update_state()

if self.check_game_end(num_chess_moves, curr_state['turn_index']):
    break
...

# This way, the conditions for game termination are encapsulated in a separate function, making the main training function cleaner and more readable.







############################################ HERE'S ANOTHER APPROACH ###########################################################

# To make the `train_rl_agents` method more modular and manageable, we can break it down into smaller helper functions. 
# Each function should have a single responsibility, making the code easier to understand, debug, and maintain. Here's a suggested breakdown:

# 1. **Initialization**:
#     - `initialize_training()`: This function will handle initializing the current game, current turn, Q-tables, and resetting the `environ`. 

# 2. **Game Loop**:
#    - We'll keep the outer loop in the `train_rl_agents` method since it's the main driving loop of the method. 
#      However, the contents of this loop will be broken down into other helper functions.

# 3. **Action Selection**:
#     - `select_action()`: This function will decide which move the agent should make. 
#       It'll take the current state and Q-table as inputs and return the selected action. 
#       This will encapsulate the logic of whether the agent should choose a random move or the move with the highest Q-value.

# 4. **Reward Assignment**:
#     - `assign_reward()`: Based on the outcome of the move, this function will determine the reward or penalty to assign to the agent. 
#       It'll take the game state and the move made as inputs and return the reward value.

# 5. **Q-value Update**:
#     - `update_q_value()`: This function will implement the Q-learning algorithm to update the Q-values in the Q-table. 
#       It'll take the current state, the action taken, the reward received, and the next state as inputs.

# 6. **Post-game Operations**:
#     - `post_game_operations()`: After a game ends, this function will handle logging the reason for the game's termination 
#       and resetting the `environ` for the next game.

# 7. **Logging and Output**:
#     - We can have separate logging functions for different purposes:
#         - `log_training_progress()`: Log the progress of the training process, such as the number of games played and the time elapsed.
#         - `log_errors()`: Log any errors or exceptions that occur during training.


# By splitting the `train_rl_agents` method into these helper functions, the main method becomes a sequence of function calls, 
# making it more readable and organized. Each helper function can be debugged and tested independently, 
# simplifying the development and maintenance process.





# User
# show me the refactored code

# ChatGPT
# Certainly! Here's how the train_rl_agents method can be refactored using the proposed helper functions:


class Bradley:

    # ... other methods ...

    def train_rl_agents(self, training_results_filepath: str) -> None:
        start_time = time.time()

        self.initialize_training()
        
        for _ in range(self.settings.training_sample_size):
            
            while self.game_on():
                self.select_action('W')
                self.select_action('B')
            
            self.post_game_operations()
            
        self.log_training_progress(start_time, training_results_filepath)

    def initialize_training(self) -> None:
        self.curr_game = 0
        self.curr_turn = "W1"
        self.W_rl_agent.reset_Q_table()
        self.B_rl_agent.reset_Q_table()
        self.environ.reset_env()

    def select_action(self, agent_color: str) -> None:
        if agent_color == 'W':
            agent = self.W_rl_agent
        else:
            agent = self.B_rl_agent
        
        # logic for selecting an action (either random or from Q-table)
        action = ... # get the action
        next_state, reward, done = self.environ.step(action)
        
        self.assign_reward(reward)
        self.update_q_value(agent, action, reward, next_state)

    def assign_reward(self, reward: int) -> None:
        # logic for assigning rewards based on game state and action taken
        ...

    def update_q_value(self, agent, action, reward, next_state) -> None:
        # logic for updating Q-values using the Q-learning algorithm
        ...

    def post_game_operations(self) -> None:
        # logic for handling operations after a game ends
        ...

    def log_training_progress(self, start_time, training_results_filepath: str) -> None:
        # logic for logging training progress
        ...


# This refactoring aims to make the code more modular and readable by breaking the main training loop into smaller, 
# more manageable helper functions. Each function has a single, clear responsibility, 
# which makes the code easier to understand, debug, and maintain.

# Note: The above code is a conceptual refactoring and uses placeholder comments (...) 
# where the actual implementation details would go. The actual implementation 
# would need to be filled in based on the original method's logic.


