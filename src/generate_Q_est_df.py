def generate_Q_est_df(self) -> None:
    try:
        ### FOR EACH GAME IN THE TRAINING SET ###
        for game_num_str in self.chess_data.index:
            num_chess_moves_curr_training_game: int = self.chess_data.at[game_num_str, 'Num Moves']

            try:
                curr_state = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write(f'An error occurred at self.environ.get_curr_state: {e}\n')
                self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                raise Exception from e

            ### LOOP PLAYS THROUGH ONE GAME ###
            while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                ##################### WHITE'S TURN ####################
                # choose action a from state s, using policy
                W_chess_move = self.W_rl_agent.choose_action(curr_state, game_num_str)
                if not W_chess_move:
                    raise ValueError(f'W_chess_move is empty at turn {curr_state["curr_turn"]}')

                ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_plays_move(W_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    break # and go to the next game. this game is over.

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    raise Exception from e
                
                # find the estimated Q value for White, but first check if game ended
                if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                    break # and go to next game
                else: # current game continues
                    try:
                        W_est_Qval: int = self.find_estimated_Q_value()
                        if game_settings.PRINT_Q_EST:
                            self.q_est_log.write(f'W_est_Qval: {W_est_Qval}\n')
                    except Exception as e:
                        self.errors_file.write(f'An error occurred while retrieving W_est_Qval: {e}\n')
                        self.errors_file.write(f"at White turn {curr_state['curr_turn']}, failed to find_estimated_Q_value\n")
                        self.errors_file.write(f'curr state is:{curr_state}\n')
                        self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                        break

                ##################### BLACK'S TURN ####################
                # choose action a from state s, using policy
                B_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                if not B_chess_move:
                    raise ValueError(f'B_chess_move is empty at turn {curr_state["curr_turn"]}')

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_plays_move(B_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    break 

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at environ.get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    self.errors_file.write("========== Bye from Bradley.train_rl_agents ===========\n\n\n")
                    raise Exception from e

                # find the estimated Q value for Black, but first check if game ended
                if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                    break # and go to next game
                else: # current game continues
                    try:
                        B_est_Qval: int = self.find_estimated_Q_value()
                        if game_settings.PRINT_Q_EST:
                            self.q_est_log.write(f'B_est_Qval: {B_est_Qval}\n') 
                    except Exception as e:
                        self.errors_file.write(f"at Black turn, failed to find_estimated_Qvalue because error: {e}\n")
                        self.errors_file.write(f'curr turn is:{curr_state["curr_turn"]}\n')
                        self.errors_file.write(f'turn index is: {curr_state["turn_index"]}\n')
                        self.errors_file.write(f'curr game is: {game_num_str}\n')
                        break # too many issues with this step, simply jump to next game.

                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n") 
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    raise Exception from e
            ### END OF CURRENT GAME LOOP ###

            # create a newline between games in the Q_est log file.
            if game_settings.PRINT_Q_EST:
                self.q_est_log.write('\n')

            self.environ.reset_environ() # reset and go to next game in training set

        # end of training, all games in database have been processed
        self.W_rl_agent.is_trained = True
        self.B_rl_agent.is_trained = True
    
    finally:
        self.engine.quit()