# CST499-40_FA22-Capstone-BradleyChess
A reinforced learning implementation of a chess engine. The implementation uses the SARSA algorithm.
This project was completed as part of the capstone course at CSU, Monterey Bay, for the completion of the Bachelor's of Computer Science program.

This project is a continuation of previous work completed in the data science course, https://github.com/abecsumb/DataScienceProject/blob/main/Chess_Data_Preparation.ipynb 

This repo contains the machine learning part of the capstone project (my part). However, the overall project submitted to faculty was a team effort. It was a web application implemented using Flask and React. Sarom Thin (https://github.com/lom360) completed work for the backend, and Mehar Rekhi (https://github.com/mehr998) completed work on the frontend. The web application can be found here, https://www.bradleychess.com/ . There is also a video presentation at this link, https://www.youtube.com/watch?v=HlXtLt6fiLE .The code in this repo represents the core of the BradleyChess team project, and it can also be a standalone application. 

The chess reinforced learning agents learn by playing games from a chess database exactly as shown. That's the first step of training in a two-step process. The second part of training lets the reinforced learning agents choose their own chess moves. The agents (White and Black players) train each other by playing against each other. 

The file main.py has all information necessary to run this program. You will need to change the filepaths in main.py, and also in Settings.py.

The chess database is already in the folder, chess_data, but you can make a bigger or different chess database. You will need to make sure the formatting is the same as shown in the file, Chess_Data_Preparation.ipynb file linked above.

### To Run the Program
1. Start at the main.py file and change the file paths shown. Also, go to the Settings.py file and change the path of the Stockfish chess engine. Setting.py shows the hyperparamers that you can change before initial training and during additional training. Also, the Stockfish engine is already included in this repo. Stockfish is used to assign points to different positions during the training periods.
2. Main.py contains 4 commented out portions of code. The first time you run this, uncomment the first part (train new agent), and adjust training_sample_size to your preference (recommend you start small, training can take hours or days). Run Main.py and the first part of the training phase will be complete. 
3. For part two of training, make sure to comment out the 'train new agents' section and uncomment the next part, 'bootstrap and continue training agents'. Adjust the variable, agent_vs_agent_num_games to your preference (again, I recommend you start small). Run Main.py and then you will have completed phase 2 of training. You can continue training, meanwhile adjusting hyperparameters (see Settings.py). 
4. Once again, comment out the previous portion of code, and uncomment the part labeled 'bootstrap and play against human'.
5. You can also have the agents play against each other by uncommenting the last portion of code, 'bootstrap agents and have them play each other.
