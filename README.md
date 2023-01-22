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
2. main.py contains 4 commented out portions of code. The first time you run this, uncomment the first part (train new agent), and adjust training_sample_size to your preference (recommend you start small, training can take hours or days). Run main.py and the first part of the training phase will be complete. 
3. For part two of training, make sure to comment out the 'train new agents' section and uncomment the next part, 'bootstrap and continue training agents'. Adjust the variable, agent_vs_agent_num_games to your preference (again, I recommend you start small). Run main.py and then you will have completed phase 2 of training. You can continue training, meanwhile adjusting hyperparameters (see Settings.py). 
4. Once again, comment out the previous portion of code, and uncomment the part labeled 'bootstrap and play against human'.
5. You can also have the agents play against each other by uncommenting the last portion of code, 'bootstrap agents and have them play each other.

### Main Components of Program
#### Bradley Class
This is a composite class that manages components and data flow during training and game play. All communication with external applications is also managed by this class. The methods of this class are focused on coordinating actions during the training and gameplay phase. However, the methods don't change the chessboard or choose the chess moves.

#### Environ Class
This class manages the chessboard and is the only part of the program that will actually change the chessboard. It also sets and manages what is called the state. In this application the state is the current chessboard configuration, the current turn, and the legal moves at each turn. 

#### Agent Class
This class is reponsible for choosing the chess moves. The first training phase (remember, there are two training phases) the agents will play through a database of games exactly as shown and learn that way. I tried many different versions of this part of the project, and this implemenation was the most effective. This style of training teaches the agents to play good positional chess. Later during phase two, the user can change hyperparameters to make the agents more aggressive or even to prefer certain openings for example.

### Rationale for Project
I chose this project for my capstone because I am pursuing a career in data science. Specifically machine learning, and ideally, reinforced learning. I structured the program as a sandbox that would be interesting and useful for students who are learning the basics of data science and to get an introduction to reinforced learning. I made sure to keep the code as simple as possible, but also as flexible as possible. The student can adjust the different hyperparameters and observe how the agents behave. Their behaviour changes dramatically based on training time and the value of variables like learning rate and the discount factor. The student can also change the way the agent chooses actions during the game play mode. Currently, the agent will choose the largest value in the Q table at each turn. Also, this project and also the data science class project (linked above) cover the fundamental topics of an introductory data science course that uses Python as the language. The student will recognize the use of Pandas, Matplotlib, Seaborn, Numpy, and some basic machine learning algorithms like kNN classification and decision trees. 

Data science enthusiast will recogize ways to make this project better and to create more sophisticated agents. I started this project knowing nothing about reinforced learning. Therefore, the implementation is basic. I did not fully understand the math and theory behind the SARSA algorithm either. Regardless, I believe the form of the SARSA algorithm is implemented correctly.

Finally, the single point of communication between this program and something external (like a web app) is facilitated by two methods, Bradley.recv_opp_move() and Bradley.rl_agent_chess_move().

### Future Plans
I will make a webGL application for this project. I think this is easier and looks better than a React app. Also, the training time could be faster, and I would like to train the agents on hundreds of thousands of games instead of tens of thousands. It could be that the only way to accomplish this is to use Julia or some other language. Also, I may find a way to use Numpy arrays instead of Pandas DataFrames. Right now it seems like I need DFs for what I am doing.
