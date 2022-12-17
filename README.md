# CST499-40_FA22-Capstone-BradleyChess
A reinforced learning implementation of a chess engine. The implementation uses the SARSA algorithm.
This project was completed as part of the capstone course at CSU, Monterey Bay, for the completion of the Bachelor's of Computer Science program.

This project is a continuation of previous work completed in the data science course, https://github.com/abecsumb/DataScienceProject/blob/main/Chess_Data_Preparation.ipynb 

This repo contains the machine learning part of the capstone project (my part). However, the overall project submitted to faculty was a team effort. It was a web application implemented using Flask and React. Sarom Thin (https://github.com/lom360) completed work for the backend, and Mehar Rekhi (https://github.com/mehr998) completed work on the frontend. Web application can be found here, https://www.bradleychess.com/ . The code in this repo represents the core of the BradleyChess team project, and it can also be a standalone application. 

The chess reinforced learning agents learn by playing games from a chess database exactly as shown. That's the first step of training in a two-step process. The second part of training lets the reinforced learning agents choose their own chess moves. The agents (White and Black players) train each other by playing against each other. 

The file main.py has all information necessary to run this program. You will need to change the filepaths in main.py, and also in Settings.py.

The chess database is already in the folder, chess_data, but you can make a bigger or different chess database. You will need to make sure the formatting is the same as shown in the file, Chess_Data_Preparation.ipynb file linked above.

### To Run the Program
