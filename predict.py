import pandas as pd
import numpy as np

import sqlite3
from pathlib import Path

import datetime as dt
from dateutil import parser

import requests

#Model training
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#Model saving
import joblib

#Helper functions
from process import get_team_data

Path('data\db.db').touch()
conn = sqlite3.connect('data\db.db')
c = conn.cursor()

#Also should probably try this as a logistic regression or a random tree classifier instead of a linear regression (for winning)
#3 functions to build models for win, total goals, win margin
#1 function to predict all three by calling these 3 saved models


def predict_winners(date=None, last_n_games = 10):
    #This function will predict the winner of a game 
    #Outcome: 0 -> 1 with 1 being 100% certain the home team wins, 0 100% certain the away team wins
    
    #Load the model 
    winner_model = joblib.load('models\\winner_model')
    
    #We want today's games
    if date==None:
        date = str(dt.date.today())

    url = 'https://api-web.nhle.com/v1/schedule/now'
    response = requests.get(url).json()['gameWeek']

    for game_date in response:
        #Just care about the day's games
        if game_date['date'] == date:
            #Check to see there are games being played
            if game_date['numberOfGames'] == 0:
                print("No games today!")
                return
            
            games = pd.DataFrame()

            #Get the info on each game
            for game in game_date['games']:
                game_id = game['id']
                home_team,away_team = game['homeTeam']['abbrev'],game['awayTeam']['abbrev']

                games = pd.concat([games,pd.DataFrame.from_dict({'game_id':[game_id],
                                                                'home_team':[home_team],
                                                                'away_team':[away_team]})])
            
            for i, game in games.iterrows():
                home_team, away_team = game['home_team'],game['away_team']
                home_stats, away_stats = get_team_data(home_team), get_team_data(away_team)



                col_names_single = home_stats.columns
                col_names_double = []
                for i in range(len(col_names_single)):
                    col_names_double.append("home_"+col_names_single[i])
                    col_names_double.append("away_"+col_names_single[i])
                col_names_double.sort()

                away_stats.columns = [f'a{i}' for i in range(len(away_stats.columns))]
                home_stats.columns = [f'h{i}' for i in range(len(home_stats.columns))]

                model_input = pd.concat([away_stats, home_stats],axis=1,ignore_index=True)
                model_input.columns = col_names_double

                #Make the winner prediction
                home_win_chance = round(100*winner_model.predict(model_input)[0],1)

                if home_win_chance > 50:
                    print(f"{away_team} @ [{home_team}] | {round(100-home_win_chance,1)} - {home_win_chance}")
                else: print(f"[{away_team}] @ {home_team} | {round(100-home_win_chance,1)} - {home_win_chance}")



                




    #Load the data for the two teams that are playing

    return 




def predict_winning_margin(home_team, away_team):
    #This functino will predict the margin the home team will win by
    #Negative if the home team will lose

    print("nothing yet!")


def predict_total_score(home_team, away_team):
    #This function will predict the total score of a game
    #Real number from 0 -> ? that is the total goals scored across both teams

    print("nothing yet!")

def winner_model(verbose=False):
    #This function will predict the winner of a game, 1 = 100% home chance, .5 = too close to call, 0 = away 100% chance

    #Pull all the data we're using from the games table
    games = pd.read_sql('SELECT * \
                                     FROM games_adv', con=conn)
    
    #Split it up to use for training
    x = games[['home_gpg_last_n','away_gpg_last_n','home_point_pct_last_n','away_point_pct_last_n','home_point_pct','away_point_pct']]
    x = x.reindex(sorted(x.columns), axis=1)

    y = games['home_win']

    #Quick correlation matrix
    df = pd.concat([x,y],axis=1)
    print(df.corr())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, shuffle=True)

    linear_model = LinearRegression().fit(x_train, y_train)
    log_model = LogisticRegression().fit(x_train, y_train)
    rf_model = RandomForestClassifier().fit(x_train, y_train)
    gb_model = GradientBoostingClassifier().fit(x_train, y_train)

    y_pred_lin = linear_model.predict(x_test)
    y_pred_lin = [round(y) for y in y_pred_lin]

    y_pred_log = log_model.predict(x_test)
    y_pred_log = [round(y) for y in y_pred_log]

    y_pred_rf = rf_model.predict(x_test)
    y_pred_rf = [round(y) for y in y_pred_rf]

    y_pred_gb = gb_model.predict(x_test)
    y_pred_gb = [round(y) for y in y_pred_gb]

    if verbose:
        print("Accuracy Scores")
        print(f"Linear Model: {accuracy_score(y_test,y_pred_lin)}")
        print(f"Log Model: {accuracy_score(y_test,y_pred_log)}")
        print(f"RF Model: {accuracy_score(y_test,y_pred_rf)}")
        print(f"GB Model: {accuracy_score(y_test,y_pred_gb)}")

        print("F1 Scores")
        print(f"Linear Model: {f1_score(y_test,y_pred_lin)}")
        print(f"Log Model: {f1_score(y_test,y_pred_log)}")
        print(f"RF Model: {f1_score(y_test,y_pred_rf)}")
        print(f"GB Model: {f1_score(y_test,y_pred_gb)}")

        print("AUC Scores")
        print(f"Linear Model: {roc_auc_score(y_test,y_pred_lin)}")
        print(f"Log Model: {roc_auc_score(y_test,y_pred_log)}")
        print(f"RF Model: {roc_auc_score(y_test,y_pred_rf)}")
        print(f"GB Model: {roc_auc_score(y_test,y_pred_gb)}")

    joblib.dump(log_model, 'models\\winner_model')

    return


def margin_model():

    print("nothing yet")

def total_model():

    print("nothing yet!")