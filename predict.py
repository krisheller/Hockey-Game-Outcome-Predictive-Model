import pandas as pd
import numpy as np

import sqlite3
from pathlib import Path

import datetime as dt
from dateutil import parser
import os

import requests

#Model training
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
#from tensorflow import keras
#from keras import layers

#Model saving
import joblib

#Helper functions
from process import get_winner_model_prediction_data, convert_decimal_to_america

Path('data\db.db').touch()
conn = sqlite3.connect('data\db.db')

Path('data\db_processed.db').touch()
#conn_proc = sqlite3.connect('data\db_processed.db')
conn_proc = sqlite3.connect('data\db_processed.db')
c = conn_proc.cursor()

#1 function to predict all three by calling these 3 saved models
#Should be a classifier that predicts different lines
#As well as over/under total goals

def predict_winners(date=None, safety_threshold = 1.15, verbose=1):
    #This function will predict the winner of a game 
    #Outcome: 0 -> 1 with 1 being 100% certain the home team wins, 0 100% certain the away team wins
    
    #Load the model 
    winner_model = joblib.load('models\\winner_model')
    
    #We want today's games
    if date==None:
        date = str(dt.date.today())

    url = 'https://api-web.nhle.com/v1/schedule/now'
    response = requests.get(url).json()['gameWeek']

    #Load in our betting information
    odds_h2h = pd.read_sql('SELECT * \
                                        FROM odds_h2h', con=conn)
    
    team_dict = pd.read_sql('SELECT * \
                                        FROM team_dictionary', con=conn)
    
    #Df to contain our output analysis
    output_df = pd.DataFrame()
    bets_to_make = []

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
            
                model_input = get_winner_model_prediction_data(home_team,away_team,5)

                home_win_chance = round(100*winner_model.predict_proba(model_input)[0][1],1)

                #What is the odds set for this game?
                home_full_name = team_dict[team_dict['team']==home_team]['team_full_name'].values[0]
                

                try:
                    this_game_home_odds = odds_h2h[(odds_h2h['date'] == date) & (odds_h2h['home_team']==home_full_name)]['home_odds'].values[0]
                    this_game_away_odds = odds_h2h[(odds_h2h['date'] == date) & (odds_h2h['home_team']==home_full_name)]['away_odds'].values[0]
                except:
                    continue



                #Calculate the breakpoint odds considering our risk/safety threshold
                home_profit_breakpoint_odds = 100/home_win_chance
                away_profit_breakpoint_odds = 100/(100-home_win_chance)

                if safety_threshold*home_profit_breakpoint_odds < this_game_home_odds:
                    home_good_bet = 1
                else: home_good_bet = 0

                if safety_threshold*away_profit_breakpoint_odds < this_game_away_odds:
                    away_good_bet = 1
                else: away_good_bet = 0

                if home_win_chance > 50:
                   predicted_winner = home_team
                else: predicted_winner = away_team
                
                home_expected_profit = (round((this_game_home_odds * home_win_chance),1) / 100)
                away_expected_profit = (round(this_game_away_odds * (100-home_win_chance),1) / 100)

                temp = {
                    'Matchup': [f'{home_team} vs. {away_team}'],
                    'Predicted Winner': [predicted_winner],
                    'Home Win %': [home_win_chance],
                    'Away Win %': [100-home_win_chance],
                    'Home Odds':[f'{convert_decimal_to_america(this_game_home_odds)} / {this_game_home_odds:.2f}'],
                    'Away Odds':[f'{convert_decimal_to_america(this_game_away_odds)} / {this_game_away_odds:.2f}'],
                    'Home Expected Profit':[f'{home_expected_profit} units'],
                    'Away Expected Profit':[f'{away_expected_profit} units'],
                    'Home ML Good Bet?': [home_good_bet],
                    'Away ML Good Bet?': [away_good_bet],
                }

                temp = pd.DataFrame.from_dict(temp)

                output_df = pd.concat([output_df,temp])

                if home_good_bet:
                    bets_to_make.append(f'{home_team} at {convert_decimal_to_america(safety_threshold*home_profit_breakpoint_odds)} or better')

                if away_good_bet:
                    bets_to_make.append(f'{away_team} at {convert_decimal_to_america(safety_threshold*away_profit_breakpoint_odds)} or better')


                #if home_win_chance > 50:
                #    print(f"{away_team} @ [{home_team}] | {round(100-home_win_chance,1)} - {home_win_chance}")
                #else: print(f"[{away_team}] @ {home_team} | {round(100-home_win_chance,1)} - {home_win_chance}")

    if verbose:
        print(output_df)

        #Summarize the bets to be made
        if(output_df['Home ML Good Bet?'].sum() + output_df['Away ML Good Bet?'].sum() > 0):
            print("\nSummary of bets to make:")
            for bet in bets_to_make:
                print(bet)

    return output_df

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

    training_data = pd.read_sql('SELECT * FROM training_data_5', con=conn_proc)
    training_data.drop(columns=['game_id','index'],inplace=True)

    #Split out x vs. y
    target_columns = [col for col in training_data if (('hometeam' in col) or ('total' in col))]    
    x = training_data.drop(columns=target_columns)
    y = training_data['hometeam_winner']
        
    split_cols = [col.replace('home_','') for col in x.columns if 'home_' in col]

    delta_df = pd.DataFrame()
    #delta_df.columns = split_cols

    for i, game in x.iterrows():
        val_list = []
        for col in split_cols:
            val_list.append(game['home_'+col] - game['away_'+col])

        temp = pd.DataFrame([val_list])

        delta_df = pd.concat([delta_df, temp],ignore_index=True)

    delta_df.columns = split_cols

    delta_df['hometeam_winner'] = y
    
    #print(delta_df.groupby(by=['is_winner']).mean())
    corr_table = delta_df.corr()['hometeam_winner'].sort_values(ascending=False)

    delta_df.drop(columns='hometeam_winner',inplace=True)
    #Drop some features with very low correlation to is_winner
    #Adding saves here because its going to be very highly correlated to shots allowed and goals allowed
    
    low_corr_cols = corr_table[abs(corr_table) < .05].index.tolist()
    print(low_corr_cols)
    delta_df.drop(columns=low_corr_cols,inplace=True)
    
    x_train, x_test, y_train, y_test = train_test_split(delta_df, y, test_size = 0.2, shuffle=True)

    print("Fitting models...")
    
    log_model = LogisticRegression(max_iter=3000).fit(x_train, y_train)
    rf_model = RandomForestClassifier().fit(x_train, y_train)
    gb_model = GradientBoostingClassifier(n_estimators=500).fit(x_train, y_train)
    
    '''

    #Neural net
    model = keras.Sequential([
        layers.Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],1)),
        layers.Dense(x_train.shape[1]*12, activation='relu'),
        layers.Dropout(.5),
        layers.Dense(x_train.shape[1]*12, activation='relu'),
        layers.Dropout(.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.build()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    x_train_main, x_train_val = x_train[0:12000],x_train[12001:]
    y_train_main, y_train_val = y_train[0:12000],y_train[12001:]

    print(model.summary())
    callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
    model.fit(x_train_main,y_train_main,epochs=20,validation_data=(x_train_val,y_train_val),verbose=1,callbacks=[callback])


    print("done")
    '''

    y_pred_log = log_model.predict(x_test)
    y_pred_log = [round(y) for y in y_pred_log]

    y_pred_rf = rf_model.predict(x_test)
    y_pred_rf = [round(y) for y in y_pred_rf]

    y_pred_gb = gb_model.predict(x_test)
    y_pred_gb = [round(y) for y in y_pred_gb]

    if verbose:
        print("Accuracy Scores")
        print(f"Log Model: {accuracy_score(y_test,y_pred_log)}")
        print(f"RF Model: {accuracy_score(y_test,y_pred_rf)}")
        print(f"GB Model: {accuracy_score(y_test,y_pred_gb)}")

        print("F1 Scores")
        print(f"Log Model: {f1_score(y_test,y_pred_log)}")
        print(f"RF Model: {f1_score(y_test,y_pred_rf)}")
        print(f"GB Model: {f1_score(y_test,y_pred_gb)}")

        print("AUC Scores")
        print(f"Log Model: {roc_auc_score(y_test,y_pred_log)}")
        print(f"RF Model: {roc_auc_score(y_test,y_pred_rf)}")
        print(f"GB Model: {roc_auc_score(y_test,y_pred_gb)}")

    joblib.dump(gb_model, 'models\\winner_model')
    

    return

def margin_model(verbose=False):
    #This function will predict how much the home team wins (or loses if -) by 

    #Pull all the data we're using from the games table
    games = pd.read_sql('SELECT * \
                                     FROM games_adv_scaled', con=conn_proc)
    
    #Drop any nas that could come in
    games.dropna(axis=0,inplace=True)
    
    #Split it up to use for training
    x = games.drop(['index','game_id','game_length_mins','home_team','away_team','home_win','home_win_margin','total_goals'],axis=1)
    x_cols = [col for col in x.columns if 'after' not in col]
    x = x[x_cols]
    x = x.reindex(sorted(x.columns), axis=1)

    #Target variable
    y = games['home_win_margin']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True)

    linear_model = LinearRegression().fit(x_train, y_train)
    rf_model = RandomForestRegressor().fit(x_train, y_train)
    gb_model = GradientBoostingRegressor().fit(x_train, y_train)
    #Neural net

    y_pred_lin = linear_model.predict(x_test)
    y_pred_lin = [round(y) for y in y_pred_lin]


    y_pred_rf = rf_model.predict(x_test)
    y_pred_rf = [round(y) for y in y_pred_rf]

    y_pred_gb = gb_model.predict(x_test)
    y_pred_gb = [round(y) for y in y_pred_gb]

    if verbose:
        print("Mean Square Error")
        print(f"Linear Model: {mean_squared_error(y_test, y_pred_lin, squared=False)}")
        print(f"Random Forest Model: {mean_squared_error(y_test, y_pred_rf, squared=False)}")
        print(f"GB Model: {mean_squared_error(y_test, y_pred_gb, squared=False)}")
        
    joblib.dump(linear_model, 'models\\margin_model')

    return

def total_model(verbose=False):
    #This function will predict how many total goals are scored in the match

    #Pull all the data we're using from the games table
    games = pd.read_sql('SELECT * \
                                     FROM games_adv_scaled', con=conn_proc)
    
    #Drop any nas that could come in
    games.dropna(axis=0,inplace=True)
    
    #Split it up to use for training
    x = games.drop(['index','game_id','game_length_mins','home_team','away_team','home_win','home_win_margin','total_goals'],axis=1)
    x_cols = [col for col in x.columns if 'after' not in col]
    x = x[x_cols]
    x = x.reindex(sorted(x.columns), axis=1)

    #Target variable
    y = games['total_goals']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True)

    print(x_train.shape)

    linear_model = LinearRegression().fit(x_train, y_train)
    rf_model = RandomForestRegressor().fit(x_train, y_train)
    gb_model = GradientBoostingRegressor().fit(x_train, y_train)
    #Neural net

    y_pred_lin = linear_model.predict(x_test)
    y_pred_lin = [round(y) for y in y_pred_lin]


    y_pred_rf = rf_model.predict(x_test)
    y_pred_rf = [round(y) for y in y_pred_rf]

    y_pred_gb = gb_model.predict(x_test)
    y_pred_gb = [round(y) for y in y_pred_gb]

    if verbose:
        print("Mean Square Error")
        print(f"Linear Model: {mean_squared_error(y_test, y_pred_lin, squared=False)}")
        print(f"Random Forest Model: {mean_squared_error(y_test, y_pred_rf, squared=False)}")
        print(f"GB Model: {mean_squared_error(y_test, y_pred_gb, squared=False)}")
        
    joblib.dump(linear_model, 'models\\total_model')

    return

#Predictive model to determine the xG for a given shot
def expected_goals(verbose=False):

    #Load in the shots database
    shots = pd.read_sql("SELECT * FROM shots \
                        WHERE home_team_side != 'N/A'", con=conn)
    games = pd.read_sql("SELECT game_id, home_team FROM games", con=conn)

    print(shots.shape)
    #Let's do it by game
    game_list = list(set(shots['game_id'].tolist()))

    for game_id in game_list:
        home_team = games[games['game_id']==game_id]['home_team'].values[0]

        home_shots = shots[(shots['game_id']==game_id) & (shots['shooter_team']==home_team)]
        away_shots = shots[(shots['game_id']==game_id) & (shots['shooter_team']!=home_team)]
        
        for shot in home_shots:
            if shot['home_team_side'] == 'right':
                shot['x'] = -shot['x']
                shot['y'] = -shot['y']

        for shot in home_shots:
            if shot['home_team_side'] == 'right':
                shot['x'] = -shot['x']
                shot['y'] = -shot['y']


        
       # if game_id == 2023020774:
            
            
        

        #Calculate the time since the last shot from the same team

        #Calculate the location 


        #Calculate the distance from net

    
    #print(shots)


    return

def get_distance_to_goal(x,y,zone):

    print("nothing yet")
    return

#expected_goals()

#winner_model(verbose=1)
#margin_model(verbose=1)
#total_model(verbose=1)
#predict_winners()
#predict_winners(safety_threshold=1.1)