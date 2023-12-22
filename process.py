#Process the raw data that we have available to us
import requests
import json
import os
import datetime as dt

import pandas as pd 
import numpy as np
import seaborn as sns

import sqlite3
from pathlib import Path


from sklearn.preprocessing import StandardScaler
Path('data\db.db').touch()
conn = sqlite3.connect('data\db.db')
c = conn.cursor()

####
#Todo

#We should probably make  function that can get each team's last n opponents point % 
#Should be getting goals allowed and goals scored for last 5 not just goals scored
#Other variables that may be good to incorporate
    #expected goals per game
    #whether or not its a back to back game?
    #power play minutes per game last 5
    #power play % last 5
    #penalty minutes last 5
    #giveaways/takeaways pg last 5
    #faceoff win % last 5
    #blocks pg last 5
    #product of ppl5 and opponent pp l5 ?
    #goal differential?

#Some of this stuff can be taken from the skater_games and goalie_games files but things like team penalty mins or # of power plays is going to have to be calculated.

#Function to determine if players are happening on the powerplay or shorthanded
#Maybe it also creates a full table of all powerplay situations and whether or not a goal is scored?
def process_special_teams(overwrite_db=False):

    #Get the games we already have
    try:
        special_teams = pd.read_sql('SELECT id \
                                    FROM special_teams',con=conn)['id'].tolist()
        
        if overwrite_db:
            raise Exception("Overwriting special teams db")
        
    except:
        special_teams = []

    #For now we're going to cut down the list for testing
    all_games = pd.read_sql("SELECT id \
                            FROM games \
                            LIMIT 1",con=conn)['id'].tolist()

    
    
    
    #This is the list of games that we need to get the data for
    game_list = list(set(all_games) - set(special_teams))
    game_list.sort()

    #If there's nothing to do, exit
    if len(game_list) == 0:
        print("No special teams to process.")
        return
    
    #Let's get all the penalties committed in the games of interest
    query_list = ', '.join(str(game) for game in game_list)
    
    penalties = pd.read_sql(f'SELECT * \
                            FROM penalties \
                            WHERE game_id in ({query_list})',con=conn)
    
    #We also need to get the games to know who is home and who is away 
    games = pd.read_sql(f'SELECT * \
                            FROM games \
                            WHERE id in ({query_list})',con=conn)
    

    #Drop penalties that don't actually result in powerplays
    penalties = penalties[penalties['duration']!='N/A'].reset_index(drop=True)
    penalties.drop(['index'],axis=1,inplace=True)

    #Now we iterate over each of these penalties and update the state
    special_teams = pd.DataFrame()

    for game in game_list:
        #New df for each game
        game_st = pd.DataFrame()

        #Let's get the home team and away teams
        home_team = games[games['id']==game]['home_team'].item()
        away_team = games[games['id']==game]['away_team'].item()

        temp = penalties[penalties['game_id']==game]
        
        #We're always going to start the game in 5v5
        current_time = 1200
        current_period = 1
        home_skaters_off_ice = 0
        away_skaters_off_ice = 0
        period = 1

        game_st = pd.concat([game_st,pd.DataFrame.from_dict({
            'game_id':[game],
            'home_team':[home_team],
            'away_team':[away_team],
            'period':[period],
            'situation_start_time':[current_time],
            'home_skaters_off_ice':[home_skaters_off_ice],
            'away_skaters_off_ice':[away_skaters_off_ice],
            'home_skaters_off_ice_diff':[home_skaters_off_ice-away_skaters_off_ice]           
        })])


        #Now within the game, we have to iterate over every penalty and set a new state for each one that occurs
        for i,penalty in temp.iterrows():
            
            #New start time will just be the time the penalty is committed
            current_time = penalty['time_remaining']
            period = penalty['period']
            
            #Determine who is losing a skater
            if penalty['committed_team'] == home_team:
                home_skaters_off_ice += 1
            else: away_skaters_off_ice += 1

            game_st = pd.concat([game_st,pd.DataFrame.from_dict({
                'game_id':[game],
                'home_team':[home_team],
                'away_team':[away_team],
                'period':[period],
                'situation_start_time':[current_time],
                'home_skaters_off_ice':[home_skaters_off_ice],
                'away_skaters_off_ice':[away_skaters_off_ice],
                'home_skaters_off_ice_diff':[home_skaters_off_ice-away_skaters_off_ice]            
            })])

            #Have to figure out how to handle multiple penalties committed within a similar period
            #Second-by-second is probably too intensive and gets confusing when adding in OT
            #situation start is preferable but we have to figure out some kind of sorting 
            #Does this require recursion?

            #Check if another penalty is committed within this special teams period
            #Account for period changeovers
            situation_end_time = current_time - int(penalty['duration'])
            if situation_end_time < 0:
                situation_end_time += 1200
                period += 1

            if i+1 < len(temp):
                if temp.loc[i+1, 'time_remaining'] > situation_end_time and period == temp.loc[i+1, 'period']:
                    print("concurrent")

        special_teams = pd.concat([special_teams, game_st])

    return special_teams, penalties

#Helper Function to get situation status based on # of players on ice
def get_situation_label(home, away):
    home = int(home)
    away = int(away)
    if home > away:
        return 'SH'
    elif away > home:
        return 'PP'
    else:
        return 'EV'

def process_games_data(n=None, last_n_games = 10, overwrite_db=False):
    #This will pull the data that we have in our sql database and setup to get ready for predictions
    #n: total number of games to pull
    #last_n_games: will look back this many games for each team to generate stats
    
    
    #See what games don't need to be replaced
    try:
        processed_games = pd.read_sql('SELECT id \
                                     FROM games_adv', con=conn)['id'].tolist()
        
        if(overwrite_db):
            raise Exception("Overwriting database, toss to except statement")
    except:
        processed_games = []

    #Get the list of games from the games database
    if n != None:
        all_games = pd.read_sql(f'SELECT * \
                            FROM games \
                            WHERE season >= 2015 \
                            ORDER BY id DESC \
                            LIMIT {n}', con=conn)
    else:
        all_games = pd.read_sql(f'SELECT * \
                            FROM games \
                            WHERE season >= 2015 \
                            ORDER BY id DESC', con=conn)
        
    #Get the difference between the two for the new games that we need to get advanced stats for
    game_list = list(set(all_games['id'].tolist()) - set(processed_games))

    #If we're up to date, don't do anything else!
    if len(game_list) == 0:
        print("Advanced game stats table up to date.")
        return

    games = all_games[all_games['id'].isin(game_list)]

    for i,game in games.iterrows():

        #Re-index and sort the games so they're moving forward in time
        games.drop(['index'], axis=1, inplace=True)
        games.sort_values(by='id', ascending=True, inplace=True)
        games.reset_index(inplace=True, drop=True)

        #we can quickly calculate our target variables we're going to eventually try to predict
        games['home_win'] = np.where(games['home_team'] == games['winner'], 1, 0)
        games['home_win_margin'] = games['home_score'] - games['away_score']
        games['total_goals'] = games['home_score'] + games['away_score']
        
        #Let's just worry about training the model on regular season games played home/away
        games = games.loc[(games['neutral_site']==0) & (games['playoffs']==0)]

        #Drop columns we don't need anymore
        drop_cols = ['start_time','tz','link', 'winner','neutral_site','playoffs','round','series_game','top_seed','bottom_seed','home_wins','away_wins']
        games.drop(drop_cols, axis=1, inplace=True)

        away = game['away_team']
        home = game['home_team']
        date = game['date']
    
        #Now let's calculate some team metrics we want to know

        #Shooting %  & shots per 60 last N games



        #Save % last N games




    
    
    start_game = 200




    #This is for giving teams away wins
    switch_dict = {0:1,
                1:0}

    for i, game in games.iterrows():

        #We'll not count in the first 100 games of the season to make sure each team has ~5 games in their history already
        #if i < start_game: 
        #    continue



        #Get the last 
        home_dict = games[((games['home_team']==home) | (games['away_team']==home)) & (games['date'] < date)].iloc[-last_n_games:,]
        away_dict = games[((games['home_team']==away) | (games['away_team']==away)) & (games['date'] < date)].iloc[-last_n_games:,]
        
        home_gpg_last_n = 0
        away_gpg_last_n = 0
        home_point_pct_last_n = 0
        away_point_pct_last_n = 0

        for j, g in home_dict.iterrows():
            if home == g['home_team']:
                home_gpg_last_n += g['home_score']/last_n_games
                home_point_pct_last_n += 2*g['home_win']/last_n_games
                if g['full_time'] != 'REG' and g['home_win'] == 0:
                    home_point_pct_last_n += g['home_win']/last_n_games
            else:
                home_gpg_last_n += g['away_score']/last_n_games
                home_point_pct_last_n += 2*switch_dict[g['home_win']]/last_n_games
                if g['full_time'] != 'REG' and g['home_win'] == 1:
                    home_point_pct_last_n += switch_dict[g['home_win']]/last_n_games

        for j, g in away_dict.iterrows():
            if away == g['home_team']:
                away_gpg_last_n += g['home_score']/last_n_games
                away_point_pct_last_n += 2*g['home_win']/last_n_games
                if g['full_time'] != 'REG' and g['home_win'] == 0:
                    away_point_pct_last_n += g['home_win']/last_n_games
            else:
                away_gpg_last_n += g['away_score']/last_n_games
                away_point_pct_last_n += 2*switch_dict[g['home_win']]/last_n_games
                if g['full_time'] != 'REG' and g['home_win'] == 1:
                    away_point_pct_last_n += switch_dict[g['home_win']]/last_n_games

        games.loc[i ,'home_gpg_last_n'] = home_gpg_last_n
        games.loc[i ,'away_gpg_last_n'] = away_gpg_last_n
        games.loc[i ,'home_point_pct_last_n'] = home_point_pct_last_n
        games.loc[i ,'away_point_pct_last_n'] = away_point_pct_last_n

    games = games.iloc[start_game-1:,]
    games.dropna(axis=0, how='any')

    gpg_list = games['home_gpg_last_n'].tolist() + games['away_gpg_last_n'].tolist()
    ppg_list = games['home_point_pct_last_n'].tolist() + games['away_point_pct_last_n'].tolist()

    gpg_mean = np.average(gpg_list)
    gpg_sd = np.std(gpg_list)
    ppg_mean = np.average(ppg_list)
    ppg_sd = np.std(ppg_list)

    games['home_gpg_last_n'] = (games['home_gpg_last_n'] - gpg_mean)/gpg_sd
    games['away_gpg_last_n'] = (games['away_gpg_last_n'] - gpg_mean)/gpg_sd
    games['home_point_pct_last_n'] = (games['home_point_pct_last_n'] - ppg_mean)/ppg_sd
    games['away_point_pct_last_n'] = (games['away_point_pct_last_n'] - ppg_mean)/ppg_sd

    #games['home_gpg_diff'] = games['home_gpg_last_n'] - games['away_gpg_last_n']
    #games['home_ppg_diff'] = games['home_point_pct_last_n'] - games['away_point_pct_last_n']

    #We'll also normalize the columns for win margin and total_goals
    games[['home_win_margin']] = StandardScaler().fit_transform(games[['home_win_margin']])
    games[['total_goals']] = StandardScaler().fit_transform(games[['total_goals']])

    #Drop the helper columns we made along hte way
    #helper_cols = ['home_gpg_last_n','away_gpg_last_n','home_point_pct_last_n','away_point_pct_last_n']
    #games.drop(helper_cols,axis=1, inplace=True)

    #Drop the other columns that are just giving us info at this point and nothing predictive
    info_cols = ['season','date','home_team','full_time','home_score','away_score','away_team']
    games.drop(info_cols,axis=1,inplace=True)

    return games

print(process_special_teams())