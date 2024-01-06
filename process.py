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
import joblib





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
    #win/loss streak
    #high danger shot/save%?





def process_games_data(n=None, last_n_games = 10, overwrite_db=False):
    #This will pull the data that we have in our sql database and setup to get ready for predictions
    #n: total number of games to pull
    #last_n_games: will look back this many games for each team to generate stats
    
    #Raw data
    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')

    #Processed data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')

    try:
        processed_games = pd.read_sql('SELECT game_id \
                                     FROM games_adv', con=conn_proc)['game_id'].tolist()
        
        if(overwrite_db):
            raise Exception("Overwriting database, toss to except statement")
    except:
        overwrite_db = True
        processed_games = []

        #We erase the games_adv table from the database
        conn_proc.close()
        os.remove('data\db_processed.db')
        
        #Start a new one
        Path('data\db_processed.db').touch()
        conn_proc = sqlite3.connect('data\db_processed.db')
        d = conn_proc.cursor()

    #Get the list of games from the games database
    if n != None:
        games = pd.read_sql(f'SELECT * \
                            FROM games \
                            ORDER BY game_id ASC \
                            LIMIT {n}', con=conn)
    else:
        games = pd.read_sql(f'SELECT * \
                            FROM games \
                            ORDER BY game_id ASC', con=conn)
        


    #Re-index and sort the games so they're moving forward in time
    games.drop(['index'], axis=1, inplace=True)
    games.sort_values(by='game_id', ascending=True, inplace=True)
    games.reset_index(inplace=True, drop=True)

    #we can quickly calculate our target variables we're going to eventually try to predict
    games['home_win'] = np.where(games['home_team'] == games['winner'], 1, 0)
    games['home_win_margin'] = games['home_score'] - games['away_score']
    games['total_goals'] = games['home_score'] + games['away_score']
    
    #Let's just worry about training the model on regular season games
    games = games.loc[(games['playoffs']==0)]

    #Drop columns we don't need anymore
    drop_cols = ['start_time','tz','link', 'winner','neutral_site','playoffs','round','series_game','top_seed','bottom_seed','home_wins','away_wins']
    games.drop(drop_cols, axis=1, inplace=True)
    
    #Get the difference between the two for the new games that we need to get advanced stats for
    game_list = list(set(games['game_id'].tolist()) - set(processed_games))

    ##This is breaking when trying to update because it's not pulling all the games data to draw from

    #If we're up to date, don't do anything else!
    if len(game_list) == 0:
        print("Advanced game stats table up to date.")
        return

    #Tracking: get the time now
    now = dt.datetime.now()

    print(len(game_list))

    #Iterate over each game and collect stats about the state before it
    for i, game in games.iterrows():
        
        #Only care about the games that we don't already have data for
        if game['game_id'] not in game_list:
            continue

        away = game['away_team']
        home = game['home_team']
        season = game['season']
        id = game['game_id']

        #Get each team's point percentage prior to the game being played
        home_prior = games[(games['season']==season) & (games['game_id']<id) & ((games['home_team']==home) | (games['away_team']==home))]
        away_prior = games[(games['season']==season) & (games['game_id']<id) & ((games['home_team']==away) | (games['away_team']==away))]

        ######################################
        #Point percentages prior to this game#
        ######################################
        home_total_points = 0
        away_total_points = 0

        try:
            for j, g in home_prior.iterrows():
                if g['home_win'] == 1 and g['home_team']==home or g['home_win'] == 0 and g['home_team']!=home:
                   home_total_points += 2
                elif g['full_time'] != 'REG':
                    home_total_points += 1
            home_point_pct = home_total_points / (home_prior.shape[0]*2) 
        except: home_point_pct = np.nan #Handles the first game of the season

        try:
            for j, g in away_prior.iterrows():
                if g['home_win'] == 1 and g['home_team']==away or g['home_win'] == 0 and g['home_team']!=away:
                    away_total_points += 2
                elif g['full_time'] != 'REG':
                    away_total_points += 1
            away_point_pct = away_total_points / (away_prior.shape[0]*2) 
        except: away_point_pct = np.nan #Handles the first game of the season

        games.loc[i,'home_point_pct'] = home_point_pct
        games.loc[i,'away_point_pct'] = away_point_pct
 
        #Might as well throw in the "afters" as well since we will need them at some point later
        if game['home_win'] == 1:
            home_total_points += 2
            if game['full_time'] != 'REG':
                away_total_points+=1
        else:
            away_total_points += 2
            if game['full_time'] != 'REG':
                home_total_points+=1

        games.loc[i,'home_point_pct_after'] = home_total_points / ((home_prior.shape[0]+1)*2)
        games.loc[i,'away_point_pct_after'] = away_total_points / ((away_prior.shape[0]+1)*2)

        ##############################################################################################################################
        #Shooting %, goals, shots per 60 (can calculate total game time from the shots table, last shot ends the game for total time)#
        ##############################################################################################################################

        #First home team
        if home_prior.shape[0] > 0:
            
            #Get all the shots from the games in the prior games list
            id_list = ', '.join(str(id) for id in home_prior['game_id'].tolist())
               
            total_shots = pd.read_sql(f'SELECT COUNT(result)  \
                                    FROM shots \
                                    WHERE game_id in ({id_list}) AND shooter_team == "{home}"', con=conn).iloc[0,0]
            
            total_shots_against = pd.read_sql(f'SELECT COUNT(result)  \
                                    FROM shots \
                                    WHERE game_id in ({id_list}) AND shooter_team != "{home}" AND result != "miss"', con=conn).iloc[0,0]

            total_goals = pd.read_sql(f'SELECT COUNT(result)  \
                        FROM shots \
                        WHERE game_id in ({id_list}) AND shooter_team == "{home}" AND result == "goal"', con=conn).iloc[0,0]
            
            total_goals_against = pd.read_sql(f'SELECT COUNT(result)  \
                                    FROM shots \
                                    WHERE game_id in ({id_list}) AND shooter_team != "{home}" AND result == "goal"', con=conn).iloc[0,0]
            
            total_giveaways = pd.read_sql(f'SELECT COUNT(event_id)  \
                                    FROM giveaways \
                                    WHERE game_id in ({id_list}) AND giver_team == "{home}"', con=conn).iloc[0,0]
            
            total_takeaways = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM takeaways \
                        WHERE game_id in ({id_list}) AND taker_team == "{home}"', con=conn).iloc[0,0]
            
            total_penalty_mins = pd.read_sql(f'SELECT SUM(duration)  \
                        FROM penalties \
                        WHERE game_id in ({id_list}) AND committed_team == "{home}"', con=conn).iloc[0,0]
            
            total_faceoffs = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM faceoffs \
                        WHERE game_id in ({id_list})', con=conn).iloc[0,0]
            
            total_faceoff_wins = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM faceoffs \
                        WHERE game_id in ({id_list}) AND winner_team == "{home}"', con=conn).iloc[0,0]
            
            total_blocks = pd.read_sql(f'SELECT SUM(event_id)  \
                        FROM blocks \
                        WHERE game_id in ({id_list}) AND blocker_team == "{home}"', con=conn).iloc[0,0]
            
            total_hits = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM hits \
                        WHERE game_id in ({id_list}) AND hitter_team == "{home}"', con=conn).iloc[0,0]
            

            #Get the total minutes to calculate rates            
            total_minutes = pd.read_sql(f'SELECT SUM(game_length_mins)  \
                        FROM games \
                        WHERE game_id in ({id_list})', con=conn).iloc[0,0]




            try:
                #Calculate all the rate stats
                home_shots_per_60, home_goals_per_60 = 60*total_shots/total_minutes,60*total_goals/total_minutes
                home_shots_against_per_60 = 60*total_shots_against/total_minutes
                home_takeaways_per_60, home_giveaways_per_60 = 60*total_takeaways/total_minutes, 60*total_giveaways/total_minutes
                home_pen_mins_per_60 = total_penalty_mins/total_minutes
                home_hits_per_60, home_blocks_per_60 = 60*total_hits/total_minutes, 60*total_blocks/total_minutes

                home_shooting_pct = 100*total_goals/total_shots
                home_save_pct = 100-(100*total_goals_against/total_shots_against)
                home_faceoff_pct = 100*total_faceoff_wins/total_faceoffs
            except:
                home_shooting_pct,home_save_pct, home_faceoff_pct = np.nan, np.nan, np.nan

        else: 
            home_shots_per_60, home_goals_per_60 = np.nan, np.nan
            home_shooting_pct, home_save_pct = np.nan, np.nan
            home_shots_against_per_60 = np.nan
            home_takeaways_per_60, home_giveaways_per_60 = np.nan, np.nan
            home_pen_mins_per_60 = np.nan
            home_hits_per_60, home_blocks_per_60 = np.nan, np.nan
            home_faceoff_pct = np.nan

        games.loc[i,'home_shooting_pct'] = home_shooting_pct
        games.loc[i,'home_shots_per_60'] = home_shots_per_60
        games.loc[i,'home_goals_per_60'] = home_goals_per_60
        games.loc[i,'home_save_pct'] = home_save_pct
        games.loc[i,'home_shots_against_per_60'] = home_shots_against_per_60
        games.loc[i,'home_giveaways_per_60'] = home_giveaways_per_60
        games.loc[i,'home_takeaways_per_60'] = home_takeaways_per_60
        games.loc[i,'home_pen_mins_per_60'] = home_pen_mins_per_60
        games.loc[i,'home_hits_per_60'] = home_hits_per_60
        games.loc[i,'home_faceoff_pct'] = home_faceoff_pct
        games.loc[i,'home_blocks_per_60'] = home_blocks_per_60

        #Now away team
        if away_prior.shape[0] > 0:
            
            #Get all the shots from the games in the prior games list
            id_list = ', '.join(str(id) for id in away_prior['game_id'].tolist())
               
            total_shots = pd.read_sql(f'SELECT COUNT(result)  \
                                    FROM shots \
                                    WHERE game_id in ({id_list}) AND shooter_team == "{away}"', con=conn).iloc[0,0]
            
            total_shots_against = pd.read_sql(f'SELECT COUNT(result)  \
                                    FROM shots \
                                    WHERE game_id in ({id_list}) AND shooter_team != "{away}" AND result != "miss"', con=conn).iloc[0,0]

            total_goals = pd.read_sql(f'SELECT COUNT(result)  \
                        FROM shots \
                        WHERE game_id in ({id_list}) AND shooter_team == "{away}" AND result == "goal"', con=conn).iloc[0,0]
            
            total_goals_against = pd.read_sql(f'SELECT COUNT(result)  \
                                    FROM shots \
                                    WHERE game_id in ({id_list}) AND shooter_team != "{away}" AND result == "goal"', con=conn).iloc[0,0]
            
            total_giveaways = pd.read_sql(f'SELECT COUNT(event_id)  \
                                    FROM giveaways \
                                    WHERE game_id in ({id_list}) AND giver_team == "{away}"', con=conn).iloc[0,0]
            
            total_takeaways = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM takeaways \
                        WHERE game_id in ({id_list}) AND taker_team == "{away}"', con=conn).iloc[0,0]
            
            total_penalty_mins = pd.read_sql(f'SELECT SUM(duration)  \
                        FROM penalties \
                        WHERE game_id in ({id_list}) AND committed_team == "{away}"', con=conn).iloc[0,0]
            
            total_faceoffs = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM faceoffs \
                        WHERE game_id in ({id_list})', con=conn).iloc[0,0]
            
            total_faceoff_wins = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM faceoffs \
                        WHERE game_id in ({id_list}) AND winner_team == "{away}"', con=conn).iloc[0,0]
            
            total_blocks = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM blocks \
                        WHERE game_id in ({id_list}) AND blocker_team == "{away}"', con=conn).iloc[0,0]
            
            total_hits = pd.read_sql(f'SELECT COUNT(event_id)  \
                        FROM hits \
                        WHERE game_id in ({id_list}) AND hitter_team == "{away}"', con=conn).iloc[0,0]
            

            #Get the total minutes to calculate rates            
            total_minutes = pd.read_sql(f'SELECT SUM(game_length_mins)  \
                        FROM games \
                        WHERE game_id in ({id_list})', con=conn).iloc[0,0]


            try:
                #Calculate all the rate stats
                away_shots_per_60, away_goals_per_60 = 60*total_shots/total_minutes,60*total_goals/total_minutes
                away_shots_against_per_60 = 60*total_shots_against/total_minutes
                away_takeaways_per_60, away_giveaways_per_60 = 60*total_takeaways/total_minutes, 60*total_giveaways/total_minutes
                away_pen_mins_per_60 = total_penalty_mins/total_minutes
                away_hits_per_60, away_blocks_per_60 = 60*total_hits/total_minutes, 60*total_blocks/total_minutes

                away_shooting_pct = 100*total_goals/total_shots
                away_save_pct = 100-(100*total_goals_against/total_shots_against)
                away_faceoff_pct = 100*total_faceoffs/total_faceoff_wins
            except:
                away_shooting_pct,away_save_pct, away_faceoff_pct = np.nan, np.nan, np.nan

        else: 
            away_shots_per_60, away_goals_per_60 = np.nan, np.nan
            away_shooting_pct, away_save_pct = np.nan, np.nan
            away_shots_against_per_60 = np.nan
            away_takeaways_per_60, away_giveaways_per_60 = np.nan, np.nan
            away_pen_mins_per_60 = np.nan
            away_hits_per_60, away_blocks_per_60 = np.nan, np.nan
            away_faceoff_pct = np.nan

        games.loc[i,'away_shooting_pct'] = away_shooting_pct
        games.loc[i,'away_shots_per_60'] = away_shots_per_60
        games.loc[i,'away_goals_per_60'] = away_goals_per_60
        games.loc[i,'away_save_pct'] = away_save_pct
        games.loc[i,'away_shots_against_per_60'] = away_shots_against_per_60
        games.loc[i,'away_giveaways_per_60'] = away_giveaways_per_60
        games.loc[i,'away_takeaways_per_60'] = away_takeaways_per_60
        games.loc[i,'away_pen_mins_per_60'] = away_pen_mins_per_60
        games.loc[i,'away_hits_per_60'] = away_hits_per_60
        games.loc[i,'away_faceoff_pct'] = away_faceoff_pct
        games.loc[i,'away_blocks_per_60'] = away_blocks_per_60

        if (i+1)%25 == 0:
            updated_time = dt.datetime.now()
            time_delta = updated_time - now
            pct_complete = 100*((i+1)/len(game_list))
            anticipated_length = (100/pct_complete) * time_delta
            anticipated_finish = now + anticipated_length
            print(f"Done with {i+1} of {len(game_list)} games {round(pct_complete,1)}%; anticipated finish: {anticipated_finish}")
    
    
    #Drop the other columns that are just giving us info at this point and nothing predictive
    info_cols = ['season','date','home_team','full_time','home_score','away_score','away_team']
    games.drop(info_cols,axis=1,inplace=True)



    #Let's add the difference to a database
    #Only keep the ones that were in the list of games we're pulling
    games = games[games['game_id'].isin(game_list)]

    if overwrite_db:
        games.to_sql('games_adv',conn_proc,if_exists='replace')
        overwrite_db = False
    else:
        games.to_sql('games_adv',conn_proc,if_exists='append')

    return games

def fill_na_adv_data():
    #Replace the NAs with the appropriate values calculated from other data

    ##########
    print("nothing yet!")

def scale_adv_data():
#Scaling all the variables
    #margin_scaler = StandardScaler().fit(np.array(games[['home_win_margin']]).reshape(-1,1))
    #total_scaler = StandardScaler().fit(np.array(games[['total_goals']]).reshape(-1,1))

    #Save the scalers out to transform back to raw numbers later on; only do this if we're working with hte large dataset
    #if overwrite_db:
    #    joblib.dump(margin_scaler, 'models\\margin_scaler')
    #    joblib.dump(total_scaler, 'models\\total_scaler')

    #games[['home_win_margin']] = margin_scaler.transform(np.array(games['home_win_margin']).reshape(-1,1))
    #games[['total_goals']] = total_scaler.transform(np.array(games['total_goals']).reshape(-1,1))
    print("nothing yet!")



def get_team_data(team, last_n_games=5):

    #Read in data for this team from SQL
    team_data = pd.read_sql(f"SELECT *\
                            FROM games \
                            WHERE home_team='{team}' OR away_team='{team}' \
                            ORDER BY id DESC \
                            LIMIT {last_n_games}", con=conn)
    
    #Was this team home or away in their last game? Did they win?
    last_game_id = team_data.iloc[-1,1]
    if team_data.iloc[-1,8]==team:
        team_point_pct = pd.read_sql(f"SELECT home_point_pct_after\
                                FROM games_adv \
                                WHERE id = {last_game_id}", con=conn).values[0,0]
    else:
        team_point_pct = pd.read_sql(f"SELECT away_point_pct_after\
                        FROM games_adv \
                        WHERE id = {last_game_id}", con=conn).values[0,0]
        
    total_points = 0
    total_goals = 0
    
    for i,game in team_data.iterrows():
        if game['home_team'] == team:
            total_goals += game['home_score']
        else: total_goals += game['away_score']
            
        if game['winner'] == team:
            total_points += 2
        elif game['full_time'] != 'REG':
            total_points += 1

    ppg = total_points/(last_n_games*2)
    gpg = total_goals/last_n_games


    df = pd.DataFrame.from_dict({'gpg_last_n':[gpg],
                                'point_pct_last_n':[ppg],
                                'point_pct':[team_point_pct]})

    return df

