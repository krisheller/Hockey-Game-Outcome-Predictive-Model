#Process the raw data that we have available to us
import requests
import json
import os
import datetime as dt
from dateutil import parser

import pandas as pd 
import numpy as np
import seaborn as sns

import sqlite3
from pathlib import Path

from collections import Counter

from sklearn.preprocessing import StandardScaler
import joblib





####
#Todo

#We should probably make  function that can get each team's last n opponents point % 
#Should be getting goals allowed and goals scored for last 5 not just goals scored
#Other variables that may be good to incorporate
    #expected goals per game - assign expected goals value to each shot taken
    #power play minutes per game last 5
    #power play % last 5
    #product of ppl5 and opponent pp l5 ?
    #high danger shot/save%?

###################
#Primary Functions#
###################

#Aggregate totals from events for each game
def aggregate_games_data(overwrite_db=False):
    #Raw data
    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')

    #Processed data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()

    #Determine which games need to be aggregated
    try:
        processed_games = pd.read_sql('SELECT game_id \
                                        FROM games_agg', con=conn_proc)['game_id'].tolist()
        

    
        if(overwrite_db):
            raise Exception("Overwriting database, toss to except statement")
    except:
        overwrite_db = True
        processed_games = []
        c.execute("DROP TABLE IF EXISTS games_agg")

    #Get the full list of games that we already have
    games_full = pd.read_sql('SELECT * \
                    FROM games', con=conn) 

    #Difference is what we have to get the aggregate data for
    games_to_process = list(set(games_full['game_id'].tolist())-set(processed_games))
    games_to_process.sort()
    
    #If there's nothing to do, we just return
    if len(games_to_process) == 0:
        return 
    
    #Now we are going to want to get aggregate stats for each team
    games = games_full[games_full['game_id'].isin(games_to_process)].copy()
    games.drop('index',axis=1,inplace=True)
    games.sort_values(by='game_id',inplace=True)

    #Dataframe to store it all
    games_agg = pd.DataFrame()
  
    #Have to pass in the list of game IDs into the SQL query as comma-sep list
    game_list = ', '.join(str(game_id) for game_id in games['game_id'].tolist())

    #The different stats we will want to count
    shots = pd.read_sql(f"SELECT game_id, result, COUNT(result) AS count, shooter_team, goalie_team \
                        FROM shots \
                        WHERE game_id in ({game_list}) \
                        GROUP BY game_id, shooter_team, result", con=conn)
    
    faceoffs = pd.read_sql(f"SELECT game_id, winner_team, COUNT(winner_team) AS count\
                        FROM faceoffs \
                        WHERE game_id in ({game_list}) \
                        GROUP by game_id, winner_team", con=conn)
    
    giveaways = pd.read_sql(f"SELECT game_id, giver_team, COUNT(giver_team) AS count\
                        FROM giveaways \
                        WHERE game_id in ({game_list}) \
                        GROUP by game_id, giver_team", con=conn)
    
    takeaways = pd.read_sql(f"SELECT game_id, taker_team, COUNT(taker_team) AS count\
                        FROM takeaways \
                        WHERE game_id in ({game_list}) \
                        GROUP by game_id, taker_team", con=conn)
    
    hits = pd.read_sql(f"SELECT game_id, hitter_team, COUNT(hitter_team) AS count\
                        FROM hits \
                        WHERE game_id in ({game_list}) \
                        GROUP by game_id, hitter_team", con=conn)

    '''
    Forgot blocks???
    '''
    
    penalties = pd.read_sql(f"SELECT game_id, committed_team, COUNT(committed_team) AS count, SUM(duration) as duration\
                        FROM penalties \
                        WHERE game_id in ({game_list}) AND infraction != 'N/A' \
                        GROUP by game_id, committed_team", con=conn)
    
    #Now iterate over each game and assign based on these 
    for game_id in games['game_id'].tolist():

        home_team = games[games['game_id']==game_id]['home_team'].iloc[0]
        
        #Data from shots table
        home_shots = shots[(shots['shooter_team']==home_team) & (shots['game_id']==game_id)]['count'].sum()
        away_shots = shots[(shots['shooter_team']!=home_team) & (shots['game_id']==game_id)]['count'].sum()

        home_shots_on_goal = shots[(shots['shooter_team']==home_team) & (shots['game_id']==game_id) & (shots['result']!='miss')]['count'].sum()
        away_shots_on_goal = shots[(shots['shooter_team']!=home_team) & (shots['game_id']==game_id) & (shots['result']!='miss')]['count'].sum()

        home_saves = shots[(shots['goalie_team']==home_team) & (shots['result']=='save') & (shots['game_id']==game_id)]['count'].sum()
        away_saves = shots[(shots['goalie_team']!=home_team) & (shots['result']=='save') & (shots['game_id']==game_id)]['count'].sum()

        #Data from faceoffs table
        home_faceoff_wins = faceoffs[(faceoffs['game_id']==game_id) & (faceoffs['winner_team']==home_team)]['count'].sum()
        away_faceoff_wins = faceoffs[(faceoffs['game_id']==game_id) & (faceoffs['winner_team']!=home_team)]['count'].sum()

        home_faceoffs = faceoffs[(faceoffs['game_id']==game_id)]['count'].sum()
        away_faceoffs = faceoffs[(faceoffs['game_id']==game_id)]['count'].sum()

        #Data from giveaways table
        home_giveaways = giveaways[(giveaways['game_id']==game_id) & (giveaways['giver_team']==home_team)]['count'].sum()
        away_giveaways = giveaways[(giveaways['game_id']==game_id) & (giveaways['giver_team']!=home_team)]['count'].sum()

        #Data from takeaways table
        home_takeaways = takeaways[(takeaways['game_id']==game_id) & (takeaways['taker_team']==home_team)]['count'].sum()
        away_takeaways = takeaways[(takeaways['game_id']==game_id) & (takeaways['taker_team']!=home_team)]['count'].sum()

        #Data from hits table
        home_hits = hits[(hits['game_id']==game_id) & (hits['hitter_team']==home_team)]['count'].sum()
        away_hits = hits[(hits['game_id']==game_id) & (hits['hitter_team']!=home_team)]['count'].sum()

        #Penalties table
        home_penalties = penalties[(penalties['game_id']==game_id) & (penalties['committed_team']==home_team)]['count'].sum()
        home_penalties_duration = penalties[(penalties['game_id']==game_id) & (penalties['committed_team']==home_team)]['duration'].sum()
        away_penalties = penalties[(penalties['game_id']==game_id) & (penalties['committed_team']!=home_team)]['count'].sum()
        away_penalties_duration = penalties[(penalties['game_id']==game_id) & (penalties['committed_team']!=home_team)]['duration'].sum()

        #Define the target variables
        winning_team = games[games['game_id']==game_id]['winner'].values[0]
        if winning_team == home_team:
            hometeam_winner = 1
        else: hometeam_winner = 0

        total_goals = games[games['game_id']==game_id]['home_score'].values[0] + games[games['game_id']==game_id]['away_score'].values[0]
        hometeam_winning_margin = games[games['game_id']==game_id]['home_score'].values[0] - games[games['game_id']==game_id]['away_score'].values[0]

        #Setup some over/unders while we're in here
        total_goals_over_twopointfive = int(total_goals > 2.5)
        total_goals_over_threepointfive = int(total_goals > 3.5)
        total_goals_over_fourpointfive = int(total_goals > 4.5)
        total_goals_over_fivepointfive = int(total_goals > 5.5)
        total_goals_over_sixpointfive = int(total_goals > 6.5)
        total_goals_over_sevenpointfive = int(total_goals > 7.5)
        total_goals_over_eightpointfive = int(total_goals > 8.5)
        total_goals_over_ninepointfive = int(total_goals > 9.5)

        hometeam_plus_threepointfive = int(hometeam_winning_margin > -3.5)
        hometeam_plus_twopointfive = int(hometeam_winning_margin > -2.5)
        hometeam_plus_onepointfive = int(hometeam_winning_margin > -1.5)
        hometeam_plus_zeropointfive = int(hometeam_winning_margin > -0.5)
        hometeam_minus_zeropointfive = int(hometeam_winning_margin > 0.5)
        hometeam_minus_onepointfive = int(hometeam_winning_margin > 1.5)
        hometeam_minus_twopointfive = int(hometeam_winning_margin > 2.5)
        hometeam_minus_threepointfive = int(hometeam_winning_margin > 3.5)


        target_vars = [hometeam_winner,total_goals, hometeam_winning_margin, total_goals_over_twopointfive,total_goals_over_threepointfive,
                       total_goals_over_fourpointfive, total_goals_over_fivepointfive, total_goals_over_sixpointfive, total_goals_over_sevenpointfive, total_goals_over_eightpointfive, total_goals_over_ninepointfive,
                       hometeam_plus_threepointfive,hometeam_plus_twopointfive,
                       hometeam_plus_onepointfive, hometeam_plus_zeropointfive,
                       hometeam_minus_zeropointfive, hometeam_minus_onepointfive,
                       hometeam_minus_twopointfive, hometeam_minus_threepointfive]
        
        target_vars_name_list = ['hometeam_winner','total_goals', 'hometeam_winning_margin', 'total_goals_over_twopointfive',
                                 'total_goals_over_threepointfive','total_goals_over_fourpointfive', 'total_goals_over_fivepointfive', 'total_goals_over_sixpointfive', 'total_goals_over_sevenpointfive', 'total_goals_over_eightpointfive', 'total_goals_over_ninepointfive','hometeam_plus_threepointfive','hometeam_plus_twopointfive','hometeam_plus_onepointfive', 'hometeam_plus_zeropointfive','hometeam_minus_zeropointfive', 'hometeam_minus_onepointfive','hometeam_minus_twopointfive', 'hometeam_minus_threepointfive']

        #Setup lists to turn into df
        var_list = [game_id, home_shots,away_shots,home_shots_on_goal, away_shots_on_goal,home_saves,away_saves,home_faceoffs,away_faceoffs,
                    home_faceoff_wins,away_faceoff_wins,home_giveaways,away_giveaways, home_takeaways, away_takeaways, home_hits, away_hits,
                    home_penalties, home_penalties_duration,away_penalties,away_penalties_duration]
        
        var_name_list = ['game_id','home_shots','away_shots','home_shots_on_goal','away_shots_on_goal','home_saves','away_saves','home_faceoffs','away_faceoffs','home_faceoff_wins',
                         'away_faceoff_wins','home_giveaways','away_giveaways','home_takeaways','away_takeaways','home_hits','away_hits',
                         'home_penalties','home_penalties_duration','away_penalties','away_penalties_duration']
        
        var_list = var_list + target_vars
        var_name_list = var_name_list + target_vars_name_list
        
        games_agg = add_to_df(var_list, var_name_list,games_agg)


    #Finally merge with the full dataframe 
    games = games.merge(games_agg, on='game_id')

    #Add to our database
    games.to_sql('games_agg',con=conn_proc,if_exists='append')

    return
    
#Function to split each game into data for each individual team to more easily access later on
def split_agg_games(overwrite_db=False):

    #Processed data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()

    #Determine which games need to be split
    try:
        split_games = pd.read_sql('SELECT game_id \
                                        FROM games_split', con=conn_proc)['game_id'].tolist()
    
        if(overwrite_db):
            raise Exception("Overwriting database, toss to except statement")
    except:
        overwrite_db = True
        split_games = []
        c.execute("DROP TABLE IF EXISTS games_split")

    #Get the full list of games that we already have
    agg_games = pd.read_sql('SELECT * \
                    FROM games_agg', con=conn_proc) 

    #Difference is what we have to get the split data for
    games_to_split = list(set(agg_games['game_id'].tolist())-set(split_games))
    games_to_split.sort()
    
    #If there's nothing to do, we just return
    if len(games_to_split) == 0:
        return 
    
    #Games that we'll be iterating over
    games = agg_games[agg_games['game_id'].isin(games_to_split)].copy()
    games.drop('index',axis=1,inplace=True)
    games.sort_values(by='game_id',inplace=True)

    #Empty df to store data
    games_split = pd.DataFrame()

    for game_id in games['game_id'].tolist():

        #Define the teams
        home_team = games[games['game_id']==game_id]['home_team'].values[0]
        away_team = games[games['game_id']==game_id]['away_team'].values[0]
        teams = [home_team, away_team]
        
        #Information we're going to want to keep for both entries
        season = games[games['game_id']==game_id]['season'].values[0]
        date = games[games['game_id']==game_id]['date'].values[0]
        start_time = games[games['game_id']==game_id]['start_time'].values[0]
        tz = games[games['game_id']==game_id]['tz'].values[0]
        full_time = games[games['game_id']==game_id]['full_time'].values[0]
        game_length_mins = games[games['game_id']==game_id]['game_length_mins'].values[0]

        columns = games.columns
        #Now iterate over the teams for their specific variables
        for i in range(len(teams)):
            if i == 0:
                prefix = 'home_'
            else: prefix = 'away_'

            columns_subset = [col for col in columns if prefix in col]

            #Drop a couple that won't matter
            del columns_subset[0]
            del columns_subset[1]

            #Add a few that we don't have but could be helpful
            if i == 0:
                is_home = 1
                team = home_team
                goals_allowed = games[games['game_id']==game_id]['away_score'].values[0]
                if games[games['game_id']==game_id]['winner'].values[0] == home_team:
                    is_winner = 1
                else: is_winner = 0
            else: 
                is_home = 0
                team = away_team
                goals_allowed = games[games['game_id']==game_id]['home_score'].values[0]
                if games[games['game_id']==game_id]['winner'].values[0] != home_team:
                    is_winner = 1
                else: is_winner = 0

            #Select the  columns from the full game that will be helpful
            temp = games[games['game_id']==game_id][games.columns.intersection(columns_subset)]

            #reformat the columns
            for i in range(len(columns_subset)):
                columns_subset[i] = columns_subset[i].replace(prefix,'')

            temp.columns = columns_subset

            #Our shared variables
            labels = ['game_id','team','season','date','start_time','tz','full_time','game_length_mins','is_home','is_winner','goals_allowed']
            vals = [game_id,team,season,date,start_time,tz,full_time,game_length_mins,is_home,is_winner,goals_allowed]

            #Add in the ones from the temp dataframe
            labels = labels + columns_subset
            vals = vals + temp.iloc[0,].values.tolist()

            games_split = add_to_df(vals,labels,games_split)

    games_split.to_sql('games_split',con=conn_proc,if_exists='append')

    return

#Next is to setup the historical stats behind each game 
def aggregate_last_n(last_n=None,overwrite_db=False):

    #Load in the split data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()

    if last_n != None:
        table_name = f'games_input_last_{last_n}'
    else: table_name = 'games_input_full_season'

    #Determine which games need to be processed
    try:
        processed_games = pd.read_sql(f'SELECT game_id \
                                        FROM {table_name}', con=conn_proc)['game_id'].tolist()
    
        if(overwrite_db):
            raise Exception("Overwriting database, toss to except statement")
    except:
        overwrite_db = True
        processed_games = []
        c.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    #processed_games = set(processed_games)

    #Now the full set of games
    games_split = pd.read_sql('SELECT * \
                            FROM games_split', con=conn_proc)
    games_split_ids = set(games_split['game_id'].tolist())

    game_list = games_split_ids - set(processed_games)
    games = games_split[games_split['game_id'].isin(game_list)].copy()

    #Empty dataframe to store
    df = pd.DataFrame()

    #Columns we don't want to get the averages for
    drop_cols = ['index','season']
    
    for i, game in games.iterrows():
        game_id = game['game_id']
        team = game['team']
        season = game['season']
        date = game['date']

        prior = games[(games['game_id'] < game_id) & (games['team']==team) & (games['season'] == season)]

        #If we need to we can pull from the last season, unless its the first season then we have to just skip this loop
        if last_n != None:
            if prior.shape[0] < last_n:
                if season == 2010:
                    continue
                else:
                    #Get the team's info from the last season as a baseline
                    prior_season = games[(games['game_id'] < game_id) & (games['team']==team) & (games['season'] == season-1)]
                    prior_season_avg = prior_season.mean(numeric_only=True).to_frame().T

                    #Resample it to get the prior df to have last_n rows
                    prior_season_avg = pd.concat([prior_season_avg for i in range(last_n-prior.shape[0])])
                    prior = pd.concat([prior, prior_season_avg])

        #Do a cutoff if we need
        if last_n != None:
            prior = prior.tail(last_n)

        #Now get the averages
        prior_avg = prior.mean(numeric_only=True).to_frame().T

        #Drop the columns that we don't need
        prior_avg.drop(columns=drop_cols,inplace=True)

        #Get the current points % for the team, looking back to earlier df that's not edited
        prior_pts_streak = games[(games['game_id'] < game_id) & (games['team']==team) & (games['season'] == season)]
        
        #Calculate some figures
        total_avail_pts = prior_pts_streak.shape[0]*2
        num_wins = prior_pts_streak['is_winner'].sum()
        num_otl = prior_pts_streak[(prior_pts_streak['full_time']!='REG') & (prior_pts_streak['is_winner']==0)].shape[0]
        total_points = num_wins*2 + num_otl
        point_pct = total_points/total_avail_pts
        
        #Get the win (or loss) streak
        streak_broken = False
        streak = 0
        while prior_pts_streak.shape[0] > 0 and not streak_broken:
            last_game = prior_pts_streak.iloc[-1]['is_winner']
            if last_game == 0:
                if streak > 0:
                    streak_broken = True
                else: 
                    streak -= 1
                    prior_pts_streak = prior_pts_streak.head(prior_pts_streak.shape[0]-1)
            else: 
                if streak < 0:
                    streak_broken = True
                else: 
                    streak += 1
                    prior_pts_streak = prior_pts_streak.head(prior_pts_streak.shape[0]-1)

        #Did the team play the day before?
        prior_pts_streak = games[(games['game_id'] < game_id) & (games['team']==team) & (games['season'] == season)]

        if prior_pts_streak.shape[0]==0:
            had_game_yesterday=0
        else:
            if (parser.parse(date)-parser.parse(prior_pts_streak.tail(1)['date'].values[0])).days > 1:
                had_game_yesterday=0
            else: had_game_yesterday=1

        #Assign these
        prior_avg['streak'] = streak
        prior_avg['point_pct'] = point_pct
        prior_avg['total_points'] = total_points
        prior_avg['had_game_yesterday'] = had_game_yesterday

        #Assign this game id and team
        prior_avg['game_id'] = game_id
        prior_avg['team'] = team
        prior_avg['date'] = date

        #Add to our df
        df = pd.concat([df,prior_avg])

    #Before saving to sql, we should only keep games if we have data from both sides (all ids should appear twice)
    df = df[df['game_id'].map(df['game_id'].value_counts()) > 1]

    #Save out to sql with appending / replacement
    df.to_sql(f'{table_name}',con=conn_proc,if_exists='append')

    return

#Generate the training data that we will be using as inputs
def generate_training_data(last_n=None, overwrite_db=False):

    #This will take in the split games table from SQL and generate corresponding training data for each game

    #Key difference with chance for improvement: do not infer any data from previous season
    
    #Load in the split data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()

    #This will be the name of our data table
    #Last n will add columns for recent trends in addition to the full season
    if last_n != None:
        table_name = f'training_data_{last_n}'
    else: table_name = 'training_data_full_season'


    #Pull in our data
    games_split = pd.read_sql(f'SELECT * \
                        FROM games_split', con=conn_proc)
    
    
    #This contains our target variables
    games_agg = pd.read_sql(f'SELECT * \
                        FROM games_agg', con=conn_proc)
    
    #Have we already done any of this?
    try:
        games_done_already = pd.read_sql(f'SELECT game_id, team, date \
                        FROM {table_name}', con=conn_proc)
        
        dates_processed = games_done_already['date']
        dates_processed.sort()

        last_game_date = dates_processed[-1]
        games_to_process = games_split[not (games_split['game_id'].isin(games_done_already['game_id'].tolist())) 
                                       & (games_split['date'] >= last_game_date)]
        
        if(overwrite_db):
            raise Exception("Overwriting database, toss to except statement")
        
    #If we don't have any games processed already, don't 
    except:
        games_to_process = games_split.copy()
        c.execute(f"DROP TABLE IF EXISTS {table_name}")
        
    #Get the list of game IDs that we are going to be processing
    game_list = list(set(games_to_process['game_id'].tolist()))
    game_list.sort()
    
    #Empty dataframe that will hold our values
    data = pd.DataFrame()

    #These are the columns we are not going to calculate an average for
    non_avg_columns = ['game_id','team','season','date','start_time','tz','full_time','game_length_mins','is_home','is_winner']


    #Iterate over these games and pull the data we need
    for game in game_list:
        team_stats = games_to_process[(games_to_process['game_id']==game)].sort_values(by='is_home',ascending=False)
        
        #Check that we have a full record here
        if team_stats.shape[0] != 2:
            continue
        
        date = team_stats['date'].iloc[0]

        #Now we can go back and get this teams stats up to this date
        home_season = get_team_current_season_stats(team_stats['team'].iloc[0],date,games_split)
        away_season = get_team_current_season_stats(team_stats['team'].iloc[1],date,games_split)

        if home_season.shape[0] < 5 or away_season.shape[0] < 5:
            continue

        #Empty dataframe to store info to add to larger 'df'
        temp = pd.DataFrame()
            
        for frame in [home_season, away_season]:
            #print(frame)
            #We won't bother if there's nothing to pull from in this season
            if frame.shape[0] < 1:
                continue
            
            #Or if we dont have enough to pull from
            if last_n:
                if frame.shape[0] < last_n:
                    continue
            
            #Calculate some variables
            total_points = get_total_points(frame[['full_time','is_winner']])
            point_pct = total_points / (frame.shape[0]*2)

            #Determine if the team had a game on the day before
            last_game_date = frame.tail(1)['date'].values[0]
            days_between = (parser.parse(date) - parser.parse(last_game_date)).days
            if days_between == 1:
                had_game_yesterday=1
            else:
                had_game_yesterday=0
            
            avg_frame = frame.drop(columns=non_avg_columns)
            avg_frame = avg_frame.mean().to_frame().T

            avg_frame['total_points'] = total_points
            avg_frame['point_pct'] = point_pct
            avg_frame['had_game_yesterday'] = had_game_yesterday

            #Save pct
            avg_frame['save_pct'] = (avg_frame['saves'].sum()/avg_frame['shots_allowed'].sum())

            #Faceoff win pct
            avg_frame['faceoff_win_pct'] = (avg_frame['faceoff_wins'].sum()/avg_frame['faceoffs'].sum())
    
            #Now we do the last_n stuff
            if last_n:
                frame_last_n = frame.tail(last_n)

                #Calculate some variables
                total_points_last_n = get_total_points(frame_last_n[['full_time','is_winner']])
                point_pct_last_n = total_points_last_n / (frame_last_n.shape[0]*2)

                avg_frame_last_n = frame_last_n.drop(columns=non_avg_columns)
                avg_frame_last_n = avg_frame_last_n.mean().to_frame().T

                avg_frame_last_n[f'total_points'] = total_points_last_n
                avg_frame_last_n[f'point_pct'] = point_pct_last_n

                #Save pct
                avg_frame_last_n[f'save_pct'] = (avg_frame_last_n['saves'].sum()/avg_frame_last_n['shots_allowed'].sum())

                #Faceoff win pct
                avg_frame_last_n[f'faceoff_win_pct'] = (avg_frame_last_n['faceoff_wins'].sum()/avg_frame_last_n['faceoffs'].sum())
                
                #Setup columns
                avg_frame_last_n_col_list = [col+f"_last_{last_n}" for col in avg_frame_last_n]
                avg_frame_last_n.columns = avg_frame_last_n_col_list

                avg_frame = pd.concat([avg_frame, avg_frame_last_n],axis=1)


            #Add this to our temp frame
            temp = pd.concat([temp, avg_frame],axis=1)

        #SKip this if we don't have enough
        if temp.shape[0] < 1:
            continue

        #Now we come in and add things that apply to both
        hometeam_winner = games_agg[(games_agg['game_id']==game)]['hometeam_winner'].values[0]
        total_goals = games_agg[(games_agg['game_id']==game)]['total_goals'].values[0]
        hometeam_winning_margin = games_agg[(games_agg['game_id']==game)]['hometeam_winning_margin'].values[0]

        temp['hometeam_winner'] = hometeam_winner
        temp['total_goals'] = total_goals
        temp['hometeam_winning_margin'] = hometeam_winning_margin
        temp['game_id'] = game

        #print(temp)

        if temp.shape[1] == 4 + avg_frame.shape[1]*2:
            data = pd.concat([data,temp])


        #print(data)
        
    #Rework the columns for the file
    home_cols = ['home_'+col for col in avg_frame.columns]
    away_cols = ['away_'+col for col in avg_frame.columns]

    #print(data)
    output_cols = home_cols + away_cols

    data_leftover_cols = data.columns[len(output_cols):].tolist()
    
    final_cols = output_cols + data_leftover_cols


    data.columns = final_cols

    data.to_sql(f'{table_name}',if_exists='append',con=conn_proc)
                
        
    return data

#Pull and format the data for model training on the 'winner' model
#This will assume the game is today for any team that is fed into it
def get_winner_model_prediction_data(home_team, away_team,last_n,split=False):

    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()

    #Pull in our data
    games_split = pd.read_sql(f'SELECT * \
                        FROM games_split', con=conn_proc)
    
    date = str(dt.date.today())
    
    home_stats = get_team_current_season_stats(home_team,date,games_split)
    away_stats = get_team_current_season_stats(away_team,date,games_split)

    #add later what to do if it's early in the aseason
    
    #Helper to hold data 
    temp = pd.DataFrame()

    #Columns we will drop
    non_avg_columns = ['game_id','team','season','date','start_time','tz','full_time','game_length_mins','is_home','is_winner']

    for frame in [home_stats,away_stats]:
        avg_frame = frame.drop(columns=non_avg_columns)
        avg_frame = avg_frame.mean().to_frame().T

        #Calculate some variables
        total_points = get_total_points(frame[['full_time','is_winner']])
        point_pct = total_points / (frame.shape[0]*2)

        #Determine if the team had a game on the day before
        last_game_date = frame.tail(1)['date'].values[0]
        days_between = (parser.parse(date) - parser.parse(last_game_date)).days
        if days_between == 1:
            had_game_yesterday=1
        else:
            had_game_yesterday=0
            

        avg_frame['total_points'] = total_points
        avg_frame['point_pct'] = point_pct
        avg_frame['had_game_yesterday'] = had_game_yesterday

        #Save pct
        avg_frame['save_pct'] = (avg_frame['saves'].sum()/avg_frame['shots_allowed'].sum())

        #Faceoff win pct
        avg_frame['faceoff_win_pct'] = (avg_frame['faceoff_wins'].sum()/avg_frame['faceoffs'].sum())

        frame_last_n = frame.tail(last_n)

        #Calculate some variables
        total_points_last_n = get_total_points(frame_last_n[['full_time','is_winner']])
        point_pct_last_n = total_points_last_n / (frame_last_n.shape[0]*2)

        avg_frame_last_n = frame_last_n.drop(columns=non_avg_columns)
        avg_frame_last_n = avg_frame_last_n.mean().to_frame().T

        avg_frame_last_n[f'total_points'] = total_points_last_n
        avg_frame_last_n[f'point_pct'] = point_pct_last_n

        #Save pct
        avg_frame_last_n[f'save_pct'] = (avg_frame_last_n['saves'].sum()/avg_frame_last_n['shots_allowed'].sum())

        #Faceoff win pct
        avg_frame_last_n[f'faceoff_win_pct'] = (avg_frame_last_n['faceoff_wins'].sum()/avg_frame_last_n['faceoffs'].sum())
        
        #Setup columns
        avg_frame_last_n_col_list = [col+f"_last_{last_n}" for col in avg_frame_last_n]
        avg_frame_last_n.columns = avg_frame_last_n_col_list

        avg_frame = pd.concat([avg_frame, avg_frame_last_n],axis=1)

        temp = pd.concat([temp, avg_frame],axis=0)

    if split:
        print(temp)
        
    input = (temp.iloc[0] - temp.iloc[1]).to_frame().T
    
    #Now we're dropping specific inputs based on what we've done in our model traingin
    low_corr_cols = ['faceoff_wins_last_5', 'faceoff_win_pct_last_5', 'faceoffs', 'takeaways', 'faceoffs_last_5', 'takeaways_last_5', 
                     'penalties', 'giveaways_last_5', 'save_pct_last_5', 'hits', 'penalties_last_5', 'giveaways', 'penalties_duration', 
                     'penalties_duration_last_5', 'hits_last_5','total_points','total_points_last_5']
    
    input.drop(columns=low_corr_cols,inplace=True)

    

    return input


##################
#Helper Functions#
##################

#Helper function to assign teams based on the sql queries made in aggregate_games_data()
def assign_teams(home_team, away_team, value_counts):

    if value_counts.shape[0] == 2:
        if value_counts.index[0] == home_team:
            return [value_counts[0],value_counts[1]]
        else: return [value_counts[1],value_counts[0]]
    elif value_counts.shape[0] == 1:
        if value_counts.index[0] == home_team:
            return [value_counts[0],0]
        else: return [0,value_counts[0]]
    else: return [0,0]

#Helper function to get what season it is from the date
def get_season_from_date(date):
    
    date = parser.parse(date)

    if date.month > 7:
        season = date.year
    else:
        season = date.year-1

    return season

#Helper function to grab a team's stats from the current season up to a given date
    #Note: pass in the full splits dataframe to avoid multiple sql pulls
def get_team_current_season_stats(team, date, df):

    season = get_season_from_date(date)

    games = df[(df['team']==team) & (df['season']==season) & (df['date'] < date)].copy()
    games.sort_values(by='date',inplace=True)
    games.drop(columns='index',inplace=True)
    games.reset_index(drop=True,inplace=True)

    #Add in some other helpful values

    #shots allowed per game
    shots_allowed = df[(df['team']!=team) & (df['game_id'].isin(games['game_id'].tolist()))]['shots_on_goal']
    shots_allowed.reset_index(drop=True,inplace=True)
    games['shots_allowed'] = shots_allowed

    return games

#Helper function to grab a team's stats from the previous season in total
     #Note: pass in the full splits dataframe to avoid multiple sql pulls
def get_team_prev_season_stats(team, date, df):

    season = get_season_from_date(date)-1
    
    games = df[(df['team']==team) & (df['season']==season)]
    games.sort_values(by='date',inplace=True)
    games.drop(columns='index',inplace=True)
    games.reset_index(drop=True,inplace=True)

    #Add in some other helpful values

    #shots allowed per game
    shots_allowed = df[(df['team']!=team) & (df['game_id'].isin(games['game_id'].tolist()))]['shots']
    shots_allowed.reset_index(drop=True,inplace=True)
    games['shots_allowed'] = shots_allowed
    
    return games

#Helper function to determine a team's total points given their outcomes
def get_total_points(outcome_df):

    win_points = outcome_df['is_winner'].sum()*2
    otl_points = outcome_df[(outcome_df['is_winner']==0) & (outcome_df['full_time']!='REG')].shape[0]

    return win_points + otl_points

#Helper function to convert from decimal to american style odds
def convert_decimal_to_america(odds):

    if odds > 2:
        odds = "+"+str(round((odds - 1)*100))
    else:
        odds = str(round((-100)/(odds-1)))
    
    return odds

#Helper function to returns the list of current nhl teams
def get_list_of_teams():

    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()

    team_list = pd.read_sql("select distinct home_team from games_agg \
                                where season >= 2015 \
                                group by home_team \
                                having count(home_team) > 10", con=conn_proc).tolist()

    return team_list

#Function to set new scalers for variables
def set_scalers():

    #Processed data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')
    c = conn_proc.cursor()
    
    data = pd.read_sql('SELECT * \
                                FROM games_input_last_5', con=conn_proc)
    
    #Columns that we're not going to scale
    non_scale = ['index', 'game_id', 'team', 'date', 'game_length_mins', 'is_home', 'is_winner',
                 'had_game_yesterday','point_pct','total_points']
    
    scale_cols = [col for col in data.columns if col not in non_scale]
    data_scale = data[data.columns.intersection(scale_cols)]

    for col in scale_cols:
        scaler = StandardScaler().fit(np.array(data[col]).reshape(-1,1))
        joblib.dump(scaler, f'data\\scalers\\{col}')

    return

#Helper function to ease creation of dataframes in larger functions
def add_to_df(var_list, var_name_list, df):
    temp = {}
    for i in range(len(var_list)):
        temp[var_name_list[i]]=[var_list[i]]
    
    df = pd.concat([df, pd.DataFrame.from_dict(temp)])

    return df

