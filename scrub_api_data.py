import requests

import os
import shutil
import datetime as dt
from dateutil import parser

import pandas as pd 
from collections import Counter 

import sqlite3
from pathlib import Path

def get_game_summaries(replace_table=False):
    #Capture the outcomes and other data (date, home/away, etc. about each game)

    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')
    c = conn.cursor()

    #If the table already exists in SQL, we won't bother
    try:
        most_recent_game = pd.read_sql('SELECT MAX(id) \
                    FROM games', con=conn).iloc[0,0]
        
        most_recent_game_date = parser.parse(pd.read_sql(f'SELECT date \
                                            FROM games \
                                            WHERE id = {most_recent_game} \
                                            LIMIT 1', con=conn).iloc[0,0])
        
        #We'll go back 1 month just to be sure we didn't miss any games 
        date = (most_recent_game_date - dt.timedelta(days=32)).date()
        
        if(replace_table):
            raise Exception("Replacing historical 'games' table in SQL database...")
    except:
        #We'd start in 2012 to replace all if we break above
        date = dt.date(2011, 1, 1)

    #Run through today
    end_date = dt.date.today()
    
    #Df to hold the data
    games = pd.DataFrame()
    save_every_n_games = 500

    #df holding full list of games we have in the db
    try:
        full_game_list = pd.read_sql('SELECT id \
                                     FROM games', con=conn)['id'].tolist()
        
    except:
        full_game_list = []
    #This is the list of games for this year that we already have

    game_count = 0
    while date <= end_date + dt.timedelta(days=6):
        
        url = f'https://api-web.nhle.com/v1/schedule/{date.year}-{date.month:02d}-{date.day:02d}'
        
        try:
            response = requests.get(url).json()
        
            #Iterate over each day in the week
            for game_date in response['gameWeek']:

                #Check to make sure games were played
                if game_date['numberOfGames'] > 0:
                    for game in game_date['games']:
                        game_count += 1
                        #print(game_date['date'], game['id'], game['gameState'])
                        #Check to see if the game got canceled and that it has already occured
                        if (game['gameScheduleState'] == 'OK' and game['gameState'] == 'OFF'):

                            #Capture all this information on the game
                            id, start_date, start_time, neutral_site, tz = game['id'], game_date['date'], game['startTimeUTC'], game['neutralSite'], game['venueTimezone']
                            home_team, home_score, away_team, away_score = game['homeTeam']['abbrev'], game['homeTeam']['score'], game['awayTeam']['abbrev'], game['awayTeam']['score']
                            winner = home_team if home_score>away_score else away_team
                            season = date.year-1 if date.month < 7 else date.year
                            game_link = game['gameCenterLink']
                            full_time = game['gameOutcome']['lastPeriodType']

                            #Try to pull playoff info if its a playoff game
                            try:
                                round = game['seriesStatus']['round']
                                playoffs = 1
                                series_game = game['seriesStatus']['gameNumberOfSeries']
                                
                                top_seed_id = game['seriesStatus']['topSeedTeamId']
                                if top_seed_id == game['homeTeam']['id']:
                                    top_seed, bottom_seed = home_team, away_team
                                else: top_seed, bottom_seed = away_team, home_team
                                

                                if top_seed == home_team:
                                    home_wins, away_wins = game['seriesStatus']['topSeedWins'], game['seriesStatus']['bottomSeedWins']
                                else: home_wins, away_wins = game['seriesStatus']['bottomSeedWins'], game['seriesStatus']['topSeedWins']

                            except:
                                playoffs = 0
                                round = 'N/A'
                                series_game = 'N/A'
                                top_seed = 'N/A'
                                bottom_seed = 'N/A'
                                home_wins = 'N/A'
                                away_wins = 'N/A'

                            
                            if game['id'] not in full_game_list:
                                temp_dict = {'id':[id],
                                        'season':[season],
                                        'date':[start_date],
                                        'neutral_site':[neutral_site],
                                        'start_time':[start_time],
                                        'tz':[tz],
                                        'full_time':[full_time],
                                        'home_team':[home_team],
                                        'home_score':[home_score],
                                        'away_team':[away_team],
                                        'away_score':[away_score],
                                        'winner':[winner],
                                        'playoffs': [playoffs],
                                        'round':[round],
                                        'series_game':[series_game],
                                        'top_seed':[top_seed],
                                        'bottom_seed':[bottom_seed],
                                        'home_wins':[home_wins],
                                        'away_wins':[away_wins],
                                        'link':[f'nhl.com{game_link}']}
                                
                                temp = pd.DataFrame.from_dict(temp_dict)   
                                games = pd.concat([games, temp])

                                games.set_index('id',drop=True)

                        #We write it to SQL database if it's not there, for more recent data we will rely on another function
                        if game_count % save_every_n_games == 0 and game_count > 0:
                            try:
                                games.to_sql('games', conn, if_exists='append')
                            except:
                                pass

                            if not(os.path.isfile('data\games\games.csv')):
                                    games.to_csv('data\games\games.csv', index=False, mode='w', header=True)
                            else:  games.to_csv('data\games\games.csv', index=False, mode='a', header=False)

                            games = pd.DataFrame()

            #Iterate to the next week
            date += dt.timedelta(days=7)
            
        except:
            pass

    #Save out at the end after all is wrapped up
    try:
        games.to_sql('games', conn, if_exists='append')
    except:
        pass

    if not(os.path.isfile('data\games\games.csv')):
            games.to_csv('data\games\games.csv', index=False, mode='w', header=True)
    else:  games.to_csv('data\games\games.csv', index=False, mode='a', header=False)

    #Let's make a backup of the database for good measure
    shutil.copy('data\db.db',f'data\\backups\db_backup_{dt.date.today()}.db')

    return

def get_stats(replace_table=False): 
    #Get stats from each game by each skater and goalie as well as sub-game data (boxscore by period)
    #replace_table -> will make fresh csv files and replace each table in the database

    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')
    c = conn.cursor()

    #If the table already exists in SQL, we won't bother getting all the games we've already pulled
    try:
        game_list_exist = pd.read_sql('SELECT game_id \
                    FROM box_scores', con=conn)['game_id'].to_list()
        
        game_list_full = pd.read_sql('SELECT id \
                                     FROM games', con=conn)['id'].to_list()
        
        games = list(set(game_list_full) - set(game_list_exist))
        games.sort()
        
        if(replace_table):
            raise Exception("Replacing historical plays tables in SQL database...")
        else:
            print(f"Adding data to tables starting with game id: {games[0]}")
    
    except:
        print("Replacing historical plays tables in SQL database...")
        games = pd.read_sql('SELECT id \
                    FROM games', con=conn)['id'].to_list()
        
    #As housecleaning, we need to follow orders about starting from scratch or not
    if(replace_table):
        #Events
        for f in os.listdir('data\events'):
            if not f.endswith(".csv"):
                continue
            os.remove(os.path.join('data\events', f))

        #Box scores
        for f in os.listdir('data\games'):
            if not f.endswith('ox_scores.csv') and not f.endswith('_games.csv'):
                continue
            os.remove(os.path.join('data\games', f))

        #SQL tables
        sql_tables = ['shots','faceoffs','penalties','hits','giveaways','takeaways','blocks','box_scores']
        for table in sql_tables:
            c.execute(f"DROP TABLE IF EXISTS {table}")
    

    
    #Now we go over the list of games we're looking to pull and start getting info
    shots_df = pd.DataFrame()
    faceoffs_df = pd.DataFrame()
    penalties_df = pd.DataFrame()
    hits_df = pd.DataFrame()
    giveaways_df = pd.DataFrame()
    takeaways_df = pd.DataFrame()
    blocks_df = pd.DataFrame()

    box_score = pd.DataFrame()

    skater_stats = pd.DataFrame()
    goalie_stats = pd.DataFrame()

    save_every_n_games = 5
    takeaway_count = 0
    for idx, id in enumerate(games):

        #Url for play-by-play data, this gets us things like hits, shots, etc.
        url = f'https://api-web.nhle.com/v1/gamecenter/{id}/play-by-play'
        
        try:
            response = requests.get(url).json()

            #Save the team ID-abbrev pairs for easy reference
            team_dict = {response['homeTeam']['id']:response['homeTeam']['abbrev'],
                            response['awayTeam']['id']:response['awayTeam']['abbrev']}
            
            event_list = response['plays']

            #Iterate over the tracked plays in the game
            for event in event_list:
                
                #Capture some things that are available for every type of event
                event_id = event['eventId']
                period = event['periodDescriptor']['number']
                try:
                    home_team_side = event['homeTeamDefendingSide']
                except:
                    home_team_side = 'N/A'

                try:
                    x = event['details']['xCoord']
                    y = event['details']['yCoord']
                    zone = event['details']['zoneCode']
                except:
                    x = 'N/A'
                    y = 'N/A'
                    zone = 'N/A'
                
                #Format in terms of seconds instead of a string
                time_split = event['timeRemaining'].split(":")
                time_remaining = int(time_split[0])*60 + int(time_split[1])

                #Shots
                if event['typeDescKey'] in ['shot-on-goal','missed-shot','goal']:
                    result = event['typeDescKey']
                    
                    #Reformat a little bit
                    if result == 'shot-on-goal':
                        result = 'save'
                    elif result == 'missed-shot':
                        result = 'miss'

                    try:
                        shot_type = event['details']['shotType']
                    except:
                        shot_type = 'N/A'

                    #different labels for scored goals vs. non-score
                    try:
                        shooter = event['details']['shootingPlayerId']
                    except: 
                        shooter = event['details']['scoringPlayerId']

                    shooter_team = team_dict[event['details']['eventOwnerTeamId']]

                    #if its a goal get the assister(s)
                    if result == 'goal':
                        try:
                            assist1 = event['details']['assist1PlayerId']
                        except:
                            assist1 = 'N/A'
                        try:
                            assist2 = event['details']['assist2PlayerId']
                        except:
                            assist2 = 'N/A'
                    else:
                        assist1 = 'N/A'
                        assist2 = 'N/A'
                    #Accounting for empty net
                    try:
                        goalie = event['details']['goalieInNetId']
                    except:
                        goalie = 0

                    temp = {
                        'game_id':id,
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'result':[result],
                        'shot_type':[shot_type],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                        'shooter':[shooter],
                        'shooter_team':[shooter_team],
                        'goalie':[goalie],
                        'assist1':[assist1],
                        'assist2':[assist2]
                    }
                    shots_df = pd.concat([shots_df, pd.DataFrame.from_dict(temp)])

                #Faceoffs
                elif event['typeDescKey'] == 'faceoff':
                    winner = event['details']['winningPlayerId']
                    loser = event['details']['losingPlayerId']

                    #Get the winning and losing teams
                    winner_team = team_dict[event['details']['eventOwnerTeamId']]
                    if winner_team == list(team_dict.values())[0]:
                        loser_team = list(team_dict.values())[1]
                    else: loser_team = list(team_dict.values())[0]
                    
                    temp = {
                        'game_id':id,
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'winner':[winner],
                        'loser':[loser],
                        'winner_team':[winner_team],
                        'loser_team':[loser_team],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                    }
                    faceoffs_df = pd.concat([faceoffs_df, pd.DataFrame.from_dict(temp)])

                #Penalties
                elif event['typeDescKey'] in ['penalty','delayed-penalty']:
                    #Apparently some penalties are not committed by players
                    try:
                        committed_by = event['details']['committedByPlayerId']
                        severity = event['details']['typeCode']
                        infraction = event['details']['descKey']
                        duration = int(event['details']['duration'])*60
                    except:
                        committed_by = 'N/A'
                        severity = 'N/A'
                        infraction = 'N/A'
                        duration = 'N/A'

                    #Not all penalties are committed against a player
                    try:
                        drawn_by = event['details']['drawnByPlayerId']
                    except:
                        drawn_by = "N/A"

                    #Get the winning and losing teams
                    committed_team = team_dict[event['details']['eventOwnerTeamId']]
                    if committed_team == list(team_dict.values())[0] and drawn_by != "N/A":
                        drawn_team = list(team_dict.values())[1]
                    elif drawn_by != "N/A": drawn_team = list(team_dict.values())[0]
                    else: drawn_team = drawn_by
                    
                    temp = {
                        'game_id':id,
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'committed_by':[committed_by],
                        'drawn_by':[drawn_by],
                        'committed_team':[committed_team],
                        'drawn_team':[drawn_team],
                        'severity':[severity],
                        'infraction':[infraction],
                        'duration':[duration],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                    }
                    penalties_df = pd.concat([penalties_df, pd.DataFrame.from_dict(temp)])
                
                #Hits
                elif event['typeDescKey'] == 'hit':
                    hitter = event['details']['hittingPlayerId']
                    hittee = event['details']['hitteePlayerId']

                    #Get the hitting and hittee teams
                    hitter_team = team_dict[event['details']['eventOwnerTeamId']]
                    if hitter_team == list(team_dict.values())[0]:
                        hittee_team = list(team_dict.values())[1]
                    else: hittee_team = list(team_dict.values())[0]
                    
                    temp = {
                        'game_id':id,
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'hitter':[hitter],
                        'hittee':[hittee],
                        'hitter_team':[hitter_team],
                        'hittee_team':[hittee_team],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                    }
                    hits_df = pd.concat([hits_df, pd.DataFrame.from_dict(temp)])
                
                #Giveaways
                elif event['typeDescKey'] == 'giveaway':

                    giver = event['details']['playerId']
                    giver_team = team_dict[event['details']['eventOwnerTeamId']]
                
                    if giver_team == list(team_dict.values())[0]:
                        receiver_team = list(team_dict.values())[1]
                    else: receiver_team = list(team_dict.values())[0]
                    
                    temp = {
                        'game_id':id,
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'giver':[giver],
                        'giver_team':[giver_team],
                        'receiver_team':[receiver_team],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                    }
                    
                    giveaways_df = pd.concat([giveaways_df, pd.DataFrame.from_dict(temp)])

                #Takeaways
                elif event['typeDescKey'] == 'takeaway':
                    takeaway_count += 1
                    taker = event['details']['playerId']
                    taker_team = team_dict[event['details']['eventOwnerTeamId']]
                
                    if taker_team == list(team_dict.values())[0]:
                        loser_team = list(team_dict.values())[1]
                    else: loser_team = list(team_dict.values())[0]

                    temp = {
                        'game_id':id,
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'taker':[taker],
                        'taker_team':[taker_team],
                        'loser_team':[loser_team],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                    }
                    
                    takeaways_df = pd.concat([takeaways_df, pd.DataFrame.from_dict(temp)])
                
                #Blocks
                elif event['typeDescKey'] == 'blocked-shot':
                    try:
                        blocker = event['details']['blockingPlayerId']
                    except: 
                        blocker = 'N/A'
                    try:
                        shooter = event['details']['shootingPlayerId']
                    except:
                        shooter = 'N/A'

                    try:
                        blocker_team = team_dict[event['details']['eventOwnerTeamId']]
                        if blocker_team == list(team_dict.values())[0]:
                            shooter_team = list(team_dict.values())[1]
                        else: shooter_team = list(team_dict.values())[0]
                    except:
                        blocker_team = 'N/A'
                        shooter_team = 'N/A'

                    
                    temp = {
                        'game_id':[id],
                        'id':[event_id],
                        'period':[period],
                        'time_remaining':[time_remaining],
                        'home_team_side':[home_team_side],
                        'blocker':[blocker],
                        'shooter':[shooter],
                        'blocker_team':[blocker_team],
                        'shooter_team':[shooter_team],
                        'x':[x],
                        'y':[y],
                        'zone':[zone],
                    }
                    
                    blocks_df = pd.concat([blocks_df, pd.DataFrame.from_dict(temp)])    
        except:
            pass
        
        
        
        #Now we can go and get the box scores as well, this gets us full-game scores and player-by-player game stats
        url = f'https://api-web.nhle.com/v1/gamecenter/{id}/boxscore'

        try:
            response = requests.get(url).json()
            
            team_dict = {response['homeTeam']['id']:response['homeTeam']['abbrev'],
                            response['awayTeam']['id']:response['awayTeam']['abbrev']}
            
            home_team = list(team_dict.values())[0]
            away_team = list(team_dict.values())[1]

            try:
                total_away_score = response['boxscore']['linescore']['totals']['away']
                total_home_score = response['boxscore']['linescore']['totals']['home']
            except:
                total_away_score = 'N/A'
                total_home_score = 'N/A'

            try:
                if total_away_score != 'N/A':
                    if total_home_score > total_away_score:
                        winner, loser = home_team, away_team
                    else: winner, loser = away_team, home_team
                else:
                    winner = 'N/A'
                    loser = 'N/A'
            except:
                    winner = 'N/A'
                    loser = 'N/A'

            try:
                per1_away_score = response['boxscore']['linescore']['byPeriod'][0]['away']
                per1_home_score = response['boxscore']['linescore']['byPeriod'][0]['home']
            except:
                per1_away_score = 'N/A'
                per1_home_score = 'N/A'

            try:
                per2_away_score = response['boxscore']['linescore']['byPeriod'][1]['away']
                per2_home_score = response['boxscore']['linescore']['byPeriod'][1]['home']
            except:
                per2_away_score = 'N/A'
                per2_home_score = 'N/A'

            try:
                per3_away_score = response['boxscore']['linescore']['byPeriod'][2]['away']
                per3_home_score = response['boxscore']['linescore']['byPeriod'][2]['home']
            except:
                per3_away_score = 'N/A'
                per3_home_score = 'N/A'

            #Check about OT and SO results
            if response['gameOutcome']['lastPeriodType'] != 'REG':
                try:
                    ot_home_score = response['boxscore']['linescore']['byPeriod'][3]['home']
                    ot_away_score = response['boxscore']['linescore']['byPeriod'][3]['away']
                except: 
                    ot_home_score = 'N/A'
                    ot_away_score = 'N/A'
            else:
                ot_home_score = 'N/A'
                ot_away_score = 'N/A'

            if response['gameOutcome']['lastPeriodType'] == 'SO':
                try:
                    so_home_score = response['boxscore']['linescore']['byPeriod'][-1]['home']
                    so_away_score = response['boxscore']['linescore']['byPeriod'][-1]['away']
                except:
                    so_home_score = 'N/A'
                    so_away_score = 'N/A'
            else:
                so_home_score = 'N/A'
                so_away_score = 'N/A'

            temp = pd.DataFrame.from_dict({
                'game_id':[id],
                'home_team':[home_team],
                'away_team':[away_team],
                'total_home_score':[total_home_score],
                'total_away_score':[total_away_score],
                'winner':[winner],
                'loser':[loser],
                'per1_home_score':[per1_home_score],
                'per1_away_score':[per1_away_score],
                'per2_home_score':[per2_home_score],
                'per2_away_score':[per2_away_score],
                'per3_home_score':[per3_home_score],
                'per3_away_score':[per3_away_score],
                'ot_home_score':[ot_home_score],
                'ot_away_score':[ot_away_score],
                'so_home_score':[so_home_score],
                'so_away_score':[so_away_score],
            })
            box_score = pd.concat([box_score,temp])
            
            #Now, it also would proably be helpful to construct game-by-game stats for players that are in here too
            teams = response['boxscore']['playerByGameStats']
            for t in list(teams.keys()):
                for role in teams[t]:
                        for player in teams[t][role]:
                            try:
                                #Get all the things that are available for each player
                                if t == 'awayTeam':
                                    opp, team = list(team_dict.keys())
                                    home = 0
                                else: 
                                    team, opp = list(team_dict.keys())
                                    home = 1

                                player_id = player['playerId']
                                number = player['sweaterNumber']
                                name = player['name']['default']
                                position = player['position']
                                toi = int(player['toi'].split(":")[0])*60 + int(player['toi'].split(":")[1])
                                penalty_minutes = player['pim']

                                #For forwards and defencemen they have the same dictionary keys
                                if role == 'forwards' or role == 'defense' :
                                    goals = player['goals']
                                    assists = player['assists']
                                    points = player['points']
                                    plus_minus = player['plusMinus']
                                    hits = player['hits']
                                    blocks = player['blockedShots']
                                    ppg = player['powerPlayGoals']
                                    ppp = player['powerPlayPoints']
                                    shg = player['shorthandedGoals']
                                    shp = player['shPoints']
                                    shots = player['shots']
                                    faceoff_wins, faceoffs = player['faceoffs'].split('/')
                                    pp_toi = int(player['powerPlayToi'].split(":")[0])*60 + int(player['powerPlayToi'].split(":")[1])
                                    sh_toi = int(player['shorthandedToi'].split(":")[0])*60 + int(player['shorthandedToi'].split(":")[1])


                                    temp = pd.DataFrame.from_dict({
                                        'game_id': [id],
                                        'id': [player_id],
                                        'number': [number],
                                        'name':[name],
                                        'position':[position],
                                        'player_team':[team],
                                        'opp_team':[opp],
                                        'home':[home],
                                        'toi':[toi],
                                        'penalty_minutes':[penalty_minutes],
                                        'hits':[hits],
                                        'blocks':[blocks],
                                        'goals':[goals],
                                        'assists':[assists],
                                        'points':[points],
                                        'plus_minus':[plus_minus],
                                        'ppg':[ppg],
                                        'ppp':[ppp],
                                        'shg':[shg],
                                        'shp':[shp],
                                        'faceoffs':[faceoffs],
                                        'faceoff_wins':[faceoff_wins],
                                        'pp_toi':[pp_toi],
                                        'sh_toi':[sh_toi]})
                                    
                                    skater_stats = pd.concat([skater_stats,temp])
                                    


                                #For goalies obviously differet
                                elif role == 'goalies':
                                    es_sa = player['evenStrengthShotsAgainst'].split("/")[0]
                                    pp_sa = player['evenStrengthShotsAgainst'].split("/")[0]
                                    sh_sa = player['shorthandedShotsAgainst'].split("/")[0]
                                    saves = player['saveShotsAgainst']
                                    save_pct = player['savePctg']
                                    es_ga = player['evenStrengthGoalsAgainst']
                                    pp_ga = player['evenStrengthGoalsAgainst']
                                    sh_ga = player['shorthandedGoalsAgainst']
                                    ga = player['goalsAgainst']

                                    temp = pd.DataFrame.from_dict({
                                        'game_id': [id],
                                        'id': [player_id],
                                        'number': [number],
                                        'name':[name],
                                        'position':[position],
                                        'player_team':[team],
                                        'opp_team':[opp],
                                        'home':[home],
                                        'toi':[toi],
                                        'penalty_minutes':[penalty_minutes],
                                        'es_sa':[es_sa],
                                        'pp_sa':[pp_sa],
                                        'sh_sa':[sh_sa],
                                        'saves':[saves],
                                        'save_pct':[save_pct],
                                        'es_ga':[es_ga],
                                        'pp_ga':[pp_ga],
                                        'sh_ga':[sh_ga],
                                        'ga':[ga]})
                                    
                                    goalie_stats = pd.concat([goalie_stats,temp])
                                

                            
                            except:
                                pass
        except:
            pass
        
        #Now we do all of our saving! Only once every n times to avoid taking up too much time 
        if idx%save_every_n_games == 0 and idx>0:
            try:
                if skater_stats.shape[0] > 0:
                    if not(os.path.isfile('data\games\skater_games.csv')):
                        skater_stats.to_csv('data\games\skater_games.csv', index=False, mode='w', header=True)
                    else: skater_stats.to_csv('data\games\skater_games.csv', index=False, mode='a', header=False)

                    skater_stats.to_sql('skater_games',conn,if_exists='append')
                    skater_stats = pd.DataFrame()
            except:
                pass

            try:
                if goalie_stats.shape[0] > 0:
                    if not(os.path.isfile('data\games\goalie_games.csv')):
                        goalie_stats.to_csv('data\games\goalie_games.csv', index=False, mode='w', header=True)
                    else: goalie_stats.to_csv('data\games\goalie_games.csv', index=False, mode='a', header=False)

                    goalie_stats.to_sql('goalie_games',conn,if_exists='append')
                    goalie_stats = pd.DataFrame()
            except:
                pass

            try:
                if box_score.shape[0] > 0:
                    if not(os.path.isfile('data\games\\box_scores.csv')):
                        box_score.to_csv('data\games\\box_scores.csv', index=False, mode='w', header=True)
                    else: box_score.to_csv('data\games\\box_scores.csv', index=False, mode='a', header=False)

                    box_score.to_sql('box_scores',conn,if_exists='append')

                    box_score = pd.DataFrame()
            except:
                pass
            

            tables = {'shots':shots_df,
                'faceoffs':faceoffs_df,
                'penalties':penalties_df,
                'hits':hits_df,
                'giveaways':giveaways_df,
                'takeaways':takeaways_df,
                'blocks':blocks_df}
            
            for key, value in tables.items():
                try:
                    #if file exists, we'll write the header
                    if value.shape[0] > 0:
                        if not(os.path.isfile(f'data\events\{key}.csv')):
                            value.to_csv(f'data\events\{key}.csv', index=False, mode='w', header=True)
                        else: value.to_csv(f'data\events\{key}.csv', index=False, mode='a', header=False)
                        
                        value.to_sql(key, conn, if_exists='append')
                except:
                    pass 

            #Reinitialize these to save space in RAM
            shots_df = pd.DataFrame()
            faceoffs_df = pd.DataFrame()
            penalties_df = pd.DataFrame()
            hits_df = pd.DataFrame()
            giveaways_df = pd.DataFrame()
            takeaways_df = pd.DataFrame()
            blocks_df = pd.DataFrame()

def update_db():
    #Pull all new games that are not yet stored in the database via cllas to above 2 functions
    get_game_summaries()
    get_stats()

    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')
    c = conn.cursor()

    most_recent_game = pd.read_sql('SELECT MAX(id) \
                        FROM games', con=conn).iloc[0,0]
    
    most_recent_game_date = parser.parse(pd.read_sql(f'SELECT date \
                                    FROM games \
                                    WHERE id = {most_recent_game} \
                                    LIMIT 1', con=conn).iloc[0,0]).date()
    
    now = dt.datetime.now().time().strftime('%I:%M %p')
    

    
    print(f"Data up to date with finished games as of {now} ET on {most_recent_game_date}. \nID of most recent game: {most_recent_game}")
    


    return

def get_team_info():
    #Get the team to ID mapping and other info like conference and division
    print("nothing yet!")

def get_player_info():
    #See if there's any other player information that we don't have that we can grab from here
    print("Nothing yet!")