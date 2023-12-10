import requests
import json
import os
import pandas as pd 
import datetime as dt
import sqlite3
from pathlib import Path

#Connect to database
Path('data\db.db').touch()
conn = sqlite3.connect('data\db.db')
c = conn.cursor()

#Start on January 1, 2007 and run through today
def get_historical_data():

    #If the table already exists in SQL, we won't bother
    try:
        games = pd.read_sql('SELECT * \
                    FROM games', con=conn)
    except:
        date = dt.date(2007, 1, 1)
        #end_date = dt.date(2010, 1, 1)
        end_date = dt.date.today()

        
        games = pd.DataFrame()

        while date < end_date:
            url = f'https://api-web.nhle.com/v1/schedule/{date.year}-{date.month:02d}-{date.day:02d}'
            pull = True
            
            try:
                response = requests.get(url).json()
            except:
                pull = False
            
            if(pull):
                #Iterate over each day in the week
                for game_date in response['gameWeek']:

                    #Check to make sure games were played
                    if game_date['numberOfGames'] > 0:
                        for game in game_date['games']:

                            #Check to see if the game got canceled and that it has already occured
                            if (game['gameScheduleState'] == 'OK' and game['gameState'] == 'OFF'):

                                #Capture all this information on the game
                                id, start_date, start_time, neutral_site, tz  = game['id'], game_date['date'], game['startTimeUTC'], game['neutralSite'], game['venueTimezone']
                                home_team, home_score, away_team, away_score = game['homeTeam']['abbrev'], game['homeTeam']['score'], game['awayTeam']['abbrev'], game['awayTeam']['score']
                                winner = home_team if home_score>away_score else away_team
                                season = date.year-1 if date.month < 5 else date.year
                                game_link = game['gameCenterLink']
                                full_time = game['gameOutcome']['lastPeriodType']

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
                                        'winner':winner,
                                        'link':[f'nhl.com{game_link}']}
                                
                                temp = pd.DataFrame.from_dict(temp_dict)                    
                                games = pd.concat([games, temp])
                
                #Iterate to the next week
                date += dt.timedelta(days=7)

        games.set_index('id',drop=True)
        

        #We write it to SQL database if it's not there, for more recent data we will rely on another function
        try:
            games.to_sql('games', conn, if_exists='fail')
        except:
            pass

        games.to_csv('data\games.csv', index=False)

    #Return the file
    return games

#Check what game data we already have and pull everything that's more recent
games = get_historical_data()

print(games[games.season == 2006])


#Get data from recent games by checking the DB
#def get_recent_data():


#Schedule of games on a given day
#response = requests.get("https://api-web.nhle.com/v1/schedule/2023-07-23")

#All data on a single game
#response = requests.get("https://api-web.nhle.com/v1/gamecenter/2023020293/play-by-play")