from scrub_api_data import get_game_summaries, get_stats, get_odds
from process import process_games_data

import sqlite3
from pathlib import Path
import datetime as dt
import pandas as pd

def update_db():

    #Raw data
    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')

    #Processed data
    Path('data\db_processed.db').touch()
    conn_proc = sqlite3.connect('data\db_processed.db')

    #Pull all new games that are not yet stored in the database
    get_game_summaries()

    #Get the event and game-level stats for all the games we just pulled in
    get_stats()

    #Update the processed data
    process_games_data()

    #Get today's odds 
    get_odds()

    #Print some information about what we just did
    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')
    c = conn.cursor()

    most_recent_game = pd.read_sql('SELECT * \
                        FROM games \
                        ORDER BY game_id DESC \
                        LIMIT 1', con=conn)
    
    home_team, away_team= most_recent_game['home_team'].values[0],most_recent_game['away_team'].values[0]
    recent_id = most_recent_game.loc[0,'game_id']
    
    now = dt.datetime.now().time().strftime('%I:%M %p')
    now_date = dt.datetime.now().date()
    
    print(f"Data up to date with finished games as of {now} ET on {now_date}. \nMost recent game: {recent_id}, {away_team} @ {home_team}")
    

update_db()

