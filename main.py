from scrub_api_data import get_game_summaries, get_stats, get_odds
from process import aggregate_games_data, split_agg_games, aggregate_last_n
from predict import predict_winners

import sqlite3
from pathlib import Path
import datetime as dt
import pandas as pd

def update_db():

    #Raw data
    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')

    #Pull all new games that are not yet stored in the database
    get_game_summaries()
    get_stats()

    #Update the processed data and get the scaling done 
    aggregate_games_data()
    split_agg_games()
    #aggregate_last_n()

    #Get today's odds for record keeping
    get_odds()

    most_recent_game = pd.read_sql('SELECT * \
                        FROM games \
                        ORDER BY game_id DESC \
                        LIMIT 1', con=conn)
    
    #home_team, away_team= most_recent_game['home_team'].values[0],most_recent_game['away_team'].values[0]
    #recent_id = most_recent_game.loc[0,'game_id']
    
    now = dt.datetime.now().time().strftime('%I:%M %p')
    now_date = dt.datetime.now().date()
    
    print(f"Data up to date with finished games as of {now} ET on {now_date}.")

    print("Today's picks:")
    predict_winners(safety_threshold=1.15, verbose=1)
    

update_db()

