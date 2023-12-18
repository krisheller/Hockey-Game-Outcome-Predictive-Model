import requests
import os

from dateutil import parser
import datetime as dt

import pandas as pd

import sqlite3
from pathlib import Path

#This will get odds for the upcoming games on today's date
def get_odds():

    #Connect to database in case we'll use it
    Path('data\db.db').touch()
    conn = sqlite3.connect('data\db.db')
    c = conn.cursor()

    #Do we already have the odds for today?
    try:
        a = pd.read_sql('SELECT date \
                        FROM odds_h2h \
                        ORDER BY date DESC \
                        LIMIT 1', con=conn, )['date']

        if str(a.iloc[0]) == str(dt.date.today()):
            return
        
    except:
        pass

    #Setup api to pull
    api_key = os.getenv("ODDS_API_KEY")
    sport = 'icehockey_nhl'
    regions = 'us'
    markets = 'h2h,spreads,totals'
    odds_format = 'decimal'

    #Set up the time to stop pulling the odds to only get today's games
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)

    start_time = f'{today.year}-{today.month:02d}-{today.day:02d}T08:00:00Z'
    stop_time = f'{tomorrow.year}-{tomorrow.month:02d}-{tomorrow.day:02d}T08:00:00Z'

    try:
        response = requests.get(f'https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={api_key}&regions={regions}&markets={markets}&oddsFormat={odds_format}&commenceTimeFrom={start_time}&commenceTimeTo={stop_time}')
    except:
        response = "N/A"


    #Set some variables
    preferred_book = 'draftkings'  #We prefer to get the draft kings odds if they are available just because that's what i use 
    h2h_df = pd.DataFrame()
    spreads_df = pd.DataFrame()
    totals_df = pd.DataFrame()

    for game in response.json():

        book_list = {}
        for i,book in enumerate(game['bookmakers']):
            book_list[book['key']] = i
        
        #We'll use the preferred book if we can
        if preferred_book in book_list:
            book = preferred_book
            book_index = book_list[preferred_book]
        
        #Otherwise we'll take whichever one is available first and hope it's close enough
        else:
            book = list(book_list.keys())[0]
            book_index = book_list[book]

        #Get the odds for that game and necesssary info to match up
        home_team = game['home_team']
        away_team = game['away_team']

        #Iterate over the markets
        markets = game['bookmakers'][book_index]['markets']
        for market in markets:
            type = market['key']
            odds_as_of = parser.parse(market['last_update']).time()
            
            #We're going to have different kinds of odds for different markets
            if type == 'h2h' or type == 'spreads':
                if market['outcomes'][0]['name'] == home_team:
                    home_odds, away_odds = market['outcomes'][0]['price'],market['outcomes'][1]['price']
                else: away_odds, home_odds = market['outcomes'][0]['price'],market['outcomes'][1]['price']

                if type == 'h2h':
                    home_spread, away_spread = 0, 0
                else:
                    if market['outcomes'][0]['name'] == home_team:
                        home_spread, away_spread = market['outcomes'][0]['point'], market['outcomes'][1]['point']
                    else: away_spread, home_spread= market['outcomes'][0]['point'], market['outcomes'][1]['point']

                temp = pd.DataFrame.from_dict({
                    'sportsbook':[book],
                    'date':[today],
                    'odds_as_of':[odds_as_of],
                    'home_team':[home_team],
                    'away_team':[away_team],
                    'home_odds':[home_odds],
                    'home_spread':[home_spread],
                    'away_odds':[away_odds],
                    'away_spread':[away_spread]
                })

                if type == 'h2h':
                    h2h_df = pd.concat([h2h_df, temp])
                else:
                    spreads_df = pd.concat([spreads_df, temp])

            elif type == 'totals':
                line = market['outcomes'][0]['point']
                if market['outcomes'][0]['name'] == 'Over':
                    over_odds, under_odds = market['outcomes'][0]['price'], market['outcomes'][1]['price']
                else: under_odds, over_odds = market['outcomes'][0]['price'], market['outcomes'][1]['price']

                temp = pd.DataFrame.from_dict({
                    'sportsbook':[book],
                    'date':[today],
                    'odds_as_of':[odds_as_of],
                    'home_team':[home_team],
                    'away_team':[away_team],
                    'line':[line],
                    'over_odds':[over_odds],
                    'under_odds':[under_odds]
                })

                totals_df = pd.concat([totals_df,temp])

    #Let's save these historical odds!
    df_dict = {'odds_h2h':h2h_df,
               'odds_spreads':spreads_df, 
               'odds_totals':totals_df}
    for key, value in df_dict.items():
        try:
            if value.shape[0] > 0:
                if not(os.path.isfile(f'data\odds\\{key}.csv')):
                    value.to_csv(f'data\odds\\{key}.csv', index=False, mode='w', header=True)
                else: value.to_csv(f'data\odds\\{key}.csv', index=False, mode='a', header=False)

                value.to_sql(f'{key}',conn,if_exists='append')
        except:
            pass
    
        
    return