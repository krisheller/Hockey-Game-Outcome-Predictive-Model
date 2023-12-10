import pandas as pd
import os
import sqlite3
from pathlib import Path

#Read in the historical data in .csv format we have available from MoneyPuck and combine the annual files with each other 
#for skaters, goalies, and teams

def format_raw_data(type):
    path = os.path.join(os.getcwd(),f'data\{type}')
    files = [file for file in os.listdir(path)]

    df = pd.DataFrame()

    for file in files:
        temp = pd.read_csv(path+'/'+file)
        df = pd.concat([df, temp])

    df.reset_index(inplace=True,drop=True)

    return df

#We will only lead the data from csv if the SQL database doesn't exist
write_to_db = False
if not(os.path.isfile('data\db.db')):
    write_to_db = True
    print("Loading data...")
    player_info = pd.read_csv('data/allPlayersLookup.csv')
    skaters, goalies, teams, shots = format_raw_data('skaters'), format_raw_data('goalies'), format_raw_data('teams'), format_raw_data('shots')
    print("Data loaded.")

#Create the SQL database if it doesn't exist
Path('data\db.db').touch()

#Connect to the table and start adding data with the cursor
conn = sqlite3.connect('data\db.db')
c = conn.cursor()

#Again, only write to the database if these don't exist already
if (write_to_db):
    #Dictionary of table names and associated DFs
    dict = {'skaters': skaters,
            'goalies':goalies,
            'teams':teams,
            'shots':shots,
            'player_info': player_info}

    #Since historical data is not going to be updated we fail if the table already exists
    print("Writing to database...")
    for key, value in dict.items():
        try:
            value.to_sql(key, conn, if_exists='fail')
        except:
            pass

#Else we can just pull from the db
players = pd.read_sql('SELECT * \
                    FROM player_info \
                    LIMIT 100', con=conn)

#Great!

