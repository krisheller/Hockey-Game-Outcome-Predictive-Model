import sqlite3
from pathlib import Path
import pandas as pd

Path('data\db.db').touch()
conn = sqlite3.connect('data\db.db')
c = conn.cursor()

game_list_exist = pd.read_sql('SELECT game_id \
            FROM box_scores', con=conn)['game_id'].to_list()
#print(game_list_exist)

game_list_full = pd.read_sql('SELECT id \
                                FROM games', con=conn)['id'].to_list()
#print(game_list_full)

games = list(set(game_list_full) - set(game_list_exist))
games.sort()
print(games)