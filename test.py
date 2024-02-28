import sqlite3
from pathlib import Path
import pandas as pd
import requests
from collections import Counter

url =f'https://api.nhle.com/stats/rest/en/team'

response = requests.get(url).json()

print(response)

situation_codes = []
time_list = []
period_list = []


for play in response:
    #print(play['situationCode'])
    try:
        situation_codes.append(play['situationCode'])
        period_list.append(play['period'])
        time_list.append(play['timeRemaining'])
        
        if play['situationCode']=='1441':
        #if play['typeDescKey']=='goal':
            print(play)

    except: pass


df = pd.DataFrame({
    'code':situation_codes,
    'period':period_list,
    'time':time_list
})

#print(df.to_string())

print(Counter(situation_codes))