#import re
import requests
import json
#import os
import pandas as pd
#import numpy as np

#response = requests.get("https://api-web.nhle.com/v1/gamecenter/2007010001/boxscore")
response = requests.get("https://api-web.nhle.com/v1/player/8467478/landing")


print(json.dumps(response.json(), indent=2))

#Define the base link
#report_type=2
#date_from= ""
#date_to = ""

#Iterate over this link to get all of the data for players on each date
#base_link =  f"https://www.nhl.com/stats/skaters"\
#            f"?report={report_type}"\
#            f"&reportType=game"\
#            f"&dateFrom={date_from}"\
#            f"&dateTo={date_to}"\
#            f"&gameType=2"\
#            f"&filter=gamesPlayed,gte,0"\
#            f"&page=0"\
#            f"&pageSize=100"

#url = 'https://www.nhl.com/stats/skaters?reportType=game&dateFrom=2023-10-10&dateTo=2023-10-10&gameType=2&filter=gamesPlayed,gte,0&sort=points,goals,assists&page=0&pageSize=100'