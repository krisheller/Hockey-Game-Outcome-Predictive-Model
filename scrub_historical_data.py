import re
from requests import session
import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import os

header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}


#s = session()

#Define the base link
report_type=2
date_from= ""
date_to = ""

#Iterate over this link to get all of the data for players on each date
base_link =  f"https://www.nhl.com/stats/skaters"\
            f"?report={report_type}"\
            f"&reportType=game"\
            f"&dateFrom={date_from}"\
            f"&dateTo={date_to}"\
            f"&gameType=2"\
            f"&filter=gamesPlayed,gte,0"\
            f"&page=0"\
            f"&pageSize=100"

url = 'https://www.nhl.com/stats/skaters?reportType=game&dateFrom=2023-10-10&dateTo=2023-10-10&gameType=2&filter=gamesPlayed,gte,0&sort=points,goals,assists&page=0&pageSize=100'
r = requests.get(url, headers=header)
print(pd.read_html(r.text))

#r = s.get(link, headers={'User-Agent': 'Mozzila/5.0'})
#soup = BeautifulSoup(r.text, "html.parser" )