import sqlite3
from pathlib import Path
import pandas as pd
import requests

Path('data\db.db').touch()
conn = sqlite3.connect('data\db.db')
c = conn.cursor()

url = 'https://api-web.nhle.com/v1/schedule/now'
response = requests.get(url).json()

print(response)