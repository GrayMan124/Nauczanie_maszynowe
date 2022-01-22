
import re
import requests
import itertools
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


df=pd.read_csv('headlines_reuters')
df=df.drop("Unnamed: 0",axis=1)
headline_regex = re.compile('h3 class="story-title"')

url='https://www.reuters.com/news/archive/eurMktRpt?view=page&page=1&pageSize=10'
response = requests.get(url,timeout=5)
soup = BeautifulSoup(response.text, 'html.parser')


headlines = soup.find('body').find_all('h3','class'=='story-title')
time1 = soup.find('body').find_all('time')

l=0

for k in range(1,1438):
    time.sleep(5)
    url='https://www.reuters.com/news/archive/eurMktRpt?view=page&page='+str(k)+'&pageSize=10'
    response = requests.get(url,timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find('body').find_all('h3','class'=='story-title')
    time1 = soup.find('body').find_all('time')
    print("Jestem na stronie: "+ str(k))
    print("Skocz se po kawke, to troche zajmie")
    for i in range(len(time1)):
        df.loc[l]=[time1[i].text.strip()]+[headlines[i].text.strip()]
        l+=1


print(df)
df.to_csv('headlines_reuters')
