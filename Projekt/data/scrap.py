
import re
import requests
import itertools
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


df=pd.DataFrame(columns=['Date','Headline'])

print(df)

headline_regex = re.compile('h3 class="story-title"')

url='https://www.reuters.com/news/archive/eurMktRpt?view=page&page=1&pageSize=10'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')


headlines = soup.find('body').find_all('h3','class'=='story-title')
time = soup.find('body').find_all('time')



for i in range(len(time)):

    df.loc[i]=[time[i].text.strip()]+[headlines[i].text.strip()]


print(df)
df.to_csv('headlines')
