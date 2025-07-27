from requests_html import HTMLSession, PyQuery as pq
from datetime import datetime, timedelta
import json
import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()  

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


session = HTMLSession()

startDate = datetime(2025,1,1)
endDate = datetime(2025,2,1) # datetime.now().date()
dt = timedelta(days=2)

'''for i in np.arange(startDate, endDate, dt).astype(datetime):
    print(type(str(i.date())))
    break'''

f = open('./data/news/headlines.json','w', encoding='utf-8')
allData = {}
for i in np.arange(startDate, endDate, dt).astype(datetime):
    while True:
        temp = f'https://newsapi.org/v2/everything?q=tesla&from=2025-06-27&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
        print(temp)
        r = session.get(temp)
        articles = r.html.find('article')
        
        if len(articles) > 0:
            break
        
    print(f"{str(i.date())} > {len(articles)} articles fetched")
    allData[str(i.date())] = []
    
    for article in articles:
        t = pq(article.html)
        headingText = t('h2.story__title a.story__link').text()
        spanId = t('span').eq(0).attr('id')
        label = spanId.lower() if spanId is not None else None
        #print(f"Extracted heading: {headingText}, label: {label}")  # debug print
        if len(headingText) > 0 and label in ["pakistan", "canada", "international", "usa", "united states of america"]:
            allData[str(i.date())].append({
                "heading": headingText,
                "label": label,
            })
    
json.dump(allData, f, ensure_ascii=False)
f.close()

print("news crapping done")