from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import browserhistory as bh
import re
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pathlib
import os

dict_obj = bh.get_browserhistory()
chrome = dict_obj['chrome']
urls = []
titles = []
times = []

for address in chrome:
    try:
        url = re.search('http[s]?://(\S+?)/', address[0]).group(1)
        url = url.replace("www.", "")
        urls.append(url)
        titles.append(address[1])
        times.append(address[2])
    except:
        pass


df = pd.DataFrame()
df['Site'] = urls
df['Title'] = titles
df['Time'] = times

df['Time'] = pd.to_datetime(
    df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df = df.dropna()

six_months = date.today() + relativedelta(months=-6)
six_months = datetime.combine(six_months, datetime.min.time())
df = df[df['Time'] >= six_months]

site_list = set(list(df.Site))


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def process_title(text):
    tokens = word_tokenize(text.lower())
    lemm_tokens = list(map(lemmatizer.lemmatize, tokens))
    without_stop = [
        word for word in lemm_tokens if word not in stop_words and len(word) > 2]
    return without_stop


df['Words'] = df['Title'].apply(lambda x: process_title(str(x)))

exploded = df.explode('Words').drop(['Title'], axis=1)
g_df = exploded.groupby(['Site', 'Words'], as_index=False).size()

print("Excluded words")
dfs = []
for site in site_list:
    site_df = g_df[g_df.Site == site]
    sigma = float(site_df['size'].values.std())
    mu = float(site_df['size'].values.mean())
    outliers = 100000
    for val in site_df['size'].values:
        z = (val-mu)/sigma
        if z >= 4 or z <= -4:
            outliers = min(outliers, val)

    outlier_words = site_df[site_df['size'] >= outliers].Words.values
    if len(outlier_words) > 0:
        print(site, outlier_words)

    dfs.append(exploded[(exploded.Site == site) & ~
               (exploded.Words.isin(outlier_words))])

final_df = pd.concat(dfs, ignore_index=True)

filepath = pathlib.Path(__file__).resolve().parent
final_df.to_csv(os.path.join(filepath, 'word_counts.csv'))
