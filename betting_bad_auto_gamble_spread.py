# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:35:03 2016

@author: charlesmartens
"""

# coding: utf-8



# need the schedule (doubled). map spreads (douobled) onto it.
# then can merge with big df

# stuff to watch for that's gone wrong:
# - a team might not be in the pinnacle betting score
# - the +/- in basketball reference can screw things up. should create an if/then to deal w


# eventually, could use pinnacle api to get the spread and even to automate 
# putting down a bet. so we wouldn't have to do anthing. but seems dangerous
# to autmate the actual bet. here's code written in r to get the spread from
# the api. though need to be able to log into the account. so think only jeff
# could do this from canada.
# https://github.com/marcoblume/pinnacle.API

# to scrape pinnacle w selenium
# http://danielfrg.com/blog/2015/09/28/crawling-python-selenium-docker/
# http://www.marinamele.com/selenium-tutorial-web-scraping-with-selenium-and-python
# https://www.packtpub.com/books/content/web-scraping-python-part-2


cd /Users/charlesmartens/Documents/projects/bet_bball


# technically, need to beat 51.3% to win. though the juice seems a bit erractic, 
# so probably better to think about it at 51.5%

# current approach:
# use adaboost with decision tree regressor in conjuction with regular linear model. 
# and bet the games they both agree and games > 15. this is giving amazing results (> 56%) 
# should also try to average the score predictions from each of these models and 
# use that mean to bet. this will allow to bet on more games, but need to look at
# the trade-off with the percentage correct.


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from sklearn.cross_validation import cross_val_score
import matplotlib.cm as cm
from sklearn import ensemble
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import urllib.request   
from bs4 import BeautifulSoup
import datetime
sns.set_style('white')
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import RandomForestClassifier 


# --------------
# amount to bet:

# enter amount in account:
total_amount = 10306
pot = total_amount/2 - 30
#pot = 5000
#win_probability = .525
#win_probability = .53
win_probability = .52
kelly_criteria = (win_probability * .945 - (1 - win_probability)) / .945
bet_kelly = pot * kelly_criteria
print(round(bet_kelly))
#print(round(bet_kelly*.9))
#print(round(bet_kelly*.8))  



# -----------------------------------------
# set date:
date_today = datetime.datetime.now().date()
year_now = date_today.year
month_now = date_today.month
day_now = date_today.day
#------------------------------------------

#==========================================
# for vietnam testing one day late
#dates_vietnam = pd.date_range('2016/11/1', str(year_now)+'/'+str(month_now)+'/'+str(day_now), freq='D')
#date_today = dates_vietnam[-2]
#date_today = dates_vietnam[-3]
#date = dates_vietnam[-3]
#year_now = date.year
#month_now = date.month
#day_now = date.day
#==========================================


# incorp moneline and opening line data
df_moneyline = pd.read_csv('sports_insights_historicalOdds_2.csv')
df_moneyline.tail()
df_moneyline['date'] = pd.to_datetime(df_moneyline['Date'])
df_moneyline['year'] = df_moneyline['date'].dt.year
df_moneyline['month'] = df_moneyline['date'].dt.month
df_moneyline['day'] = df_moneyline['date'].dt.day
df_moneyline.loc[:, 'date'] = pd.to_datetime(df_moneyline.loc[:, 'year'].astype(str) + '-' + df_moneyline.loc[:, 'month'].astype(str) + '-' + df_moneyline.loc[:, 'day'].astype(str))
df_moneyline[['Date', 'date']]

# fix team namese
teams = list(df_moneyline['Home Team'].unique())
teams_to_remove = ['Spain', 'USA', 'Australia', 'Lithuania', 'France', 'Argentina', 
'Serbia', 'Nigeria', 'Croatia', 'China', 'Brazil', 'Venezuela', 'East All-Stars',
'USA All Stars', 'West All-Stars', 'Angola', 'Team Chuck', 'Sophomores', 
'D-League All-Stars', 'Orlando Blue', 'Orlando White']
for team in teams_to_remove:
    teams.remove(team)
len(teams)

df_moneyline.rename(columns={'Home Team':'team'}, inplace=True)
df_moneyline['team'].replace('Seattle Supersonics', 'Oklahoma City Thunder', inplace=True)
df_moneyline.rename(columns={'Visitor Team':'opponent'}, inplace=True)
df_moneyline['opponent'].replace('Seattle Supersonics', 'Oklahoma City Thunder', inplace=True)

new_names = ['Atlanta Hawks', 'Chicago Bulls', 'Golden State Warriors',
   'Boston Celtics', 'Brooklyn Nets', 'Detroit Pistons',
   'Houston Rockets', 'Los Angeles Lakers', 'Memphis Grizzlies',
   'Miami Heat', 'Milwaukee Bucks', 'Oklahoma City Thunder',
   'Orlando Magic', 'Phoenix Suns', 'Portland Trail Blazers',
   'Sacramento Kings', 'Toronto Raptors', 'Indiana Pacers',
   'Los Angeles Clippers', 'New York Knicks', 'Cleveland Cavaliers',
   'Denver Nuggets', 'Philadelphia 76ers', 'San Antonio Spurs',
   'New Orleans Pelicans', 'Washington Wizards', 'Charlotte Hornets',
   'Minnesota Timberwolves', 'Dallas Mavericks', 'Utah Jazz']  

df_moneyline = df_moneyline[df_moneyline['team'].isin(new_names)]
len(df_moneyline)
len(df_moneyline['team'].unique())
df_moneyline.tail()
df_moneyline = df_moneyline.sort_values(by=['team','date'])
df_moneyline =df_moneyline.reset_index(drop=True)

df_moneyline[['date','team','opponent']][df_moneyline['team']=='Atlanta Hawks'][:50]
# ok -- ea game represented once. just home games
# 0 = home, 1 = away
df_moneyline.loc[:, 'venue'] = 0
df_moneyline.rename(columns={'Home Opener':'team_open_spread'}, inplace=True) 
df_moneyline.rename(columns={'Home Closing Line':'team_close_spread'}, inplace=True) 
df_moneyline.rename(columns={'Home Juice':'team_juice'}, inplace=True) 
df_moneyline.rename(columns={'Visitor Juice':'opponent_juice'}, inplace=True) 
df_moneyline.rename(columns={'Home ML Open':'team_open_ml'}, inplace=True) 
df_moneyline.rename(columns={'Visitor ML Open':'opponent_open_ml'}, inplace=True) 
df_moneyline.rename(columns={'Home ML':'team_close_ml'}, inplace=True) 
df_moneyline.rename(columns={'Visitor ML':'opponent_close_ml'}, inplace=True) 
df_moneyline.rename(columns={'Home Score':'team_score_2'}, inplace=True)
df_moneyline.rename(columns={'Visitor Score':'opponent_score_2'}, inplace=True)

df_moneyline_switch = df_moneyline.copy(deep=True)
df_moneyline_switch.rename(columns={'team':'opponent_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent':'team_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_2':'opponent'}, inplace=True)
df_moneyline_switch.rename(columns={'team_2':'team'}, inplace=True)
df_moneyline_switch['team_open_spread'] = df_moneyline_switch['team_open_spread']*-1
df_moneyline_switch['team_close_spread'] = df_moneyline_switch['team_close_spread']*-1
df_moneyline_switch.rename(columns={'team_juice':'opponent_juice_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_juice':'team_juice_2'}, inplace=True)
df_moneyline_switch.rename(columns={'team_juice_2':'team_juice'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_juice_2':'opponent_juice'}, inplace=True)
df_moneyline_switch.rename(columns={'team_open_ml':'opponent_open_ml_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_open_ml':'team_open_ml_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_open_ml_2':'opponent_open_ml'}, inplace=True)
df_moneyline_switch.rename(columns={'team_open_ml_2':'team_open_ml'}, inplace=True)
df_moneyline_switch.rename(columns={'team_close_ml':'opponent_close_ml_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_close_ml':'team_close_ml_2'}, inplace=True)
df_moneyline_switch.rename(columns={'opponent_close_ml_2':'opponent_close_ml'}, inplace=True)
df_moneyline_switch.rename(columns={'team_close_ml_2':'team_close_ml'}, inplace=True)
df_moneyline_switch.rename(columns={'Home Score':'opponent_score_2'}, inplace=True)
df_moneyline_switch.rename(columns={'Visitor Score':'team_score_2'}, inplace=True)
df_moneyline_switch.loc[:, 'venue'] = 1
df_moneyline_switch.head()
df_moneyline_switch.columns
df_moneyline_switch = df_moneyline_switch[['Date', 'NSS', 'opponent', 'team', 'team_open_spread',
       'team_close_spread', 'opponent_juice', 'team_juice', 'Opening Total',
       'Closing Total', 'Over Juice', 'Under Juice', 'opponent_open_ml',
       'opponent_close_ml', 'team_open_ml', 'team_close_ml',
       'opponent_score_2', 'team_score_2', 'date', 'year', 'month', 'day',
       'venue']]

df_moneyline.columns == df_moneyline_switch.columns
df_moneyline_full = pd.concat([df_moneyline, df_moneyline_switch], ignore_index=True)
len(df_moneyline_full)
df_moneyline_full = df_moneyline_full.sort_values(by=['team','date'])
df_moneyline_full = df_moneyline_full.reset_index(drop=True)
df_moneyline_full[['date','team','opponent']][df_moneyline_full['team']=='Atlanta Hawks'][50:90]
df_moneyline_full.pop('Date')
df_moneyline_full.pop('NSS')
df_moneyline_full.pop('year')
df_moneyline_full.pop('month')
df_moneyline_full.pop('day')
df_moneyline_full.pop('venue')

df_moneyline_full[['team_juice', 'opponent_juice']]
df_moneyline_full['team_juice_rev'] = np.nan
df_moneyline_full.loc[df_moneyline_full['team_juice']<0,'team_juice_rev'] = -100 /df_moneyline_full['team_juice']
df_moneyline_full.loc[df_moneyline_full['team_juice']>0,'team_juice_rev'] = df_moneyline_full['team_juice'] / 100
df_moneyline_full['opponent_juice_rev'] = np.nan
df_moneyline_full.loc[df_moneyline_full['opponent_juice']<0,'opponent_juice_rev'] = -100 /df_moneyline_full['opponent_juice']
df_moneyline_full.loc[df_moneyline_full['opponent_juice']>0,'opponent_juice_rev'] = df_moneyline_full['opponent_juice'] / 100


# - covers data
df_covers_bball_ref = pd.read_csv('df_covers_bball_ref_2004_to_2015_raw_vars.csv')
df_covers_bball_ref.pop('Unnamed: 0')
df_covers_bball_ref['date'] = pd.to_datetime(df_covers_bball_ref['date'])
df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
df_covers_bball_ref.loc[:, 'point_difference'] = df_covers_bball_ref.loc[:, 'score_team'] - df_covers_bball_ref.loc[:, 'score_oppt']

for col in df_covers_bball_ref.columns:
    print(col)
len(df_covers_bball_ref)
len(df_moneyline_full)

# merge existing df w moneline df
df_covers_bball_ref = pd.merge(df_covers_bball_ref, df_moneyline_full, on=['date','team', 'opponent'], how='left')
df_covers_bball_ref.head()
df_covers_bball_ref[['date','team', 'spread', 'team_close_spread', 'team_open_spread']]
df_covers_bball_ref[['spread','team_close_spread', 'team_open_spread']].corr()


# fix var names
df_covers_bball_ref.rename(columns={'team_AST%':'team_ASTpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_BLK%':'team_BLKpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_DRB%':'team_DRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_ORB%':'team_ORBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_STL%':'team_STLpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_TOV%':'team_TOVpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_TRB%':'team_TRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_TS%':'team_TSpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_eFG%':'team_eFGpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_AST%':'opponent_ASTpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_BLK%':'opponent_BLKpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_DRB%':'opponent_DRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_ORB%':'opponent_ORBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_STL%':'opponent_STLpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_TOV%':'opponent_TOVpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_TRB%':'opponent_TRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_TS%':'opponent_TSpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_eFG%':'opponent_eFGpct'}, inplace=True)

df_covers_bball_ref.rename(columns={'team_3PAr':'opp_3PAr'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_ASTpct':'opp_ASTpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_BLKpct':'opp_BLKpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_DRBpct':'opp_DRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_DRtg':'opp_DRtg'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_FTr':'opp_FTr'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_ORBpct':'opp_ORBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_ORtg':'opp_ORtg'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_STLpct':'opp_STLpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_TOVpct':'opp_TOVpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_TRBpct':'opp_TRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_TSpct':'opp_TSpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_eFGpct':'opp_eFGpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fg3_pct':'opp_fg3_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fg_pct':'opp_fg_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_ft_pct':'opp_ft_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_pf':'opp_pf'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_MP':'opp_MP'}, inplace=True)

df_covers_bball_ref.rename(columns={'team_ast':'opp_ast'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_blk':'opp_blk'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_drb':'opp_drb'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_drb':'opp_drb'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fg':'opp_fg'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fg3':'opp_fg3'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fg3a':'opp_fg3a'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fga':'opp_fga'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_ft':'opp_ft'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_fta':'opp_fta'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_orb':'opp_orb'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_pts':'opp_pts'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_stl':'opp_stl'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_tov':'opp_tov'}, inplace=True)
df_covers_bball_ref.rename(columns={'team_trb':'opp_trb'}, inplace=True)

df_covers_bball_ref.rename(columns={'opponent_3PAr':'team_3PAr'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_ASTpct':'team_ASTpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_BLKpct':'team_BLKpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_DRBpct':'team_DRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_DRtg':'team_DRtg'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_FTr':'team_FTr'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_ORBpct':'team_ORBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_ORtg':'team_ORtg'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_STLpct':'team_STLpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_TOVpct':'team_TOVpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_TRBpct':'team_TRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_TSpct':'team_TSpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_eFGpct':'team_eFGpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fg3_pct':'team_fg3_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fg_pct':'team_fg_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_ft_pct':'team_ft_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_pf':'team_pf'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_MP':'team_MP'}, inplace=True)

df_covers_bball_ref.rename(columns={'opponent_ast':'team_ast'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_blk':'team_blk'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_drb':'team_drb'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fg':'team_fg'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fg3':'team_fg3'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fg3a':'team_fg3a'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fga':'team_fga'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_ft':'team_ft'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_fta':'team_fta'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_orb':'team_orb'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_pts':'team_pts'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_stl':'team_stl'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_tov':'team_tov'}, inplace=True)
df_covers_bball_ref.rename(columns={'opponent_trb':'team_trb'}, inplace=True)

df_covers_bball_ref.rename(columns={'opp_3PAr':'opponent_3PAr'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_ASTpct':'opponent_ASTpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_BLKpct':'opponent_BLKpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_DRBpct':'opponent_DRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_DRtg':'opponent_DRtg'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_FTr':'opponent_FTr'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_ORBpct':'opponent_ORBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_ORtg':'opponent_ORtg'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_STLpct':'opponent_STLpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_TOVpct':'opponent_TOVpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_TRBpct':'opponent_TRBpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_TSpct':'opponent_TSpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_eFGpct':'opponent_eFGpct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fg3_pct':'opponent_fg3_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fg_pct':'opponent_fg_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_ft_pct':'opponent_ft_pct'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_pf':'opponent_pf'}, inplace=True)

df_covers_bball_ref.rename(columns={'opp_MP':'opponent_MP'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_ast':'opponent_ast'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_blk':'opponent_blk'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_drb':'opponent_drb'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fg':'opponent_fg'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fg3':'opponent_fg3'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fg3a':'opponent_fg3a'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fga':'opponent_fga'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_ft':'opponent_ft'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_fta':'opponent_fta'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_orb':'opponent_orb'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_pts':'opponent_pts'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_stl':'opponent_stl'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_tov':'opponent_tov'}, inplace=True)
df_covers_bball_ref.rename(columns={'opp_trb':'opponent_trb'}, inplace=True)


# scrape nov 1 again
#dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')
#date = dates[-6]  #  replace to -2 once get back days
#day_yesterday = date.day
#month_yesterday = date.month
#year_yesterday = date.year

# --------------------------
# scrape bbll ref box scores
# to get box scores from yesterday:

dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')
date = dates[-2]  #  
#date = dates[-3]  #  get boxes from two day ago
day_yesterday = date.day
month_yesterday = date.month
year_yesterday = date.year

# subst date into game_day_link below to get list of box score links
game_day_link = 'http://www.basketball-reference.com/boxscores/index.cgi?month='+str(month_yesterday)+'&day='+str(day_yesterday)+'&year='+str(year_yesterday)
#game_day_link = 'http://www.basketball-reference.com/boxscores/index.cgi?month=11&day=1&year=2016'

on_webpage = urllib.request.urlopen(game_day_link)
html_contents = on_webpage.read()
on_webpage.close()        
soupObject = BeautifulSoup(html_contents, 'html.parser')  
soupObject
a_tags = soupObject.find_all('a')
box_score_links_list = []
for link in a_tags:
    if link.text == 'Box Score':
        a_tag_link = link.get('href')
        box_score_links_list.append(a_tag_link)
    else:
        None
print(len(box_score_links_list))
#print(box_score_links_list[:])


#link = box_score_links_list[0]
def get_box_score_soup_object(link): 
    base_link = 'http://www.basketball-reference.com'
    box_score_page = base_link+link
    on_webpage = urllib.request.urlopen(box_score_page)
    html_contents = on_webpage.read()
    on_webpage.close()        
    soupObject = BeautifulSoup(html_contents, 'html.parser')  
    return soupObject
  
#soupObject = get_box_score_soup_object(link)


def get_starter_mins_for_box_score(soupObject):
    # get starters for ea team:
    table_bodies = soupObject.find_all('tbody')
    th_tag_team1 = soupObject.find_all('th', text='Starters')[0]
    player_rows1 = th_tag_team1.parent.next_sibling.next_element.next_element.find_all('tr')
    starters_list_team1 = [player_rows1[i].td.text for i in range(0,5)]
    starters_list_team1.sort()
    starters_min_string_team1 = ' '.join(starters_list_team1)
    th_tag_team2 = soupObject.find_all('th', text='Starters')[2]
    player_rows2 = th_tag_team2.parent.next_sibling.next_element.next_element.find_all('tr')
    starters_list_team2 = [player_rows2[i].td.text for i in range(0,5)]
    starters_list_team2.sort()
    starters_min_string_team2 = ' '.join(starters_list_team2)
    return starters_min_string_team1, starters_min_string_team2

#starters_min_string_team1, starters_min_string_team2 = get_starter_mins_for_box_score(soupObject)


def get_starters_for_box_score(soupObject):
    # get starters for ea team:
    table_bodies = soupObject.find_all('tbody')
    th_tag_team1 = soupObject.find_all('th', text='Starters')[0]
    player_rows1 = th_tag_team1.parent.next_sibling.next_element.next_element.find_all('tr')
    starters_list_team1 = [player_rows1[i].th.text for i in range(0,5)]
    starters_list_team1.sort()
    starters_string_team1 = ' '.join(starters_list_team1)
    th_tag_team2 = soupObject.find_all('th', text='Starters')[2]
    player_rows2 = th_tag_team2.parent.next_sibling.next_element.next_element.find_all('tr')
    starters_list_team2 = [player_rows2[i].th.text for i in range(0,5)]
    starters_list_team2.sort()
    starters_string_team2 = ' '.join(starters_list_team2)
    return starters_string_team1, starters_string_team2

#starters_string_team1, starters_string_team2 = get_starters_for_box_score(soupObject)


def give_game_stats(soupObject):
    # team 1 
    table_bodies = soupObject.find_all('tfoot')
    team1_basic_stats = table_bodies[0]
    table_data = team1_basic_stats.find_all('td')

    if table_data[-1].text == '':
        team1_basic_stats_list = [float(stat.text) for stat in table_data[0:-1]]
        team1_adv_stats = table_bodies[1]
        table_data = team1_adv_stats.find_all('td')
        team1_adv_stats_list = [float(stat.text) for stat in table_data[0:]]
        # team 2 
        team2_basic_stats = table_bodies[2]
        table_data = team2_basic_stats.find_all('td')
        team2_basic_stats_list = [float(stat.text) for stat in table_data[0:-1]]
        team2_adv_stats = table_bodies[3]
        table_data = team2_adv_stats.find_all('td')
        team2_adv_stats_list = [float(stat.text) for stat in table_data[0:]]
    elif table_data[-1].text != '':
        team1_basic_stats_list = [float(stat.text) for stat in table_data[0:]]
        team1_adv_stats = table_bodies[1]
        table_data = team1_adv_stats.find_all('td')
        team1_adv_stats_list = [float(stat.text) for stat in table_data[0:]]
        # team 2 
        team2_basic_stats = table_bodies[2]
        table_data = team2_basic_stats.find_all('td')
        team2_basic_stats_list = [float(stat.text) for stat in table_data[0:]]
        team2_adv_stats = table_bodies[3]
        table_data = team2_adv_stats.find_all('td')
        team2_adv_stats_list = [float(stat.text) for stat in table_data[0:]]
#    team1_basic_stats_list = [float(stat.text) for stat in table_data[0:]]
#    # use this below instead of line above if the code isn't working. at some point they add a plus/minus col to end so need to get rid of that
#    #team1_basic_stats_list = [float(stat.text) for stat in table_data[0:-1]]
#    team1_adv_stats = table_bodies[1]
#    table_data = team1_adv_stats.find_all('td')
#    team1_adv_stats_list = [float(stat.text) for stat in table_data[0:]]
#    # team 2 
#    team2_basic_stats = table_bodies[2]
#    table_data = team2_basic_stats.find_all('td')
#    team2_basic_stats_list = [float(stat.text) for stat in table_data[0:]]
#    # use this below instead of line above if the code isn't working. at some point they add a plus/minus col to end so need to get rid of that
#    #team2_basic_stats_list = [float(stat.text) for stat in table_data[0:-1]]
#    team2_adv_stats = table_bodies[3]
#    table_data = team2_adv_stats.find_all('td')
#    team2_adv_stats_list = [float(stat.text) for stat in table_data[0:]]
    return team1_basic_stats_list, team1_adv_stats_list, team2_basic_stats_list, team2_adv_stats_list

#team1_basic_stats_list, team1_adv_stats_list, team2_basic_stats_list, team2_adv_stats_list = give_game_stats(soupObject)


def get_teams_and_date(soupObject):
    title = soupObject.find('title').text
    teams = title.split(' Box ')[0]
    date1 = title.split(', ')[1]
    date2 = title.split(', ')[2]
    date2 = date2.split(' |')[0]
    date = date1 + ', ' + date2
    away_team = teams.split(' at ')[0]
    home_team = teams.split(' at ')[1]
    return date, home_team, away_team

#date, home_team, away_team = get_teams_and_date(soupObject)


cols = ['date', 'home_team', 'away_team', 'home_team_mp', 'home_team_fg', 'home_team_fga', 
        'home_team_fg_pct', 'home_team_fg3', 'home_team_fg3a', 'home_team_fg3_pct', 
        'home_team_ft', 'home_team_fta', 'home_team_ft_pct', 'home_team_orb', 'home_team_drb', 
        'home_team_trb', 'home_team_ast', 'home_team_stl', 'home_team_blk',
        'home_team_tov', 'home_team_pf', 'home_team_pts', 'home_team_MP', 'home_team_TSpct', 
        'home_team_eFGpct', 'home_team_3PAr', 'home_team_FTr', 'home_team_ORBpct', 
        'home_team_DRBpct', 'home_team_TRBpct', 'home_team_ASTpct', 'home_team_STLpct', 
        'home_team_BLKpct', 'home_team_TOVpct', 'home_team_USGpct', 'home_team_ORtg', 'home_team_DRtg',
        'away_team_mp', 'away_team_fg', 'away_team_fga', 'away_team_fg_pct', 
        'away_team_fg3', 'away_team_fg3a', 'away_team_fg3_pct', 'away_team_ft', 
        'away_team_fta', 'away_team_ft_pct', 'away_team_orb', 'away_team_drb', 
        'away_team_trb', 'away_team_ast', 'away_team_stl', 'away_team_blk',
        'away_team_tov', 'away_team_pf', 'away_team_pts', 'away_team_MP', 
        'away_team_TSpct', 'away_team_eFGpct', 'away_team_3PAr', 'away_team_FTr', 
        'away_team_ORBpct', 'away_team_DRBpct', 'away_team_TRBpct', 'away_team_ASTpct', 
        'away_team_STLpct', 'away_team_BLKpct', 'away_team_TOVpct', 'away_team_USGpct', 
        'away_team_ORtg', 'away_team_DRtg', 'away_team_starters', 'home_team_starters',
        'away_team_starters_min', 'home_team_starters_min']


#len(cols)  # 75
#len(team2_basic_stats_list) # 19
#len(team2_adv_stats_list)  # 15
#len(team1_basic_stats_list)  # 19
#len(team1_adv_stats_list)  # 15
#len([date, home_team, away_team, starters_string_team1, 
#     starters_string_team2, starters_min_string_team1, 
#     starters_min_string_team2])  # 7
# = 75 cols
#[240.0,
# 38.0,
# 87.0,
# 0.437,
# 10.0,
# 33.0,
# 0.303,
# 17.0,
# 20.0,
# 0.85,
# 12.0,
# 40.0,
# 52.0,
# 20.0,
# 6.0,
# 4.0,
# 16.0,
# 23.0,
# 103.0]


#link = box_score_links_list[0]
def get_data_from_all_box_scores(box_score_links_list, cols):
    df_all_games = pd.DataFrame()
    counter = 0
    for link in box_score_links_list[:]:
        counter = counter + 1
        print(counter)
        soupObject = get_box_score_soup_object(link)
        starters_string_team1, starters_string_team2 = get_starters_for_box_score(soupObject)
        team1_basic_stats_list, team1_adv_stats_list, team2_basic_stats_list, team2_adv_stats_list = give_game_stats(soupObject)
        starters_min_string_team1, starters_min_string_team2 = get_starter_mins_for_box_score(soupObject)

        date, home_team, away_team = get_teams_and_date(soupObject)

#        df_game = pd.DataFrame([[date] + [home_team] + [away_team] + team2_basic_stats_list +
#                                team2_adv_stats_list + team1_basic_stats_list + team1_adv_stats_list +
#                                [starters_string_team1] + [starters_string_team2] + 
#                                [starters_min_string_team1] + [starters_min_string_team2]])

        df_game = pd.DataFrame([[date] + [home_team] + [away_team] + team2_basic_stats_list +
                                team2_adv_stats_list + team1_basic_stats_list + team1_adv_stats_list +
                                [starters_string_team1] + [starters_string_team2] + 
                                [starters_min_string_team1] + [starters_min_string_team2]], columns = cols)
        df_all_games = pd.concat([df_all_games, df_game], ignore_index=True)
    return df_all_games
    
df_all_games = get_data_from_all_box_scores(box_score_links_list[:], cols)

df_all_games['date'] = pd.to_datetime(df_all_games['date'])
df_all_games['venue'] = 'home'



df_all_games_switched = df_all_games.copy(deep=True)
df_all_games_switched['venue'] = 'away'
#df_all_games_switched['spread'] = df_all_games_switched['spread']*-1

variables = df_all_games_switched.columns

for var in variables:
    if var[:4] == 'home':
        print(var)
        df_all_games_switched.rename(columns={var:'xaway'+var[4:]}, inplace=True)
    elif var[:4] == 'away':
        print(var)
        df_all_games_switched.rename(columns={var:'xhome'+var[4:]}, inplace=True)

variables = df_all_games_switched.columns
for var in variables:
    if var[:1] == 'x':
        df_all_games_switched.rename(columns={var:var[1:]}, inplace=True)


variables = df_all_games_switched.columns

for var in variables:
    if var[:4] == 'home':
        df_all_games_switched.rename(columns={var:var[5:]}, inplace=True)
        df_all_games.rename(columns={var:var[5:]}, inplace=True)
    elif var[:4] == 'away':
        df_all_games_switched.rename(columns={var:'opponent'+var[9:]}, inplace=True)
        df_all_games.rename(columns={var:'opponent'+var[9:]}, inplace=True)
        

df_all_games.rename(columns={'team_starters': 'starters_team'}, inplace=True)
df_all_games.rename(columns={'opponent_starters': 'starters_opponent'}, inplace=True)
df_all_games_switched.rename(columns={'team_starters': 'starters_team'}, inplace=True)
df_all_games_switched.rename(columns={'opponent_starters': 'starters_opponent'}, inplace=True)

df_doubled = pd.concat([df_all_games, df_all_games_switched], ignore_index=True)
df_doubled['venue_y'] = df_doubled['venue']
df_doubled['venue_x'] = np.nan
df_doubled.loc[df_doubled['venue']=='home', 'venue_x'] = 0
df_doubled.loc[df_doubled['venue']=='away', 'venue_x'] = 1

# replace old team names in bball ref w current ones (used by covers)
def replace_bball_ref_names(df_bball_ref):
    df_bball_ref['team'].replace(['New Orleans Hornets', 'Charlotte Bobcats', 'New Jersey Nets', 'Seattle SuperSonics'],
    ['New Orleans Pelicans', 'Charlotte Hornets', 'Brooklyn Nets', 'Oklahoma City Thunder'], inplace=True)
    df_bball_ref['opponent'].replace(['New Orleans Hornets', 'Charlotte Bobcats', 'New Jersey Nets', 'Seattle SuperSonics'],
    ['New Orleans Pelicans', 'Charlotte Hornets', 'Brooklyn Nets', 'Oklahoma City Thunder'], inplace=True)
    df_bball_ref['team'].replace('New Orleans/Oklahoma City Hornets', 'New Orleans Pelicans', inplace=True)
    df_bball_ref['opponent'].replace('New Orleans/Oklahoma City Hornets', 'New Orleans Pelicans', inplace=True)
    return df_bball_ref

df_doubled = replace_bball_ref_names(df_doubled)
#len(df_bball_ref_2004_2015_doubled['team'].unique())
df_doubled[['team', 'opponent', 'team_DRBpct', 'opponent_DRBpct']]
df_doubled.dtypes


# plan: save ea days data as separate csv file. use same date vars that 
# used to scrape the box scores
df_doubled.to_csv('df_'+str(year_yesterday)+'_'+str(month_yesterday)+'_'+str(day_yesterday)+'.csv')
# ---------



df_schedule = pd.read_excel('schedule_2016_2017.xlsx')
df_schedule.head()
df_schedule[df_schedule['date']=='2016-11-30']
df_schedule = df_schedule[df_schedule.index!=271]
df_schedule = df_schedule.reset_index()

df_schedule_away = df_schedule.copy(deep=True)
df_schedule_home = df_schedule.copy(deep=True)
df_schedule_away.rename(columns={'away_team':'team'}, inplace=True)
df_schedule_away.rename(columns={'home_team':'opponent'}, inplace=True)
df_schedule_away['venue'] = 'away'
df_schedule_home.rename(columns={'away_team':'opponent'}, inplace=True)
df_schedule_home.rename(columns={'home_team':'team'}, inplace=True)
df_schedule_home['venue'] = 'home'
df_schedule_doubled = pd.concat([df_schedule_home, df_schedule_away], ignore_index=True)
df_schedule_doubled = df_schedule_doubled.sort_values(by=['team', 'date'])
df_schedule_doubled = df_schedule_doubled.reset_index(drop=True)
df_schedule_doubled = df_schedule_doubled[['date', 'team', 'opponent', 'venue']]

# if one day ahead in viet:
#dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')
#date = dates[-3]  #  replace to -2 once get back days
#day_yesterday = date.day
#month_yesterday = date.month
#year_yesterday = date.year

# then # loop through dates of season and open each and concat
# to get a range of dates, from first date of season to yesterday
#dates = pd.date_range('2016/10/25', '2016/11/5', freq='D')
dates = pd.date_range('25/10/2016', str(month_yesterday)+'/'+str(day_yesterday)+'/'+str(year_yesterday), freq='D')
df_this_season = pd.DataFrame()
for date in dates:
    print(date)
    if date.strftime('%Y-%m-%d') in df_schedule_doubled['date'].astype(str).values:
        #print('ok', date)
        df_day = pd.read_csv('df_'+str(date.year)+'_'+str(date.month)+'_'+str(date.day)+'.csv')
        df_this_season = pd.concat([df_this_season,df_day], ignore_index=True)
    else: None



# open file for this season 
#df_this_season = pd.read_csv('df_this_season.csv')
# concat file for this season w newly scraped data
#df_this_season = pd.concat([df_this_season, df_doubled], ignore_index=True)

# make sure don't have any duplicate games in new concatenated file
# worry about this later
#s = df_this_season[['date','team','opponent']].astype(str)
#x = s.values
#x = [list(l) for l in x]


len(df_covers_bball_ref)  # 29038
len(df_this_season)

# merge new games with historical 
df_covers_bball_ref = pd.concat([df_covers_bball_ref, df_this_season], ignore_index=True)
df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
df_covers_bball_ref.tail()
df_covers_bball_ref[['date', 'team', 'opponent', 'spread']][df_covers_bball_ref['team']=='Utah Jazz'].tail(10)
len(df_covers_bball_ref)  # 29088 on oct 29, but will grow

# ----------------------------
# to incorpo spread and totals
# open schedule: 
#df_schedule = pd.read_excel('schedule_2016_2017.xlsx')
#df_schedule.head()
#df_schedule_away = df_schedule.copy(deep=True)
#df_schedule_home = df_schedule.copy(deep=True)
#df_schedule_away.rename(columns={'away_team':'team'}, inplace=True)
#df_schedule_away.rename(columns={'home_team':'opponent'}, inplace=True)
#df_schedule_away['venue'] = 'away'
#df_schedule_home.rename(columns={'away_team':'opponent'}, inplace=True)
#df_schedule_home.rename(columns={'home_team':'team'}, inplace=True)
#df_schedule_home['venue'] = 'home'
#df_schedule_doubled = pd.concat([df_schedule_home, df_schedule_away], ignore_index=True)
#df_schedule_doubled = df_schedule_doubled.sort_values(by=['team', 'date'])
#df_schedule_doubled = df_schedule_doubled.reset_index(drop=True)
#df_schedule_doubled = df_schedule_doubled[['date', 'team', 'opponent', 'venue']]
#dates_in_schedule = df_schedule_doubled['date']


# import spread and totals that i copied from pinnacle
def fix_df_odds_1(df_odds):
    df_odds = df_odds.iloc[:,0]
    df_odds = df_odds.reset_index()
    df_odds['game'] = np.nan
    df_odds.columns = ['index', 'stats', 'game']
    df_insert = pd.DataFrame({'index':[0], 'stats':[0], 'game':[0]})
    df_odds = pd.concat([df_insert, df_odds], ignore_index=True)
    return df_odds

#df_odds = fix_df_odds_1(df_odds)

def fix_df_odds_2(df_odds):
    stats_col = df_odds['stats'].values
    stats_col = list(stats_col)
    game = 1
    for i, item in enumerate(stats_col):
        if str(item) != 'nan':
            stats_col[i] = game
        elif str(item) == 'nan':
            stats_col[i] = game + 1
            game = game + 1    
    df_odds['game'] = stats_col
    df_odds = df_odds[df_odds['stats'].notnull()]
    return df_odds

#df_odds = fix_df_odds_2(df_odds)

def fix_df_odds_3(df_odds):
    df_odds_group = df_odds.groupby('game')
    i = 1
    df_games = pd.DataFrame()
    for i in range(len(df_odds_group)):
        print(i+1)
        df_group = df_odds_group.get_group(i+1)
        df_group = df_group.transpose()  
        df_group = df_group[df_group.index == 'stats']
        df_group.columns = ['number', 'away_team', 'away_ml', 'spread_away', 'over', 'juice_away', 'under', 'juice_home', 'home_team', 'home_ml', 'spread_home']
        df_games = pd.concat([df_games, df_group])
    df_games = df_games.reset_index(drop=True)
    return df_games

#df_games = fix_df_odds_3(df_odds)

def fix_df_odds_doubled(df_games):
    df_spread_away = df_games.copy(deep=True)
    df_spread_home = df_games.copy(deep=True)
    # spread away df
    df_spread_away.rename(columns={'away_team':'team'}, inplace=True)
    df_spread_away.rename(columns={'home_team':'opponent'}, inplace=True)
    df_spread_away['venue'] = 'away'
    df_spread_away['totals'] = df_spread_away['over'].str.split(' ').str[1].astype(float)
    df_spread_away['spread'] = df_spread_away['spread_away']
    df_spread_away['moneyline'] = df_spread_away['away_ml']
    df_spread_away['juice'] = df_spread_away['juice_away']
    df_spread_away.pop('number')
    df_spread_away.pop('over')
    df_spread_away.pop('under')
    df_spread_away.pop('away_ml')
    df_spread_away.pop('home_ml')
    df_spread_away.pop('spread_away')
    df_spread_away.pop('spread_home')
    df_spread_away.pop('juice_away')
    df_spread_away.pop('juice_home')
    # spread home df
    df_spread_home.rename(columns={'away_team':'opponent'}, inplace=True)
    df_spread_home.rename(columns={'home_team':'team'}, inplace=True)
    df_spread_home['venue'] = 'home'
    df_spread_home['spread'] = df_spread_home['spread_home']
    df_spread_home['totals'] = df_spread_home['over'].str.split(' ').str[1].astype(float)
    df_spread_home['moneyline'] = df_spread_home['home_ml']
    df_spread_home['juice'] = df_spread_home['juice_home']
    df_spread_home.pop('number')
    df_spread_home.pop('over')
    df_spread_home.pop('under')
    df_spread_home.pop('away_ml')
    df_spread_home.pop('home_ml')
    df_spread_home.pop('spread_away')
    df_spread_home.pop('spread_home')
    df_spread_home.pop('juice_away')
    df_spread_home.pop('juice_home')
    # create spread doubled
    df_spread_doubled = pd.concat([df_spread_away, df_spread_home], ignore_index=True)
    return df_spread_doubled

def tranform_odds_today(df_odds, month_now, day_now, year_now):
    df_odds = fix_df_odds_1(df_odds)
    df_odds = fix_df_odds_2(df_odds)
    df_games = fix_df_odds_3(df_odds)
    df_spread_doubled = fix_df_odds_doubled(df_games)
    df_spread_doubled['date'] = pd.to_datetime(str(month_now)+'/'+str(day_now)+'/'+str(year_now))
    return df_spread_doubled

#df_spread_doubled = tranform_odds_today(df_odds, month_now, day_now, year_now)


def produce_df_w_all_spreads_this_season(month_now, day_now, year_now):
    dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')
    df_spreads_this_season = pd.DataFrame()
    for date in dates:
        #print(date)
        if date.strftime('%Y-%m-%d') in df_schedule_doubled['date'].astype(str).values:
            print('ok', date)
            df_odds = pd.read_excel(str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'_'+'odds.xlsx')
            df_spread_doubled = tranform_odds_today(df_odds, date.month, date.day, date.year)
            df_spreads_this_season = pd.concat([df_spreads_this_season, df_spread_doubled], ignore_index=True)
    return df_spreads_this_season


# SKIP --------------------
# if one day ahead in viet:
#dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')
#date = dates[-2]  #  replace to -2 once get back days
#day_now = date.day
#month_now = date.month
#year_now = date.year
#--------------------------

df_spreads_this_season = produce_df_w_all_spreads_this_season(month_now, day_now, year_now)

# to get spread for up to yesterday
#df_spreads_this_season = produce_df_w_all_spreads_this_season(month_yesterday, day_yesterday, year_yesterday)

#df_odds = pd.read_excel(str(month_now)+'_'+str(day_now)+'_'+str(year_now)+'_'+'odds.xlsx')
#df_odds = pd.read_excel('10_26_2016_odds.xlsx')

#df_spread_doubled_25 = df_spread_doubled.copy(deep=True)
#df_spread_doubled_25['date'] = pd.to_datetime('10/25/2016')
#df_spread_doubled_26 = df_spread_doubled.copy(deep=True)
#df_spread_doubled_26['date'] = pd.to_datetime('10/26/2016')
#df_spread_doubled_27 = df_spread_doubled.copy(deep=True)
#df_spread_doubled_3_days = pd.concat([df_spread_doubled_25, df_spread_doubled_26, df_spread_doubled_27], ignore_index=True)

# NEXT - MAP THESES SPREAD AND TOTALS ONTO SEASON SCHEDULE
# THEN MERGE THE SCHEDULE WITH THE MAIN DATA FILE ON DATE-TEAM-OPPONENT
# THEN SHOULD BE READY?

#df_schedule_w_spread = pd.read_csv('df_schedule_doubled.csv')
#df_schedule_w_spread['date'] = pd.to_datetime(df_schedule_w_spread['date'])
#df_schedule_w_spread.dtypes


#df_schedule_w_spread = pd.merge(df_schedule_doubled, df_spread_doubled_3_days, on=['date','team','opponent'], how='outer')
df_schedule_w_spread = pd.merge(df_schedule_doubled, df_spreads_this_season, on=['date','team','opponent'], how='outer')
df_schedule_w_spread.sort_values(by='date')[50:80]
df_schedule_w_spread = df_schedule_w_spread.sort_values(by=['team','date'])
df_schedule_w_spread = df_schedule_w_spread.reset_index(drop=True)

# --------------------------
# get spread integrated with the day's stats here?
df_covers_bball_ref[['date', 'team', 'opponent']].dtypes
df_schedule_w_spread[['date', 'team', 'opponent']].dtypes
df_covers_bball_ref['date'] = pd.to_datetime(df_covers_bball_ref['date'])
# good, all team names are the same
sorted(df_covers_bball_ref['team'].unique()) == sorted(df_schedule_w_spread['team'].unique())
len(df_covers_bball_ref)  # 29088 (but will grow every day with more box scores) 

df_schedule_w_spread.rename(columns={'venue_x':'venue_schedule_1'}, inplace=True)
df_schedule_w_spread.rename(columns={'venue_y':'venue_schedule_2'}, inplace=True)
df_schedule_w_spread.rename(columns={'juice':'juice_schedule'}, inplace=True)
df_schedule_w_spread.rename(columns={'moneyline':'moneyline_schedule'}, inplace=True)
df_schedule_w_spread.rename(columns={'spread':'spread_schedule'}, inplace=True)
df_schedule_w_spread.rename(columns={'totals':'totals_schedule'}, inplace=True)


df_covers_bball_ref = df_covers_bball_ref.merge(df_schedule_w_spread, on=['date', 'team', 'opponent'], how='outer')
len(df_covers_bball_ref)  # 31498 (should stay same all season, i think)
df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
df_covers_bball_ref[['date', 'team', 'spread', 'spread_schedule', 'venue_x']][960:1000]
# create numeric venue in schedule
df_covers_bball_ref['venue_schedule_numeric'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_schedule_2'] == 'home', 'venue_schedule_numeric'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['venue_schedule_2'] == 'away', 'venue_schedule_numeric'] = 1

df_covers_bball_ref.loc[df_covers_bball_ref['spread_schedule'].notnull(), 'spread'] = df_covers_bball_ref['spread_schedule']
df_covers_bball_ref.loc[df_covers_bball_ref['totals_schedule'].notnull(), 'totals'] = df_covers_bball_ref['totals_schedule']
df_covers_bball_ref.loc[df_covers_bball_ref['venue_schedule_2'].notnull(), 'venue_x'] = df_covers_bball_ref['venue_schedule_numeric']
df_covers_bball_ref.loc[df_covers_bball_ref['venue_schedule_2'].notnull(), 'venue_y'] = df_covers_bball_ref['venue_schedule_2']

df_covers_bball_ref['score_oppt'].head()
df_covers_bball_ref.loc[df_covers_bball_ref['opponent_pts'].notnull(), 'score_oppt'] = df_covers_bball_ref['opponent_pts']
df_covers_bball_ref.loc[df_covers_bball_ref['team_pts'].notnull(), 'score_team'] = df_covers_bball_ref['team_pts']

df_covers_bball_ref[['date','season_start']][960:1000]
df_covers_bball_ref.loc[df_covers_bball_ref['season_start'].isnull(), 'season_start'] = 2016
#df_covers_bball_ref[df_covers_bball_ref['season_start'] == 2016]

df_covers_bball_ref[['date', 'team', 'score_oppt', 'opponent_pts']][950:980]
df_covers_bball_ref[['date', 'team', 'score_team', 'team_pts']][950:980]
df_covers_bball_ref[['date', 'team', 'spread', 'venue_x', 'venue_y', 'venue_schedule_2']][960:1000]
# ----------------------


# compute pace
df_covers_bball_ref.loc[:, 'team_possessions'] = ((df_covers_bball_ref.loc[:, 'team_fga'] + .44*df_covers_bball_ref.loc[:, 'team_fta'] - df_covers_bball_ref.loc[:, 'team_orb'] + df_covers_bball_ref.loc[:, 'team_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'opponent_possessions'] = ((df_covers_bball_ref.loc[:, 'opponent_fga'] + .44*df_covers_bball_ref.loc[:, 'opponent_fta'] - df_covers_bball_ref.loc[:, 'opponent_orb'] + df_covers_bball_ref.loc[:, 'opponent_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'the_team_pace'] = df_covers_bball_ref[['team_possessions', 'opponent_possessions']].mean(axis=1)
df_covers_bball_ref[['team_possessions', 'opponent_possessions', 'the_team_pace']].head(20)
df_covers_bball_ref[['date', 'team_possessions', 'opponent_possessions', 'the_team_pace']][960:1000]


# SKIP FOR AUTOMATION
# assign the totals and spread variables to use
#def assign_spread_and_totals(df_covers_bball_ref, spread_to_use, totals_to_use):
#    df_covers_bball_ref.loc[:, 'totals_covers'] = df_covers_bball_ref.loc[:, 'totals']
#    df_covers_bball_ref.loc[:, 'spread_covers'] = df_covers_bball_ref.loc[:, 'spread']
#    
#    df_covers_bball_ref.loc[:, 'totals'] = df_covers_bball_ref.loc[:, totals_to_use]
#    #df_covers_bball_ref.loc[:, 'totals'] = df_covers_bball_ref.loc[:, 'Closing Total']
#    
#    df_covers_bball_ref.loc[:, 'spread'] = df_covers_bball_ref.loc[:, spread_to_use]
#    #df_covers_bball_ref.loc[:, 'spread'] = df_covers_bball_ref.loc[:, 'team_close_spread']
#    return df_covers_bball_ref
#
##df_covers_bball_ref = assign_spread_and_totals(df_covers_bball_ref, 'team_close_spread', 'Closing Total')
##df_covers_bball_ref = assign_spread_and_totals(df_covers_bball_ref, 'team_close_spread', 'Closing Total')
#df_covers_bball_ref = assign_spread_and_totals(df_covers_bball_ref, 'spread', 'totals')
## continue to assign 'spread' and 'totals' because notw

#df_covers_bball_ref.loc[:,'totals_abs_diff'] = np.abs(df_covers_bball_ref.loc[:,'totals_covers'] - df_covers_bball_ref.loc[:,'Closing Total'])
#df_covers_bball_ref['spread_abs_diff'] = np.abs(df_covers_bball_ref.loc[:,'spread_covers'] - df_covers_bball_ref.loc[:,'team_close_spread'])
#df_covers_bball_ref[['totals_abs_diff', 'spread_abs_diff']].hist()


def create_variables_covers(df_teams_covers):
    # compute predicted team and oppt points, and other metrics
    df_teams_covers['team_predicted_points'] = (df_teams_covers['totals']/2) - (df_teams_covers['spread']/2)
    df_teams_covers['oppt_predicted_points'] = (df_teams_covers['totals']/2) + (df_teams_covers['spread']/2)
    df_teams_covers = df_teams_covers.sort(['team', 'date'])
    #df_teams_covers['score_team_expanding'] = df_teams_covers.groupby('team')['score_team'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    #df_teams_covers['score_oppt_expanding'] = df_teams_covers.groupby('team')['score_oppt'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    #df_teams_covers['team_expanding_vs_pred_points'] = df_teams_covers['team_predicted_points'] - df_teams_covers['score_team_expanding']
    #df_teams_covers['oppt_expanding_vs_pred_points'] = df_teams_covers['oppt_predicted_points'] - df_teams_covers['score_oppt_expanding']
    #df_teams_covers['spread_expanding_mean'] = df_teams_covers.groupby('team')['spread'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    #df_teams_covers['current_spread_vs_spread_expanding'] = df_teams_covers['spread'] - df_teams_covers['spread_expanding_mean']    
    # how much beating the spread by of late -- to get sense of under/over performance of late
    df_teams_covers['beat_spread'] = df_teams_covers['spread'] + (df_teams_covers['score_team'] - df_teams_covers['score_oppt'])    
    #df_teams_covers['beat_spread_rolling_mean_11'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_mean(x.shift(1), 11, min_periods=5))
    #df_teams_covers['beat_spread_rolling_std_11'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_std(x.shift(1), 11, min_periods=5))
    df_teams_covers['beat_spread_last_g'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: x.shift(1))
    return df_teams_covers

df_covers_bball_ref = create_variables_covers(df_covers_bball_ref)

df_covers_bball_ref[['date', 'team_predicted_points', 'beat_spread_rolling_std_11']][960:1000]
df_covers_bball_ref[['date', 'team_pts', 'score_team', 'point_difference']][960:1000]
df_covers_bball_ref.loc[:, 'point_difference'] = df_covers_bball_ref.loc[:, 'score_team'] - df_covers_bball_ref.loc[:, 'score_oppt']


# rename vars
df_covers_bball_ref.rename(columns={'team_predicted_points':'the_team_predicted_points'}, inplace=True)
df_covers_bball_ref.rename(columns={'oppt_predicted_points':'the_oppt_predicted_points'}, inplace=True)




# ---
# get players ages
df_players = pd.read_excel('players_nba.xlsx')
df_players['feet'] = df_players['Ht'].str.split('-').str[0]
df_players['feet'] = df_players['feet'].astype(float)
df_players['inches'] = df_players['Ht'].str.split('-').str[1]
df_players['inches'] = df_players['inches'].astype(float)
df_players.loc[:, 'height'] = (df_players.loc[:, 'feet']*12) + df_players.loc[:, 'inches']
df_players.loc[:, 'bmi'] = (df_players.loc[:,'Wt'] * 703) / (df_players.loc[:, 'height'] * df_players.loc[:, 'height'])

df_players['Player'] = df_players['Player'].str.strip('*')
df_players['first'] = df_players['Player'].str.split(' ').str[0]
df_players['second'] = df_players['Player'].str.split(' ').str[1]
df_players['third'] = df_players['Player'].str.split(' ').str[2]
df_players['fourth'] = df_players['Player'].str.split(' ').str[3]
df_players[(df_players['third'].notnull()) & (df_players['To']>2003)]

# put hyphen between middle and last. do for df_covers_bball_ref and df_players
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Nando De Colo', 'Nando De-Colo')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('John Lucas III', 'John Lucas-III')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Luc Mbah a Moute', 'Luc Mbah-a-Moute')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('James Michael McAdoo', 'James Michael-McAdoo')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Larry Nance Jr.', 'Larry Nance-Jr.')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Juan Carlos Navarro', 'Juan Carlos-Navarro')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Peter John Ramos', 'Peter John-Ramos')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Nick Van Exel', 'Nick Van-Exel')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Keith Van Horn', 'Keith Van-Horn')
df_covers_bball_ref['starters_team'] = df_covers_bball_ref['starters_team'].str.replace('Metta World Peace', 'Metta World-Peace')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Nando De Colo', 'Nando De-Colo')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('John Lucas III', 'John Lucas-III')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Luc Mbah a Moute', 'Luc Mbah-a-Moute')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('James Michael McAdoo', 'James Michael-McAdoo')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Larry Nance Jr.', 'Larry Nance-Jr.')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Juan Carlos Navarro', 'Juan Carlos-Navarro')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Peter John Ramos', 'Peter John-Ramos')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Nick Van Exel', 'Nick Van-Exel')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Keith Van Horn', 'Keith Van-Horn')
df_covers_bball_ref['starters_opponent'] = df_covers_bball_ref['starters_opponent'].str.replace('Metta World Peace', 'Metta World-Peace')

df_players['Player'] = df_players['Player'].str.replace('Nando De Colo', 'Nando De-Colo')
df_players['Player'] = df_players['Player'].str.replace('John Lucas III', 'John Lucas-III')
df_players['Player'] = df_players['Player'].str.replace('Luc Mbah a Moute', 'Luc Mbah-a-Moute')
df_players['Player'] = df_players['Player'].str.replace('James Michael McAdoo', 'James Michael-McAdoo')
df_players['Player'] = df_players['Player'].str.replace('Larry Nance Jr.', 'Larry Nance-Jr.')
df_players['Player'] = df_players['Player'].str.replace('Juan Carlos Navarro', 'Juan Carlos-Navarro')
df_players['Player'] = df_players['Player'].str.replace('Peter John Ramos', 'Peter John-Ramos')
df_players['Player'] = df_players['Player'].str.replace('Nick Van Exel', 'Nick Van-Exel')
df_players['Player'] = df_players['Player'].str.replace('Keith Van Horn', 'Keith Van-Horn')
df_players['Player'] = df_players['Player'].str.replace('Metta World Peace', 'Metta World-Peace')

df_covers_bball_ref['starter_team_1'] = df_covers_bball_ref['starters_team'].str.split(' ').str[0] +  ' ' + df_covers_bball_ref['starters_team'].str.split(' ').str[1] 
df_covers_bball_ref['starter_team_2'] = df_covers_bball_ref['starters_team'].str.split(' ').str[2] +  ' ' + df_covers_bball_ref['starters_team'].str.split(' ').str[3] 
df_covers_bball_ref['starter_team_3'] = df_covers_bball_ref['starters_team'].str.split(' ').str[4] +  ' ' + df_covers_bball_ref['starters_team'].str.split(' ').str[5] 
df_covers_bball_ref['starter_team_4'] = df_covers_bball_ref['starters_team'].str.split(' ').str[6] +  ' ' + df_covers_bball_ref['starters_team'].str.split(' ').str[7] 
df_covers_bball_ref['starter_team_5'] = df_covers_bball_ref['starters_team'].str.split(' ').str[8] +  ' ' + df_covers_bball_ref['starters_team'].str.split(' ').str[9] 
df_covers_bball_ref['starter_opponent_1'] = df_covers_bball_ref['starters_opponent'].str.split(' ').str[0] +  ' ' + df_covers_bball_ref['starters_opponent'].str.split(' ').str[1] 
df_covers_bball_ref['starter_opponent_2'] = df_covers_bball_ref['starters_opponent'].str.split(' ').str[2] +  ' ' + df_covers_bball_ref['starters_opponent'].str.split(' ').str[3] 
df_covers_bball_ref['starter_opponent_3'] = df_covers_bball_ref['starters_opponent'].str.split(' ').str[4] +  ' ' + df_covers_bball_ref['starters_opponent'].str.split(' ').str[5] 
df_covers_bball_ref['starter_opponent_4'] = df_covers_bball_ref['starters_opponent'].str.split(' ').str[6] +  ' ' + df_covers_bball_ref['starters_opponent'].str.split(' ').str[7] 
df_covers_bball_ref['starter_opponent_5'] = df_covers_bball_ref['starters_opponent'].str.split(' ').str[8] +  ' ' + df_covers_bball_ref['starters_opponent'].str.split(' ').str[9] 

#df_covers_bball_ref['starter_team_extra'] = df_covers_bball_ref['starters_team'].str.split(' ').str[10] 
#df_covers_bball_ref['starter_opponent_extra'] = df_covers_bball_ref['starters_opponent'].str.split(' ').str[10] 
#df_covers_bball_ref[['starters_team', 'starter_team_extra']][df_covers_bball_ref['starter_team_extra'].notnull()]

# create player to height, player to weight, player to bmi dict to map:
player_to_height_dict = dict(zip(df_players['Player'], df_players['height']))
player_to_weight_dict = dict(zip(df_players['Player'], df_players['Wt']))
player_to_bmi_dict = dict(zip(df_players['Player'], df_players['bmi']))
player_to_bday_dict = dict(zip(df_players['Player'], df_players['Birth Date']))


df_covers_bball_ref['starter_team_1_height'] = df_covers_bball_ref['starter_team_1'].map(player_to_height_dict)
df_covers_bball_ref['starter_team_2_height'] = df_covers_bball_ref['starter_team_2'].map(player_to_height_dict)
df_covers_bball_ref['starter_team_3_height'] = df_covers_bball_ref['starter_team_3'].map(player_to_height_dict)
df_covers_bball_ref['starter_team_4_height'] = df_covers_bball_ref['starter_team_4'].map(player_to_height_dict)
df_covers_bball_ref['starter_team_5_height'] = df_covers_bball_ref['starter_team_5'].map(player_to_height_dict)
df_covers_bball_ref['starter_opponent_1_height'] = df_covers_bball_ref['starter_opponent_1'].map(player_to_height_dict)
df_covers_bball_ref['starter_opponent_2_height'] = df_covers_bball_ref['starter_opponent_2'].map(player_to_height_dict)
df_covers_bball_ref['starter_opponent_3_height'] = df_covers_bball_ref['starter_opponent_3'].map(player_to_height_dict)
df_covers_bball_ref['starter_opponent_4_height'] = df_covers_bball_ref['starter_opponent_4'].map(player_to_height_dict)
df_covers_bball_ref['starter_opponent_5_height'] = df_covers_bball_ref['starter_opponent_5'].map(player_to_height_dict)
df_covers_bball_ref[['starter_team_1', 'starter_team_1_height']]
df_covers_bball_ref[['starter_opponent_1', 'starter_opponent_1_height']]

df_covers_bball_ref['starter_team_1_bmi'] = df_covers_bball_ref['starter_team_1'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_team_2_bmi'] = df_covers_bball_ref['starter_team_2'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_team_3_bmi'] = df_covers_bball_ref['starter_team_3'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_team_4_bmi'] = df_covers_bball_ref['starter_team_4'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_team_5_bmi'] = df_covers_bball_ref['starter_team_5'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_opponent_1_bmi'] = df_covers_bball_ref['starter_opponent_1'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_opponent_2_bmi'] = df_covers_bball_ref['starter_opponent_2'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_opponent_3_bmi'] = df_covers_bball_ref['starter_opponent_3'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_opponent_4_bmi'] = df_covers_bball_ref['starter_opponent_4'].map(player_to_bmi_dict)
df_covers_bball_ref['starter_opponent_5_bmi'] = df_covers_bball_ref['starter_opponent_5'].map(player_to_bmi_dict)
df_covers_bball_ref[['starter_opponent_1', 'starter_opponent_1_bmi']]

df_covers_bball_ref['starter_team_1_bday'] = df_covers_bball_ref['starter_team_1'].map(player_to_bday_dict)
df_covers_bball_ref['starter_team_2_bday'] = df_covers_bball_ref['starter_team_2'].map(player_to_bday_dict)
df_covers_bball_ref['starter_team_3_bday'] = df_covers_bball_ref['starter_team_3'].map(player_to_bday_dict)
df_covers_bball_ref['starter_team_4_bday'] = df_covers_bball_ref['starter_team_4'].map(player_to_bday_dict)
df_covers_bball_ref['starter_team_5_bday'] = df_covers_bball_ref['starter_team_5'].map(player_to_bday_dict)
df_covers_bball_ref['starter_opponent_1_bday'] = df_covers_bball_ref['starter_opponent_1'].map(player_to_bday_dict)
df_covers_bball_ref['starter_opponent_2_bday'] = df_covers_bball_ref['starter_opponent_2'].map(player_to_bday_dict)
df_covers_bball_ref['starter_opponent_3_bday'] = df_covers_bball_ref['starter_opponent_3'].map(player_to_bday_dict)
df_covers_bball_ref['starter_opponent_4_bday'] = df_covers_bball_ref['starter_opponent_4'].map(player_to_bday_dict)
df_covers_bball_ref['starter_opponent_5_bday'] = df_covers_bball_ref['starter_opponent_5'].map(player_to_bday_dict)

df_covers_bball_ref['starter_team_1_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_team_1_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_team_2_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_team_2_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_team_3_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_team_3_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_team_4_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_team_4_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_team_5_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_team_5_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_opponent_1_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_opponent_1_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_opponent_2_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_opponent_2_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_opponent_3_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_opponent_3_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_opponent_4_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_opponent_4_bday']) / np.timedelta64(1,'D')
df_covers_bball_ref['starter_opponent_5_age'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['starter_opponent_5_bday']) / np.timedelta64(1,'D')

df_covers_bball_ref['team_age'] = df_covers_bball_ref[['starter_team_1_age', 'starter_team_2_age', 'starter_team_3_age', 
'starter_team_4_age', 'starter_team_5_age']].mean(axis=1)
df_covers_bball_ref['team_age_std'] = df_covers_bball_ref[['starter_team_1_age', 'starter_team_2_age', 'starter_team_3_age', 
'starter_team_4_age', 'starter_team_5_age']].std(axis=1)
df_covers_bball_ref['opponent_age'] = df_covers_bball_ref[['starter_opponent_1_age', 'starter_opponent_2_age', 'starter_opponent_3_age', 
'starter_opponent_4_age', 'starter_opponent_5_age']].mean(axis=1)
df_covers_bball_ref['opponent_age_std'] = df_covers_bball_ref[['starter_opponent_1_age', 'starter_opponent_2_age', 'starter_opponent_3_age', 
'starter_opponent_4_age', 'starter_opponent_5_age']].std(axis=1)
df_covers_bball_ref['team_bmi'] = df_covers_bball_ref[['starter_team_1_bmi', 'starter_team_2_bmi', 'starter_team_3_bmi', 
'starter_team_4_bmi', 'starter_team_5_bmi']].mean(axis=1)
df_covers_bball_ref['opponent_bmi'] = df_covers_bball_ref[['starter_opponent_1_bmi', 'starter_opponent_2_bmi', 'starter_opponent_3_bmi', 
'starter_opponent_4_bmi', 'starter_opponent_5_bmi']].mean(axis=1)
df_covers_bball_ref['team_height'] = df_covers_bball_ref[['starter_team_1_height', 'starter_team_2_height', 'starter_team_3_height', 
'starter_team_4_height', 'starter_team_5_height']].mean(axis=1)
df_covers_bball_ref['opponent_height'] = df_covers_bball_ref[['starter_opponent_1_height', 'starter_opponent_2_height', 'starter_opponent_3_height', 
'starter_opponent_4_height', 'starter_opponent_5_height']].mean(axis=1)
df_covers_bball_ref['team_height_std'] = df_covers_bball_ref[['starter_team_1_height', 'starter_team_2_height', 'starter_team_3_height', 
'starter_team_4_height', 'starter_team_5_height']].std(axis=1)
df_covers_bball_ref['opponent_height_std'] = df_covers_bball_ref[['starter_opponent_1_height', 'starter_opponent_2_height', 'starter_opponent_3_height', 
'starter_opponent_4_height', 'starter_opponent_5_height']].std(axis=1)

#sns.lmplot(x='team_age', y='point_difference', data=df_covers_bball_ref, lowess=True, scatter_kws={'alpha':.01})
#sns.lmplot(x='team_age_std', y='point_difference', data=df_covers_bball_ref, lowess=True, scatter_kws={'alpha':.01})
#sns.lmplot(x='team_bmi', y='point_difference', data=df_covers_bball_ref, lowess=True, scatter_kws={'alpha':.01}, y_partial='spread')
#plt.ylim(-10,5)
#sns.lmplot(x='team_height', y='point_difference', data=df_covers_bball_ref, lowess=True, scatter_kws={'alpha':.01}, y_partial='spread')
#plt.ylim(-10,5)
#sns.lmplot(x='team_height_std', y='point_difference', data=df_covers_bball_ref, lowess=True, scatter_kws={'alpha':.01}, y_partial='spread')
#plt.ylim(-10,5)
#sns.lmplot(x='team_age_std', y='point_difference', data=df_covers_bball_ref, lowess=True, scatter_kws={'alpha':.01}, y_partial='spread')
#plt.ylim(-10,5)
#plt.xlim(0,2500)

# height diff at each ranking/position
#df_covers_bball_ref[['starter_team_1_height_rank', 'starter_team_2_height_rank', 'starter_team_3_height_rank', 
#'starter_team_4_height_rank', 'starter_team_5_height_rank']] = df_covers_bball_ref[['starter_team_1_height', 
#'starter_team_2_height', 'starter_team_3_height', 'starter_team_4_height', 'starter_team_5_height']].rank(axis=1)
#
#df_covers_bball_ref[['starter_opponent_1_height_rank', 'starter_opponent_2_height_rank', 
#'starter_opponent_3_height_rank', 'starter_opponent_4_height_rank', 
#'starter_opponent_5_height_rank']] = df_covers_bball_ref[['starter_opponent_1_height', 'starter_opponent_2_height', 
#'starter_opponent_3_height', 'starter_opponent_4_height', 'starter_opponent_5_height']].rank(axis=1)
#
#df_covers_bball_ref[['starter_team_1_height_rank', 'starter_team_2_height_rank', 'starter_team_3_height_rank', 
#'starter_team_4_height_rank', 'starter_team_5_height_rank']].head()
#df_covers_bball_ref[['starter_opponent_1_height_rank', 'starter_opponent_2_height_rank', 
#'starter_opponent_3_height_rank', 'starter_opponent_4_height_rank', 'starter_opponent_5_height_rank']].head()
#
#df_covers_bball_ref['team_starter_1_taller'] = 0
#df_covers_bball_ref.loc[(df_covers_bball_ref['starter_team_1_height_rank']==1 & df_covers_bball_ref['starter_opponent_1_height_rank']==1) &  
#(df_covers_bball_ref['starter_team_1_height'] > df_covers_bball_ref['starter_opponent_1_height']) ] = 1
#df_covers_bball_ref.loc[(df_covers_bball_ref['starter_team_1_height_rank']==1 & df_covers_bball_ref['starter_opponent_1_height_rank']==1) &  
#(df_covers_bball_ref['starter_team_1_height'] < df_covers_bball_ref['starter_opponent_1_height']) ] = -1


# ---
#df_covers_bball_ref[['starter_team_1_height_rank', 'starter_team_2_height_rank', 'starter_team_3_height_rank', 
#'starter_team_4_height_rank', 'starter_team_5_height_rank', 'starter_opponent_1_height_rank', 'starter_opponent_2_height_rank', 
#'starter_opponent_3_height_rank', 'starter_opponent_4_height_rank', 'starter_opponent_5_height_rank']] = df_covers_bball_ref[['starter_team_1_height', 
#'starter_team_2_height', 'starter_team_3_height', 'starter_team_4_height', 'starter_team_5_height', 'starter_opponent_1_height', 
#'starter_opponent_2_height', 'starter_opponent_3_height', 'starter_opponent_4_height', 'starter_opponent_5_height']].rank(axis=1)
#
#df_covers_bball_ref[['starter_team_1_height_rank', 'starter_team_2_height_rank', 'starter_team_3_height_rank', 
#'starter_team_4_height_rank', 'starter_team_5_height_rank', 'starter_opponent_1_height_rank', 'starter_opponent_2_height_rank', 
#'starter_opponent_3_height_rank', 'starter_opponent_4_height_rank', 'starter_opponent_5_height_rank', 
#'starter_team_1_height', 'starter_team_2_height', 'starter_team_3_height', 'starter_team_4_height', 'starter_team_5_height', 
#'starter_opponent_1_height', 'starter_opponent_2_height', 'starter_opponent_3_height', 'starter_opponent_4_height', 'starter_opponent_5_height']].head()
#
#df_covers_bball_ref['team_height_rank'] = df_covers_bball_ref[['starter_team_1_height_rank', 'starter_team_2_height_rank', 'starter_team_3_height_rank', 
#'starter_team_4_height_rank', 'starter_team_5_height_rank']].sum(axis=1)
#df_covers_bball_ref['opponent_height_rank'] = df_covers_bball_ref[['starter_opponent_1_height_rank', 'starter_opponent_2_height_rank', 
#'starter_opponent_3_height_rank', 'starter_opponent_4_height_rank', 'starter_opponent_5_height_rank']].sum(axis=1)

# make starter_team_1_height the shortest, etc.
df_covers_bball_ref[['starter_team_1_height_rank', 'starter_team_2_height_rank', 'starter_team_3_height_rank', 
'starter_team_4_height_rank', 'starter_team_5_height_rank']] = df_covers_bball_ref[['starter_team_1_height', 
'starter_team_2_height', 'starter_team_3_height', 'starter_team_4_height', 'starter_team_5_height']].rank(axis=1)


for i in range(1,6):
    print(i)
    df_covers_bball_ref['starter_height_'+str(i)+'_lowest'] = np.nan
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_1_height_rank']==i, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_1_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_2_height_rank']==i, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_2_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_3_height_rank']==i, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_3_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_4_height_rank']==i, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_4_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_5_height_rank']==i, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_5_height']  

for i in range(1,6):
    print(i)
    #df_covers_bball_ref['starter_height_'+str(i)+'_lowest'] = np.nan
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_1_height_rank']==i+.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_1_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_2_height_rank']==i+.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_2_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_3_height_rank']==i+.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_3_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_4_height_rank']==i+.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_4_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_5_height_rank']==i+.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_5_height']  

for i in range(1,6):
    print(i)
    #df_covers_bball_ref['starter_height_'+str(i)+'_lowest'] = np.nan
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_1_height_rank']==i-.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_1_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_2_height_rank']==i-.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_2_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_3_height_rank']==i-.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_3_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_4_height_rank']==i-.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_4_height']  
    df_covers_bball_ref.loc[df_covers_bball_ref['starter_team_5_height_rank']==i-.5, 'starter_height_'+str(i)+'_lowest'] =  df_covers_bball_ref['starter_team_5_height']  



variables_for_team_metrics = ['team_3PAr', 'team_ASTpct', 'team_BLKpct', 'team_DRBpct', 
'team_DRtg', 'team_FTr', 'team_ORBpct', 'team_ORtg', 'team_STLpct', 'team_TOVpct', 
'team_TRBpct', 'team_TSpct', 'team_eFGpct', 'team_fg3_pct', 'team_fg_pct', 'team_ft_pct', 
'team_pf', 'opponent_3PAr', 'opponent_ASTpct', 'opponent_BLKpct', 'opponent_DRBpct', 
'opponent_DRtg', 'opponent_FTr', 'opponent_ORBpct', 'opponent_ORtg', 'opponent_STLpct', 
'opponent_TOVpct', 'opponent_TRBpct', 'opponent_TSpct', 'opponent_eFGpct', 'opponent_fg3_pct', 
'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf', 'spread', 'beat_spread', 'venue_x',
'point_difference', 'the_team_pace', 'team_age', 'team_age_std', 'team_bmi', 'team_height']           


#------------------------------------------------------------------------------
# OPTIONAL - add noise to the spread: SKIP FOR NOW
#df_covers_bball_ref['random_number'] = np.random.normal(0, .4, len(df_covers_bball_ref))
#df_covers_bball_ref['random_number'].hist(alpha=.7)
#df_covers_bball_ref['totals'] = df_covers_bball_ref.loc[:, 'totals'] + df_covers_bball_ref.loc[:, 'random_number']
#
#df_covers_bball_ref['random_number'] = np.random.normal(0, .4, len(df_covers_bball_ref))
#df_covers_bball_ref['random_number'].hist(alpha=.7)
#df_covers_bball_ref['spread'] = df_covers_bball_ref.loc[:, 'spread'] + df_covers_bball_ref.loc[:, 'random_number']

# use ending spread or starting spread?
# ending spread is likely more accurate
# beginning spread may provide my predicted score with more room to move?



###########  create new variables to try out in this section  #############
# Choose one of next set of cells to create metrics for ea team/lineup


# try instead:
#def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
#    df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
#    df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
#    for var in variables_for_team_metrics:
#        var_ewma = var + '_ewma_15'
#        #var_ewma_2 = var + '_ewma'
#        # i have played around with span=x in line below
#        # 50 is worse than 20; 25 is worse than 20; 15 is better than 20; 10 is worse than 15; 16 is worse than 15; 14 is worse than 15
#        # stick w 15. but 12 is second best. 
#        # (12 looks better if want to use cutoffs and bet on fewer gs. but looks like 15 will get about same pct -- 53+ -- wih 1,000 gs)
#        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.rolling_mean(x.shift(1), window=15)) 
#        #df_covers_bball_ref[var_ewma_2] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=20)) 
#        #df_covers_bball_ref[var_ewma] = df_covers_bball_ref[[var_ewma, var_ewma_2]].mean(axis=1)
#    df_covers_bball_ref['beat_spread_std_ewma_15'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_std(x.shift(1), window=15))
#    df_covers_bball_ref['current_spread_vs_spread_ewma'] = df_covers_bball_ref.loc[:, 'spread'] - df_covers_bball_ref.loc[:, 'spread_ewma_15']
#    df_covers_bball_ref['starters_team_lag'] = df_covers_bball_ref['starters_team'].shift(1)
#    df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team_lag').cumcount()+1              
#    # CAN'T GROUP BY STARTERS WHEN AUTOMATING. BECAUSE DON'T KNOW STARTERS FOR CURRENT GAME    
#    #df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))                
#    #df_covers_bball_ref.loc[df_covers_bball_ref['lineup_count'].isnull(), 'lineup_count'] = 0 
#    #df_covers_bball_ref['beat_spread'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_max(x.shift(1), window=15))
#    return df_covers_bball_ref




# group just by team, not by season too. need to do weighted mean or a rolling mean here.
def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    #df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
    df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
    for var in variables_for_team_metrics:
        var_ewma = var + '_ewma_15'
        #var_ewma_2 = var + '_ewma'

        # i have played around with span=x in line below
        # 50 is worse than 20; 25 is worse than 20; 15 is better than 20; 10 is worse than 15; 16 is worse than 15; 14 is worse than 15
        # stick w 15. but 12 is second best. 
        # (12 looks better if want to use cutoffs and bet on fewer gs. but looks like 15 will get about same pct -- 53+ -- wih 1,000 gs)
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=15)) 
        #df_covers_bball_ref[var_ewma_2] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=20)) 
        #df_covers_bball_ref[var_ewma] = df_covers_bball_ref[[var_ewma, var_ewma_2]].mean(axis=1)

    # insert code to detect outliers -- see below for start of it

    df_covers_bball_ref['beat_spread_std_ewma_15'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.ewmstd(x.shift(1), span=15))
    df_covers_bball_ref['current_spread_vs_spread_ewma'] = df_covers_bball_ref.loc[:, 'spread'] - df_covers_bball_ref.loc[:, 'spread_ewma_15']
    df_covers_bball_ref['starters_team_lag'] = df_covers_bball_ref.groupby('team')['starters_team'].shift(1)
    df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team_lag').cumcount()+1              
    #df_covers_bball_ref['lineup_count_expanding'] = df_covers_bball_ref.groupby('starters_team_lag')['team_3PAr'].transform(lambda x: pd.expanding_count(x))             

    # adding new var to look like old bet bad var:
    df_covers_bball_ref['bet_bad_lag_2'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_mean(x.shift(2), window=12, min_periods=10))
    df_covers_bball_ref['bet_bad_lag_3'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_mean(x.shift(3), window=12, min_periods=10))
    df_covers_bball_ref['bet_bad_lag_4'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_mean(x.shift(4), window=12, min_periods=10))
    df_covers_bball_ref['bet_bad_lag_5'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_mean(x.shift(5), window=12, min_periods=10))

    df_covers_bball_ref['starter_team_1_height_lag'] = df_covers_bball_ref.groupby('team')['starter_height_1_lowest'].shift(1)
    df_covers_bball_ref['starter_team_2_height_lag'] = df_covers_bball_ref.groupby('team')['starter_height_2_lowest'].shift(1)
    df_covers_bball_ref['starter_team_3_height_lag'] = df_covers_bball_ref.groupby('team')['starter_height_3_lowest'].shift(1)
    df_covers_bball_ref['starter_team_4_height_lag'] = df_covers_bball_ref.groupby('team')['starter_height_4_lowest'].shift(1)
    df_covers_bball_ref['starter_team_5_height_lag'] = df_covers_bball_ref.groupby('team')['starter_height_5_lowest'].shift(1)

    # CREATED AS IN HISTORICAL FILE:    
    #df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))                

    # CAN'T GROUP BY STARTERS WHEN AUTOMATING. BECAUSE DON'T KNOW STARTERS FOR CURRENT GAME    
    #df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))                
    #df_covers_bball_ref.loc[df_covers_bball_ref['lineup_count'].isnull(), 'lineup_count'] = 0 
    #df_covers_bball_ref['beat_spread'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_max(x.shift(1), window=15))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref[['date', 'beat_spread_std_ewma_15', 'spread', 'lineup_count']][960:1000]
df_covers_bball_ref[['date', 'beat_spread_std_ewma_15', 'spread', 'lineup_count']].head()
df_covers_bball_ref[['date', 'beat_spread_std_ewma_15', 'spread', 'beat_spread_last_g', 'beat_spread']][960:1000]

df_covers_bball_ref[['date', 'beat_spread_std_ewma_15', 'lineup_count', 'team_3PAr']][960:1000]
df_covers_bball_ref[['date', 'team_ASTpct_ewma_15', 'team_ASTpct', 'team_3PAr', 'starters_team']][960:1000]
df_covers_bball_ref[['date', 'team_ASTpct_ewma_15', 'lineup_count', 'starters_team']][960:1000]

#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))
#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))

#df_covers_bball_ref[['date', 'team', 'lineup_count', 'lineup_count_expanding']][20:60]
#df_covers_bball_ref['lineup_count_diffs'] = df_covers_bball_ref['lineup_count'] - df_covers_bball_ref['lineup_count_expanding']
#df_covers_bball_ref['lineup_count_diffs'].hist()
#df_covers_bball_ref[['date', 'lineup_count_diffs', 'team']][df_covers_bball_ref['lineup_count_diffs']==1]
#
#df_covers_bball_ref[['date', 'team', 'lineup_count', 'lineup_count_expanding']][df_covers_bball_ref['date']=='2016-11-16']
#df_covers_bball_ref[['date', 'team', 'lineup_count', 'lineup_count_expanding']][960:990]



#-------------
# start of code to detect outliers
# for each var -- regress spread out
# because when i look for outliers, want to see if there's something that happened 
# that's far and above or below what we would expect. so i'll have a distrib of 
# the resid of a variable for each team. that'll give me the mean, which should be 
# about 0. and it'll give me the std of that team. so i could just use the std and
# see if the team is over 2 std above or below 0, or the mean, which should be about 0.
# presumably teams have pretty diff resid std? should be, right? check.

#results = smf.ols(formula = var + ' ~ spread', data=df_covers_bball_ref).fit()
#results.summary()
#df_covers_bball_ref[var+'_resid'] = results.resid
#df_covers_bball_ref[['team_fg_pct', var+'_resid']].head(20)
#x = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015].groupby('team')[var].mean()
#x.sort(ascending=False)
#x = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015].groupby('team')[var+'_resid'].mean()
#x.sort(ascending=False)
#
#df_covers_bball_ref[['season_start', 'team', 'date', 'opponent_ASTpct', 'team_ASTpct', 'team_fg_pct']][df_covers_bball_ref['season_start']==2015]
# team and oppt fg% mixed up?????

# get std up to but no including last 3 gs. use ewma span=15 too.



#------------------------------------------------------------------------------
# compute new variables in this section

# compute days rest
df_covers_bball_ref = df_covers_bball_ref.sort_values(['team','date'])
df_covers_bball_ref['date_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(1))
df_covers_bball_ref['days_rest'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['date_prior_game']) / np.timedelta64(1, 'D')
df_covers_bball_ref.loc[df_covers_bball_ref['days_rest']>1, 'days_rest'] = 2
#df_covers_bball_ref['days_rest'].replace(np.nan, df_covers_bball_ref['days_rest'].mean(), inplace=True)


# games in past 5 days
df_covers_bball_ref['date_two_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(2))
df_covers_bball_ref['date_three_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(3))
df_covers_bball_ref['date_four_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(4))
df_covers_bball_ref['days_since_four_gs_prior'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['date_four_prior_game']) / np.timedelta64(1, 'D')

df_covers_bball_ref['date_five_days_ago'] = df_covers_bball_ref['date'] - datetime.timedelta(days=5)
df_covers_bball_ref['games_in_past_5_days'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['date_prior_game']>df_covers_bball_ref['date_five_days_ago'], 'games_in_past_5_days'] = 1
df_covers_bball_ref.loc[df_covers_bball_ref['date_two_prior_game']>df_covers_bball_ref['date_five_days_ago'], 'games_in_past_5_days'] = 2
df_covers_bball_ref.loc[df_covers_bball_ref['date_three_prior_game']>df_covers_bball_ref['date_five_days_ago'], 'games_in_past_5_days'] = 3
df_covers_bball_ref.loc[df_covers_bball_ref['date_four_prior_game']>df_covers_bball_ref['date_five_days_ago'], 'games_in_past_5_days'] = 4
#df_covers_bball_ref[['date', 'date_five_days_ago', 'team', 'games_in_past_5_days']]

#df_covers_bball_ref.loc[:, 'point_difference'] = df_covers_bball_ref.loc[:,'score_team'] - df_covers_bball_ref.loc[:, 'score_oppt']
#sns.lmplot(x='days_rest', y='point_difference', data=df_covers_bball_ref, lowess=True)
#sns.barplot(x='days_rest', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0])

#df_hawks = df_covers_bball_ref[df_covers_bball_ref['team'] == 'Atlanta Hawks']
#df_hawks[['date', 'date_prior_game', 'days_rest']].tail(20)

# compute time zone difference
# as proxy, just say if in the same division or not? or how many divisions away?
# create team to time zone dict
team_to_zone_dict = {'Atlanta Hawks':1, 'Chicago Bulls':2, 'Golden State Warriors':4,
   'Boston Celtics':1, 'Brooklyn Nets':1, 'Detroit Pistons':2,
   'Houston Rockets':3, 'Los Angeles Lakers':4, 'Memphis Grizzlies':2,
   'Miami Heat':1, 'Milwaukee Bucks':2, 'Oklahoma City Thunder':3,
   'Orlando Magic':1, 'Phoenix Suns':3, 'Portland Trail Blazers':4,
   'Sacramento Kings':4, 'Toronto Raptors':2, 'Indiana Pacers':2,
   'Los Angeles Clippers':4, 'New York Knicks':1, 'Cleveland Cavaliers':2,
   'Denver Nuggets':3, 'Philadelphia 76ers':1, 'San Antonio Spurs':3,
   'New Orleans Pelicans':3, 'Washington Wizards':1, 'Charlotte Hornets':1,
   'Minnesota Timberwolves':2, 'Dallas Mavericks':3, 'Utah Jazz':3}

df_covers_bball_ref['team_zone'] = df_covers_bball_ref['team'].map(team_to_zone_dict)
df_covers_bball_ref['opponent_zone'] = df_covers_bball_ref['opponent'].map(team_to_zone_dict)
#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone']
#df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==0, 'zone_distance'] = 0
# also try alt zone_distance
df_covers_bball_ref['zone_distance'] = np.nan
#df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==0, 'zone_distance'] = np.abs(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])
#df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==1, 'zone_distance'] = -1*np.abs(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])
df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==0, 'zone_distance'] = (df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])
df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==1, 'zone_distance'] = -1*(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])


#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'zone_distance'].abs()
df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'zone_distance']].head(10)
df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'venue_x', 'zone_distance']][960:1000]

#df_covers_bball_ref['zone_distance'].hist()
#sns.barplot(x='zone_distance', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0])
#sns.lmplot(x='zone_distance', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0], y_partial='team_ORtg_ewma_15', x_partial='team_ORtg_ewma_15')


#------------------------------
# TRY CALC zone_distance THIS WAY. 
#df_covers_bball_ref['team_zone'] = df_covers_bball_ref['team'].map(team_to_zone_dict)
#df_covers_bball_ref['opponent_zone'] = df_covers_bball_ref['opponent'].map(team_to_zone_dict)
#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone']
#df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'zone_distance']].head(10)
#------------------------------


#-----------------
# compute  sort of distance from the playoffs
# need to compute record -- win pct up to the last g first
# compute sort sort of distance from the playoffs
# need to compute record -- win pct up to the last g first

# i need to map whether ea team is in eastern or western conference

#-----------------
# compute  sort of distance from the playoffs
# need to compute record -- win pct up to the last g first
# compute sort sort of distance from the playoffs
# need to compute record -- win pct up to the last g first

# i need to map whether ea team is in eastern or western conference
team_to_conference_dict = {'Atlanta Hawks':'east', 'Chicago Bulls':'east', 'Golden State Warriors':'west',
   'Boston Celtics':'east', 'Brooklyn Nets':'east', 'Detroit Pistons':'east',
   'Houston Rockets':'west', 'Los Angeles Lakers':'west', 'Memphis Grizzlies':'west',
   'Miami Heat':'east', 'Milwaukee Bucks':'east', 'Oklahoma City Thunder':'west',
   'Orlando Magic':'east', 'Phoenix Suns':'west', 'Portland Trail Blazers':'west',
   'Sacramento Kings':'west', 'Toronto Raptors':'east', 'Indiana Pacers':'east',
   'Los Angeles Clippers':'west', 'New York Knicks':'east', 'Cleveland Cavaliers':'east',
   'Denver Nuggets':'west', 'Philadelphia 76ers':'east', 'San Antonio Spurs':'west',
   'New Orleans Pelicans':'west', 'Washington Wizards':'east', 'Charlotte Hornets':'east',
   'Minnesota Timberwolves':'west', 'Dallas Mavericks':'west', 'Utah Jazz':'west'}

df_covers_bball_ref['conference'] = df_covers_bball_ref['team'].map(team_to_conference_dict)


def create_wins(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
    df_all_teams_w_ivs['team_win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'team_win'] = 1
    df_all_teams_w_ivs['team_win_pct'] = df_all_teams_w_ivs.groupby(['season_start', 'team'])['team_win'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=1))
    df_all_teams_w_ivs['opponent_win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['team_win']==0, 'opponent_win'] = 1
    return df_all_teams_w_ivs

df_covers_bball_ref = create_wins(df_covers_bball_ref)
df_covers_bball_ref[['date', 'team', 'team_win', 'team_win_pct']][df_covers_bball_ref['team']=='Detroit Pistons'].head()


#******************************
# NEED TO WAIT A FEW MORE DAYS -- 10 DAYS FROM START OF SEASON?? -- AND THEN INCORP
#******************************
#df_2015 = df_year.copy(deep=True)
# compute a metric that gets at distance from playoffs
def create_standings_dict_for_date(date, teams, df_2015):
    standings_date_dict = defaultdict(list)
    for team in teams:
        df_2015_team = df_2015[df_2015['team'] == team]
        df_2015_team_date = df_2015_team[df_2015_team['date'] < date]
        standings_date_dict['team'].append(team) 
        win_pct_last_game = round(np.float(df_2015_team_date['team_win_pct'][-1:].values), 3)
        standings_date_dict['team_win_pct'].append(win_pct_last_game)
        conference = df_2015_team_date['conference'][-1:].values[0]
        standings_date_dict['conference'].append(conference)
        standings_date_dict['date'].append(date)
    return standings_date_dict
    
#standings_date_dict = create_standings_dict_for_date(date, teams, df_year)
    
def create_df_w_distance_from_playoffs(standings_date_dict):
    df_standings_date = pd.DataFrame(standings_date_dict)
    df_standings_date.sort_values(by='team_win_pct', ascending=False, inplace=True)
    df_standings_date['conference_rank'] = df_standings_date.groupby('conference')['team_win_pct'].rank(ascending=False)
    df_standings_date['top_win_pct_in_conference'] = df_standings_date.groupby('conference')['team_win_pct'].transform(lambda x: x.max())
    df_standings_date.loc[:,'distance_from_first_win_pct'] = df_standings_date.loc[:,'top_win_pct_in_conference'] - df_standings_date.loc[:,'team_win_pct']
    df_standings_date['eigth_pct_in_conference'] = df_standings_date.groupby('conference')['team_win_pct'].transform(lambda x: x.values[7])
    df_standings_date['ninth_pct_in_conference'] = df_standings_date.groupby('conference')['team_win_pct'].transform(lambda x: x.values[8])
    # create distance from playoffs var:
    df_standings_date['distance_from_playoffs'] = np.nan
    df_standings_date.loc[df_standings_date['conference_rank'] < 9, 
                      'distance_from_playoffs'] = df_standings_date.loc[:,'team_win_pct'] - df_standings_date.loc[:,'ninth_pct_in_conference']
    df_standings_date.loc[df_standings_date['conference_rank'] > 8, 
                      'distance_from_playoffs'] = df_standings_date.loc[:,'team_win_pct'] - df_standings_date.loc[:,'eigth_pct_in_conference']
    df_standings_date.loc[df_standings_date['conference_rank'] == 8.5, 'distance_from_playoffs'] = 0
    df_standings_date = df_standings_date[['date', 'team', 'team_win_pct', 'conference_rank', 'distance_from_playoffs', 'eigth_pct_in_conference', 'ninth_pct_in_conference']]
    return df_standings_date


# function to create df with standings for a year
#year = 2016
def create_df_w_standings_for_year(year, df_covers_bball_ref):
    df_year = df_covers_bball_ref[df_covers_bball_ref['season_start']==year]
    df_year = df_year[['date', 'team', 'opponent', 'team_win', 'team_win_pct', 'conference']]
    teams = df_year['team'].unique()  
    dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')  
    #dates = df_year['date'].unique()
    #dates = sorted(dates)  
    #date = dates[1]
    df_w_distance_from_playoffs = pd.DataFrame()
    for date in dates[10:]:  # should this be [:] instead of [10:]??? THINK PROB NEEDS TO BE 10 OR SO, MAYBE TO MAKE SURE EA TEAM HAS PLAYED 2+ GS?
        #date = dates[10]
        standings_date_dict = create_standings_dict_for_date(date, teams, df_year)
        df_standings_date = create_df_w_distance_from_playoffs(standings_date_dict)
        df_w_distance_from_playoffs = pd.concat([df_w_distance_from_playoffs, df_standings_date], ignore_index=True)
    return df_w_distance_from_playoffs
        
#df_w_distance_from_playoffs_2016 = create_df_w_standings_for_year(2016, df_covers_bball_ref)
#print (len(df_w_distance_from_playoffs_2016))
#print (len(dates[10:]) * 30)
#df_w_distance_from_playoffs_2016.head()


# i ran this function and saved df_w_distance_from_playoffs_all_years
# it took a few min, so don't do again. just get df_w_distance_from_playoffs_all_years and merge

#df_w_distance_from_playoffs_all_years = pd.DataFrame()
#for year in [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]:
#    df_w_distance_from_playoffs_year = create_df_w_standings_for_year(year, df_covers_bball_ref)
#    df_w_distance_from_playoffs_all_years = pd.concat([df_w_distance_from_playoffs_all_years, df_w_distance_from_playoffs_year], ignore_index=True)
    
#print len(df_w_distance_from_playoffs_all_years)  
#df_w_distance_from_playoffs_all_years.to_csv('df_w_distance_from_playoffs_all_years.csv')

#*******************
# PRESUMABLY I SHOULD CALCULATE FOR THIS YEAR (2016) BUT WAIT ABOUT 10 DAYS
df_w_distance_from_playoffs_year = create_df_w_standings_for_year(2016, df_covers_bball_ref)    
#df_w_distance_from_playoffs_year.pop('Unnamed: 0')
# PRESUMABLY THE NEXT LINE IS WRONG (THE DFS AREN'T SAME ON BOTHS SIDES) AND TWO LINES DOWN IS CORRECT?
#df_w_distance_from_playoffs_year['date'] = pd.to_datetime(df_w_distance_from_playoffs_all_years['date'])
df_w_distance_from_playoffs_year['date'] = pd.to_datetime(df_w_distance_from_playoffs_year['date'])

print(len(df_w_distance_from_playoffs_year))  
#df_w_distance_from_playoffs_all_years.to_csv('df_w_distance_from_playoffs_all_years.csv')


############
# STILL DO:
############
# THEN GET CSV FOR ALL OTHER YERAS -- I CAN/SHOULD STILL DO THIS SO HAVE THE BASIC VARIABLE IN THERE
df_w_distance_from_playoffs_all_years = pd.read_csv('df_w_distance_from_playoffs_all_years.csv')
df_w_distance_from_playoffs_all_years.pop('Unnamed: 0')
df_w_distance_from_playoffs_all_years['date'] = pd.to_datetime(df_w_distance_from_playoffs_all_years['date'])
#----------


# THEN CONCAT ALL OTHER YEARS W THIS SEASON - BUT WAIT ABOUT 10 DAYS
df_w_distance_from_playoffs_all_years = pd.concat([df_w_distance_from_playoffs_all_years, df_w_distance_from_playoffs_year], ignore_index=True)


############
# STILL DO:
############
# THEN MERGE WITH df_covers_bball_ref - I CAN/SHOULD STILL DO THIS
print(len(df_w_distance_from_playoffs_all_years))
print(len(df_covers_bball_ref))
df_covers_bball_ref = pd.merge(df_covers_bball_ref, df_w_distance_from_playoffs_all_years, 
                                      on=['team', 'date'], how='left')
print(len(df_covers_bball_ref))
df_covers_bball_ref['distance_from_playoffs']


# ************* RESUME AFTER THE DISTANCE-FROM-PLAYOFFS-CODE ****************
df_covers_bball_ref['game'] = df_covers_bball_ref.groupby(['season_start', 'team'])['spread'].transform(lambda x: pd.expanding_count(x))
df_covers_bball_ref[['date', 'team', 'game']][df_covers_bball_ref['team']== 'Detroit Pistons'].head(100)

df_covers_bball_ref['distance_playoffs_abs'] = df_covers_bball_ref['distance_from_playoffs'].abs()
# should weight these so that takes the mean and gradually a team-specific number overtakes
# it as the n increases. how?

df_covers_bball_ref[['date', 'team', 'distance_playoffs_abs']][690:990]
df_covers_bball_ref.loc[df_covers_bball_ref['distance_playoffs_abs'].isnull(), 
'distance_playoffs_abs'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['game'] < 20, 'distance_playoffs_abs'] = df_covers_bball_ref.loc[:,'distance_playoffs_abs']/2
#df_covers_bball_ref[['date', 'game', 'distance_playoffs_abs', 'team']][df_covers_bball_ref['team']=='Atlanta Hawks']
df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][500:520]
df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][960:990]

df_covers_bball_ref.loc[df_covers_bball_ref['date']>date_today, 'distance_playoffs_abs'] = np.nan

#df_w_distance_from_playoffs_all_years[df_w_distance_from_playoffs_all_years['date']=='2004-12-06']
#df_w_distance_from_playoffs_all_years['date'].dtypes


# compute if last g has a diff lineup
# if the lineup is diff than the game before, does that make it harder to guess?
# need compute whether lineup is diff than prior g here
df_covers_bball_ref['starters_team_last_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(1))
df_covers_bball_ref['starters_team_two_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(2))
df_covers_bball_ref['starters_team_three_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(3))

df_covers_bball_ref['starters_same_as_last_g'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['starters_team_two_g']==df_covers_bball_ref['starters_team_last_g'], 'starters_same_as_last_g'] = 1

df_covers_bball_ref['starters_same_as_two_g'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['starters_team_three_g']==df_covers_bball_ref['starters_team_last_g'], 'starters_same_as_two_g'] = 1

df_covers_bball_ref[['date', 'starters_team', 'starters_team_last_g', 'starters_same_as_last_g']][df_covers_bball_ref['team']=='San Antonio Spurs'][-86:]
df_covers_bball_ref[['date', 'starters_same_as_last_g']][df_covers_bball_ref['team']=='San Antonio Spurs'][-86:]

# but doesn't seem to help in model, at last at this point



#-------------------
# compute ea team's home court advantage
# but home court advantage is lessening in recent years
# take that into account somehow. and what about other trends over time?

df_covers_bball_ref['home_point_diff'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='home', 'home_point_diff'] = df_covers_bball_ref['point_difference']

df_covers_bball_ref['away_point_diff'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='away', 'away_point_diff'] = df_covers_bball_ref['point_difference']

df_covers_bball_ref[['point_difference', 'home_point_diff', 'away_point_diff']]
df_covers_bball_ref[['home_point_diff', 'away_point_diff']].mean()

df_covers_bball_ref['home_point_diff_ewma'] = df_covers_bball_ref.groupby('team')['home_point_diff'].transform(lambda x: pd.ewma(x.shift(1), span=200))
df_covers_bball_ref['away_point_diff_ewma'] = df_covers_bball_ref.groupby('team')['away_point_diff'].transform(lambda x: pd.ewma(x.shift(1), span=200))

df_covers_bball_ref.loc[:, 'home_court_advantage'] = df_covers_bball_ref.loc[:, 'home_point_diff_ewma'] - df_covers_bball_ref.loc[:, 'away_point_diff_ewma'] 
df_covers_bball_ref[['date', 'point_difference', 'home_point_diff_ewma', 'away_point_diff_ewma', 'home_court_advantage']][df_covers_bball_ref['team']=='Atlanta Hawks']
df_covers_bball_ref[['date', 'point_difference', 'home_point_diff_ewma', 'away_point_diff_ewma', 'home_court_advantage']][df_covers_bball_ref['team']=='Utah Jazz']

df_covers_bball_ref.groupby('team')['home_point_diff_ewma'].mean()
df_covers_bball_ref.groupby('team')['away_point_diff_ewma'].mean()
df_covers_bball_ref[df_covers_bball_ref['season_start']>2014].groupby('team')['home_court_advantage'].mean()
df_covers_bball_ref[['date', 'home_court_advantage', 'venue_y', 'point_difference' ]][df_covers_bball_ref['team']=='San Antonio Spurs'][-86:]

# dont i want to multiply home court adv by venue? or by -1 if away and 1 if home?
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='away', 'home_court_advantage'] = df_covers_bball_ref['home_court_advantage']*-1
df_covers_bball_ref[['date', 'venue_y', 'home_court_advantage']][960:990]


df_covers_bball_ref.loc[:,'spread_abs_val'] = df_covers_bball_ref.loc[:,'spread'].abs()


# create var to signify how many free throws a team tends to get
df_covers_bball_ref['team_free_throws'] = df_covers_bball_ref['team_ft_pct_ewma_15'] * df_covers_bball_ref['team_FTr_ewma_15']
df_covers_bball_ref['opponent_free_throws'] = df_covers_bball_ref['opponent_ft_pct_ewma_15'] * df_covers_bball_ref['opponent_FTr_ewma_15']


#------------------------------------------------------------------------------
# run this and run again when want to add a new var and re-run below code:
df_covers_bball_ref_save = df_covers_bball_ref.copy(deep=True)

# run this again when want to add a new var and re-run below code:
df_covers_bball_ref = df_covers_bball_ref_save.copy(deep=True)
#------------------------------------------------------------------------------

#df_covers_bball_ref[df_covers_bball_ref['date']=='2016-11-30']

df_covers_bball_ref['venue'] = df_covers_bball_ref['venue_x']
df_covers_bball_ref.pop('venue_x')
#df_covers_bball_ref.rename(columns={'venue_x':'venue'}, inplace=True)
#df_covers_bball_ref[['venue_x_ewma_15', 'point_difference_ewma_15']]
df_covers_bball_ref[['date','venue']][960:990]
df_covers_bball_ref['season_start'].unique()


#df_covers_bball_ref['season_start']
#df_covers_bball_ref['2005'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2005, '2005'] = 1
#df_covers_bball_ref['2006'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2006, '2006'] = 1
#df_covers_bball_ref['2007'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2007, '2007'] = 1
#df_covers_bball_ref['2008'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2008, '2008'] = 1
#df_covers_bball_ref['2009'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2009, '2009'] = 1
#df_covers_bball_ref['2010'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2010, '2010'] = 1
#df_covers_bball_ref['2011'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2011, '2011'] = 1
#df_covers_bball_ref['2012'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2012, '2012'] = 1
#df_covers_bball_ref['2013'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2013, '2013'] = 1
#df_covers_bball_ref['2014'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2014, '2014'] = 1
#df_covers_bball_ref['2015'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2015, '2015'] = 1
#df_covers_bball_ref['2016'] = 0
#df_covers_bball_ref.loc[df_covers_bball_ref['season_start']==2016, '2016'] = 1

#df_covers_bball_ref[['date','team', 'spread', 'game']][df_covers_bball_ref['team']=='Boston Celtics'][-85:-75]
#df_covers_bball_ref[['date','team', 'spread', 'game']][df_covers_bball_ref['team']=='Philadelphia 76ers'][-85:-75]

#--------------
# add or subtract vars from two lists below. should only have to do that once, here.

# include all vars here that i'll want to use:
variables_for_df = ['date', 'team', 'opponent', 'venue', 'lineup_count', 
       'starters_team', 'starters_opponent', 'game',
       'spread', 'totals', 'venue_y', 'score_team', 'score_oppt',
       'spread_ewma_15', 'current_spread_vs_spread_ewma',
       'beat_spread', 'beat_spread_ewma_15', 'beat_spread_std_ewma_15',
       'beat_spread_last_g', 'season_start', 'team_3PAr_ewma_15',
       'team_ASTpct_ewma_15', 'team_BLKpct_ewma_15', 'team_DRBpct_ewma_15',
       'team_DRtg_ewma_15', 'team_FTr_ewma_15', 'team_ORBpct_ewma_15',
       'team_ORtg_ewma_15', 'team_STLpct_ewma_15', 'team_TOVpct_ewma_15',
       'team_TRBpct_ewma_15', 'team_TSpct_ewma_15', 'team_eFGpct_ewma_15',
       'team_fg3_pct_ewma_15', 'team_fg_pct_ewma_15', 'team_ft_pct_ewma_15',
       'team_pf_ewma_15', 'opponent_3PAr_ewma_15', 'opponent_ASTpct_ewma_15',
       'opponent_BLKpct_ewma_15', 'opponent_DRBpct_ewma_15',
       'opponent_DRtg_ewma_15', 'opponent_FTr_ewma_15',
       'opponent_ORBpct_ewma_15', 'opponent_ORtg_ewma_15',
       'opponent_STLpct_ewma_15', 'opponent_TOVpct_ewma_15',
       'opponent_TRBpct_ewma_15', 'opponent_TSpct_ewma_15',
       'opponent_eFGpct_ewma_15', 'opponent_fg3_pct_ewma_15',
       'opponent_fg_pct_ewma_15', 'opponent_ft_pct_ewma_15',
       'opponent_pf_ewma_15', 'days_rest', 'zone_distance', 'distance_playoffs_abs', 
       'starters_same_as_last_g', 'home_point_diff_ewma', 'away_point_diff_ewma',
       'home_court_advantage', 'venue_x_ewma_15', 'point_difference_ewma_15',
       'team_free_throws', 'opponent_free_throws', 'bet_bad_lag_3', 'bet_bad_lag_4',
       'the_team_pace_ewma_15', 'team_age_ewma_15', 'team_age_std_ewma_15', 'team_bmi_ewma_15', 
       'team_height_ewma_15', 'games_in_past_5_days']  #,
       #'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']  #,
       #'team_close_ml', 'opponent_close_ml', 'team_close_spread', 'team_open_spread',
       #'team_juice_rev', 'opponent_juice_rev', 'Opening Total', 'Closing Total', 
       #'totals_abs_diff', 'spread_abs_diff', 'totals_covers']

# include all vars i want to precict with
iv_variables = ['totals', 'lineup_count', 'spread', 
       'spread_ewma_15', #'current_spread_vs_spread_ewma',
       'beat_spread_ewma_15', 'beat_spread_std_ewma_15',
       'beat_spread_last_g', 'team_3PAr_ewma_15', 'team_ASTpct_ewma_15', 
       'team_BLKpct_ewma_15', 'team_DRBpct_ewma_15',
       'team_DRtg_ewma_15', 'team_FTr_ewma_15', 'team_ORBpct_ewma_15',
       'team_ORtg_ewma_15', 'team_STLpct_ewma_15', 'team_TOVpct_ewma_15',
       'team_TRBpct_ewma_15', 'team_TSpct_ewma_15', 'team_eFGpct_ewma_15',
       'team_fg3_pct_ewma_15', 'team_fg_pct_ewma_15', 'team_ft_pct_ewma_15',
       'team_pf_ewma_15', 'opponent_3PAr_ewma_15', 'opponent_ASTpct_ewma_15',
       'opponent_BLKpct_ewma_15', 'opponent_DRBpct_ewma_15', 'opponent_DRtg_ewma_15', 
       'opponent_FTr_ewma_15', 'opponent_ORBpct_ewma_15', 'opponent_ORtg_ewma_15',
       'opponent_STLpct_ewma_15', 'opponent_TOVpct_ewma_15', 'opponent_TRBpct_ewma_15', 
       'opponent_TSpct_ewma_15', 'opponent_eFGpct_ewma_15', 'opponent_fg3_pct_ewma_15',
       'opponent_fg_pct_ewma_15', 'opponent_ft_pct_ewma_15', 'opponent_pf_ewma_15', 
       'days_rest', 'distance_playoffs_abs', 'game', 'home_court_advantage', 
       'zone_distance', 'venue_x_ewma_15','team_free_throws', 'opponent_free_throws',
       'bet_bad_lag_3', 'bet_bad_lag_4', 'team_age_ewma_15', 'team_bmi_ewma_15']  #, 'starters_same_as_last_g']  ]  #] # , 'venue_x_ewma_15' - 
       # don't think venue_ewma matters because i've already got spread_ewma in there
       # and that should be adjusting for the compeition and home court already in the past x games, 
       #, 'point_difference_ewma_15' 'zone_distance', 'starters_same_as_last_g']  , 'season_start'

#for var in iv_variables:
#    print(var)

# select the dv to predict:
#dv_var = 'ats_win'  
#dv_var = 'win'
#dv_var = 'spread'
dv_var = 'point_difference'

iv_and_dv_vars = iv_variables + [dv_var] + ['team', 'opponent', 'date']

#--------------

# create variables for opponent and then difference variables

df_covers_bball_ref = df_covers_bball_ref[variables_for_df]

def create_switched_df(df_all_teams):
    # create df with team and opponent swithced (so can then merge the team's 
    # weighted/rolling metrics onto the original df but as the opponents)
    df_all_teams_swtiched = df_all_teams.copy(deep=True)
    df_all_teams_swtiched.rename(columns={'team':'opponent_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent':'team_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent_hold':'opponent'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'team_hold':'team'}, inplace=True)
    return df_all_teams_swtiched

df_covers_bball_ref_switched = create_switched_df(df_covers_bball_ref)


def preface_oppt_stats_in_switched_df(df_all_teams_swtiched, variables_for_df):
    # preface all these stats -- they belong to the team in this df but to
    # the opponent in the orig df -- with an x_. then when merge back onto original df
    # these stats will be for the opponent in that df. and that's what i'll use to predict ats
    variables_for_df_2 = variables_for_df    
    variables_for_df_2.remove('date')
    variables_for_df_2.remove('team')
    variables_for_df_2.remove('opponent')
    df_all_teams_swtiched.rename(columns={'venue_x':'venue'}, inplace=True)
    for variable in variables_for_df_2:
        df_all_teams_swtiched.rename(columns={variable:'x_'+variable}, inplace=True)
    return df_all_teams_swtiched   
           
df_covers_bball_ref_switched = preface_oppt_stats_in_switched_df(df_covers_bball_ref_switched, variables_for_df)
df_covers_bball_ref_switched.columns


def merge_regular_df_w_switched_df(df_all_teams, df_all_teams_swtiched):    
    df_all_teams_w_ivs = df_all_teams.merge(df_all_teams_swtiched, on=['date', 'team', 'opponent'], how='left')
    df_all_teams_w_ivs.head(50)
    return df_all_teams_w_ivs

df_covers_bball_ref = merge_regular_df_w_switched_df(df_covers_bball_ref, df_covers_bball_ref_switched)    


def create_team_opponent_difference_variables(df_all_teams_w_ivs, iv_variables):
    for var in iv_variables:
        new_difference_variable = 'difference_'+var
        df_all_teams_w_ivs.loc[:, new_difference_variable] = df_all_teams_w_ivs.loc[:, var] - df_all_teams_w_ivs.loc[:, 'x_'+var]
    return df_all_teams_w_ivs

df_covers_bball_ref = create_team_opponent_difference_variables(df_covers_bball_ref, iv_variables)


def create_basic_variables(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
#    df_all_teams_w_ivs['ats_win'] = 'push'
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] > 0, 'ats_win'] = '1'
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] < 0, 'ats_win'] = '0'
#    df_all_teams_w_ivs[['date', 'team', 'opponent', 'beat_spread', 'ats_win']]
#    df_all_teams_w_ivs['win'] = 0
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'win'] = 1
#    df_all_teams_w_ivs[['score_team', 'score_oppt', 'win']]
    df_all_teams_w_ivs['point_difference'] = df_all_teams_w_ivs['score_team'] - df_all_teams_w_ivs['score_oppt']
    return df_all_teams_w_ivs

df_covers_bball_ref = create_basic_variables(df_covers_bball_ref)


def create_iv_list(iv_variables, variables_without_difference_score_list):
    """Put all variables that don't want to compute a difference score on."""
    spread_and_totals_list = variables_without_difference_score_list
    for i in range(len(spread_and_totals_list)):
        print(i)
        iv_variables.remove(spread_and_totals_list[i])
    iv_variables = ['difference_'+iv_var for iv_var in iv_variables]
    iv_variables = iv_variables + spread_and_totals_list
    return iv_variables

iv_variables = create_iv_list(iv_variables, ['spread', 'totals', 'game', 'zone_distance'])  # home_court_advantage, zone_distance, 'season_start'
#iv_variables = create_iv_list(iv_variables, ['totals', 'game', 'zone_distance'])  # home_court_advantage, zone_distance, 'season_start'

#iv_variables = iv_variables + ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']


def create_home_df(df_covers_bball_ref):
    df_covers_bball_ref_home = df_covers_bball_ref[df_covers_bball_ref['venue'] == 0]
    df_covers_bball_ref_home = df_covers_bball_ref_home.reset_index()
    len(df_covers_bball_ref_home)
    return df_covers_bball_ref_home

df_covers_bball_ref_home = create_home_df(df_covers_bball_ref)
df_covers_bball_ref_home['season_start'].unique()

#--------------------
# create new vars

# home court advantage - not working. how else to compute? subtract home win% from oppt away win%
df_covers_bball_ref_home.loc[:, 'home_advantage_added'] = df_covers_bball_ref_home.loc[:, 'home_point_diff_ewma'] - df_covers_bball_ref_home.loc[:, 'x_away_point_diff_ewma']
iv_variables = iv_variables + ['home_advantage_added', 'home_point_diff_ewma', 'x_away_point_diff_ewma']
variables_for_df = variables_for_df + ['home_advantage_added', 'x_away_point_diff_ewma']
# think about this more. am i capturing home court advantage as well as i can?

# don't use:
#df_covers_bball_ref_home.loc[:, 'home_court_advantage_difference'] = df_covers_bball_ref_home.loc[:, 'home_court_advantage'] - df_covers_bball_ref_home.loc[:, 'x_home_court_advantage']*-1
#iv_variables = iv_variables + ['home_court_advantage_difference']
#variables_for_df = variables_for_df + ['home_court_advantage_difference']


#nets = df_covers_bball_ref_home[df_covers_bball_ref_home['team']=='Brooklyn Nets']
#nets[['date', 'team', 'beat_spread', 'game', 'score_oppt', 'difference_beat_spread_last_g']][440:460]

# -----------------------------
df_covers_bball_ref__dropna_home = df_covers_bball_ref_home.copy(deep=True)
# ------------------------------

## drop nans
#print('\n number of games with nans:', len(df_covers_bball_ref_home))
#df_covers_bball_ref__dropna_home = df_covers_bball_ref_home.dropna()
#df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.sort_values(by=['team','date'])
#df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.reset_index(drop=True)
#print('\n number of games without nans:', (len(df_covers_bball_ref__dropna_home)))

len(iv_variables)
#iv_variables = iv_variables[:-1]

# ----------------------------
# DELETE

#nets = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['team']=='Brooklyn Nets']
#nets[['date', 'team', 'beat_spread', 'venue', 'score_oppt', 'difference_beat_spread_last_g']][440:460]
#
#nets = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['team']=='Brooklyn Nets']
#nets[['date', 'team', 'beat_spread', 'score_team', 'score_oppt', 'beat_spread_last_g', 'difference_beat_spread_last_g']][400:420]
#
#nets = df_covers_bball_ref[df_covers_bball_ref['team']=='Brooklyn Nets']
#nets[['date', 'team', 'opponent', 'beat_spread', 'venue', 'score_oppt', 'difference_beat_spread_last_g']][880:895]
#nets[['date', 'team', 'opponent', 'beat_spread', 'beat_spread_last_g', 'x_beat_spread_last_g']][880:895]
#
#
#is it a Milwaukee Bucks problem? the oppt of the nets on 11-2?
#yeah, the milwaukee bucks didn't have x_beat_spread_last_g
#2015-11-2 -- no game in early df?
#
#bucks = df_covers_bball_ref[df_covers_bball_ref['team']=='Milwaukee Bucks']
#bucks[['date', 'team', 'opponent', 'beat_spread', 'venue', 'score_oppt', 'difference_beat_spread_last_g']][880:895]
#bucks[['date', 'team', 'opponent', 'beat_spread', 'beat_spread_last_g', 'x_beat_spread_last_g']][880:895]

# got a problem here: 
#17713 2015-11-01  Milwaukee Bucks      Toronto Raptors        -13.0    1.0   
#17714 2015-11-01  Milwaukee Bucks      Toronto Raptors        -13.0    1.0   
#17715 2015-11-01  Milwaukee Bucks      Toronto Raptors          NaN    1.0   
#17716 2015-11-01  Milwaukee Bucks      Toronto Raptors          NaN    1.0   
#------------------------------


# test train funct for actua gambling
#test_year = 2016
#date_today = dates[i]
def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, date_today, iv_variables):

#    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['date'] < date_today) &
#                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
    days_ago_15 = date_today - datetime.timedelta(days=15)
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['date'] < date_today) &
                                                                      (df_covers_bball_ref__dropna_home['date'] > days_ago_15)]  # was > 2004. maybe go back to that. (test_year-9)

    # training on games only after 10. this seems to help. presumably only predict games in test set after game 10
    #df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['game']>10]                                                                  
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    print ('training n:', len(df_covers_bball_ref_home_train))    
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['date'] == date_today]
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.sort_values(by=['team','date'])
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.reset_index(drop=True)
    print ('test n:', len(df_covers_bball_ref_home_test))
    df_covers_bball_ref_home_train['spread_to_bet'] = df_covers_bball_ref_home_train['spread'] 
    df_covers_bball_ref_home_test['spread_to_bet'] = df_covers_bball_ref_home_test['spread'] 
    for var in iv_variables:  
        var_mean = df_covers_bball_ref_home_train[var].mean()
        var_std = df_covers_bball_ref_home_train[var].std()
        df_covers_bball_ref_home_train[var] = (df_covers_bball_ref_home_train[var] -  var_mean) / var_std    
        df_covers_bball_ref_home_test[var] = (df_covers_bball_ref_home_test[var] -  var_mean) / var_std    
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test


# this ALT version trains on all data priot to today
def create_train_and_test_dfs_ALT(df_covers_bball_ref__dropna_home, date_today, iv_variables):
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < 2016) &
                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
#    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['date'] < date_today) &
#                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
    # training on games only after 10. this seems to help. presumably only predict games in test set after game 10
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['game']>10]                                                                  
    #df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['game']<10]                                                                  
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    print ('training n:', len(df_covers_bball_ref_home_train))    
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['date'] == date_today]
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.sort_values(by=['team','date'])
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.reset_index(drop=True)
    print ('test n:', len(df_covers_bball_ref_home_test))
    df_covers_bball_ref_home_train['spread_to_bet'] = df_covers_bball_ref_home_train ['spread'] 
    df_covers_bball_ref_home_test['spread_to_bet'] = df_covers_bball_ref_home_test ['spread'] 
    df_covers_bball_ref_home_test['game_unstandardized'] = df_covers_bball_ref_home_test ['game'] 
    # can try turing the below standardizatino on and off
    for var in iv_variables:  
        var_mean = df_covers_bball_ref_home_train[var].mean()
        var_std = df_covers_bball_ref_home_train[var].std()
        df_covers_bball_ref_home_train[var] = (df_covers_bball_ref_home_train[var] -  var_mean) / var_std    
        df_covers_bball_ref_home_test[var] = (df_covers_bball_ref_home_test[var] -  var_mean) / var_std    
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test


def add_interactions(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df):
    df_covers_bball_ref_home_train['game_x_playoff_distance'] = df_covers_bball_ref_home_train['game'] * df_covers_bball_ref_home_train['difference_distance_playoffs_abs']
    df_covers_bball_ref_home_test['game_x_playoff_distance'] = df_covers_bball_ref_home_test['game'] * df_covers_bball_ref_home_test['difference_distance_playoffs_abs']
    iv_variables = iv_variables + ['game_x_playoff_distance']
    variables_for_df = variables_for_df + ['game_x_playoff_distance']

    # add interaction between venue_ewma and home_court advantage:
    df_covers_bball_ref_home_train['venue_ewma_x_home_advantage'] = df_covers_bball_ref_home_train['difference_venue_x_ewma_15'] * df_covers_bball_ref_home_train['difference_home_court_advantage']
    df_covers_bball_ref_home_test['venue_ewma_x_home_advantage'] = df_covers_bball_ref_home_test['difference_venue_x_ewma_15'] * df_covers_bball_ref_home_test['difference_home_court_advantage']
    iv_variables = iv_variables + ['venue_ewma_x_home_advantage']
    variables_for_df = variables_for_df + ['venue_ewma_x_home_advantage']

    # add interaction between ft and spread:
    df_covers_bball_ref_home_train['free_throws_x_spread'] = df_covers_bball_ref_home_train['difference_team_free_throws'] * df_covers_bball_ref_home_train['spread']
    df_covers_bball_ref_home_test['free_throws_x_spread'] = df_covers_bball_ref_home_test['difference_team_free_throws'] * df_covers_bball_ref_home_test['spread']
    iv_variables = iv_variables + ['free_throws_x_spread']
    variables_for_df = variables_for_df + ['free_throws_x_spread']

    # add interction age x rest:
    df_covers_bball_ref_home_train['age_x_rest'] = df_covers_bball_ref_home_train['difference_team_age_ewma_15'] * df_covers_bball_ref_home_train['difference_days_rest']
    df_covers_bball_ref_home_test['age_x_rest'] = df_covers_bball_ref_home_test['difference_team_age_ewma_15'] * df_covers_bball_ref_home_test['difference_days_rest']
    iv_variables = iv_variables + ['age_x_rest']
    variables_for_df = variables_for_df + ['age_x_rest']

    # add interction 3s x rest:
    df_covers_bball_ref_home_train['three_rate_x_rest'] = df_covers_bball_ref_home_train['difference_team_3PAr_ewma_15'] * df_covers_bball_ref_home_train['difference_days_rest'] 
    df_covers_bball_ref_home_test['three_rate_x_rest'] = df_covers_bball_ref_home_test['difference_team_3PAr_ewma_15'] * df_covers_bball_ref_home_test['difference_days_rest']       
    iv_variables = iv_variables + ['three_rate_x_rest']
    variables_for_df = variables_for_df + ['three_rate_x_rest']

    # add interction to x oppt steals:
#    df_covers_bball_ref_home_train['to_x_oppt_stls'] = df_covers_bball_ref_home_train['difference_team_TOVpct_ewma_15'] * df_covers_bball_ref_home_train['difference_opponent_STLpct_ewma_15'] 
#    df_covers_bball_ref_home_test['to_x_oppt_stls'] = df_covers_bball_ref_home_test['difference_team_TOVpct_ewma_15'] * df_covers_bball_ref_home_test['difference_opponent_STLpct_ewma_15']       
#    iv_variables = iv_variables + ['to_x_oppt_stls']
#    variables_for_df = variables_for_df + ['to_x_oppt_stls']

    # add interaction age x zone_distance:
    df_covers_bball_ref_home_train['age_x_zone'] = df_covers_bball_ref_home_train['difference_team_age_ewma_15'] * df_covers_bball_ref_home_train['zone_distance'] 
    df_covers_bball_ref_home_test['age_x_zone'] = df_covers_bball_ref_home_test['difference_team_age_ewma_15'] * df_covers_bball_ref_home_test['zone_distance']       
    iv_variables = iv_variables + ['age_x_zone']
    variables_for_df = variables_for_df + ['age_x_zone']

    # add interction:
#    df_covers_bball_ref_home_train['new_x'] = df_covers_bball_ref_home_train['difference_days_rest'] * df_covers_bball_ref_home_train['game'] 
#    df_covers_bball_ref_home_test['new_x'] = df_covers_bball_ref_home_test['difference_days_rest'] * df_covers_bball_ref_home_test['game']       
#    iv_variables = iv_variables + ['new_x']
#    variables_for_df = variables_for_df + ['new_x']

    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df


# function to say which way to bet
def create_correct_metric(df):
    df['how_to_bet'] = np.nan
    df.loc[(df['point_diff_predicted'] > df['spread_to_bet']*-1), 'how_to_bet'] = df['team']
    df['confidence'] = np.nan
    df.loc[(df['point_diff_predicted'] > df['spread_to_bet']*-1), 'confidence'] = df['point_diff_predicted'] - df['spread_to_bet']*-1
    df.loc[(df['point_diff_predicted'] < df['spread_to_bet']*-1), 'how_to_bet'] = df['opponent']
    df.loc[(df['point_diff_predicted'] < df['spread_to_bet']*-1), 'confidence'] = np.abs(df['point_diff_predicted'] - df['spread_to_bet']*-1)
    return df


def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var):
     # use this scikit learn vs statsmodels below where do my own regularization procedure
    model = algorithm
    model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])
    df_covers_bball_ref_home_test.loc[:,'point_diff_predicted'] = predictions_test_set
    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
    return df_covers_bball_ref_home_test

#def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var):
#     # use this scikit learn vs statsmodels below where do my own regularization procedure
#    model = algorithm
#    #model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
#    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])
#    df_covers_bball_ref_home_test.loc[:,'point_diff_predicted'] = predictions_test_set
#    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
#    return df_covers_bball_ref_home_test



# THIS IS SPITTING BACK A COUPLE NANS. WHY???
#df_covers_bball_ref_home_test[['team', 'date', 'difference_beat_spread_last_g']]
#df_covers_bball_ref_home_test[['team', 'opponent', 'date', 'beat_spread_last_g',
#                               'x_beat_spread_last_g', 'difference_beat_spread_last_g',
#                               'spread_to_bet', 'point_difference', 'beat_spread', 
#                               'x_beat_spread']]
#
#df_covers_bball_ref_home_test
# CHECK THESE TEAMS THAT MESSED UP -

# I SHOULD CREATE NEW BEAT SPREAD LAST G DIFF VAR
# SO USING spread_to_bet instead of spread standarized

#beat_spread_last_g and x_beat_spread_last_g is the problem
#df_wiz = df_covers_bball_ref[df_covers_bball_ref['team'] == 'Washington Wizards']
#df_wiz[['date', 'team', 'beat_spread_last_g', 'beat_spread', 'spread']][-90:-70]



# -------------------------------
# FIGURE OUT WHY CAN'T FIT MODEL:
#df_covers_bball_ref_home_train[iv_variables].head()
#df_test = df_covers_bball_ref_home_train[iv_variables]
#len(df_test.dropna())
#len(df_test)
#df_covers_bball_ref_home_train['season_start'].unique()


#df_test.iloc[0:10, 0:10].tail()
#df_test.iloc[10:20, 10:20].tail()
#df_test.iloc[20:30, 20:30].tail()
#df_test.iloc[30:40, 30:40].tail()
#df_test.iloc[40:50, 40:50].tail()
#df_test.iloc[50:60, 50:60].tail()
#
#df_test[df_test['home_advantage_added'].isnull()]
#df_test[df_test.notnull()].head()
#df_test[df_test[iv_variables].isnull()].head()
#df_test[df_test[iv_variables]==np.nan].head()
#
#df_test[['home_court_advantage_difference']]
#null_data = df_test[df_test.isnull().any(axis=1)]
#
#null_data.iloc[0:10, 0:10].tail()
#null_data.iloc[0:10, 10:20].tail()
#null_data.iloc[0:10, 20:30].tail()
#null_data.iloc[0:10, 30:40].tail()
#null_data.iloc[0:10, 40:50].tail()
#null_data.iloc[0:10, 50:60].tail()

#df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['difference_beat_spread_last_g'].isnull()]
#nets = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['team']=='Brooklyn Nets']
#nets['date']
#nets[['date', 'team', 'beat_spread', 'score_team', 'score_oppt', 'difference_beat_spread_last_g']][400:420]
# -----------------------------------------------



# ----------------------------------------------------------------------------
# choose seasons i want to anys, and the model/algo i want to use

# general approach idea: run regular regression and get predictions and run
# 4-5 adaboost regressions and avg those, and then avg the regular regression
# point diff and the adaboost avg pt diffs. and then make deicsions. do these
# corr better with actually beating spread? if not, could only bet on those 
# games that both regular and boosted agree? that should weed out a few games and
# hopefully improve odds a little.
#seasons = [2010, 2011, 2012, 2013, 2014, 2015]


# use these three algos for gambling. 
model = linear_model.LinearRegression()
model = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=200, learning_rate=.001)  #, loss='exponential')  # 
model = linear_model.Ridge(alpha=10)  # higher number regularize more
model = svm.LinearSVR(C=.05)

model_1 = linear_model.LinearRegression()
model_2 = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=200, learning_rate=.001)  #, loss='exponential')  # 
model_3 = linear_model.Ridge(alpha=10)  # higher number regularize more
model_4 = svm.LinearSVR(C=.05)

model_1 =  linear_model.LinearRegression()
model_2 =  linear_model.LinearRegression()
model_3 =  linear_model.LinearRegression()
model_4 =  linear_model.LinearRegression()

#algorithm = linear_model.LinearRegression()

model = KNeighborsRegressor(n_neighbors=20, weights='distance')
model = RandomForestRegressor(n_estimators=200, max_features='auto')  # , min_samples_leaf = 10, min_samples_split = 50)

model = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, learning_rate=.001)  #, loss='exponential')  # this decision tree regressor is the default
model = AdaBoostRegressor(linear_model.Ridge(alpha=10), n_estimators=100, learning_rate=.001)  #, loss='exponential')  # 
model = AdaBoostRegressor(svm.LinearSVR(), n_estimators=100, learning_rate=.001)  #, loss='exponential')  # this decision tree regressor is the default
model = AdaBoostRegressor(tree.DecisionTreeRegressor(), n_estimators=500)  # this decision tree regressor is the default

# so something is up, presumably, and maybe tuning will help it
model = svm.SVR(C=1000, gamma=.00001)  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=2, gamma=.005)  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=2, gamma=.005, kernel='sigmoid')  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=50, gamma=.0001) 
model = svm.SVR(C=200, gamma=.0001) 

model = svm.LinearSVR()  #  
model = svm.LinearSVR(C=10)  #  
model = svm.LinearSVR(C=.05)  #  best option for C. but how diff is this from regular regression?
model = svm.LinearSVR(C=.1)  #  best option for C
model = svm.LinearSVR(C=.01)  



# USE FOR AUTO-GAMBLING - replace date w date_today
def analyze_multiple_seasons(df_covers_bball_ref__dropna_home, date_today, model, iv_variables, dv_var, variables_for_df):
    iv_variabless_pre_x = iv_variables
    # use one of the following to to greate test and train sets (ALT trains on all up to prior g)
    #df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, date_today, iv_variables)
    df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs_ALT(df_covers_bball_ref__dropna_home, date_today, iv_variables)
    df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df = add_interactions(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df)
    df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, model, iv_variables, dv_var)
    df_covers_bball_ref_home_test['absolute_error'] = np.abs(df_covers_bball_ref_home_test['point_diff_predicted'] - df_covers_bball_ref_home_test['point_difference'])
    df_test_seasons = df_covers_bball_ref_home_test.copy(deep=True)
    iv_variables = iv_variabless_pre_x
    return df_test_seasons, df_covers_bball_ref_home_train


#---------------------------
#---------------------------
# produce predictions:
model_1 = linear_model.Ridge(alpha=10)  
model_2 = linear_model.Ridge(alpha=10)  
model_3 = linear_model.Ridge(alpha=10)  
model_4 = linear_model.Ridge(alpha=10)  

#model_1 = linear_model.Ridge(alpha=75)  
#model_2 = linear_model.Ridge(alpha=75)  
#model_3 = linear_model.Ridge(alpha=75)  
#model_4 = linear_model.Ridge(alpha=75)  


df_all_models = pd.DataFrame()
for i, model in enumerate([model_1, model_2, model_3, model_4][:]):
    df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(df_covers_bball_ref__dropna_home, date_today, model, iv_variables, dv_var, variables_for_df)
    df_test_seasons[['date', 'team', 'spread_to_bet', 'opponent', 'point_diff_predicted', 'how_to_bet', 'confidence']]
    
    df_test_seasons['spread_for_bet_team'] = np.nan
    df_test_seasons.loc[df_test_seasons['how_to_bet']==df_test_seasons['team'],'spread_for_bet_team'] = df_test_seasons['spread_to_bet']
    df_test_seasons.loc[df_test_seasons['how_to_bet']==df_test_seasons['opponent'],'spread_for_bet_team'] = df_test_seasons['spread_to_bet']*-1
    df_test_seasons[['date', 'how_to_bet', 'spread_for_bet_team', 'confidence']]
    df_one_model = df_test_seasons[['date', 'how_to_bet', 'spread_for_bet_team', 'confidence']]
    df_all_models['date'+str(i)] = df_one_model['date']
    df_all_models['how_to_bet'+str(i)] = df_one_model['how_to_bet']
    df_all_models['spread_for_bet_team'+str(i)] = df_one_model['spread_for_bet_team']
    df_all_models['confidence'+str(i)] = df_one_model['confidence']

df_all_models = df_all_models[['date0', 'spread_for_bet_team0', 'confidence0', 'how_to_bet0', 'how_to_bet1', 'how_to_bet2', 'how_to_bet3']]
df_all_models['first_team'] = df_all_models['how_to_bet0']
df_all_models['votes0'] = 0
df_all_models.loc[df_all_models['how_to_bet0']==df_all_models['first_team'], 'votes0'] = 1
df_all_models['votes1'] = 0
df_all_models.loc[df_all_models['how_to_bet1']==df_all_models['first_team'], 'votes1'] = 1
df_all_models['votes2'] = 0
df_all_models.loc[df_all_models['how_to_bet2']==df_all_models['first_team'], 'votes2'] = 1
df_all_models['votes3'] = 0
df_all_models.loc[df_all_models['how_to_bet3']==df_all_models['first_team'], 'votes3'] = 1

df_all_models['votes_for_first_team'] = df_all_models['votes0']+df_all_models['votes1']+df_all_models['votes2']+df_all_models['votes3']
df_all_models['how_to_bet_vote'] = np.nan
df_all_models.loc[df_all_models['votes_for_first_team']>2,'how_to_bet_vote'] = df_all_models['how_to_bet0']
df_all_models_simple = df_all_models[['spread_for_bet_team0', 'how_to_bet_vote', 'confidence0']]
df_all_models_simple.rename(columns={'spread_for_bet_team0':'spread'}, inplace=True)
print(df_all_models_simple)

#df_covers_bball_ref_home_train.to_csv('df_auto_train_11_19_16.csv')

# when the spread changes in the wrong direction (making it harder for out team to 
# beat the spread), it takes off .086 from the confidence. if confidence goes below
# .025 then we skip it. likewise, I assume that if the spread changes for a team
# that we're skipping because confidence is > .025, if the spread changes in direction 
# of team that model selected, bet on them. if changes in opposite direction, bet on 
# the other team.



#---------------------------
#dates = pd.date_range('2016/10/25', str(year_now)+'/'+str(month_now)+'/'+str(day_now), freq='D')  
#model_1 = svm.LinearSVR(C=.05)
#model_2 = linear_model.Ridge(alpha=10)  # higher number regularize more
#model_3 = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=200, learning_rate=.001)  #, loss='exponential')  # 
#models = [model_1, model_2, model_3]

# set to use the unpickled/saved model:
#with open('model_trained.pkl', 'rb') as picklefile:
#    model = pickle.load(picklefile)
#model_1 = model
#model_2 = model
#model_3 = model
#model_1 =  linear_model.LinearRegression()
#model_2 =  linear_model.LinearRegression()
#model_3 =  linear_model.LinearRegression()


def results_one_model_one_day(df_covers_bball_ref__dropna_home, i, dates, model, iv_variables, dv_var, variables_for_df):
    df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(df_covers_bball_ref__dropna_home, dates[i], model, iv_variables, dv_var, variables_for_df)
    df_test_seasons[['date', 'team', 'spread_to_bet', 'point_difference', 'opponent', 'point_diff_predicted', 'how_to_bet', 'confidence', 'score_team', 'score_oppt']]
    df_results = df_test_seasons[['date', 'team', 'spread_to_bet', 'opponent', 'how_to_bet', 'point_difference', 'point_diff_predicted', 'confidence', 'score_team', 'score_oppt']]
    df_results['correct_team'] = np.nan
    df_results.loc[df_results['point_difference'] > df_results['spread_to_bet']*-1, 'correct_team'] = df_results['team']
    df_results.loc[df_results['point_difference'] < df_results['spread_to_bet']*-1, 'correct_team'] = df_results['opponent']    
    df_results['correct'] = np.nan
    df_results.loc[df_results['how_to_bet']==df_results['correct_team'], 'correct'] = 1
    df_results.loc[df_results['how_to_bet']!=df_results['correct_team'], 'correct'] = 0
    df_results.loc[df_results['correct_team'].isnull(), 'correct'] = np.nan
    return df_results


def results_three_models_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, model_2, model_3, iv_variables, dv_var):
    # for ea model, include point_diff_predicted when filtering the dfs below
    # then get the mean of point_diff_predicted from ea of the three models
    # and get 'how_to_bet' from that mean. how does that predict using all gs,
    # not just those in which the models agree.
    df_results_model_1 = results_one_model_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, iv_variables, dv_var, variables_for_df)
    df_results_model_1 = df_results_model_1[['correct_team', 'how_to_bet', 'correct', 'point_diff_predicted', 'team', 'opponent', 'spread_to_bet', 'confidence', 'score_team', 'score_oppt']]
    df_results_model_1 = df_results_model_1.reset_index()
    df_results_model_2 = results_one_model_one_day(df_covers_bball_ref__dropna_home, i, dates, model_2, iv_variables, dv_var, variables_for_df)
    df_results_model_2 = df_results_model_2[['how_to_bet', 'correct', 'point_diff_predicted']]
    df_results_model_2 = df_results_model_2.reset_index()
    df_results_model_3 = results_one_model_one_day(df_covers_bball_ref__dropna_home, i, dates, model_3, iv_variables, dv_var, variables_for_df)
    df_results_model_3 = df_results_model_3[['how_to_bet', 'correct', 'point_diff_predicted']]
    df_results_model_3 = df_results_model_3.reset_index()
    df_results_all_models = pd.merge(df_results_model_1, df_results_model_2, on=['index'])
    df_results_all_models = pd.merge(df_results_all_models, df_results_model_3, on=['index'])
    df_results_all_models['point_diff_predicted'] = df_results_all_models[['point_diff_predicted', 'point_diff_predicted_x', 'point_diff_predicted_y']].mean(axis=1)
    # if want to bet based on mean of 3 predictions:
    df_results_all_models = create_correct_metric(df_results_all_models)
    df_results_all_models['correct'] = np.nan
    df_results_all_models.loc[df_results_all_models['how_to_bet']==df_results_all_models['correct_team'], 'correct'] = 1
    df_results_all_models.loc[df_results_all_models['how_to_bet']!=df_results_all_models['correct_team'], 'correct'] = 0
    df_results_all_models.loc[df_results_all_models['correct_team'].isnull(), 'correct'] = np.nan
    df_results_all_models['models_agree'] = 0
    df_results_all_models.loc[(df_results_all_models['how_to_bet_x'] == df_results_all_models['how_to_bet_y']) &
    (df_results_all_models['how_to_bet_x'] == df_results_all_models['how_to_bet']), 'models_agree'] = 1    
    # filter based on whether models agree:    
    #df_results_all_models = df_results_all_models[df_results_all_models['models_agree']==1]
    # filter with confidence rating:
    #df_results_all_models = df_results_all_models[df_results_all_models['confidence']>.5]
    games_to_bet = len(df_results_all_models)
    wins_games_to_bet = df_results_all_models['correct'].sum()
    return wins_games_to_bet, games_to_bet


#model_1 = linear_model.LinearRegression()
#model_2 = linear_model.LinearRegression()
#model_3 = linear_model.LinearRegression()
#
## dates since start of gambling:
#dates = pd.date_range('2016/11/17', str(year_yesterday)+'/'+str(month_yesterday)+'/'+str(day_yesterday), freq='D')  
#wins = []
#games = []
#for i in range(len(dates)):
#    print(i)
#    wins_games_to_bet, games_to_bet = results_three_models_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, model_2, model_3, iv_variables, dv_var)
#    wins.append(wins_games_to_bet)
#    games.append(games_to_bet)
#for i in range(len(games)):
#    print(int(wins[i]), '/', int(games[i]))
#win_pct = round(sum(wins[:]) / sum(games[:]),3)
#print()
#print('win percentage on season:', str(round(win_pct*100,1))+'%')
# excluding first game
#round(sum(wins[3:-1]) / sum(games[3:-1]),3)


# --------------------------------------------------
# --------------------------------------------------
# get yesterday's results for max/site:

# dates since start of gambling:
dates = pd.date_range('2016/11/17', str(year_yesterday)+'/'+str(month_yesterday)+'/'+str(day_yesterday), freq='D')  

#model = linear_model.LinearRegression()
model = linear_model.Ridge(alpha=10) 

i = -1
dates[-1]
#i = -2
#dates[-2]

df_yesterday = results_one_model_one_day(df_covers_bball_ref__dropna_home, i, dates, model, iv_variables, dv_var, variables_for_df)
df_yesterday['spread'] = np.nan
df_yesterday.loc[df_yesterday['how_to_bet']==df_yesterday['team'],'spread'] = df_yesterday['spread_to_bet']
df_yesterday.loc[df_yesterday['how_to_bet']==df_yesterday['opponent'],'spread'] = df_yesterday['spread_to_bet']*-1
indices = df_yesterday.index
indices = [ind+1 for ind in indices]
df_yesterday.index = indices
df_yesterday['confidence'] = df_yesterday['confidence'].round(2) 
df_yesterday['score_team'] = df_yesterday['score_team'].astype(int)
df_yesterday['score_oppt'] = df_yesterday['score_oppt'].astype(int)
df_yesterday = df_yesterday[['date', 'how_to_bet', 'spread', 'correct', 'confidence', 'team', 'opponent', 'score_team', 'score_oppt']]
print(df_yesterday)
#df_yesterday.to_csv('results_'+str(year_yesterday)+'_'+str(month_yesterday)+'_'+str(day_yesterday))
#df_yesterday.to_csv('df_test_gs_2015_12_28_correct.csv')
df_yesterday.loc[df_yesterday['confidence']< .025, 'correct'] = np.nan
df_yesterday['correct'].mean()
#df_yesterday = df_yesterday[df_yesterday['confidence']>.4]
#df_yesterday = df_yesterday.reset_index(drop=True)

old_names = ['ATL', 'CHI', 'GSW', 'BOS', 'BRK', 'DET', 'HOU', 'LAL', 'MEM',
             'MIA', 'MIL', 'OKC', 'ORL', 'PHO', 'POR', 'SAC', 'TOR', 'IND', 
             'LAC', 'NYK', 'CLE', 'DEN', 'PHI', 'SAS', 'NOP', 'WAS', 'CHO', 
             'MIN', 'DAL', 'UTA'] 
new_names = ['Atlanta Hawks', 'Chicago Bulls', 'Golden State Warriors',
       'Boston Celtics', 'Brooklyn Nets', 'Detroit Pistons',
       'Houston Rockets', 'Los Angeles Lakers', 'Memphis Grizzlies',
       'Miami Heat', 'Milwaukee Bucks', 'Oklahoma City Thunder',
       'Orlando Magic', 'Phoenix Suns', 'Portland Trail Blazers',
       'Sacramento Kings', 'Toronto Raptors', 'Indiana Pacers',
       'Los Angeles Clippers', 'New York Knicks', 'Cleveland Cavaliers',
       'Denver Nuggets', 'Philadelphia 76ers', 'San Antonio Spurs',
       'New Orleans Pelicans', 'Washington Wizards', 'Charlotte Hornets',
       'Minnesota Timberwolves', 'Dallas Mavericks', 'Utah Jazz']  

long_to_short_name_dict = {}
for num in range(len(old_names)):
    long_to_short_name_dict[new_names[num]] = old_names[num]

df_yesterday['how_to_bet_abbrev'] = df_yesterday['how_to_bet'].map(long_to_short_name_dict)
df_yesterday.rename(columns={'team':'home_team'}, inplace=True)
df_yesterday.rename(columns={'opponent':'away_team'}, inplace=True)
df_yesterday['home_team_abbrev'] = df_yesterday['home_team'].map(long_to_short_name_dict)
df_yesterday['away_team_abbrev'] = df_yesterday['away_team'].map(long_to_short_name_dict)
df_yesterday.to_csv('results_'+str(dates[i])[:10]+'.csv')
print(df_yesterday[['how_to_bet','spread', 'confidence', 'correct']])
# get these in the df so can give to max: 'score_team', 'score_oppt', 



# ---------
#  plot winning in graph along with historical winnings
#  plot winnings graph so can see it updated daily
#  need to estimate how much would be betting based on initial investment
#  use my and jeff's investment of 10,0000 canadian. but do it it conservatively
#  by altering the return to .94 and by setting the proba of winning low 
#  (e.g., 52? 52.5 max) and also by multiplying the results by a fraction, e.g.,. 
#  by .75. maybe can afford to use 52.5% as preicted win proba if also
#  multiplying the amount to bet by .75. or .5.

dates = pd.date_range('17/11/2016', str(month_yesterday)+'/'+str(day_yesterday)+'/'+str(year_yesterday), freq='D')
dates[-1]
df_bets_this_season = pd.DataFrame()
for date in dates[:]:
    print(date)
    if date.strftime('%Y-%m-%d') in df_schedule_doubled['date'].astype(str).values:
        #df_day = pd.read_csv('df_'+str(date.year)+'_'+str(date.month)+'_'+str(date.day)+'.csv')
        #df_day = pd.read_csv('results_'+str(date.year)+'-'+str(date.month)+'-'+str(date.day)+'.csv')
        df_day = pd.read_csv('results_'+str(date)[:10]+'.csv')
        df_bets_this_season = pd.concat([df_bets_this_season, df_day], ignore_index=True)

df_bets_this_season['date'] = pd.to_datetime(df_bets_this_season['date'])
print('percent correct this season:', str(round(df_bets_this_season['correct'].mean()*100, 1))+'%')

df_results_groupby_date = df_bets_this_season.groupby('date')['correct'].sum()
df_results_groupby_date = df_results_groupby_date.reset_index()
df_results_groupby_date_count = df_bets_this_season.groupby('date')['correct'].count()
df_results_groupby_date_count = df_results_groupby_date_count.reset_index()
df_results_groupby_date_count.rename(columns={'correct':'games'}, inplace=True)
df_results_groupby_date = pd.merge(df_results_groupby_date, df_results_groupby_date_count, on='date')

print('percent correct this season:', round(100 * df_results_groupby_date['correct'].sum() / df_results_groupby_date['games'].sum(), 1))
win_pct_season = round(100 * df_results_groupby_date['correct'].sum() / df_results_groupby_date['games'].sum(), 1)


pot = 5000
win_probability = .52
kelly_criteria = (win_probability * .945 - (1 - win_probability)) / .945
bet_kelly = pot * kelly_criteria
bet_kelly = bet_kelly  # *.75
total_pot_kelly = 0
total_winnings_kelly_list = [0]

for day in dates:
    if day.strftime('%Y-%m-%d') in df_schedule_doubled['date'].astype(str).values:
        wins = df_results_groupby_date[df_results_groupby_date['date']==day]['correct'].values[0]
        games = df_results_groupby_date[df_results_groupby_date['date']==day]['games'].values[0]
        losses = games - wins
        winnings_for_day = wins*.945*bet_kelly
        losses_for_day = losses*bet_kelly*-1
        total_pot_kelly = total_pot_kelly + winnings_for_day + losses_for_day
        total_winnings_kelly_list.append(total_pot_kelly)
        pot = pot + winnings_for_day + losses_for_day
        bet_kelly = pot * kelly_criteria
        bet_kelly = bet_kelly  # *.75
        print(day, bet_kelly)
    

plt.plot(total_winnings_kelly_list, alpha=.4, color='red', linewidth=4)
plt.xlabel('\n days', fontsize=18)
plt.ylabel('money won/lost', fontsize=18)
plt.xlim(0, 50)
#plt.xlim(0,500)
plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1000)
#plt.ylim(min(total_winnings_kelly_list)-5000,70000)
plt.axhline(.5, linestyle='--', color='blue', linewidth=1.5, alpha=.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(axis='y', alpha=.5)
plt.grid(axis='y', which='minor', alpha=.05, color='grey', linewidth=1)
plt.minorticks_on()
plt.title('win percentage: '+str(win_pct_season)+'%'+'\n\nThe line below shows the amount of money won/lost \nbased on a hypothetical inital investiment of $5,000', fontsize=18)
#plt.title('total pot: $' + str(int(money)))
sns.despine()



# --------
# ---------------------------------------------------------
# to get graph of winnings over time starting on Dec 7

dates = pd.date_range('12/7/2016', str(month_yesterday)+'/'+str(day_yesterday)+'/'+str(year_yesterday), freq='D')
dates[-1]
df_bets_this_season = pd.DataFrame()
for date in dates[:]:
    print(date)
    if date.strftime('%Y-%m-%d') in df_schedule_doubled['date'].astype(str).values:
        df_day = pd.read_csv('results_'+str(date)[:10]+'.csv')
        df_bets_this_season = pd.concat([df_bets_this_season, df_day], ignore_index=True)

df_bets_this_season['date'] = pd.to_datetime(df_bets_this_season['date'])
print('percent correct this season:', str(round(df_bets_this_season['correct'].mean()*100, 1))+'%')

df_results_groupby_date = df_bets_this_season.groupby('date')['correct'].sum()
df_results_groupby_date = df_results_groupby_date.reset_index()
df_results_groupby_date_count = df_bets_this_season.groupby('date')['correct'].count()
df_results_groupby_date_count = df_results_groupby_date_count.reset_index()
df_results_groupby_date_count.rename(columns={'correct':'games'}, inplace=True)
df_results_groupby_date = pd.merge(df_results_groupby_date, df_results_groupby_date_count, on='date')

print('percent correct this season:', round(100 * df_results_groupby_date['correct'].sum() / df_results_groupby_date['games'].sum(), 1))
win_pct_season = round(100 * df_results_groupby_date['correct'].sum() / df_results_groupby_date['games'].sum(), 1)


#  plot winnings graph so can see it updated daily

pot = 5000
win_probability = .52
kelly_criteria = (win_probability * .945 - (1 - win_probability)) / .945
bet_kelly = pot * kelly_criteria
bet_kelly = bet_kelly  # *.75
total_pot_kelly = 0
total_winnings_kelly_list = [0]

for day in dates:
    if day.strftime('%Y-%m-%d') in df_schedule_doubled['date'].astype(str).values:
        wins = df_results_groupby_date[df_results_groupby_date['date']==day]['correct'].values[0]
        games = df_results_groupby_date[df_results_groupby_date['date']==day]['games'].values[0]
        losses = games - wins
        winnings_for_day = wins*.945*bet_kelly
        losses_for_day = losses*bet_kelly*-1
        total_pot_kelly = total_pot_kelly + winnings_for_day + losses_for_day
        total_winnings_kelly_list.append(total_pot_kelly)
        pot = pot + winnings_for_day + losses_for_day
        bet_kelly = pot * kelly_criteria
        bet_kelly = bet_kelly  # *.75
        print(day, bet_kelly)
    

plt.plot(total_winnings_kelly_list, alpha=.4, color='red', linewidth=4)
plt.xlabel('\n days', fontsize=18)
plt.ylabel('money won/lost', fontsize=18)
plt.xlim(0, 50)
#plt.xlim(0,500)
plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1000)
#plt.ylim(min(total_winnings_kelly_list)-5000,70000)
plt.axhline(.5, linestyle='--', color='blue', linewidth=1.5, alpha=.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(axis='y', alpha=.5)
plt.grid(axis='y', which='minor', alpha=.05, color='grey', linewidth=1)
plt.minorticks_on()
plt.title('win percentage: '+str(win_pct_season)+'%'+'\n\nThe line below shows the amount of money won/lost \nbased on a hypothetical inital investiment of $5,000', fontsize=18)
#plt.title('total pot: $' + str(int(money)))
sns.despine()








#-------------------------------
#-------------------------------
# for past seasons(s):

def results_one_models_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, iv_variables, dv_var):
    df_results_model_1 = results_one_model_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, iv_variables, dv_var, variables_for_df)
    df_results_model_1 = df_results_model_1[['correct_team', 'how_to_bet', 'correct', 'point_diff_predicted', 'team', 'opponent', 'spread_to_bet', 'confidence']]
    df_results_model_1 = df_results_model_1.reset_index()        
    df_results_all_models = df_results_model_1    
    #df_results_all_models = create_correct_metric(df_results_all_models)
#    df_results_all_models['correct'] = np.nan
#    df_results_all_models.loc[df_results_all_models['how_to_bet']==df_results_all_models['correct_team'], 'correct'] = 1
#    df_results_all_models.loc[df_results_all_models['how_to_bet']!=df_results_all_models['correct_team'], 'correct'] = 0
    # to weed out super low confidece:    
    #df_results_all_models = df_results_all_models[df_results_all_models['confidence']>.5]
    games_to_bet = len(df_results_all_models[df_results_all_models['correct'].notnull()])  # is this working -- don't want to couunt games in which the pt diff equals spread
    wins_games_to_bet = df_results_all_models['correct'].sum()    
    return wins_games_to_bet, games_to_bet, df_results_all_models

#model_1 = linear_model.LinearRegression()
model_1 = linear_model.Ridge(alpha=10)
model_1 = linear_model.Ridge(alpha=75) 
model_1 = linear_model.Ridge(alpha=100) 
model_1 = tree.DecisionTreeRegressor()
model_1 = linear_model.LinearRegression()

year = 2016

#dates = pd.date_range('2015/10/27', '2016/4/13', freq='D')  
dates = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start']==year]['date'].unique()
dates = sorted(dates)
dates = [pd.to_datetime(date) for date in dates]
#dates = dates[20:]

wins = []
games = []
date_of_games = []
df_results_season = pd.DataFrame()
len(dates)
for i in range(len(dates[:])):
    if i < -1:
        None
    else:
        print(i)
        wins_games_to_bet, games_to_bet, df_results_day = results_one_models_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, iv_variables, dv_var)
        #wins_games_to_bet, games_to_bet = results_three_models_one_day(df_covers_bball_ref__dropna_home, i, dates, model_1, model_2, model_3, iv_variables, dv_var)
        wins.append(wins_games_to_bet)
        games.append(games_to_bet)
        date_of_games.append(dates[i])
        df_results_day['date'] = dates[i]    
        df_results_season = pd.concat([df_results_season, df_results_day], ignore_index=True)

for i in range(len(games)):
    print(date_of_games[i], int(wins[i]), '/', int(games[i]))

df_results_season = df_results_season[['date', 'how_to_bet', 'point_diff_predicted', 'spread_to_bet', 'confidence', 'correct']]
df_dates = df_results_season.groupby('date')['correct'].mean()
df_dates = df_dates.reset_index()
df_dates = df_dates.reset_index()
date_to_number_dict = dict(zip(df_dates['date'], df_dates['index']))
df_results_season['day'] = df_results_season['date'].map(date_to_number_dict)
#df_results_season[['date', 'correct', 'day']][150:175]

df_results_season['correct'].mean()
df_results_season['correct'][df_results_season['day']>5].mean()
df_results_season['correct'][df_results_season['day']>21].mean()
df_results_season['correct'][df_results_season['confidence']>.1].mean()
df_results_season['correct'][(df_results_season['day']>22) & (df_results_season['confidence']>.1)].mean()


df_results_season['correct'][300:320]
df_results_season['correct'][df_results_season['day']<120].mean()
df_results_season['correct'][df_results_season.index==15]=0
df_results_season['correct'][df_results_season.index==22]=0
df_results_season['correct'][df_results_season.index==529]=0
df_results_season['correct'][df_results_season.index==530]=0
df_results_season['correct'][df_results_season.index==540]=0
df_results_season['correct'][df_results_season.index==541]=0
df_results_season['correct'][df_results_season.index==810]=0
df_results_season['correct'][df_results_season.index==811]=0
df_results_season['correct'][df_results_season.index==311]=0
df_results_season['correct'][df_results_season.index==319]=0

df_results_season['rolling'] = pd.rolling_mean(df_results_season['correct'], window=75, min_periods=5)
df_results_season['rolling'].plot()
plt.axhline(.515)

df_results_season_x = df_results_season[['day', 'correct']][df_results_season['day']<45]
df_results_season_x['correct'].mean()

df_results_season_x['rolling'] = pd.rolling_mean(df_results_season_x['correct'], window=50, min_periods=5)
df_results_season_x['rolling'].plot()
plt.axhline(.515)


sns.lmplot(x='day', y='correct', data=df_results_season[df_results_season['confidence']>.1], lowess=True)
plt.ylim(.25,.63)
plt.xlim(-5,165)
plt.axhline(.515, linestyle='--', linewidth=1, color='grey', alpha=.75)

sns.lmplot(x='confidence', y='correct', data=df_results_season[df_results_season['day']>5], lowess=True)
plt.ylim(.25,.65)
plt.xlim(-.1,3)
plt.axhline(.515, linestyle='--', linewidth=1, color='grey', alpha=.75)


df_results_season_regular = df_results_season.copy(deep=True)
df_results_season_alt = df_results_season.copy(deep=True)
df_results_season_regression = df_results_season.copy(deep=True)

df_results_season_regular = df_results_season_regular.reset_index()
df_results_season_alt = df_results_season_alt.reset_index()
df_results_season_regression = df_results_season_regression.reset_index()


df_results_merge_regular_alt = pd.merge(df_results_season_regular, df_results_season_alt, on=['date','index'], how='outer')
df_results_merge_regular_alt.tail()

df_results_merge_regular_alt = pd.merge(df_results_merge_regular_alt, df_results_season_regression, on=['date','index'], how='outer')

df_results_merge_regular_alt[df_results_merge_regular_alt['how_to_bet_x']!=df_results_merge_regular_alt['how_to_bet_y']]
df_results_merge_regular_alt['disagree'] = 0
df_results_merge_regular_alt.loc[(df_results_merge_regular_alt['how_to_bet_x']!=df_results_merge_regular_alt['how_to_bet_y']), 'disagree'] = 1

df_results_merge_regular_alt['new_correct'] = np.nan
df_results_merge_regular_alt.loc[(df_results_merge_regular_alt['disagree']==1) & 
(df_results_merge_regular_alt['confidence_x']>df_results_merge_regular_alt['confidence_y']), 'new_correct'] = df_results_merge_regular_alt['correct_x']
df_results_merge_regular_alt.loc[(df_results_merge_regular_alt['disagree']==1) & 
(df_results_merge_regular_alt['confidence_x']<df_results_merge_regular_alt['confidence_y']), 'new_correct'] = df_results_merge_regular_alt['correct_y']
df_results_merge_regular_alt.loc[df_results_merge_regular_alt['disagree']==0, 'new_correct'] = df_results_merge_regular_alt['correct_x']

df_results_merge_regular_alt[['correct', 'correct_x', 'correct_y', 'new_correct']].mean()
df_results_merge_regular_alt[['correct_x', 'correct_y', 'new_correct']][df_results_merge_regular_alt['day_x']>21].mean()
df_results_merge_regular_alt[['correct_x', 'correct_y', 'new_correct']][df_results_merge_regular_alt['confidence_x']>.1].mean()
df_results_merge_regular_alt[['correct_x', 'correct_y', 'new_correct']][(df_results_merge_regular_alt['day_x']>22) & (df_results_merge_regular_alt['confidence_x']>.1)].mean()

df_results_merge_regular_alt[['correct_x', 'correct_y', 'new_correct']][df_results_merge_regular_alt['day_x']<30].mean()


# for decision tree:
#sns.lmplot(x='confidence', y='correct', data=df_results_season[df_results_season['day']>21], lowess=True)
#plt.xlim(-1,30)
#plt.ylim(.35,.6)
#plt.axhline(.515, linestyle='--', alpha=.5, color='grey')
#df_results_season['confidence'].hist()

#df_results_season.to_csv('df_results_ridge_2012_auto.csv')
#df_results_season = pd.read_csv('df_results_ridge_2013_auto.csv')
#df_results_season = pd.read_csv('df_results_ridge_2014_auto.csv')


# -----------------
# graph winnings in prev season (so can compare with current)

round(sum(wins[:]) / sum(games[:]),3)
# excluding first game
round(sum(wins[3:]) / sum(games[3:]),3)
round(sum(wins[30:]) / sum(games[30:]),3)
round(sum(wins[60:]) / sum(games[60:]),3)
round(sum(wins[80:]) / sum(games[80:]),3)

df = pd.DataFrame({'date':date_of_games, 'wins':wins, 'games':games})
df.loc[:,'win_pct_day'] = df.loc[:,'wins'] / df.loc[:,'games']
#df['wins_moving_sum_10'] = pd.rolling_sum(df['wins'], 10)
#df['games_moving_sum_10'] = pd.rolling_sum(df['games'], 10)
#df['pct_moving_avg_10'] = df['wins_moving_sum_10'] / df['games_moving_sum_10']
df_results_groupby_date = df.groupby('date')['wins'].sum()
df_results_groupby_date = df_results_groupby_date.reset_index()
df_results_groupby_date_count = df.groupby('date')['games'].sum()
df_results_groupby_date_count = df_results_groupby_date_count.reset_index()
df_results_groupby_date = pd.merge(df_results_groupby_date, df_results_groupby_date_count, on='date')

print('percent correct this season:', round(100 * df_results_groupby_date['wins'].sum() / df_results_groupby_date['games'].sum(), 1))
win_pct_season = round(100 * df_results_groupby_date['wins'].sum() / df_results_groupby_date['games'].sum(), 1)


pot = 10000
win_probability = .52
kelly_criteria = (win_probability * .94 - (1 - win_probability)) / .94
bet_kelly = pot * kelly_criteria
bet_kelly = bet_kelly*.75
total_pot_kelly = 0
total_winnings_kelly_list = [0]

def create_results_by_day_for_plot_x(df_results_groupby_date, kelly_criteria, bet_kelly, pot, total_pot_kelly, total_winnings_kelly_list):
    for day in df_results_groupby_date['date']:
        print(day)
        wins = df_results_groupby_date[df_results_groupby_date['date']==day]['wins'].values[0]
        games = df_results_groupby_date[df_results_groupby_date['date']==day]['games'].values[0]
        losses = games - wins
        winnings_for_day = wins*.95*bet_kelly
        losses_for_day = losses*bet_kelly*-1
        total_pot_kelly = total_pot_kelly + winnings_for_day + losses_for_day
        total_winnings_kelly_list.append(total_pot_kelly)
        pot = pot + winnings_for_day + losses_for_day
        bet_kelly = pot * kelly_criteria
        bet_kelly = bet_kelly*.75
        print(bet_kelly)
    return total_winnings_kelly_list

def plot_results_by_day_x(total_winnings_kelly_list, alpha, color):
    plt.plot(total_winnings_kelly_list, alpha=alpha, color=color, linewidth=4)
    plt.xlabel('\n days', fontsize=18)
    plt.ylabel('money won/lost', fontsize=18)
    plt.xlim(0, 170)
    #plt.xlim(0,50)
    plt.ylim(-2000,4700)
    plt.ylim(-7000,9200)
    #plt.ylim(min(total_winnings_kelly_list)-5000,70000)
    plt.axhline(.5, linestyle='--', color='blue', linewidth=1.5, alpha=.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='y', alpha=.5)
    plt.grid(axis='y', which='minor', alpha=.05, color='grey', linewidth=1)
    plt.minorticks_on()
    #plt.title('Performance if gambled in the five prior seaons = grey \n Current performance = red', fontsize=18)
    #plt.title('total pot: $' + str(int(money)))
    sns.despine()




#df_results_groupby_date.to_csv('winnings_2015_all_gs_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2014_all_gs_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2013_all_gs_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2012_all_gs_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2011_alll_gs_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2010_all_gs_to_compare.csv')

# these are seasons with the ridge model, all games (no confidence restriction)
df_results_groupby_date_2015 = pd.read_csv('winnings_2015_all_gs_to_compare.csv')
df_results_groupby_date_2014 = pd.read_csv('winnings_2014_all_gs_to_compare.csv')
df_results_groupby_date_2013 = pd.read_csv('winnings_2013_all_gs_to_compare.csv')
df_results_groupby_date_2012 = pd.read_csv('winnings_2012_all_gs_to_compare.csv')
df_results_groupby_date_2011 = pd.read_csv('winnings_2011_alll_gs_to_compare.csv')
df_results_groupby_date_2010 = pd.read_csv('winnings_2010_all_gs_to_compare.csv')


#df_results_groupby_date.to_csv('winnings_2015_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2014_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2013_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2012_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2011_to_compare.csv')
#df_results_groupby_date.to_csv('winnings_2010_to_compare.csv')

# there are seasons with the ridge model and only games with confidence > .5
df_results_groupby_date_2015 = pd.read_csv('winnings_2015_to_compare.csv')
df_results_groupby_date_2014 = pd.read_csv('winnings_2014_to_compare.csv')
df_results_groupby_date_2013 = pd.read_csv('winnings_2013_to_compare.csv')
df_results_groupby_date_2012 = pd.read_csv('winnings_2012_to_compare.csv')
df_results_groupby_date_2011 = pd.read_csv('winnings_2011_to_compare.csv')
df_results_groupby_date_2010 = pd.read_csv('winnings_2010_to_compare.csv')


# tweak df from graph of winnigns so far. compare prior with current.
df_results_groupby_date.rename(columns={'correct':'wins'}, inplace=True)

for results_df, color in zip([df_results_groupby_date_2015, df_results_groupby_date_2014, 
                   df_results_groupby_date_2013, df_results_groupby_date_2012,
                   df_results_groupby_date_2011, df_results_groupby_date], ['grey',
                   'grey','grey','grey','grey','red']): #, df_results_groupby_date_2010]:
#for results_df, color in zip([df_results_groupby_date_2015, df_results_groupby_date_2014, 
#                   df_results_groupby_date_2013, df_results_groupby_date_2012,
#                   df_results_groupby_date, df_results_groupby_date_2010], ['grey',
#                   'grey','grey','grey','grey','red']): #, df_results_groupby_date_2010]:
    if color == 'grey':
        results_df = results_df.iloc[20:,:]
        alpha = .2
    else:
        results_df = results_df
        alpha = .5
    pot = 10000
    win_probability = .52
    kelly_criteria = (win_probability * .94 - (1 - win_probability)) / .94
    bet_kelly = pot * kelly_criteria
    bet_kelly = bet_kelly*.75
    total_pot_kelly = 0
    total_winnings_kelly_list = [0]
    total_winnings_kelly_list = create_results_by_day_for_plot_x(results_df, kelly_criteria, bet_kelly, pot, total_pot_kelly, total_winnings_kelly_list)
    plot_results_by_day_x(total_winnings_kelly_list, alpha, color)

# to bring it to 50%
df_results_groupby_date_2010['wins'].sum() / df_results_groupby_date_2010['games'].sum()
df_results_groupby_date_2010['wins'][df_results_groupby_date_2010.index==145] = 0
df_results_groupby_date_2010['wins'][df_results_groupby_date_2010.index==29] = 0
df_results_groupby_date_2010['wins'][df_results_groupby_date_2010.index==136] = 0


# plot all gs after 20th day, not just those with confidence > .5
for results_df in [df_results_groupby_date_2015, df_results_groupby_date_2014, 
                   df_results_groupby_date_2013, df_results_groupby_date_2010]: #, df_results_groupby_date_2010]: #, df_results_groupby_date_2010]:
    results_df = results_df.iloc[:,:]
    alpha = .2
    color='grey'
    pot = 10000
    win_probability = .525
    kelly_criteria = (win_probability * .94 - (1 - win_probability)) / .94
    bet_kelly = pot * kelly_criteria
    bet_kelly = bet_kelly*.75
    total_pot_kelly = 0
    total_winnings_kelly_list = [0]
    total_winnings_kelly_list = create_results_by_day_for_plot_x(results_df, kelly_criteria, bet_kelly, pot, total_pot_kelly, total_winnings_kelly_list)
    plot_results_by_day_x(total_winnings_kelly_list, alpha, color)







df[df['date']=='2015-12-14']
# LEFT OFF -- HASN'T FIXED. SHOULD BE 3/9. FIND OUT WHY. SOMEHOW NOT
# TAKING INTO ACCUNT THAT SPREAD == POINT DIFF HERE.

plt.plot(df['pct_moving_avg_10'], alpha=.75)
plt.axhline(.515, linestyle='--', alpha=.5, linewidth=1, color='grey')
sns.despine()

plt.plot(df['win_pct_day'], alpha=.75)
plt.axhline(.515, linestyle='--', alpha=.5, linewidth=1, color='grey')


# save results from 2013, 2014, and 2015 seasons in order to compare them
# to the historical. use model_1 = linear_model.Ridge(alpha=10)
df.to_csv('df_2015_results_by_day.csv')
#df.to_csv('df_2014_results_by_day_ridge_10.csv')
#df.to_csv('df_2013_results_by_day_ridge_10.csv')
#df.to_csv('df_2012_results_by_day_ridge_10.csv')



## -----------------------------------------------------------------------------
#  plot winnings graph so can see it updated daily
#  need to estimate how much would be betting based on initial investment
#  use my and jeff's investment of 10,0000 canadian. but do it it conservatively
#  by altering the return to .94 and by setting the proba of winning low 
#  (e.g., 52? 52.5 max) and also by multiplying the results by a fraction, e.g.,. 
#  by .75. maybe can afford to use 52.5% as preicted win proba if also
#  multiplying the amount to bet by .75. or .5.

pot = 10000
win_probability = .52
kelly_criteria = (win_probability * .94 - (1 - win_probability)) / .94
money_kelly = 10000
bet_kelly = pot * kelly_criteria
bet_kelly = bet_kelly*.8

total_pot_kelly = 0
total_winnings_kelly_list = [0]


dates = df['date']
len(dates)
for i in range(0,160):
    
    
    
    
    
    







## -----------------------------------------------------------------------------
#i = 14
#i = -1
#i = -2
#
#wins = []
#games = []
#for i in range(14):
#    i = 10
#
#def results_one_day(df_covers_bball_ref__dropna_home, dates[i], model, iv_variables, dv_var, variables_for_df):
#    df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(df_covers_bball_ref__dropna_home, dates[i], model, iv_variables, dv_var, variables_for_df)
#    df_test_seasons[['date', 'team', 'spread_to_bet', 'opponent', 'point_diff_predicted', 'how_to_bet', 'confidence']]
#    # problems: i = 13, i = 11
#    # 11-7
#
#    df_results = df_test_seasons[['date', 'team', 'spread_to_bet', 'opponent', 'how_to_bet', 'point_difference']]
#    df_results['correct_team'] = np.nan
#    df_results.loc[df_results['point_difference'] > df_results['spread_to_bet']*-1, 'correct_team'] = df_results['team']
#    df_results.loc[df_results['point_difference'] < df_results['spread_to_bet']*-1, 'correct_team'] = df_results['opponent']
#
#    df_results['correct'] = np.nan
#    df_results.loc[df_results['how_to_bet']==df_results['correct_team'], 'correct'] = 1
#    df_results.loc[df_results['how_to_bet']!=df_results['correct_team'], 'correct'] = 0
#    df_results
#
#    print('win pct:', round(df_results['correct'].mean(),2))
#    print(int(df_results['correct'].sum()), '/', int(df_results['correct'].count()))
#
#    wins.append(df_results['correct'].sum())
#    games.append(df_results['correct'].count())
#
#print()
#print()
#print('wins:', int(sum(wins)), '|', 'games:', int(sum(games)))
#print('win pct:', round(sum(wins) / sum(games),3))
# 
#  
#
#
#results_list = [
#[1,3],
#[3,10],
#[1,4],
#[4,8],
#[6,9],
#[3,7],
#[3,4],
#[7,9],
#[5,10],
#[2,5],
#[1,9],
#[4,8],
#[3,6],
#[5,7]
#]
#
#
#wins = []
#games = []
#for l in results_list:
#    wins.append(l[0])
#    games.append(l[1])
#
#print('win pct:', round(sum(wins) / sum(games),3))



#------------------------------------------
# check that algo predicts on prior seasons

# when do this, remember that after about 10 games i can include the distance from 
# playoffs variable in better way to predict these past seasons and that should help

seasons = [2010, 2011, 2012, 2013, 2014, 2015]
model = svm.LinearSVR(C=.05)

def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, test_year, iv_variables):
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < test_year) &
                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['game']>10]                                                                  
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    print ('training n:', len(df_covers_bball_ref_home_train))    
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] == test_year]
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.sort_values(by=['team','date'])
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.reset_index(drop=True)
    print ('test n:', len(df_covers_bball_ref_home_test))
    df_covers_bball_ref_home_train['spread_to_bet'] = df_covers_bball_ref_home_train ['spread'] 
    df_covers_bball_ref_home_test['spread_to_bet'] = df_covers_bball_ref_home_test ['spread'] 
    df_covers_bball_ref_home_test['game_unstandardized'] = df_covers_bball_ref_home_test ['game'] 
    for var in iv_variables:  
        var_mean = df_covers_bball_ref_home_train[var].mean()
        var_std = df_covers_bball_ref_home_train[var].std()
        df_covers_bball_ref_home_train[var] = (df_covers_bball_ref_home_train[var] -  var_mean) / var_std    
        df_covers_bball_ref_home_test[var] = (df_covers_bball_ref_home_test[var] -  var_mean) / var_std    
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test


def add_interactions(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df):
    df_covers_bball_ref_home_train['game_x_playoff_distance'] = df_covers_bball_ref_home_train['game'] * df_covers_bball_ref_home_train['difference_distance_playoffs_abs']
    df_covers_bball_ref_home_test['game_x_playoff_distance'] = df_covers_bball_ref_home_test['game'] * df_covers_bball_ref_home_test['difference_distance_playoffs_abs']
    iv_variables = iv_variables + ['game_x_playoff_distance']
    variables_for_df = variables_for_df + ['game_x_playoff_distance']
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df


def create_correct_metric(df):
    df['correct'] = np.nan
    df.loc[(df['point_diff_predicted'] > df['spread_to_bet']*-1) &
                                       (df['point_difference'] > df['spread_to_bet']*-1), 'correct'] = 1
    df.loc[(df['point_diff_predicted'] > df['spread_to_bet']*-1) &
                                       (df['point_difference'] < df['spread_to_bet']*-1), 'correct'] = 0
    df.loc[(df['point_diff_predicted'] < df['spread_to_bet']*-1) &
                                       (df['point_difference'] < df['spread_to_bet']*-1), 'correct'] = 1
    df.loc[(df['point_diff_predicted'] < df['spread_to_bet']*-1) &
                                       (df['point_difference'] > df['spread_to_bet']*-1), 'correct'] = 0
    # create var to say how much my prediction deviates from actual spread:
    df['predicted_spread_deviation'] = np.abs(df['spread_to_bet'] + df['point_diff_predicted'])
    df['predicted_spread_deviation_raw'] = df['spread_to_bet'] + df['point_diff_predicted']
    return df


def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var):
    model = algorithm
    model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])
    df_covers_bball_ref_home_test.loc[:,'point_diff_predicted'] = predictions_test_set
    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
    return df_covers_bball_ref_home_test


def analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, algorithm, iv_variables, dv_var, variables_for_df):
    iv_variabless_pre_x = iv_variables
    accuracy_list = []
    df_test_seasons = pd.DataFrame()
    for season in seasons:
        print(season)
        df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, season, iv_variables)
        df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df = add_interactions(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df)
        df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var)
        df_covers_bball_ref_home_test['absolute_error'] = np.abs(df_covers_bball_ref_home_test['point_diff_predicted'] - df_covers_bball_ref_home_test['point_difference'])
        df_test_seasons = pd.concat([df_test_seasons, df_covers_bball_ref_home_test])
        accuracy_list.append((season, np.round(df_covers_bball_ref_home_test['correct'].mean(), 4)*100))
        iv_variables = iv_variabless_pre_x
        print()
    return accuracy_list, df_test_seasons, df_covers_bball_ref_home_train


def plot_accuracy(accuracy_list):
    df = pd.DataFrame(accuracy_list, columns=['season', 'accuracy'])
    for season in range(len(seasons)):
        plt.plot([accuracy_list[season][1], accuracy_list[season][1]], label=str(seasons[season]));  #, color='white');
    plt.ylim(49, 58)
    plt.xticks([])
    plt.yticks(fontsize=15)
    plt.legend(fontsize=12)
    plt.ylabel('percent correct', fontsize=18)
    plt.axhline(51.5, linestyle='--', color='grey', linewidth=1, alpha=.5)
    plt.title('accuracy of model in last six years', fontsize=15)
    sns.despine() 
    [print(str(year)+':', accuracy) for year, accuracy in accuracy_list]

accuracy_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model, iv_variables, dv_var, variables_for_df)
plot_accuracy(accuracy_list)















