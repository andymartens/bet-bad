# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:19:21 2016

@author: charlesmartens
"""

cd /Users/charlesmartens/Documents/projects/bet_bball
#----------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf 
#from statsmodels.formula.api import *
#from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import cross_val_predict
#from sklearn import linear_model
#from sklearn.preprocessing import scale
#from sklearn.cross_validation import KFold
#from sklearn.learning_curve import learning_curve
import matplotlib.cm as cm
#from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.ensemble.partial_dependence import plot_partial_dependence
#from collections import Counter
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.metrics import roc_curve
#from sklearn.metrics import auc
#from sklearn.cross_validation import KFold
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import recall_score
#from sklearn.cross_validation import cross_val_score
#from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.ensemble import GradientBoostingRegressor
# try tpot
#from tpot import TPOT
#from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
import urllib.request   
from bs4 import BeautifulSoup
import datetime
sns.set_style('white')



#-------------------------------------------
# set today's date
date_today = datetime.datetime.now().date()
year_now = date_today.year
month_now = date_today.month
day_now = date_today.day
#-------------------------------------------


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
df_moneyline_full.tail()



#df_covers_bball_ref = pd.read_csv('df_covers_bball_ref_2004_to_2015.csv')
df_covers_bball_ref = pd.read_csv('df_covers_bball_ref_2004_to_2015_raw_vars.csv')

df_covers_bball_ref.pop('Unnamed: 0')
df_covers_bball_ref.loc[:,'date'] = pd.to_datetime(df_covers_bball_ref.loc[:,'date'])
df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
#df_covers_bball_ref[['date','team']]
df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
df_covers_bball_ref.loc[:,'point_total'] = df_covers_bball_ref[['score_team', 'score_oppt']].sum(axis=1)
df_covers_bball_ref.loc[:,'over'] = df_covers_bball_ref.loc[:, 'point_total'] - df_covers_bball_ref.loc[:, 'totals']
df_covers_bball_ref[['point_total', 'totals', 'over', 'score_team', 'score_oppt']].tail()


# merge existing df w moneline df
df_covers_bball_ref = pd.merge(df_covers_bball_ref, df_moneyline_full, on=['date','team', 'opponent'], how='left')
df_covers_bball_ref.head()
df_covers_bball_ref[['date','team', 'spread', 'team_close_spread', 'team_open_spread']]
df_covers_bball_ref[['spread','team_close_spread', 'team_open_spread']].corr()
#plt.scatter(df_covers_bball_ref['spread'], df_covers_bball_ref['team_close_spread'], alpha=.4)
#plt.scatter(df_covers_bball_ref['spread'], df_covers_bball_ref['team_open_spread'], alpha=.1)
#plt.scatter(df_covers_bball_ref['team_close_spread'], df_covers_bball_ref['team_open_spread'], alpha=.1)

df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)


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




# ---------------------------------
# to get box scores from yesterday:
dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')
date = dates[-2]  #  replace to -2 once get back days
day_yesterday = date.day
month_yesterday = date.month
year_yesterday = date.year

# then # loop through dates of season and open each and concat
# to get a range of dates, from first date of season to yesterday
dates = pd.date_range('25/10/2016', str(month_yesterday)+'/'+str(day_yesterday)+'/'+str(year_yesterday), freq='D')
df_this_season = pd.DataFrame()
for date in dates:
    print(date)
    df_day = pd.read_csv('df_'+str(date.year)+'_'+str(date.month)+'_'+str(date.day)+'.csv')
    df_this_season = pd.concat([df_this_season,df_day], ignore_index=True)


len(df_covers_bball_ref)  # 29038
len(df_this_season)


# merge new games with historical 
df_covers_bball_ref = pd.concat([df_covers_bball_ref, df_this_season], ignore_index=True)
df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
df_covers_bball_ref.tail()
df_covers_bball_ref[['date', 'team', 'opponent', 'team_tov', 'totals']][df_covers_bball_ref['team']=='Utah Jazz'].tail(10)
len(df_covers_bball_ref)  # 29088 on oct 29, but will grow

# ----------------------------
# to incorpo spread and totals
# open schedule: 
df_schedule = pd.read_excel('schedule_2016_2017.xlsx')
df_schedule.head()
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
        print(date)
        df_odds = pd.read_excel(str(date.month)+'_'+str(date.day)+'_'+str(date.year)+'_'+'odds.xlsx')
        df_spread_doubled = tranform_odds_today(df_odds, date.month, date.day, date.year)
        df_spreads_this_season = pd.concat([df_spreads_this_season, df_spread_doubled], ignore_index=True)
    return df_spreads_this_season

df_spreads_this_season = produce_df_w_all_spreads_this_season(month_now, day_now, year_now)

#df_odds = pd.read_excel(str(month_now)+'_'+str(day_now)+'_'+str(year_now)+'_'+'odds.xlsx')
#df_odds = pd.read_excel('10_26_2016_odds.xlsx')


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
df_covers_bball_ref[['date', 'team', 'totals', 'venue_x', 'venue_y', 'venue_schedule_2']][960:1000]
# ----------------------



# compute pace
df_covers_bball_ref.loc[:, 'team_possessions'] = ((df_covers_bball_ref.loc[:, 'team_fga'] + .44*df_covers_bball_ref.loc[:, 'team_fta'] - df_covers_bball_ref.loc[:, 'team_orb'] + df_covers_bball_ref.loc[:, 'team_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'opponent_possessions'] = ((df_covers_bball_ref.loc[:, 'opponent_fga'] + .44*df_covers_bball_ref.loc[:, 'opponent_fta'] - df_covers_bball_ref.loc[:, 'opponent_orb'] + df_covers_bball_ref.loc[:, 'opponent_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'the_team_pace'] = df_covers_bball_ref[['team_possessions', 'opponent_possessions']].mean(axis=1)
df_covers_bball_ref[['team_possessions', 'opponent_possessions', 'the_team_pace']].head(20)


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
df_covers_bball_ref[['team_predicted_points', 'beat_spread_rolling_std_11']]


# rename vars
df_covers_bball_ref.rename(columns={'team_predicted_points':'the_team_predicted_points'}, inplace=True)
df_covers_bball_ref.rename(columns={'oppt_predicted_points':'the_oppt_predicted_points'}, inplace=True)


variables_for_team_metrics = ['team_3PAr', 'team_ASTpct', 'team_BLKpct', 'team_DRBpct', 'team_DRtg', 
    'team_FTr', 'team_ORBpct', 'team_ORtg', 'team_STLpct', 'team_TOVpct', 'team_TRBpct', 
    'team_TSpct', 'team_eFGpct', 'team_fg3_pct', 'team_fg_pct', 'team_ft_pct', 'team_pf', 
    'the_team_predicted_points', 'opponent_3PAr', 'opponent_ASTpct', 'opponent_BLKpct', 
    'opponent_DRBpct', 'opponent_DRtg', 'opponent_FTr', 'opponent_ORBpct', 'opponent_ORtg', 
    'opponent_STLpct', 'opponent_TOVpct', 'opponent_TRBpct', 'opponent_TSpct', 'opponent_eFGpct', 
    'opponent_fg3_pct', 'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf', 'team_ast',
    'team_blk', 'team_drb', 'team_fg', 'team_fg3', 'team_fg3a', 'team_fga', 'team_ft',
    'team_fta', 'team_orb', 'team_stl', 'team_tov', 'opponent_ast', 'opponent_blk',
    'opponent_drb', 'opponent_fg', 'opponent_fg3', 'opponent_fg3a', 'opponent_fga', 
    'opponent_ft', 'opponent_fta', 'opponent_orb', 'opponent_stl', 
    'opponent_tov', 'venue_x', 'the_oppt_predicted_points', 'beat_spread', 'spread', 
    'score_team', 'score_oppt', 'team_possessions', 'opponent_possessions', 'totals']  #, 'over', 'over_team_predicted_points', 'over_oppt_predicted_points']           


# loop through...
def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    #df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
    for var in variables_for_team_metrics:
        print(var)
        var_ewma = var + '_ewma_15'
        var_ewma_2 = var + '_ewma_2'

        # i have played around with span=x in line below
        # 50 is worse than 20; 25 is worse than 20; 15 is better than 20; 10 is worse than 15; 16 is worse than 15; 14 is worse than 15
        # stick w 15. but 12 is second best. 
        # (12 looks better if want to use cutoffs and bet on fewer gs. but looks like 15 will get about same pct -- 53+ -- wih 1,000 gs)
    
#        for team in df_covers_bball_ref['team'].unique():
#            print(team)
#            df_covers_bball_ref.loc[:, var_ewma] = pd.ewma(df_covers_bball_ref[var].shift(1), span=8)  # change back to 15
#
        
        df_covers_bball_ref.loc[:, var_ewma] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=4))  # change back to 15
        df_covers_bball_ref.loc[:, var_ewma_2] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=10))  # change back to 15
        # avg the ewma_15 and ewma_5
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref[[var_ewma, var_ewma_2]].mean(axis=1)

# insert code to detect outliers -- see below for start of it

#    df_covers_bball_ref['over_team_predict_points_std_ewma_15'] = df_covers_bball_ref.groupby('team')['over_team_predicted_points'].transform(lambda x: pd.ewmstd(x.shift(1), span=15))
#    df_covers_bball_ref['over_oppt_predict_points_std_ewma_15'] = df_covers_bball_ref.groupby('team')['over_oppt_predicted_points'].transform(lambda x: pd.ewmstd(x.shift(1), span=15))
    df_covers_bball_ref.loc[:, 'current_spread_vs_spread_ewma'] = df_covers_bball_ref.loc[:, 'spread'] - df_covers_bball_ref.loc[:, 'spread_ewma_15']
    df_covers_bball_ref.loc[:, 'current_totals_vs_totals_ewma'] = df_covers_bball_ref.loc[:, 'totals'] - df_covers_bball_ref.loc[:, 'totals_ewma_15']
    df_covers_bball_ref['starters_team_lag'] = df_covers_bball_ref['starters_team'].shift(1)
    df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team_lag').cumcount()+1              
    df_covers_bball_ref = df_covers_bball_ref.sort_values(['team','date'])
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)
df_covers_bball_ref[['date','team','lineup_count']]  # still in team-date order. nice.



#------------------------------------------------------------------------------
# compute new variables in this section

# compute days rest
df_covers_bball_ref = df_covers_bball_ref.sort_values(['team','date'])
df_covers_bball_ref['date_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(1))
df_covers_bball_ref['days_rest'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['date_prior_game']) / np.timedelta64(1, 'D')
df_covers_bball_ref.loc[df_covers_bball_ref['days_rest']>1, 'days_rest'] = 2
#df_covers_bball_ref['days_rest'].replace(np.nan, df_covers_bball_ref['days_rest'].mean(), inplace=True)

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
df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==0, 'zone_distance'] = np.abs(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])
df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==1, 'zone_distance'] = -1*np.abs(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])

#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'zone_distance'].abs()
df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'zone_distance']].head(10)
df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'venue_x', 'zone_distance']][960:1000]

#df_covers_bball_ref['zone_distance'].hist()
#sns.barplot(x='zone_distance', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0])
#sns.lmplot(x='zone_distance', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0], y_partial='team_ORtg_ewma_15', x_partial='team_ORtg_ewma_15')


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

df_covers_bball_ref['season_start'][960:1000]

def create_wins(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
    df_all_teams_w_ivs['team_win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'team_win'] = 1
    df_all_teams_w_ivs['team_win_pct'] = df_all_teams_w_ivs.groupby(['season_start', 'team'])['team_win'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=1))
    df_all_teams_w_ivs['opponent_win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['team_win']==0, 'opponent_win'] = 1
    return df_all_teams_w_ivs

df_covers_bball_ref = create_wins(df_covers_bball_ref)
df_covers_bball_ref[['date', 'team', 'team_win', 'team_win_pct']][df_covers_bball_ref['team']=='Detroit Pistons'][960:1000]

#df_2015 = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015]
#df_2015 = df_2015[['date', 'team', 'opponent', 'team_win', 'team_win_pct', 'conference']]
#df_2015.tail()


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
        #date = dates[1]
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
df_w_distance_from_playoffs_year.pop('Unnamed: 0')
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

# THEN CONCAT ALL OTHER YEARS W THIS SEASON - BUT WAIT ABOUT 10 DAYS
df_w_distance_from_playoffs_all_years = pd.concat([df_w_distance_from_playoffs_all_years,df_w_distance_from_playoffs_year], ignore_index=True)


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

df_covers_bball_ref[['date', 'distance_playoffs_abs']][690:990]
df_covers_bball_ref.loc[df_covers_bball_ref['distance_playoffs_abs'].isnull(), 
'distance_playoffs_abs'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['game'] < 20, 'distance_playoffs_abs'] = df_covers_bball_ref.loc[:,'distance_playoffs_abs']/2
#df_covers_bball_ref[['date', 'game', 'distance_playoffs_abs', 'team']][df_covers_bball_ref['team']=='Atlanta Hawks']
df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][500:520]
df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][960:990]

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


# if i looped twice (to adj for oppt):
if 'venue_y' not in df_covers_bball_ref.columns:
    print('true')    
    df_covers_bball_ref['venue_y'] = df_covers_bball_ref['venue_y_y']



#-------------------
# try supping this home court adv code used in spread prediction. see if better
#-------------------
# compute ea team's home court advantage
# but home court advantage is lessening in recent years
# take that into account somehow. and what about other trends over time?
df_covers_bball_ref['home_team_score'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='home', 'home_team_score'] = df_covers_bball_ref['score_team']
df_covers_bball_ref['away_team_score'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='away', 'away_team_score'] = df_covers_bball_ref['score_team']
df_covers_bball_ref[['home_team_score', 'away_team_score']].mean()

df_covers_bball_ref['home_oppt_score'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='home', 'home_oppt_score'] = df_covers_bball_ref['score_oppt']
df_covers_bball_ref['away_oppt_score'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_y']=='away', 'away_oppt_score'] = df_covers_bball_ref['score_oppt']
df_covers_bball_ref[['home_oppt_score', 'away_oppt_score']].mean()

df_covers_bball_ref.loc[:,'home_team_score_ewma'] = df_covers_bball_ref.groupby('team')['home_team_score'].transform(lambda x: pd.ewma(x.shift(1), span=50))
df_covers_bball_ref.loc[:,'away_team_score_ewma'] = df_covers_bball_ref.groupby('team')['away_team_score'].transform(lambda x: pd.ewma(x.shift(1), span=50))
df_covers_bball_ref.loc[:, 'home_team_score_advantage'] = df_covers_bball_ref.loc[:, 'home_team_score_ewma'] - df_covers_bball_ref.loc[:, 'away_team_score_ewma'] 
df_covers_bball_ref[['date', 'score_team', 'home_team_score_ewma', 'away_team_score_ewma', 'home_team_score_advantage']][df_covers_bball_ref['team']=='Atlanta Hawks']

df_covers_bball_ref.loc[:,'home_oppt_score_ewma'] = df_covers_bball_ref.groupby('team')['home_oppt_score'].transform(lambda x: pd.ewma(x.shift(1), span=50))
df_covers_bball_ref.loc[:,'away_oppt_score_ewma'] = df_covers_bball_ref.groupby('team')['away_oppt_score'].transform(lambda x: pd.ewma(x.shift(1), span=50))
df_covers_bball_ref.loc[:, 'home_oppt_score_advantage'] = df_covers_bball_ref.loc[:, 'home_oppt_score_ewma'] - df_covers_bball_ref.loc[:, 'away_oppt_score_ewma'] 
df_covers_bball_ref[['date', 'score_oppt', 'home_oppt_score_ewma', 'away_oppt_score_ewma', 'home_oppt_score_advantage']][df_covers_bball_ref['team']=='Atlanta Hawks']

df_covers_bball_ref.groupby('team')['home_team_score_ewma'].mean()
df_covers_bball_ref.groupby('team')['away_team_score_ewma'].mean()
df_covers_bball_ref[df_covers_bball_ref['season_start']>2014].groupby('team')['home_team_score_advantage'].mean()

df_covers_bball_ref.loc[:,'spread_abs_val'] = df_covers_bball_ref.loc[:,'spread'].abs()


#------------------------------------------------------------------------------
df_covers_bball_ref_save = df_covers_bball_ref.copy(deep=True)

# run this again when want to add a new var and re-run below code:
df_covers_bball_ref = df_covers_bball_ref_save.copy(deep=True)
#------------------------------------------------------------------------------

df_covers_bball_ref['venue'] = df_covers_bball_ref['venue_x']
df_covers_bball_ref.pop('venue_x')
#df_covers_bball_ref.rename(columns={'venue_x':'venue'}, inplace=True)
#df_covers_bball_ref[['venue_x_ewma_15', 'point_difference_ewma_15']]
df_covers_bball_ref[['date','venue']][960:990]
df_covers_bball_ref['season_start'].unique()

df_covers_bball_ref[['date','team', 'totals', 'game']][df_covers_bball_ref['team']=='Boston Celtics'][-85:-75]
df_covers_bball_ref[['date','team', 'totals', 'game']][df_covers_bball_ref['team']=='Philadelphia 76ers'][-85:-75]



variables_for_df = ['date', 'team', 'opponent', 'venue', 'lineup_count', 'score_team_sq_rt',
       'starters_team', 'starters_opponent', 'game', 'spread_abs_val', 'score_team_cu_rt', 'score_team_log',
       'spread', 'score_team', 'score_oppt', 'beat_spread', 'totals', 'totals_ewma_15',
       'over', 'beat_spread_last_g', 'season_start', 'days_rest', 'current_totals_vs_totals_ewma',
       'distance_playoffs_abs', 'starters_same_as_last_g', 'point_total', 'the_team_predicted_points', 
       'the_oppt_predicted_points', 'home_team_score_advantage', 'home_oppt_score_advantage',
       'team_3PAr_ewma_15','team_ASTpct_ewma_15','team_BLKpct_ewma_15','team_DRBpct_ewma_15',
       'team_DRtg_ewma_15','team_FTr_ewma_15','team_ORBpct_ewma_15','team_ORtg_ewma_15',
       'team_STLpct_ewma_15','team_TOVpct_ewma_15','team_TRBpct_ewma_15','team_TSpct_ewma_15',
       'team_eFGpct_ewma_15','team_fg3_pct_ewma_15','team_fg_pct_ewma_15','team_ft_pct_ewma_15',
       'team_pf_ewma_15','the_team_predicted_points_ewma_15','opponent_3PAr_ewma_15',
       'opponent_ASTpct_ewma_15','opponent_BLKpct_ewma_15','opponent_DRBpct_ewma_15',
       'opponent_DRtg_ewma_15','opponent_FTr_ewma_15','opponent_ORBpct_ewma_15','opponent_ORtg_ewma_15',
       'opponent_STLpct_ewma_15','opponent_TOVpct_ewma_15','opponent_TRBpct_ewma_15',
       'opponent_TSpct_ewma_15','opponent_eFGpct_ewma_15','opponent_fg3_pct_ewma_15',
       'opponent_fg_pct_ewma_15','opponent_ft_pct_ewma_15','opponent_pf_ewma_15',
       'team_ast_ewma_15','team_blk_ewma_15','team_drb_ewma_15','team_fg_ewma_15',
       'team_fg3_ewma_15','team_fg3a_ewma_15','team_fga_ewma_15','team_ft_ewma_15',
       'team_fta_ewma_15','team_orb_ewma_15','team_stl_ewma_15','team_tov_ewma_15',
       'opponent_ast_ewma_15','opponent_blk_ewma_15','opponent_drb_ewma_15','opponent_fg_ewma_15',
       'opponent_fg3_ewma_15','opponent_fg3a_ewma_15','opponent_fga_ewma_15','opponent_ft_ewma_15',
       'opponent_fta_ewma_15','opponent_orb_ewma_15','opponent_stl_ewma_15',
       'opponent_tov_ewma_15','venue_x_ewma_15','beat_spread_ewma_15',
       'spread_ewma_15','score_team_ewma_15','score_oppt_ewma_15','current_spread_vs_spread_ewma',
       'the_oppt_predicted_points_ewma_15', 'team_possessions', 'opponent_possessions', 
       'team_possessions_ewma_15', 'opponent_possessions_ewma_15']

iv_variables = ['game', 'beat_spread_last_g', 'lineup_count', 'spread', 'totals', 'spread_abs_val', # 'totals',
'days_rest', 'distance_playoffs_abs', 'the_team_predicted_points', #'totals_ewma_15',
'home_team_score_advantage', 'home_oppt_score_advantage', 'the_oppt_predicted_points', 
'team_3PAr_ewma_15','team_ASTpct_ewma_15','team_BLKpct_ewma_15','team_DRBpct_ewma_15',
'team_DRtg_ewma_15','team_FTr_ewma_15','team_ORBpct_ewma_15','team_ORtg_ewma_15',
'team_STLpct_ewma_15','team_TOVpct_ewma_15','team_TRBpct_ewma_15','team_TSpct_ewma_15',
'team_eFGpct_ewma_15','team_fg3_pct_ewma_15','team_fg_pct_ewma_15','team_ft_pct_ewma_15',
'team_pf_ewma_15','opponent_3PAr_ewma_15', 'opponent_ASTpct_ewma_15', 'the_team_predicted_points_ewma_15',
'opponent_BLKpct_ewma_15','opponent_DRBpct_ewma_15','opponent_DRtg_ewma_15','opponent_FTr_ewma_15',
'opponent_ORBpct_ewma_15','opponent_ORtg_ewma_15','opponent_STLpct_ewma_15','opponent_TOVpct_ewma_15',
'opponent_TRBpct_ewma_15','opponent_TSpct_ewma_15','opponent_eFGpct_ewma_15','opponent_fg3_pct_ewma_15',
'opponent_fg_pct_ewma_15','opponent_ft_pct_ewma_15','opponent_pf_ewma_15','venue_x_ewma_15', 'spread_ewma_15',
'team_possessions_ewma_15', 'the_oppt_predicted_points_ewma_15',  
'opponent_possessions_ewma_15', 'score_team_ewma_15', 'score_oppt_ewma_15'] 


df_covers_bball_ref.loc[:,'point_total'] = df_covers_bball_ref[['score_team', 'score_oppt']].sum(axis=1)
df_covers_bball_ref[['score_team', 'score_oppt', 'point_total']].head()

df_covers_bball_ref['score_team'].hist(bins=20, alpha=.7)

# log transform the dv
df_covers_bball_ref.loc[:, 'score_team_log'] = np.log(df_covers_bball_ref.loc[:, 'score_team'])
df_covers_bball_ref[['score_team', 'score_team_log']].head()
#df_covers_bball_ref['score_team_log'].hist(bins=20, alpha=.6)
#stats.probplot(df_covers_bball_ref['score_team'], dist="norm", plot=plt)
#stats.probplot(df_covers_bball_ref['score_team_log'], dist="norm", plot=plt)

df_covers_bball_ref.loc[:, 'score_team_sq_rt'] = np.sqrt(df_covers_bball_ref.loc[:, 'score_team'])
#df_covers_bball_ref['score_team_sq_rt'].hist(bins=20, alpha=.6)
#stats.probplot(df_covers_bball_ref['score_team_sq_rt'], dist="norm", plot=plt)
# teh square root may be better than cube root in as far as sq root has equal tails

df_covers_bball_ref.loc[:, 'score_team_cu_rt'] = np.cbrt(df_covers_bball_ref.loc[:, 'score_team'])
#df_covers_bball_ref['score_team_cu_rt'].hist(bins=20, alpha=.6)
#stats.probplot(df_covers_bball_ref['score_team_cu_rt'], dist="norm", plot=plt)

# select the dv to predict:
#dv_var = 'ats_win'  
#dv_var = 'win'
#dv_var = 'spread'
#dv_var = 'point_difference'
#dv_var = 'point_total'
#dv_var = 'score_team'
#dv_var = 'score_team_log'
#dv_var = 'score_team_sq_rt'
dv_var = 'score_team_cu_rt'
# cube root is best when compare with sq root and log and regular

iv_and_dv_vars = iv_variables + [dv_var] + ['team', 'opponent', 'date']


# ------------------
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
    df_all_teams_w_ivs = df_all_teams_w_ivs.sort_values(by=['team','date'])
    df_all_teams_w_ivs = df_all_teams_w_ivs.reset_index(drop=True)
    return df_all_teams_w_ivs

df_covers_bball_ref = merge_regular_df_w_switched_df(df_covers_bball_ref, df_covers_bball_ref_switched)    
len(df_covers_bball_ref)


def create_basic_variables(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
#    df_all_teams_w_ivs['over_win'] = 'push'
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['over'] > 0, 'over_win'] = '1'
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['over'] < 0, 'over_win'] = '0'
#    #df_all_teams_w_ivs[['date', 'team', 'opponent', 'beat_spread', 'over_win']]
#    df_all_teams_w_ivs['win'] = 0
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'win'] = 1
#    #df_all_teams_w_ivs[['score_team', 'score_oppt', 'win']]
    df_all_teams_w_ivs.loc[:,'point_difference'] = df_all_teams_w_ivs.loc[:,'score_team'] - df_all_teams_w_ivs.loc[:,'score_oppt']
    return df_all_teams_w_ivs

df_covers_bball_ref = create_basic_variables(df_covers_bball_ref)



def add_past_oppt_ewma(df_covers_bball_ref, iv_variables):
# i liked the below set of code. produced consistent resulst form year to year
    for iv in iv_variables:
        if iv[-7:]=='ewma_15' and (iv[:3]=='tea' or iv[:3]=='opp'):
            print(iv)
            iv_variables = iv_variables + ['x_'+iv]
            df_covers_bball_ref['x_'+iv+'_ewma'] = df_covers_bball_ref.groupby('team')['x_'+iv].transform(lambda x: pd.ewma(x.shift(1), span=4))
            df_covers_bball_ref['x_'+iv+'_ewma_2'] = df_covers_bball_ref.groupby('team')['x_'+iv].transform(lambda x: pd.ewma(x.shift(1), span=10))
            # avg the ewma_15 and ewma_5
            df_covers_bball_ref['x_'+iv+'_ewma'] = df_covers_bball_ref[['x_'+iv+'_ewma', 'x_'+iv+'_ewma_2']].mean(axis=1)
            iv_variables = iv_variables + ['x_'+iv+'_ewma']
    return df_covers_bball_ref, iv_variables

df_covers_bball_ref, iv_variables = add_past_oppt_ewma(df_covers_bball_ref, iv_variables)

# this below code seemed to make worse. figure out which var(s) made worse
# why don't i have oppt day's rest in here? and all other x_vars! distance from playoffs, etc.
iv_variables_more = ['days_rest' ,'spread_ewma_15', 'score_team_ewma_15',
                     'score_oppt_ewma_15']  # distance_playoffs_abs seems to screw it up
                     # 'the_team_predicted_points_ewma_15', 'the_oppt_predicted_points_ewma_15' also screw it up
                     # should i remove these from the team side? try it

#iv_variables_more = ['days_rest' ,'spread_ewma_15', 'score_team_ewma_15',
#                     'score_oppt_ewma_15', 'distance_playoffs_abs', 'the_team_predicted_points_ewma_15', 
#                     'the_oppt_predicted_points_ewma_15']

def add_more_past_oppt_ewma(df_covers_bball_ref, iv_variables, iv_variables_more):
    for iv in iv_variables_more:    
        print(iv)
        iv_variables = iv_variables + ['x_'+iv]
        df_covers_bball_ref['x_'+iv+'_ewma'] = df_covers_bball_ref.groupby('team')['x_'+iv].transform(lambda x: pd.ewma(x.shift(1), span=4))
        df_covers_bball_ref['x_'+iv+'_ewma_2'] = df_covers_bball_ref.groupby('team')['x_'+iv].transform(lambda x: pd.ewma(x.shift(1), span=10))
        # avg the ewma_15 and ewma_5
        df_covers_bball_ref['x_'+iv+'_ewma'] = df_covers_bball_ref[['x_'+iv+'_ewma', 'x_'+iv+'_ewma_2']].mean(axis=1)
        iv_variables = iv_variables + ['x_'+iv+'_ewma']
        print(iv)
    
    iv_variables.remove('the_team_predicted_points_ewma_15')
    iv_variables.remove('the_oppt_predicted_points_ewma_15')
    #iv_variables.remove('distance_playoffs_abs')
    return df_covers_bball_ref, iv_variables

df_covers_bball_ref, iv_variables = add_more_past_oppt_ewma(df_covers_bball_ref, iv_variables, iv_variables_more)



# -----------------------------
# ----------------------------
# call the main file df_covers_bball_ref__dropna_home even though it's not just
# home games. but it'll let me use that file name in code below.
df_covers_bball_ref__dropna_home = df_covers_bball_ref.copy(deep=True)
# ------------------------------

df_covers_bball_ref__dropna_home['season_start'].unique()

def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, date_today, iv_variables):
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < 2016) &
                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    print ('training n:', len(df_covers_bball_ref_home_train))
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['date'] == date_today]
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.sort_values(by=['team','date'])
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.reset_index(drop=True)
    print ('test n:', len(df_covers_bball_ref_home_test))
    df_covers_bball_ref_home_train['total_to_bet'] = df_covers_bball_ref_home_train ['totals'] 
    df_covers_bball_ref_home_test['total_to_bet'] = df_covers_bball_ref_home_test ['totals'] 
    for var in iv_variables:  
        var_mean = df_covers_bball_ref_home_train[var].mean()
        var_std = df_covers_bball_ref_home_train[var].std()
        df_covers_bball_ref_home_train[var] = (df_covers_bball_ref_home_train[var] -  var_mean) / var_std    
        df_covers_bball_ref_home_test[var] = (df_covers_bball_ref_home_test[var] -  var_mean) / var_std    
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test


def create_correct_metric(df):
    df['how_to_bet'] = np.nan
    df.loc[(df['point_total_predicted'] > df['total_to_bet']), 'how_to_bet'] = 'over'
    df['confidence'] = np.nan
    df.loc[(df['point_total_predicted'] < df['total_to_bet']), 'how_to_bet'] = 'under'
    df['confidence'] = np.abs(df['point_total_predicted'] - df['total_to_bet'])
    return df


def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var):
    model = algorithm
    model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])    
    predictions_test_set = np.power(predictions_test_set,3)  # set to 2 if cube root version
    df_covers_bball_ref_home_test.loc[:,'team_score_predicted'] = predictions_test_set
    predictions_train_set = model.predict(df_covers_bball_ref_home_train[iv_variables])
    predictions_train_set = np.power(predictions_train_set,3)  # set to 2 if sq root version
    df_covers_bball_ref_home_train.loc[:,'team_score_predicted'] = predictions_train_set    

    df_opponent_train = df_covers_bball_ref_home_train.copy(deep=True)
    df_opponent_train.loc[:,'opponent'] = df_opponent_train.loc[:,'team']
    df_opponent_train.loc[:,'oppt_score_predicted'] = df_opponent_train.loc[:,'team_score_predicted']
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.merge(df_opponent_train[['date', 'opponent', 'oppt_score_predicted']], on=['opponent', 'date'], how='outer')
    df_covers_bball_ref_home_train.loc[:, 'point_total_predicted'] = df_covers_bball_ref_home_train.loc[:, 'team_score_predicted'] + df_covers_bball_ref_home_train.loc[:, 'oppt_score_predicted']
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['venue']==0]
    df_opponent = df_covers_bball_ref_home_test.copy(deep=True)
    df_opponent.loc[:,'opponent'] = df_opponent.loc[:,'team']
    df_opponent.loc[:,'oppt_score_predicted'] = df_opponent.loc[:,'team_score_predicted']
    len(df_covers_bball_ref_home_test)  # 28560
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.merge(df_opponent[['date', 'opponent', 'oppt_score_predicted']], on=['opponent', 'date'], how='outer')
    len(df_covers_bball_ref_home_test)
    df_covers_bball_ref_home_test.loc[:, 'point_total_predicted'] = df_covers_bball_ref_home_test.loc[:, 'team_score_predicted'] + df_covers_bball_ref_home_test.loc[:, 'oppt_score_predicted']
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test[df_covers_bball_ref_home_test['venue']==0]      
    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
    return df_covers_bball_ref_home_test

#df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, linear_model.LinearRegression())


# select a model
model = linear_model.Ridge(alpha=1)  # higher number regularize more
model = linear_model.LinearRegression()


def produce_bet(df_covers_bball_ref__dropna_home, date_today, model, iv_variables, dv_var, variables_for_df):
    iv_variabless_pre_x = iv_variables

    # use one of the following to to greate test and train sets (ALT trains on all up to prior g)
    df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, date_today, iv_variables)
    #df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs_ALT(df_covers_bball_ref__dropna_home, date_today, iv_variables)

    df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, model, iv_variables, dv_var)
    df_test_seasons = df_covers_bball_ref_home_test.copy(deep=True)
    iv_variables = iv_variabless_pre_x
    return df_test_seasons, df_covers_bball_ref_home_train

df_test_seasons, df_covers_bball_ref_home_train = produce_bet(df_covers_bball_ref__dropna_home, date_today, model, iv_variables, dv_var, variables_for_df)
df_test_seasons[['date', 'team', 'opponent', 'total_to_bet', 'point_total_predicted', 'how_to_bet', 'confidence']]


# to look at the predictions for a particular day, run the range and select
# that day from the list of options. then put into funct above
dates = pd.date_range('25/10/2016', str(month_now)+'/'+str(day_now)+'/'+str(year_now), freq='D')  

df_test_seasons, df_covers_bball_ref_home_train = produce_bet(df_covers_bball_ref__dropna_home, dates[3], model, iv_variables, dv_var, variables_for_df)
df_test_seasons[['date', 'team', 'opponent', 'total_to_bet', 'point_total_predicted', 'how_to_bet', 'confidence']]
























