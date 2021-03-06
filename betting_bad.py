# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:35:03 2016

@author: charlesmartens
"""

# coding: utf-8

# ##To Do: 
# 

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
#from statsmodels.formula.api import *
#from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.ensemble import GradientBoostingRegressor
# try tpot
#from tpot import TPOT
sns.set_style('white')


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

df_moneyline_full[['team_juice', 'opponent_juice']]
df_moneyline_full['team_juice_rev'] = np.nan
df_moneyline_full.loc[df_moneyline_full['team_juice']<0,'team_juice_rev'] = -100 /df_moneyline_full['team_juice']
df_moneyline_full.loc[df_moneyline_full['team_juice']>0,'team_juice_rev'] = df_moneyline_full['team_juice'] / 100
df_moneyline_full['opponent_juice_rev'] = np.nan
df_moneyline_full.loc[df_moneyline_full['opponent_juice']<0,'opponent_juice_rev'] = -100 /df_moneyline_full['opponent_juice']
df_moneyline_full.loc[df_moneyline_full['opponent_juice']>0,'opponent_juice_rev'] = df_moneyline_full['opponent_juice'] / 100

df_moneyline_full[['team_juice', 'opponent_juice', 'team_juice_rev', 'opponent_juice_rev']].tail(10)


df_moneyline['team_open_spread'].hist(bins=20)
plt.axvline(df_moneyline['team_open_spread'].mean())

df_moneyline['team_close_spread'].hist(bins=20)
df_moneyline['team_open_spread'].hist()




#df_covers_bball_ref['spread'][df_covers_bball_ref['venue_x']==0].hist(bins=20)
#plt.axvline(df_covers_bball_ref['spread'][df_covers_bball_ref['venue_x']==0].mean())
#df_covers_bball_ref['spread'][df_covers_bball_ref['venue_x']==0].hist()


#df_covers_bball_ref = pd.read_csv('df_covers_bball_ref_2004_to_2015.csv')
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
plt.scatter(df_covers_bball_ref['spread'], df_covers_bball_ref['team_close_spread'], alpha=.4)
plt.scatter(df_covers_bball_ref['spread'], df_covers_bball_ref['team_open_spread'], alpha=.1)
plt.scatter(df_covers_bball_ref['team_close_spread'], df_covers_bball_ref['team_open_spread'], alpha=.1)

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

# compute pace
df_covers_bball_ref.loc[:, 'team_possessions'] = ((df_covers_bball_ref.loc[:, 'team_fga'] + .44*df_covers_bball_ref.loc[:, 'team_fta'] - df_covers_bball_ref.loc[:, 'team_orb'] + df_covers_bball_ref.loc[:, 'team_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'opponent_possessions'] = ((df_covers_bball_ref.loc[:, 'opponent_fga'] + .44*df_covers_bball_ref.loc[:, 'opponent_fta'] - df_covers_bball_ref.loc[:, 'opponent_orb'] + df_covers_bball_ref.loc[:, 'opponent_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'the_team_pace'] = df_covers_bball_ref[['team_possessions', 'opponent_possessions']].mean(axis=1)
df_covers_bball_ref[['team_possessions', 'opponent_possessions', 'the_team_pace']].head(20)


# assign the totals and spread variables
def assign_spread_and_totals(df_covers_bball_ref, spread_to_use, totals_to_use):
    df_covers_bball_ref.loc[:, 'totals_covers'] = df_covers_bball_ref.loc[:, 'totals']
    df_covers_bball_ref.loc[:, 'spread_covers'] = df_covers_bball_ref.loc[:, 'spread']
    
    df_covers_bball_ref.loc[:, 'totals'] = df_covers_bball_ref.loc[:, totals_to_use]
    #df_covers_bball_ref.loc[:, 'totals'] = df_covers_bball_ref.loc[:, 'Closing Total']
    
    df_covers_bball_ref.loc[:, 'spread'] = df_covers_bball_ref.loc[:, spread_to_use]
    #df_covers_bball_ref.loc[:, 'spread'] = df_covers_bball_ref.loc[:, 'team_close_spread']
    return df_covers_bball_ref

#df_covers_bball_ref = assign_spread_and_totals(df_covers_bball_ref, 'team_close_spread', 'Closing Total')
#df_covers_bball_ref = assign_spread_and_totals(df_covers_bball_ref, 'team_close_spread', 'Closing Total')
df_covers_bball_ref = assign_spread_and_totals(df_covers_bball_ref, 'spread', 'totals')

df_covers_bball_ref.loc[:,'totals_abs_diff'] = np.abs(df_covers_bball_ref.loc[:,'totals_covers'] - df_covers_bball_ref.loc[:,'Closing Total'])
df_covers_bball_ref['spread_abs_diff'] = np.abs(df_covers_bball_ref.loc[:,'spread_covers'] - df_covers_bball_ref.loc[:,'team_close_spread'])
df_covers_bball_ref[['totals_abs_diff', 'spread_abs_diff']].hist()
#df_covers_bball_ref[['date', 'spread', 'spread_covers', 'team_close_spread', 'spread_abs_diff']].sort_values(by='spread_abs_diff', ascending=False).head(40)
#df_covers_bball_ref[['date', 'totals', 'totals_covers', 'Closing Total', 'totals_abs_diff']].sort_values(by='totals_abs_diff', ascending=False).head(50)


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
    #df_teams_covers['beat_spread_last_g'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: x.shift(1))
    return df_teams_covers

df_covers_bball_ref = create_variables_covers(df_covers_bball_ref)
df_covers_bball_ref[['team_predicted_points', 'beat_spread_rolling_std_11']]


# rename vars
df_covers_bball_ref.rename(columns={'team_predicted_points':'the_team_predicted_points'}, inplace=True)
df_covers_bball_ref.rename(columns={'oppt_predicted_points':'the_oppt_predicted_points'}, inplace=True)

variables_for_team_metrics = ['team_3PAr', 'team_ASTpct', 'team_BLKpct', 'team_DRBpct', 
'team_DRtg', 'team_FTr', 'team_ORBpct', 'team_ORtg', 'team_STLpct', 'team_TOVpct', 
'team_TRBpct', 'team_TSpct', 'team_eFGpct', 'team_fg3_pct', 'team_fg_pct', 'team_ft_pct', 
'team_pf', 'opponent_3PAr', 'opponent_ASTpct', 'opponent_BLKpct', 'opponent_DRBpct', 
'opponent_DRtg', 'opponent_FTr', 'opponent_ORBpct', 'opponent_ORtg', 'opponent_STLpct', 
'opponent_TOVpct', 'opponent_TRBpct', 'opponent_TSpct', 'opponent_eFGpct', 'opponent_fg3_pct', 
'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf', 'spread', 'beat_spread', 'venue_x',
'point_difference']           



#------------------------------------------------------------------------------
# OPTIONAL - add noise to the spread:
df_covers_bball_ref['random_number'] = np.random.normal(0, .4, len(df_covers_bball_ref))
df_covers_bball_ref['random_number'].hist(alpha=.7)
df_covers_bball_ref['totals'] = df_covers_bball_ref.loc[:, 'totals'] + df_covers_bball_ref.loc[:, 'random_number']

df_covers_bball_ref['random_number'] = np.random.normal(0, .4, len(df_covers_bball_ref))
df_covers_bball_ref['random_number'].hist(alpha=.7)
df_covers_bball_ref['spread'] = df_covers_bball_ref.loc[:, 'spread'] + df_covers_bball_ref.loc[:, 'random_number']

# use ending spread or starting spread?
# ending spread is likely more accurate
# beginning spread may provide my predicted score with more room to move?



###########  create new variables to try out in this section  #############
# Choose one of next set of cells to create metrics for ea team/lineup

# group just by team, not by season too. need to do weighted mean or a rolling mean here.
def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
    for var in variables_for_team_metrics:
        var_ewma = var + '_ewma_15'
        var_ewma_2 = var + '_ewma'

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
    df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))                
    #df_covers_bball_ref['beat_spread'] = df_covers_bball_ref.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_max(x.shift(1), window=15))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))
#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))


#-------------
# start of code to detect outliers
# for each var -- regress spread out
# because when i look for outliers, want to see if there's something that happened 
# that's far and above or below what we would expect. so i'll have a distrib of 
# the resid of a variable for each team. that'll give me the mean, which should be 
# about 0. and it'll give me the std of that team. so i could just use the std and
# see if the team is over 2 std above or below 0, or the mean, which should be about 0.
# presumably teams have pretty diff resid std? should be, right? check.

results = smf.ols(formula = var + ' ~ spread', data=df_covers_bball_ref).fit()
results.summary()
df_covers_bball_ref[var+'_resid'] = results.resid
df_covers_bball_ref[['team_fg_pct', var+'_resid']].head(20)
x = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015].groupby('team')[var].mean()
x.sort(ascending=False)
x = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015].groupby('team')[var+'_resid'].mean()
x.sort(ascending=False)

df_covers_bball_ref[['season_start', 'team', 'date', 'opponent_ASTpct', 'team_ASTpct', 'team_fg_pct']][df_covers_bball_ref['season_start']==2015]
# team and oppt fg% mixed up?????

# get std up to but no including last 3 gs. use ewma span=15 too.


###################################
###################################
# skip - group by season and by team:
df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team', 'date'])

def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
    for var in variables_for_team_metrics:
        var_ewma = var + '_ewma_15'
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby(['season_start', 'team'])[var].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref['count'] = df_covers_bball_ref.groupby(['season_start', 'team'])['team_3PAr'].transform(lambda x: pd.expanding_count(x))


##################################
# skip - alt - group by lineup (not team) 
# surprisingly, this didn't seem to help

def loop_through_lineups_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    df_covers_bball_ref = df_covers_bball_ref.sort('date')
    for var in variables_for_team_metrics:
        var_ewma = var + '_lineups_15'
        #df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('starters_team')[var].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('starters_team')[var].transform(lambda x: pd.ewma(x.shift(1), span=15))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_lineups_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))

# change name of starters_team and starters_opponent to team and opponent
# so that all subsequent code will treat these as the teams and opponent
#def substitute_linups_for_team_names(df_covers_bball_ref):
#    df_covers_bball_ref['team_actual'] = df_covers_bball_ref['team']
#    df_covers_bball_ref['opponent_actual'] = df_covers_bball_ref['opponent']
#    df_covers_bball_ref['team'] = df_covers_bball_ref['starters_team']
#    df_covers_bball_ref['opponent'] = df_covers_bball_ref['starters_opponent']
#    df_covers_bball_ref[['team_actual', 'opponent_actual', 'team', 'opponent', 'starters_team']].head()
#    return df_covers_bball_ref
    
#df_covers_bball_ref = substitute_linups_for_team_names(df_covers_bball_ref)


# merge the lineups metrics with the team metrics
df_covers_bball_ref[['team_ASTpct_ewma_15', 'team_ORtg_ewma_15', 'team_pf_ewma_15', 'team_count',
                     'team_ASTpct_lineups_15', 'team_ORtg_lineups_15', 'team_pf_lineups_15', 'lineup_count']].tail()


def loop_through_vars_to_create_compsite_metrics(df_covers_bball_ref, variables_for_team_metrics):
    for var in variables_for_team_metrics:
        var_ewma_orig = var + '_ewma_15'
        var_ewma_lineup = var + '_lineups_15'
        var_ewma_comp = var + '_ewma_15_comp'
        df_covers_bball_ref[var_ewma_comp] = df_covers_bball_ref[var_ewma_orig]
        df_covers_bball_ref.loc[df_covers_bball_ref['lineup_count'] > 10, var_ewma_comp] = df_covers_bball_ref[[var_ewma_orig, var_ewma_lineup]].mean(axis=1)
    return df_covers_bball_ref
        
df_covers_bball_ref = loop_through_vars_to_create_compsite_metrics(df_covers_bball_ref, variables_for_team_metrics)


df_covers_bball_ref[['team_ASTpct_ewma_15', 'team_ORtg_ewma_15', 'team_ASTpct_lineups_15', 'team_ORtg_lineups_15', 
                     'team_ASTpct_ewma_15_comp', 'team_ORtg_ewma_15_comp', 'lineup_count']].tail()


def loop_through_vars_to_change_names(df_covers_bball_ref, variables_for_team_metrics):
    for var in variables_for_team_metrics:
        var_ewma_orig = var + '_ewma_15'
        var_ewma_comp = var + '_ewma_15_comp'
        df_covers_bball_ref[var_ewma_orig] = df_covers_bball_ref[var_ewma_comp]
    return df_covers_bball_ref
        
df_covers_bball_ref = loop_through_vars_to_change_names(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref[['team_ASTpct_ewma_15', 'team_ORtg_ewma_15', 'team_AST%_lineups_15', 'team_ORtg_lineups_15', 
                     'team_ASTpct_ewma_15_comp', 'team_ORtg_ewma_15_comp', 'lineup_count']].tail()
#################################
#################################


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


#-----------------------------------------
# SKIP IN LIEU OF ZONE_DISTANCE CALC BELOW
#df_covers_bball_ref['team_zone'] = df_covers_bball_ref['team'].map(team_to_zone_dict)
#df_covers_bball_ref['opponent_zone'] = df_covers_bball_ref['opponent'].map(team_to_zone_dict)
#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone']
#df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'zone_distance']].head(10)
#-----------------------------------------


df_covers_bball_ref['team_zone'] = df_covers_bball_ref['team'].map(team_to_zone_dict)
df_covers_bball_ref['opponent_zone'] = df_covers_bball_ref['opponent'].map(team_to_zone_dict)
# also try alt zone_distance
df_covers_bball_ref['zone_distance'] = np.nan
df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==0, 'zone_distance'] = np.abs(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])
df_covers_bball_ref.loc[df_covers_bball_ref['venue_x']==1, 'zone_distance'] = -1*np.abs(df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone'])


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

#df_2015 = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015]
#df_2015 = df_2015[['date', 'team', 'opponent', 'team_win', 'team_win_pct', 'conference']]
#df_2015.tail()


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
def create_df_w_standings_for_year(year, df_covers_bball_ref):
    df_year = df_covers_bball_ref[df_covers_bball_ref['season_start']==year]
    df_year = df_year[['date', 'team', 'opponent', 'team_win', 'team_win_pct', 'conference']]
    teams = df_year['team'].unique()
    dates = df_year['date'].unique()
    df_w_distance_from_playoffs = pd.DataFrame()
    for date in dates[10:]:
        standings_date_dict = create_standings_dict_for_date(date, teams, df_year)
        df_standings_date = create_df_w_distance_from_playoffs(standings_date_dict)
        df_w_distance_from_playoffs = pd.concat([df_w_distance_from_playoffs, df_standings_date], ignore_index=True)
    return df_w_distance_from_playoffs
        
#df_w_distance_from_playoffs_2015 = create_df_w_standings_for_year(2015, df_covers_bball_ref)
#print len(df_w_distance_from_playoffs_2015)
#print len(dates[10:]) * 30
#df_w_distance_from_playoffs_2015.head()


# i ran this function and saved df_w_distance_from_playoffs_all_years
# it took a few min, so don't do again. just get df_w_distance_from_playoffs_all_years and merge

#df_w_distance_from_playoffs_all_years = pd.DataFrame()
#for year in [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]:
#    df_w_distance_from_playoffs_year = create_df_w_standings_for_year(year, df_covers_bball_ref)
#    df_w_distance_from_playoffs_all_years = pd.concat([df_w_distance_from_playoffs_all_years, df_w_distance_from_playoffs_year], ignore_index=True)
    
#print len(df_w_distance_from_playoffs_all_years)  
#df_w_distance_from_playoffs_all_years.to_csv('df_w_distance_from_playoffs_all_years.csv')


df_w_distance_from_playoffs_all_years = pd.read_csv('df_w_distance_from_playoffs_all_years.csv')
df_w_distance_from_playoffs_all_years.pop('Unnamed: 0')
df_w_distance_from_playoffs_all_years['date'] = pd.to_datetime(df_w_distance_from_playoffs_all_years['date'])
print(len(df_w_distance_from_playoffs_all_years))
#df_covers_bball_ref_test = df_covers_bball_ref[:5000]
print(len(df_covers_bball_ref))

df_covers_bball_ref = pd.merge(df_covers_bball_ref, df_w_distance_from_playoffs_all_years, 
                                      on=['team', 'date'], how='left')
print(len(df_covers_bball_ref))

df_covers_bball_ref['game'] = df_covers_bball_ref.groupby(['season_start', 'team'])['spread'].transform(lambda x: pd.expanding_count(x))
df_covers_bball_ref[['date', 'team', 'game']][df_covers_bball_ref['team']== 'Detroit Pistons'].head(100)

df_covers_bball_ref['distance_playoffs_abs'] = df_covers_bball_ref['distance_from_playoffs'].abs()
# should weight these so that takes the mean and gradually a team-specific number overtakes
# it as the n increases. how?

df_covers_bball_ref.loc[df_covers_bball_ref['distance_playoffs_abs'].isnull(), 
'distance_playoffs_abs'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['game'] < 20, 'distance_playoffs_abs'] = df_covers_bball_ref.loc[:,'distance_playoffs_abs']/2
#df_covers_bball_ref[['date', 'game', 'distance_playoffs_abs', 'team']][df_covers_bball_ref['team']=='Atlanta Hawks']


df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][500:520]
#df_w_distance_from_playoffs_all_years[df_w_distance_from_playoffs_all_years['date']=='2004-12-06']
#df_w_distance_from_playoffs_all_years['date'].dtypes



# compute if last g has a diff lineup
# if the lineup is diff than the game before, does that make it harder to guess?
# need compute whether lineup is diff than prior g here
df_covers_bball_ref['starters_team_last_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(1))
df_covers_bball_ref['starters_team_two_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(2))

df_covers_bball_ref['starters_same_as_last_g'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['starters_team']==df_covers_bball_ref['starters_team_last_g'], 'starters_same_as_last_g'] = 1

df_covers_bball_ref['starters_same_as_two_g'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['starters_team']==df_covers_bball_ref['starters_team_two_g'], 'starters_same_as_two_g'] = 1

#df_covers_bball_ref[['date', 'starters_team', 'starters_team_last_g', 'starters_same_as_last_g']][df_covers_bball_ref['team']=='San Antonio Spurs'].tail(10)

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




# try new home court calc-----------------------
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
#------------------------------------------------





#g = df_covers_bball_ref.groupby('team')['home_court_advantage'].mean()
#g.sort()
#g.plot(kind='barh', sort_columns=True)
#plt.xlabel('home court advantage')
# makes sense -- when you play at home, you get half of this home court adv.
# and when you play away, you loose half of it.

df_covers_bball_ref.loc[:,'spread_abs_val'] = df_covers_bball_ref.loc[:,'spread'].abs()


#------------------------------------------------------------------------------
# run this and run again when want to add a new var and re-run below code:
df_covers_bball_ref_save = df_covers_bball_ref.copy(deep=True)

# run this again when want to add a new var and re-run below code:
df_covers_bball_ref = df_covers_bball_ref_save.copy(deep=True)
#------------------------------------------------------------------------------

df_covers_bball_ref.rename(columns={'venue_x':'venue'}, inplace=True)
#df_covers_bball_ref[['venue_x_ewma_15', 'point_difference_ewma_15']]


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
       'team_close_ml', 'opponent_close_ml', 'team_close_spread', 'team_open_spread',
       'team_juice_rev', 'opponent_juice_rev', 'Opening Total', 'Closing Total', 
       'totals_abs_diff', 'spread_abs_diff', 'totals_covers']

# include all vars i want to precict with
iv_variables = ['spread', 'totals', 'lineup_count', 
       'spread_ewma_15', #'current_spread_vs_spread_ewma',
       'beat_spread_ewma_15', #'beat_spread_std_ewma_15',
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
       'days_rest', 'distance_playoffs_abs', 'game', 'zone_distance', 'home_court_advantage'] # , 'venue_x_ewma_15' - 
       # don't think venue_ewma matters because i've already got spread_ewma in there
       # and that should be adjusting for the compeition and hoe court already in the past x games, 
       #, 'point_difference_ewma_15' 'zone_distance', 'starters_same_as_last_g']  , 'season_start'

for var in iv_variables:
    print(var)

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
    df_all_teams_w_ivs['ats_win'] = 'push'
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] > 0, 'ats_win'] = '1'
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] < 0, 'ats_win'] = '0'
    df_all_teams_w_ivs[['date', 'team', 'opponent', 'beat_spread', 'ats_win']]
    df_all_teams_w_ivs['win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'win'] = 1
    df_all_teams_w_ivs[['score_team', 'score_oppt', 'win']]
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

iv_variables = create_iv_list(iv_variables, ['spread', 'totals', 'game', 'zone_distance'])  # zone_distance, 'season_start'


def create_home_df(df_covers_bball_ref):
    df_covers_bball_ref_home = df_covers_bball_ref[df_covers_bball_ref['venue'] == 0]
    df_covers_bball_ref_home = df_covers_bball_ref_home.reset_index()
    len(df_covers_bball_ref_home)
    return df_covers_bball_ref_home

df_covers_bball_ref_home = create_home_df(df_covers_bball_ref)


#--------------------
# create new vars

# home court advantage - not working. how else to compute? subtract home win% from oppt away win%
df_covers_bball_ref_home.loc[:, 'home_advantage_added'] = df_covers_bball_ref_home.loc[:, 'home_point_diff_ewma'] - df_covers_bball_ref_home.loc[:, 'x_away_point_diff_ewma']
iv_variables = iv_variables + ['home_advantage_added', 'home_point_diff_ewma', 'x_away_point_diff_ewma']
variables_for_df = variables_for_df + ['home_advantage_added', 'x_away_point_diff_ewma']
#iv_variables.remove('home_point_diff_ewma')
#iv_variables.remove('x_away_point_diff_ewma')
# think about this more. am i capturing home court advantage as well as i can?

df_covers_bball_ref_home.loc[:, 'home_court_advantage_difference'] = df_covers_bball_ref_home.loc[:, 'home_court_advantage'] - df_covers_bball_ref_home.loc[:, 'x_home_court_advantage']*-1
iv_variables = iv_variables + ['home_court_advantage_difference']
variables_for_df = variables_for_df + ['home_court_advantage_difference']


# ----------------------------
# drop nans
print('\n number of games with nans:', len(df_covers_bball_ref_home))
df_covers_bball_ref__dropna_home = df_covers_bball_ref_home.dropna()
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.sort_values(by=['team','date'])
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.reset_index(drop=True)
print('\n number of games without nans:', (len(df_covers_bball_ref__dropna_home)))

## ----------------------------
## scale vars
#
##variable = iv_variables[0]
#iv_variables_z = []
#for i, variable in enumerate(iv_variables[:]):
#    iv_variables_z.append(variable+'_z')
#    df_covers_bball_ref__dropna_home.loc[:,variable +'_z'] = StandardScaler().fit_transform(df_covers_bball_ref__dropna_home[[variable]])
#    model_plot = smf.ols(formula = 'point_difference ~ ' + variable,
#                data=df_covers_bball_ref__dropna_home).fit()  
#    t = round(model_plot.tvalues[1], 3)
#    p = round(model_plot.pvalues[1], 3)
#    model_plot = smf.ols(formula = 'point_difference ~ ' + variable+'_z',
#                data=df_covers_bball_ref__dropna_home).fit()  
#    t_z = round(model_plot.tvalues[1], 3)
#    p_z = round(model_plot.pvalues[1], 3)
#    if t == t_z and p == p_z:
#        print(i, 'ok', p_z)
#    else:
#        print(i, variable)
#
#
#iv_variables_original = iv_variables
## CHANGE THIS TO iv_variables_to_analyze? I THINK SO. AND THEN DON'T SCALE IN FINAL FUNCTION
##iv_variables = iv_variables_z
##iv_variables_to_analyze = iv_variables_z
#iv_variables = iv_variables_z
#
#df_covers_bball_ref__dropna_home[[variable, variable+'_z']].hist()



# ------------------
# variable reduction -- pca
# SKIP for now. doesn't help much, though might help a little to set pca to .99 (i.e., 99% of variance)
# pca (and could try Xs after that?)
# when actually do this, should: 
# 1. take training set and fit pca
# 2. transform the training set using that pca model
# 3. transform the test set using that model

len(iv_variables)
# can tell is to give back the components that preserve, say, 95% of variance of the data
pca_stand = PCA(.99).fit(df_covers_bball_ref__dropna_home[iv_variables])

# see how many components should keep
# the amount of variance that each PC explains
variance_explained = pca_stand.explained_variance_ratio_
len(variance_explained)
# cumulative variance explained
var1 = np.cumsum(np.round(pca_stand.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)

# now can redo but set n_components to however many components want to keep
#pca_stand = PCA(n_components=25).fit(df_covers_bball_ref__dropna_home[iv_variables])
pca_stand = PCA(n_components=40).fit(df_covers_bball_ref__dropna_home[iv_variables])
variable_names = []
for i in range(40):
    variable = '_'+str(i)
    variable_names.append(variable)
df_home_pca =  pd.DataFrame(pca_stand.transform(df_covers_bball_ref__dropna_home[iv_variables]), columns=variable_names)
df_home_pca['_4']
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.reset_index()
df_covers_bball_ref__dropna_home = pd.concat([df_covers_bball_ref__dropna_home, df_home_pca], axis=1)
df_covers_bball_ref__dropna_home.head()

iv_variables = variable_names


# ---------------------------
# add interactions -- do after scaling
# TRY MORE INTERACTIONS WTIH GAME

# if standardizing, skip here and use funciton after standardization
# seemed to help, so use. and explore more -- yeah really makes a difference!!! more motivational vars???
df_covers_bball_ref__dropna_home['game_x_playoff_distance'] = df_covers_bball_ref__dropna_home['game'] * df_covers_bball_ref__dropna_home['difference_distance_playoffs_abs']
iv_variables = iv_variables + ['game_x_playoff_distance']
variables_for_df = variables_for_df + ['game_x_playoff_distance']

#df_covers_bball_ref__dropna_home['game_x_playoff_distance_z'] = df_covers_bball_ref__dropna_home['game_z'] * df_covers_bball_ref__dropna_home['difference_distance_playoffs_abs_z']
#iv_variables = iv_variables + ['game_x_playoff_distance_z']
#variables_for_df = variables_for_df + ['game_x_playoff_distance_z']

# if don't scale here, do this:
#df_covers_bball_ref__dropna_home['game_x_playoff_distance'] = df_covers_bball_ref__dropna_home['game'] * df_covers_bball_ref__dropna_home['difference_distance_playoffs_abs']
#iv_variables = iv_variables + ['game_x_playoff_distance']
#variables_for_df = variables_for_df + ['game_x_playoff_distance']



# do regression right here w and without standardizing vars. should be same.
string_for_regression = ''
for var in iv_variables:
    print(var)
    string_for_regression += ' + ' + var
string_for_regression = string_for_regression[3:]        
model_plot = smf.ols(formula = 'point_difference ~ ' + string_for_regression,
                data=df_covers_bball_ref__dropna_home).fit()  
print(model_plot.summary()) 



#------------------------------------------------------------------------------
# compare spread with diff between past spread histories
#df_covers_bball_ref__dropna_home.loc[:, 'spread_vs_past_spread_histories'] = df_covers_bball_ref__dropna_home.loc[:, 'spread'] - df_covers_bball_ref__dropna_home.loc[:, 'difference_spread_ewma_15']
#df_covers_bball_ref__dropna_home[['spread_vs_past_spread_histories', 'difference_current_spread_vs_spread_ewma']].corr()
### interesting -- these 2 aren't the same (corr ~ .6)
#results = smf.ols(formula = 'point_difference ~ spread_vs_past_spread_histories + difference_current_spread_vs_spread_ewma + spread', data=df_covers_bball_ref__dropna_home).fit()
#print(results.summary())  # p = .149
#
#df_covers_bball_ref__dropna_home.loc[:,'spread_vs_past_spread_histories' +'_z'] = StandardScaler().fit_transform(df_covers_bball_ref__dropna_home[['spread_vs_past_spread_histories']])
#
#iv_variables = iv_variables + ['spread_vs_past_spread_histories' +'_z']
#variables_for_df = variables_for_df + ['spread_vs_past_spread_histories' +'_z']

#this doesnt help

#variables_for_df = variables_for_df + ['spread_vs_past_spread_histories']
# crazy -- this var doesn't predict at all when difference_current_spread_vs_spread_ewma is in the model
# so what's special about difference_current_spread_vs_spread_ewma???


# ---------------------
# skip for now ------
df_covers_bball_ref_home.loc[:, 'starters_the_same'] = df_covers_bball_ref_home.loc[:, 'starters_same_as_last_g'] + df_covers_bball_ref_home.loc[:, 'x_starters_same_as_last_g']
iv_variables = iv_variables + ['starters_the_same']
variables_for_df = variables_for_df + ['starters_the_same']

#----------------
# skip for now
# add interactions with game. what might interact such that can predict early games better?
df_covers_bball_ref_home['game_x_spread'] = df_covers_bball_ref_home['game'] * df_covers_bball_ref_home['spread']
iv_variables = iv_variables + ['game_x_spread']
variables_for_df = variables_for_df + ['game_x_spread']

df_covers_bball_ref_home['game_x_days_rest'] = df_covers_bball_ref_home['game'] * df_covers_bball_ref_home['difference_days_rest']
iv_variables = iv_variables + ['game_x_days_rest']
variables_for_df = variables_for_df + ['game_x_days_rest']

df_covers_bball_ref_home['game_x_team_ORtg_ewma_15'] = df_covers_bball_ref_home['game'] * df_covers_bball_ref_home['difference_team_ORtg_ewma_15']
iv_variables = iv_variables + ['game_x_team_ORtg_ewma_15']
variables_for_df = variables_for_df + ['game_x_team_ORtg_ewma_15']

df_covers_bball_ref_home['game_x_opponent_ORtg_ewma_15'] = df_covers_bball_ref_home['game'] * df_covers_bball_ref_home['difference_opponent_ORtg_ewma_15']
iv_variables = iv_variables + ['game_x_opponent_ORtg_ewma_15']
variables_for_df = variables_for_df + ['game_x_opponent_ORtg_ewma_15']


#----------------
# skip for now
# add cubic -- yea think i should really try this. skip sigmoid for now. this cubic may be pretty good at modeling
df_covers_bball_ref_home['difference_team_ASTpct_ewma_15_squared'] = df_covers_bball_ref_home[['difference_team_ASTpct_ewma_15']]**2
iv_variables = iv_variables + ['game_x_playoff_distance']
variables_for_df = variables_for_df + ['game_x_playoff_distance']

df_covers_bball_ref_home[['difference_team_ASTpct_ewma_15', 'difference_team_ASTpct_ewma_15_squared']].tail()

#x = np.arange(-100,100)
#x_sq = x**2
#x_cu = x**3
#
#plt.scatter(x,x_sq)
#plt.scatter(x,x_cu)
#
#y = -5.5*x + 2.25*x_sq - 7.85*x_cu
#plt.plot(y)
#
#y = .5*x + .85*x_cu
#plt.plot(y)
#
#x_sig = x/(np.square(1+x**2))
#plt.plot(x_sig)
#
#y = .5*(x/(np.square(1+x**2)))
#plt.plot(y)
#
#plt.scatter(x_sig,x)
#----------------

# skip:
# truncated ivs -- ivs w p < .05
#all_iv_vars = [ 
#       'difference_team_ASTpct_ewma_15', 
#       'difference_team_FTr_ewma_15', 'difference_team_pf_ewma_15', 
#       'difference_opponent_ASTpct_ewma_15', 'difference_opponent_FTr_ewma_15', 
#       'difference_opponent_pf_ewma_15', 'totals', 'spread']  #  using

# Interesting: the diff between my predicted pt diff and the spread is very low -- half what it was w 
# all the vars above included. because spread has a more dominant role in the prediction. 
# likely as a consequence, the lowess prediction curve between diff of my pt diff and spread and
# being correct is worse and more non-sensical, i.e., doesn't really have a pos slope
# so seems the key is having the spread in the model, but then lots of other vars to alter the pt diff
# prediction so it's the right amount of diff from the spread itself. 

# but, the overall means look a bit more consistent. only 2010 is crappy. so don't necessarily give up on this

# test_year = 2013
# ----------------------------
# functions for final analysis
def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, test_year, iv_variables):
    #df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < test_year) &
    #                                                                  (df_covers_bball_ref__dropna_home['season_start'] > test_year-7)]
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < test_year) &
                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
    # ADDED THIS TO TRAINING ON ONLY GAMES AFTER # 15. SEE IF HELPS
    #df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['game']>10]                                                                  
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    print ('training n:', len(df_covers_bball_ref_home_train))    
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] == test_year]
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.sort_values(by=['team','date'])
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.reset_index(drop=True)
    print ('test n:', len(df_covers_bball_ref_home_test))

    # to test whether can train model on opening lines but predict w closing line
    df_covers_bball_ref_home_test['totals'] = df_covers_bball_ref_home_test['Closing Total']
    df_covers_bball_ref_home_test['spread'] = df_covers_bball_ref_home_test['team_close_spread']

# add stuff here to standardize:
# THIS IS MESSING IT UP!!! IT'S SAYING I'M HITTING AT 69%. CAN'T BE RIGHT. FIGURE OUT WHAT DOING WRONG.
#var = iv_variables[10]
#    for var in iv_variables:  
#        scalar = StandardScaler().fit(df_covers_bball_ref_home_train[[var]].values)
#        df_covers_bball_ref_home_train[var] = scalar.transform(df_covers_bball_ref_home_train[[var]].values)
#        df_covers_bball_ref_home_test[var] = scalar.transform(df_covers_bball_ref_home_test[[var]].values)
        
    df_covers_bball_ref_home_train['spread_to_bet'] = df_covers_bball_ref_home_train ['spread'] 
    df_covers_bball_ref_home_test['spread_to_bet'] = df_covers_bball_ref_home_test ['spread'] 

    for var in iv_variables:  
        var_mean = df_covers_bball_ref_home_train[var].mean()
        var_std = df_covers_bball_ref_home_train[var].std()
        df_covers_bball_ref_home_train[var] = (df_covers_bball_ref_home_train[var] -  var_mean) / var_std    
        df_covers_bball_ref_home_test[var] = (df_covers_bball_ref_home_test[var] -  var_mean) / var_std    
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test

#df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, 2012, iv_variables)


def add_interactions(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df):
    df_covers_bball_ref_home_train['game_x_playoff_distance'] = df_covers_bball_ref_home_train['game'] * df_covers_bball_ref_home_train['difference_distance_playoffs_abs']
    df_covers_bball_ref_home_test['game_x_playoff_distance'] = df_covers_bball_ref_home_test['game'] * df_covers_bball_ref_home_test['difference_distance_playoffs_abs']
    iv_variables = iv_variables + ['game_x_playoff_distance']
    variables_for_df = variables_for_df + ['game_x_playoff_distance']
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df



#df_covers_bball_ref__dropna_home[var].std()
#df_covers_bball_ref_home_train[var].std()
#from sklearn import preprocessing
#
dftest = pd.DataFrame({'a':[1,2,3,4,5,6,7], 'b':[5.,6.,7.,3.,5.,2.,6.]})
scalar = preprocessing.StandardScaler().fit(dftest[['b']].values)
dftest['b'] = scalar.transform(dftest[['b']])
dftest['b'].mean()
dftest['b'].std()
dftest['a'] = scalar.transform(dftest[['a']])
dftest['b'].mean(axis=0)


# doing this above instead
#def standardize_ivs(df_train, df_test, iv_variables):
#    df_train = df_train.reset_index(drop=True)
#    df_test = df_test.reset_index(drop=True)
#    # standardize both dfs, using mean and stdev from the training df
#    saved_standardized_model = StandardScaler().fit(df_train[iv_variables])
#    df_standardized_train = pd.DataFrame(saved_standardized_model.transform(df_train[iv_variables]), columns=iv_variables)
#    df_standardized_test = pd.DataFrame(saved_standardized_model.transform(df_test[iv_variables]), columns=iv_variables)  # standardizing the test set using training variable means and stds 
#    # WHY DID I DO THIS BELOW? DOESN'T ABOVE TAKE CARE OF IT?
#    for i, variable in enumerate(iv_variables[:]):
#        iv_variables_z.append(variable+'_z')
#        df_train.loc[:,variable +'_z'] = StandardScaler().fit_transform(df_train[[variable]])
#        df_test.loc[:,variable +'_z'] = StandardScaler().fit_transform(df_test[[variable]])
#    iv_variables_original = iv_variables
#    iv_variables_to_analyze = iv_variables_z
##        del df_train[variable]
##        del df_test[variable] 
##    df_train = pd.concat([df_train, df_standardized_train], axis=1)
##    df_test = pd.concat([df_test, df_standardized_test], axis=1)
#    return df_train, df_test, iv_variables_to_analyze
#
#def add_interaction_terms(df_train, df_test)

#dftest1 = pd.DataFrame({'a':[1,2,3], 'b':[44,3,22], 'c':[44,55,66], 'd':['aa', 'bg', 'cc']})
#dftest2 = pd.DataFrame({'home':[0,9,8], 'away':[0,1,0]})
#df_both = pd.concat([dftest1, dftest2], axis=1)


#def create_df_weight_recent_seasons_more(df_covers_bball_ref_home_train):
#    seasons = sorted(df_covers_bball_ref_home_train['season_start'].unique())
#    df_training_weighted = pd.DataFrame()
#    for i in range(len(seasons)):
#        df_season = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start'] == seasons[i]]
#        df_season_multiple = pd.DataFrame()
#        for k in range(5+i):
#            df_season_multiple = pd.concat([df_season_multiple,df_season], ignore_index=True)
#        df_training_weighted = pd.concat([df_training_weighted, df_season_multiple], ignore_index=True)
#    return df_training_weighted

#df_covers_bball_ref_home_train_weighted = create_df_weight_recent_seasons_more(df_covers_bball_ref_home_train)

#for season in seasons:
#    print str(season) +':',
#    print len(df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start'] == season]), 'to',
#    print len(df_covers_bball_ref_home_train_weighted[df_covers_bball_ref_home_train_weighted['season_start'] == season])
#    print
#print len(df_covers_bball_ref_home_train_weighted)


#model = linear_model.Ridge(alpha=2)
#model = linear_model.LinearRegression()
#model = KNeighborsRegressor(n_neighbors=125)

#def mse_in_training_set(df_covers_bball_ref_home_train, algorithm, iv_variables):
#    model = algorithm
#    mses = cross_val_score(model, df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var], 
#                           scoring = 'mean_squared_error', cv=10)
#    print ('cross validated mse training:', -1*mses.mean())

##mse_in_training_set(df_covers_bball_ref_home_train, linear_model.LinearRegression(), iv_variables)


# Translate point diff model into ats in test season(s)
# create correct metric:
#def create_correct_metric(df):
#    df['correct'] = np.nan
#    df.loc[(df['point_diff_predicted'] > df['spread']*-1) &
#                                       (df['ats_win'] == '1'), 'correct'] = 1
#    df.loc[(df['point_diff_predicted'] < df['spread']*-1) &
#                                       (df['ats_win'] == '0'), 'correct'] = 1
#    df.loc[(df['point_diff_predicted'] > df['spread']*-1) &
#                                       (df['ats_win'] == '0'), 'correct'] = 0
#    df.loc[(df['point_diff_predicted'] < df['spread']*-1) &
#                                       (df['ats_win'] == '1'), 'correct'] = 0
#    # create var to say how much my prediction deviates from actual spread:
#    df['predicted_spread_deviation'] = np.abs(df['spread'] + df['point_diff_predicted'])
#    return df

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
    return df

# alt version that uses closing spread
#def create_correct_metric(df):
#    df['correct'] = np.nan
#    df.loc[(df['point_diff_predicted'] > df['team_close_spread']*-1) &
#                                       (df['point_difference'] > df['team_close_spread']*-1), 'correct'] = 1
#    df.loc[(df['point_diff_predicted'] < df['team_close_spread']*-1) &
#                                       (df['point_difference'] < df['team_close_spread']*-1), 'correct'] = 1
#    df.loc[(df['point_diff_predicted'] > df['team_close_spread']*-1) &
#                                       (df['point_difference'] < df['team_close_spread']*-1), 'correct'] = 0
#    df.loc[(df['point_diff_predicted'] < df['team_close_spread']*-1) &
#                                       (df['point_difference'] > df['team_close_spread']*-1), 'correct'] = 0
#    # create var to say how much my prediction deviates from actual spread:
#    df['predicted_spread_deviation'] = np.abs(df['team_close_spread'] + df['point_diff_predicted'])
#    return df
#



# fit model on data up through 2014 season
# when i use alpha > 1 and scale, seems to be doing better. presumably not overfitting.
# the predicted spread deviation looks more variable. but the n is small here, so think it should
# and actually, it does look pretty similar to the one above in the training set -- peaks about 3 at 55%. cool.


def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var):
     # use this scikit learn vs statsmodels below where do my own regularization procedure
    model = algorithm
    model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])
    df_covers_bball_ref_home_test.loc[:,'point_diff_predicted'] = predictions_test_set
    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
    # new statsmodels approach so can shrink coeff by p-values
#    model = smf.ols(formula = 'point_difference ~ ' + string_for_regression, 
#                    data=df_covers_bball_ref_home_train).fit()  
#    # change params so that they're shrunk proportional to the p-value
#    for i in range(len(model.params[:])):
#        pct_to_minimize = 1 - model.pvalues[i]/20
#        model.params[i] = model.params[i] * pct_to_minimize
#    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])
#    df_covers_bball_ref_home_test.loc[:,'point_diff_predicted'] = predictions_test_set
#    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)    
    return df_covers_bball_ref_home_test

#df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, linear_model.LinearRegression(), iv_variables, dv_var)
#df_covers_bball_ref_home_test['correct'].mean()
#df_covers_bball_ref_home_test[['correct', 'spread', 'point_diff_predicted', 'point_difference']]






#print 'mean accuracy:', np.round(df_covers_bball_ref_home_test['correct'].mean(), 3)*100, 'percent'
#df_covers_bball_ref_home_test['predicted_spread_deviation'].hist(alpha=.8, bins=15);

#sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_covers_bball_ref_home_test, lowess=True)
#plt.ylim(.4, .7)
#plt.xlim(0,4)


#import inspect
#from sklearn.utils.testing import all_estimators
#for name, clf in all_estimators(type_filter='regressor'):
#    if 'sample_weight' in inspect.getargspec(clf().fit)[0]:
#       print(name)


# ----------------------
# add remove variables:
#iv_variables.remove('spread')
#iv_variables = iv_variables + ['spread']


# remove all vars with a p-value of more than some amount?
# use only the training set(s) to make this determination
#string_for_regression = ''
#for var in iv_variables:
#    print(var)
#    string_for_regression += ' + ' + var
#string_for_regression = string_for_regression[3:]        
#model_plot = smf.ols(formula = 'point_difference ~ ' + string_for_regression,
#                data=df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start']<2013]).fit()  
#print(model_plot.summary()) 
#
#print(model_plot.params)
#print(round(model_plot.pvalues,3))
#df_vars_pvals = pd.DataFrame(model_plot.pvalues, columns=['pvalue'])
#df_vars_pvals = df_vars_pvals.reset_index()
#df_vars_pvals['pvalue'] = round(df_vars_pvals.loc[:,'pvalue'],3)
#df_vars_pvals = df_vars_pvals[df_vars_pvals['pvalue']<.5]
#iv_variables = list(df_vars_pvals['index'].values[1:])
#
#
## remove vars using regilariz?
## use only the training set(s) to make this determination
#model = linear_model.Ridge(alpha=100000000)  # small positive alphas regularize more
#model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
#df_regularization = pd.DataFrame({'ivs':iv_variables, 'coefs':model.coef_})
#df_regularization
#


# ----------------------------------------------------------------------------
# choose seasons i want to anys, and the model/algo i want to use

# general approach idea: run regular regression and get predictions and run
# 4-5 adaboost regressions and avg those, and then avg the regular regression
# point diff and the adaboost avg pt diffs. and then make deicsions. do these
# corr better with actually beating spread? if not, could only bet on those 
# games that both regular and boosted agree? that should weed out a few games and
# hopefully improve odds a little.

# compute cross validated accuracy
seasons = [2010, 2011, 2012, 2013, 2014, 2015]
#seasons = [2010, 2012, 2013, 2014, 2015]  # omit 2011
seasons = [2015]
seasons = [2012, 2013, 2014, 2015]
seasons = [2010, 2011, 2012, 2013]
seasons = [2013, 2014, 2015]

model = linear_model.Ridge(alpha=.01)  # higher number regularize more

model = linear_model.LinearRegression()
#algorithm = linear_model.LinearRegression()

model = KNeighborsRegressor(n_neighbors=20, weights='distance')
model = RandomForestRegressor(n_estimators=200, max_features='auto')  # , min_samples_leaf = 10, min_samples_split = 50)
#model = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=.01, subsample=.7) # this sucked
#model = tree.DecisionTreeRegressor(min_samples_split = 50)
#model = ExtraTreesRegressor(n_estimators=500)  

# holy shit, this adaboost ups the 2015 accuracy to 55%. but took about 1/2 hr to run w 200 estimators
# I think shrinking the learning rate below 1 helps. need bigger n_estimators, though?
# using loss='square' may also be better than 'linear?'

# used this below in conjuction with regular linear model. and bet the games they both agree. 
# except it's variable and give different answers
# really slow. if use, do something else while it's working
model = AdaBoostRegressor(tree.DecisionTreeRegressor(), n_estimators=500)  # this decision tree regressor is the default
# this is predicting well for 2015 and maybe 2014. but not predicting for all. why?


# but the i did it again w 200 estimators and only got 50%?!
# nearest neighbors didn't seem good for adaboost:
#model = AdaBoostRegressor(KNeighborsRegressor(n_neighbors=500), n_estimators=100)  

# this sometimes worked well
# TRY THIS W REGULAR REGRESSION:
model = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, learning_rate=.001)  #, loss='exponential')  # this decision tree regressor is the default

model = AdaBoostRegressor(linear_model.Ridge(alpha=10), n_estimators=100, learning_rate=.001)  #, loss='exponential')  # 

model = AdaBoostRegressor(svm.LinearSVR(), n_estimators=100, learning_rate=.001)  #, loss='exponential')  # this decision tree regressor is the default

#model = AdaBoostRegressor(linear_model.Ridge(alpha=.01), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
#model = AdaBoostRegressor(linear_model.KernelRidge(alpha=.01), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
#model = AdaBoostRegressor(AdaBoostRegressor(), n_estimators=50, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default

# might try this in conjuction with regular regression
# but needs a lot of tweaking. it's predicting favored teams well and underdog teams poorly
# so something is up, presumably, and maybe tuning will help it
model = svm.SVR(C=1000, gamma=.00001)  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=2, gamma=.005)  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=2, gamma=.005, kernel='sigmoid')  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=50, gamma=.0001) 
model = svm.SVR(C=200, gamma=.0001) 
# cache_size doesn't make a diff for accuracy. leave as default
#model = svm.SVR(C=10, kernel='poly', degree=3)  # takes loonger and doesn't work as well. but considers interactions so might want to try more, esp after pca?
#model = svm.SVR(C=.1, kernel='sigmoid')  # takes loonger and doesn't work as well. but considers interactions so might want to try more, esp after pca?
#model = svm.SVR(C=10, kernel='poly', gamma=0)  # takes loonger and doesn't work as well. but considers interactions so might want to try more, esp after pca?


# TRY THIS W REGULAE REGRESSION
model = svm.LinearSVR()  #  
model = svm.LinearSVR(C=10)  #  
model = svm.LinearSVR(C=.05)  #  best option for C. but how diff is this from regular regression?
model = svm.LinearSVR(C=.1)  #  best option for C
model = svm.LinearSVR(C=.01)  

# these wouldn't finish. had to re-boot:
#model = KernelRidge(alpha=.01, gamma=.5, kernel='sigmoid')  # kernel='poly'. sigmoid and poly are making worse. but can't get to work with scaled ivs
#model = KernelRidge(alpha=.01, gamma=.5)

#The parameter learning_rate strongly interacts with the parameter n_estimators, the number 
#of weak learners to fit. Smaller values of learning_rate require larger numbers of weak 
#learners to maintain a constant training error. Empirical evidence suggests that small 
#values of learning_rate favor better test error. [HTF2009] recommend to set the learning 
#rate to a small constant (e.g. learning_rate <= 0.1) and choose n_estimators by early 
#stopping. For a more detailed discussion of the interaction between learning_rate and 
#n_estimators see [R2007].


def analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, algorithm, iv_variables, dv_var, variables_for_df):
    iv_variabless_pre_x = iv_variables
    accuracy_list = []
    mean_sq_error_list = []
    df_test_seasons = pd.DataFrame()
    #df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start']!=2011]
    for season in seasons:
        print(season)
        df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, season, iv_variables)
        df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df = add_interactions(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables, variables_for_df)
        #df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start'] != 2011]  #shortened season. helps a bit.
        #df_covers_bball_ref_home_train = create_df_weight_recent_seasons_more(df_covers_bball_ref_home_train)
        # AT MOMENT, THIS TOTALLY RUNIS IT:        
        #df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables_to_analyze = standardize_ivs(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables)
        #df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables_to_analyze)
        df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables, dv_var)

        df_covers_bball_ref_home_test['absolute_error'] = np.abs(df_covers_bball_ref_home_test['point_diff_predicted'] - df_covers_bball_ref_home_test['point_difference'])
        df_test_seasons = pd.concat([df_test_seasons, df_covers_bball_ref_home_test])
        accuracy_list.append((season, np.round(df_covers_bball_ref_home_test['correct'].mean(), 4)*100))
        mean_sq_error_list.append((season, np.round(df_covers_bball_ref_home_test['absolute_error'].mean(), 2)))
        #print season, 'mean accuracy:', np.round(df_covers_bball_ref_home_test['correct'].mean(), 3)*100, 'percent'
        #print sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_covers_bball_ref_home_test, lowess=True)
        iv_variables = iv_variabless_pre_x
        print()
    return accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train


df_test_seasons[['point_diff_predicted', 'spread', 'spread_to_bet', 'point_difference']].head(10)
sns.lmplot(x='point_diff_predicted', y='point_difference', data=df_test_seasons, scatter_kws={'alpha':.05})
plt.axhline(0, linestyle='--', alpha=.2, color='red')
plt.axvline(.5, linestyle='--', alpha=.2, color='red')

#accuracy_list, df_test_seasons = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, linear_model.Ridge(alpha=.01))
#accuracy_list, df_test_seasons = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, linear_model.Lasso(alpha=.01))

# USE THIS IF SCALING IN FUNCITON ABOVE
#accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model, iv_variables)
# USE THIS IF SCALED WAY ABOVE

accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model, iv_variables, dv_var, variables_for_df)

df = pd.DataFrame(accuracy_list, columns=['season', 'accuracy'])
df_mse = pd.DataFrame(mean_sq_error_list, columns=['season', 'error'])
df_mse['accuracy'] = df['accuracy']
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


# use this to get deviation without abs val so can get better stdev and range of deviations of predicted from spread
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
    df['predicted_spread_deviation_raw'] = df['spread_to_bet'] + df['point_diff_predicted']
    df['predicted_spread_deviation'] = np.abs(df['spread_to_bet'] + df['point_diff_predicted'])
    return df


df_test_seasons_1 = df_test_seasons['predicted_spread_deviation_raw']
df_test_seasons_2 = df_test_seasons['predicted_spread_deviation_raw']
df_test_seasons_3 = df_test_seasons['predicted_spread_deviation_raw']
df_test_seasons_4 = df_test_seasons['predicted_spread_deviation_raw']
df_test_seasons_5 = df_test_seasons['predicted_spread_deviation_raw']
df_test_seasons_6 = df_test_seasons['predicted_spread_deviation_raw']
df_test_seasons_7 = df_test_seasons['predicted_spread_deviation_raw']

df_test_seasons['pred_dev_1'] = df_test_seasons_1
df_test_seasons['pred_dev_2'] = df_test_seasons_2
df_test_seasons['pred_dev_3'] = df_test_seasons_3
df_test_seasons['pred_dev_4'] = df_test_seasons_4
df_test_seasons['pred_dev_5'] = df_test_seasons_5
df_test_seasons['pred_dev_6'] = df_test_seasons_6
df_test_seasons['pred_dev_7'] = df_test_seasons_7

df_test_seasons[['pred_dev_1', 'pred_dev_2', 'pred_dev_3', 'pred_dev_4', 'pred_dev_5', 
                 'pred_dev_6', 'pred_dev_7']].corr()


df_test_seasons['stdev_of_model_deviations'] = df_test_seasons[['pred_dev_1', 'pred_dev_2', 'pred_dev_3', 'pred_dev_4', 'pred_dev_5', 
                 'pred_dev_6', 'pred_dev_7']].std(axis=1)

df_test_seasons['range_of_model_deviations'] = df_test_seasons[['pred_dev_1', 'pred_dev_2', 'pred_dev_3', 'pred_dev_4', 'pred_dev_5', 
                 'pred_dev_6', 'pred_dev_7']].max(axis=1) - df_test_seasons[['pred_dev_1', 'pred_dev_2', 'pred_dev_3', 'pred_dev_4', 'pred_dev_5', 
                 'pred_dev_6', 'pred_dev_7']].min(axis=1)

df_test_seasons['mean_of_model_deviations'] = df_test_seasons[['pred_dev_1', 'pred_dev_2', 'pred_dev_3', 'pred_dev_4', 'pred_dev_5', 
                 'pred_dev_6', 'pred_dev_7']].mean(axis=1)

df_test_seasons['stdev_of_model_deviations'].hist(bins=15)
df_test_seasons['range_of_model_deviations'].hist(bins=15)
df_test_seasons['mean_of_model_deviations'].hist(bins=15)


plt.scatter(df_test_seasons['mean_of_model_deviations'], df_test_seasons['predicted_spread_deviation'], alpha=.1)
plt.scatter(df_test_seasons['range_of_model_deviations'], df_test_seasons['stdev_of_model_deviations'], alpha=.1)


sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.5, 'color':'blue'})
plt.ylim(.45, .6)
plt.xlim(0,4)
plt.ylabel('percent correct', fontsize=15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)


sns.lmplot(x='mean_of_model_deviations', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.5, 'color':'blue'})
plt.ylim(.45, .6)
plt.xlim(-4.5,4)
plt.ylabel('percent correct', fontsize=15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)


sns.lmplot(x='stdev_of_model_deviations', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.5, 'color':'blue'})
plt.ylim(.45, .6)
plt.xlim(0,1.25)
plt.ylabel('percent correct', fontsize=15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
# when predictions of diff models deviate a lot, they tend to fail.
# combo deviation and stdev?


sns.lmplot(x='range_of_model_deviations', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.5, 'color':'blue'})
plt.ylim(.45, .6)
plt.xlim(0,1.25)
plt.ylabel('percent correct', fontsize=15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)


#sns.lmplot(x='stdev_of_model_deviations', y='correct', data=df_test_seasons, x_bins=15, line_kws={'alpha':.5, 'color':'blue'})
#plt.ylim(.45, .6)
#plt.xlim(0,1)
#plt.ylabel('percent correct', fontsize=15)
#plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)


sns.lmplot(x='stdev_of_model_deviations', y='correct', data=df_test_seasons[df_test_seasons['season_start']>2011], hue='season_start', lowess=True, line_kws={'alpha':.55})
plt.ylim(.45, .6)
plt.xlim(0,1.2)
plt.ylabel('percent correct', fontsize=15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)


bin_number = 30
n, bins, patches = plt.hist(df_test_seasons['stdev_of_model_deviations'], bin_number, color='white')
sns.lmplot(x='stdev_of_model_deviations', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.35, 'color':'blue'})
plt.ylim(.46, .6)
plt.xlim(.025,1.25)
plt.ylabel('percent correct', fontsize=15)
#plt.grid(axis='y', linestyle='--', alpha=.75)
max_n = n.max()
plt.title('accuracy as a function of std dev of predictions')
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.xlabel('std dev of deviation that diff model predictions \n deviate from the spread', fontsize=15)
for i in range(bin_number):
    a = 1 - n[i]/max_n
    a = a*.825    
    print(a)
    plt.bar(bins[:-1][i], n[i], width=bins[1]-bins[0], alpha=a, color='b', linewidth=0)


# look at interaction between std dev of deviations and the mean deviation

df_test_seasons.predicted_spread_deviation.describe() #will give 25, 50, and 75 percentiles. Use those numbers to bin
bins = np.array([.533, .9026, 1.481]) #these three numbers are the 25 th , 50 th , and 75 th percentiles.
binned = np.digitize(df_test_seasons.mean_of_model_deviations, bins)
df_test_seasons['mean_deviation_binned'] = binned #create new

sns.lmplot(x='range_of_model_deviations', y='correct', data=df_test_seasons, hue='mean_deviation_binned', lowess=True, line_kws={'alpha':.5})
plt.xlim(0,1.25)
plt.ylim(.45,.65)

sns.lmplot(x='stdev_of_model_deviations', y='correct', data=df_test_seasons, hue='mean_deviation_binned', lowess=True, line_kws={'alpha':.5})
plt.xlim(0,1.25)
plt.ylim(.4,.65)


results = smf.ols(formula = 'correct ~ stdev_of_model_deviations * np.abs(mean_of_model_deviations)', data=df_test_seasons).fit()
print(results.summary())  # 



#------------------------
# try bootstrap to get ci
# take 20% of training and and predict and get accurcy and std of dev. then do again 
# and again, till get distribution of accuracies.

def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, test_year, iv_variables):
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < test_year) &
                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]  # was > 2004. maybe go back to that. (test_year-9)
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    
    indices_to_keep = np.random.choice(df_covers_bball_ref_home_train.index.values, 1000, replace=False)

    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train.index.isin(indices_to_keep)]
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.sort_values(by=['team','date'])
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.reset_index(drop=True)
    
    print ('training n:', len(df_covers_bball_ref_home_train))    
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] == test_year]
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.sort_values(by=['team','date'])
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.reset_index(drop=True)
    print ('test n:', len(df_covers_bball_ref_home_test))

    # to test whether can train model on opening lines but predict w closing line
    df_covers_bball_ref_home_test['totals'] = df_covers_bball_ref_home_test['Closing Total']
    df_covers_bball_ref_home_test['spread'] = df_covers_bball_ref_home_test['team_close_spread']

    df_covers_bball_ref_home_train['spread_to_bet'] = df_covers_bball_ref_home_train ['spread'] 
    df_covers_bball_ref_home_test['spread_to_bet'] = df_covers_bball_ref_home_test ['spread'] 

    for var in iv_variables:  
        var_mean = df_covers_bball_ref_home_train[var].mean()
        var_std = df_covers_bball_ref_home_train[var].std()
        df_covers_bball_ref_home_train[var] = (df_covers_bball_ref_home_train[var] -  var_mean) / var_std    
        df_covers_bball_ref_home_test[var] = (df_covers_bball_ref_home_test[var] -  var_mean) / var_std    
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test



df_test_seasons_base = df_test_seasons.copy(deep=True)
df_test_seasons_base['correct'].mean()

seasons = [2015]
df_bootstrap_predicted = pd.DataFrame()
df_bootstrap_correct = pd.DataFrame()
for i in range(100):
    print(i)
    accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model, iv_variables, dv_var, variables_for_df)
    prediction_spread_deviation = df_test_seasons['point_diff_predicted']
    #correct = df_test_seasons['correct'] 
    df_bootstrap_predicted[i] = prediction_spread_deviation 
    df_bootstrap_correct[i] = correct 
    
    
df_bootstrap_predicted['predicted_stdev'] = df_bootstrap_predicted.std(axis=1)
df_bootstrap_predicted['tenth_percentile'] = df_bootstrap_predicted.quantile(q=.1, axis=1)
df_bootstrap_predicted['ninetieth_percentile'] = df_bootstrap_predicted.quantile(q=.9, axis=1)
df_bootstrap_predicted['twenty_percentile'] = df_bootstrap_predicted.quantile(q=.2, axis=1)
df_bootstrap_predicted['eighty_percentile'] = df_bootstrap_predicted.quantile(q=.8, axis=1)
df_bootstrap_predicted['forty_percentile'] = df_bootstrap_predicted.quantile(q=.4, axis=1)
df_bootstrap_predicted['sixy_percentile'] = df_bootstrap_predicted.quantile(q=.6, axis=1)
df_bootstrap_predicted['fifty_percentile'] = df_bootstrap_predicted.quantile(q=.5, axis=1)
df_bootstrap_predicted['thirty_percentile'] = df_bootstrap_predicted.quantile(q=.3, axis=1)
df_bootstrap_predicted['seventy_percentile'] = df_bootstrap_predicted.quantile(q=.7, axis=1)



df_test_seasons_base['predicted_stdev'] = df_bootstrap_predicted['predicted_stdev']
df_test_seasons_base['tenth_percentile'] = df_bootstrap_predicted['tenth_percentile']
df_test_seasons_base['ninetieth_percentile'] = df_bootstrap_predicted['ninetieth_percentile']
df_test_seasons_base['twenty_percentile'] = df_bootstrap_predicted['twenty_percentile']
df_test_seasons_base['eighty_percentile'] = df_bootstrap_predicted['eighty_percentile']
df_test_seasons_base['forty_percentile'] = df_bootstrap_predicted['forty_percentile']
df_test_seasons_base['sixy_percentile'] = df_bootstrap_predicted['sixy_percentile']
df_test_seasons_base['fifty_percentile'] = df_bootstrap_predicted['fifty_percentile']
df_test_seasons_base['thirty_percentile'] = df_bootstrap_predicted['thirty_percentile']
df_test_seasons_base['seventy_percentile'] = df_bootstrap_predicted['seventy_percentile']

df_test_seasons_base['spread_rev'] = -1*df_test_seasons_base['spread_to_bet']
df_test_seasons_base['ci'] = df_test_seasons_base['sixy_percentile'] - df_test_seasons_base['forty_percentile'] 
df_test_seasons_base[['forty_percentile', 'sixy_percentile', 'spread_rev', 'ci']]

df_test_seasons_base['ci'].hist(bins=20)

sns.lmplot(x='ci', y='correct', data=df_test_seasons_base, lowess=True, scatter_kws={'alpha':.01})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35,.65)
plt.xlim(.5,2.5)
sns.despine()

df_test_seasons_base[df_test_seasons_base['ci']<2]['correct'].mean()
# don't bet on games with ci that's super high

# what about getinng pct of time that shoud be in one direction?
# get distrib of preiction and where does spread lie?
df_bootstrap_predicted['spread_to_bet'] = df_test_seasons_base['spread_to_bet']
df_bootstrap_predicted['spread_rev'] = -1*df_bootstrap_predicted['spread_to_bet']


df_bootstrap_predicted['spread_falls'] = np.nan
from scipy import stats
stats.percentileofscore([1, 2, 3, 4], 3)
matrix = df_bootstrap_predicted.values   
len(matrix) 
for i in range(len(matrix[:])):
    print(i)
    array = matrix[i][:100]
    spread =  df_bootstrap_predicted[df_bootstrap_predicted.index==i]['spread_rev']   
    ranking = stats.percentileofscore([a for a in array], spread.values[0])
    df_bootstrap_predicted.loc[df_bootstrap_predicted.index==i, 'spread_falls'] = ranking

df_bootstrap_predicted['spread_falls']
df_bootstrap_predicted['spread_falls_from_mid'] = np.abs(50 - df_bootstrap_predicted['spread_falls']) + 50
df_bootstrap_predicted[['spread_falls', 'spread_falls_from_mid']] 
df_bootstrap_predicted['correct'] = df_test_seasons_base['correct']

sns.lmplot(x='spread_falls_from_mid', y='correct', data=df_bootstrap_predicted, lowess=True)
plt.ylim(.4,.65)
plt.xlim(48,95)
plt.axhline(.515, alpha=.2, linestyle='--')
# is this the nicest lookin curve yet. i think. see what looks like in other years
# but this approach seems to make sense

df_bootstrap_predicted['spread_falls_from_mid'].hist(bins=15)

df_bootstrap_predicted[df_bootstrap_predicted['spread_falls_from_mid']>51]['correct'].mean()
# even making this above 1 (or 5), it makes a big leap





df_test_seasons_base['bet'] = 1
df_test_seasons_base.loc[(df_test_seasons_base['spread_rev']<df_test_seasons_base['eighty_percentile']) & (df_test_seasons_base['spread_rev']>df_test_seasons_base['twenty_percentile']), 'bet'] = 0

df_test_seasons_base['bet'] = 1
df_test_seasons_base.loc[(df_test_seasons_base['spread_rev']<df_test_seasons_base['seventy_percentile']) & (df_test_seasons_base['spread_rev']>df_test_seasons_base['thirty_percentile']), 'bet'] = 0

df_test_seasons_base['bet'].mean()
sns.barplot(x='bet', y='correct', data=df_test_seasons_base)

df_test_seasons_base[df_test_seasons_base['bet']==1]['correct'].mean()


df_test_seasons_base['deviation_from_fifty'] = np.abs(df_test_seasons_base['spread_rev'] - df_test_seasons_base['fifty_percentile'])
df_test_seasons_base['deviation_from_fifty'].hist(bins=20)

sns.lmplot(x='deviation_from_fifty', y='correct', data=df_test_seasons_base, lowess=True)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)


# need to see which way i'm voting. is my prediction jiving with the direction
# the deviation from median or ci says???

df_test_seasons_base_to_bet = df_test_seasons_base[df_test_seasons_base['bet']==1]
df_test_seasons_base_to_bet['correct_2'] = np.nan
df_test_seasons_base_to_bet.loc[(df_test_seasons_base_to_bet['spread_rev'] < df_test_seasons_base_to_bet['forty_percentile']) &
(df_test_seasons_base_to_bet['point_difference'] > df_test_seasons_base_to_bet['spread_rev']), 'correct_2'] = 1
df_test_seasons_base_to_bet.loc[(df_test_seasons_base_to_bet['spread_rev'] < df_test_seasons_base_to_bet['forty_percentile']) &
(df_test_seasons_base_to_bet['point_difference'] < df_test_seasons_base_to_bet['spread_rev']), 'correct_2'] = 0
df_test_seasons_base_to_bet.loc[(df_test_seasons_base_to_bet['spread_rev'] > df_test_seasons_base_to_bet['sixy_percentile']) &
(df_test_seasons_base_to_bet['point_difference'] < df_test_seasons_base_to_bet['spread_rev']), 'correct_2'] = 1
df_test_seasons_base_to_bet.loc[(df_test_seasons_base_to_bet['spread_rev'] > df_test_seasons_base_to_bet['sixy_percentile']) &
(df_test_seasons_base_to_bet['point_difference'] > df_test_seasons_base_to_bet['spread_rev']), 'correct_2'] = 0


len(df_test_seasons_base_to_bet[df_test_seasons_base_to_bet['correct_2'].notnull()])
df_test_seasons_base_to_bet['correct_2'].mean()
df_test_seasons_base_to_bet['correct'].mean()

df_test_seasons_base['correct'].mean()

df_test_seasons_base_to_bet[['correct', 'correct_2']].corr()  # they are the same







df_test_seasons_base['predicted_stdev'].hist(bins=20)
df_test_seasons_base['predicted_stdev'].mean()

sns.lmplot(x='predicted_stdev', y='correct', data=df_test_seasons_base, lowess=True)
plt.ylim(.45,.6)
# saying i shold bet on teams with high st dev




#--------------------------------
for season in range(len(seasons)):
    plt.plot([mean_sq_error_list[season][1], mean_sq_error_list[season][1]], label=str(seasons[season]));
plt.ylim(8.5,10.5)
plt.ylabel('error')
plt.legend()
#plt.axhline(9.0, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.axhline(9.5, linestyle='--', color='grey', linewidth=1, alpha=.5)
sns.despine() 


len(df_test_seasons['correct'][df_test_seasons['season_start']==2015])
correct_year = df_test_seasons['correct'][df_test_seasons['season_start']==2015].values





df_test_seasons['team_juice_rev'].hist(bins=10)
df_test_seasons['team_juice_rev'].min()
df_test_seasons.loc[df_test_seasons['team_juice_rev']>1.5,'team_juice_rev'] = np.nan


df_test_seasons['predict_over_spread'] = np.nan
df_test_seasons.loc[df_test_seasons['point_diff_predicted']>-1*df_test_seasons['spread'], 'predict_over_spread'] = 1
df_test_seasons.loc[df_test_seasons['point_diff_predicted']<-1*df_test_seasons['spread'], 'predict_over_spread'] = 0

# do games that i get correct give worse payout
df_test_seasons['team_juice_rev'].mean()
df_test_seasons['team_juice_rev'][df_test_seasons['correct']==1].mean()
df_test_seasons['team_juice_rev'][df_test_seasons['correct']==0].mean()

sns.barplot(x='correct', y='team_juice_rev', data=df_test_seasons, hue='predict_over_spread')
plt.ylim(.92,.97)

sns.barplot(x='correct', y='opponent_juice_rev', data=df_test_seasons, hue='predict_over_spread')
plt.ylim(.92,.97)

# do teams i bet on get worse payout?
sns.barplot(x='predict_over_spread', y='team_juice_rev', data=df_test_seasons, hue='season_start')
plt.ylim(.935,.956)
# this is saying I should probably put my return at about .95

sns.barplot(x='predict_over_spread', y='opponent_juice_rev', data=df_test_seasons, hue='season_start')
plt.ylim(.945,.969)
# when i bet on the away team, the return is getting close to .96. wow. 

#  is saying i shouldn't worry about the return/juice

df_test_seasons['team_juice_rev'].hist()
sns.lmplot(x='team_juice_rev', y='point_diff_predicted', data=df_test_seasons, lowess=True, x_partial='spread', y_partial='spread', scatter_kws={'alpha':.01})
plt.ylim(2,4)
# hmmm. this is saying that the more return (i.e., the less the juice),
# the the more the team will win by.


sns.lmplot(x='team_juice_rev', y='correct', data=df_test_seasons, lowess=True)
plt.axhline(.515, linestyle='--', color='grey')
plt.ylim(.4,.6)




df_test_seasons.to_csv('test_spred_seasons.csv')


df_test_seasons[['point_diff_predicted', 'win']]
sns.lmplot(x='point_diff_predicted', y='win', data=df_test_seasons, scatter_kws={'alpha':.05}, logistic=True)
plt.axhline(.5, linestyle='--', alpha=.2, color='red')
plt.axvline(0, linestyle='--', alpha=.2, color='red')

sns.lmplot(x='point_diff_predicted', y='win', data=df_test_seasons, scatter_kws={'alpha':.05}, lowess=True)
plt.axhline(.5, linestyle='--', alpha=.2, color='red')
plt.axvline(0, linestyle='--', alpha=.2, color='red')

sns.lmplot(x='point_diff_predicted', y='win', data=df_test_seasons, scatter_kws={'alpha':.05})
plt.axhline(.5, linestyle='--', alpha=.2, color='red')
plt.axvline(0, linestyle='--', alpha=.2, color='red')

sns.lmplot(x='spread', y='win', data=df_test_seasons, scatter_kws={'alpha':.05}, logistic=True)
plt.axhline(.5, linestyle='--', alpha=.2, color='red')
plt.axvline(0, linestyle='--', alpha=.2, color='red')

# convert point diff predicted to proba of win:
early_df = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start']<2011]
early_df.loc[:, 'spread_rev'] = early_df.loc[:, 'spread']*-1
model_logit = LogisticRegression()
model_linear = linear_model.LinearRegression()
model_logit.fit(early_df[['spread_rev']], early_df['win']) 
model_linear.fit(early_df[['spread_rev']], early_df['win']) 
proba_of_win_logit = model_logit.predict_proba(df_test_seasons[['point_diff_predicted']])
proba_of_win_logit = [proba[1] for proba in proba_of_win_logit]
proba_of_win_linear = model_linear.predict(df_test_seasons[['point_diff_predicted']])
df_test_seasons['probability_of_win_logit'] = proba_of_win_logit
df_test_seasons['probability_of_win_linear'] = proba_of_win_linear
plt.scatter(df_test_seasons['probability_of_win_logit'], df_test_seasons['probability_of_win_linear'], alpha=.1)


df_test_seasons = df_test_seasons.reset_index(drop=True)
df_test_seasons['ml_team_proba'] = np.nan
df_test_seasons.loc[df_test_seasons['team_close_ml']<0,'ml_team_proba'] = df_test_seasons['team_close_ml'] / (df_test_seasons['team_close_ml'] - 100)
df_test_seasons.loc[df_test_seasons['team_close_ml']>0,'ml_team_proba'] = 100 / (100 + df_test_seasons.loc[:, 'team_close_ml'])

df_test_seasons['ml_opponent_proba'] = np.nan
df_test_seasons.loc[df_test_seasons['opponent_close_ml']<0,'ml_opponent_proba'] = df_test_seasons['opponent_close_ml'] / (df_test_seasons['opponent_close_ml'] - 100)
df_test_seasons.loc[df_test_seasons['opponent_close_ml']>0,'ml_opponent_proba'] = 100 / (100 + df_test_seasons.loc[:, 'opponent_close_ml'])

df_test_seasons['ml_team_decimal'] = np.nan
df_test_seasons.loc[df_test_seasons['team_close_ml']<0,'ml_team_decimal'] = (100 / df_test_seasons.loc[:, 'team_close_ml'])*-1 + 1
df_test_seasons.loc[df_test_seasons['team_close_ml']>0,'ml_team_decimal'] = (df_test_seasons['team_close_ml'] / 100) + 1
df_test_seasons[['team_close_ml','ml_team_decimal']]
df_test_seasons['ml_team_decimal'] = df_test_seasons['ml_team_decimal'] - 1

df_test_seasons['ml_opponent_decimal'] = np.nan
df_test_seasons.loc[df_test_seasons['opponent_close_ml']<0,'ml_opponent_decimal'] = (100 / df_test_seasons.loc[:, 'opponent_close_ml'])*-1 + 1
df_test_seasons.loc[df_test_seasons['opponent_close_ml']>0,'ml_opponent_decimal'] = (df_test_seasons['opponent_close_ml'] / 100) + 1
df_test_seasons[['opponent_close_ml','ml_opponent_decimal']]
df_test_seasons['ml_opponent_decimal'] = df_test_seasons['ml_opponent_decimal'] - 1

df_test_seasons['predicted_proba_team_win'] = df_test_seasons['probability_of_win_logit']
df_test_seasons['predicted_proba_opponent_win'] = 1 - df_test_seasons.loc[:, 'probability_of_win_logit']
df_test_seasons[['predicted_proba_team_win', 'predicted_proba_opponent_win']]

df_test_seasons[['team_close_ml', 'opponent_close_ml', 'ml_team_proba', 'ml_opponent_proba', 'predicted_proba_team_win', 'predicted_proba_opponent_win']]
df_test_seasons.loc[:, 'ml_team_proba'] + df_test_seasons.loc[:,'ml_opponent_proba']

df_test_seasons_2015 = df_test_seasons[df_test_seasons['season_start']==2015]

df_test_seasons_2015['correct_win'] = np.nan
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']>df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']<df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==1), 'correct_win'] = 1
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']>df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']<df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==0), 'correct_win'] = 0
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']<df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']>df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==0), 'correct_win'] = 1
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']<df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']>df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==1), 'correct_win'] = 0
df_test_seasons_2015['correct_win'].mean()
len(df_test_seasons_2015[df_test_seasons_2015['correct_win'].isnull()])

df_test_seasons_2015['money'] = np.nan
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']>df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']<df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==1), 'money'] = 100 * df_test_seasons_2015['ml_team_decimal']
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']>df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']<df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==0), 'money'] = -100
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']<df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']>df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==0), 'money'] = 100 *df_test_seasons_2015['ml_opponent_decimal']
df_test_seasons_2015.loc[(df_test_seasons_2015['predicted_proba_team_win']<df_test_seasons_2015['ml_team_proba']) & (df_test_seasons_2015['predicted_proba_opponent_win']>df_test_seasons_2015['ml_opponent_proba']) & (df_test_seasons_2015['win']==1), 'money'] = -100

df_test_seasons_2015['money'].sum()
df_test_seasons_2015['money'].mean()

# for some other years, lose money with ml. hmmm.




# -------
# if useing multiple algos, run these 3 lines to save the alt one (i.e., not the linear one)
df_test_seasons_alt_model = df_test_seasons.copy(deep=True)
df_test_seasons_alt_model['point_diff_predicted_alt_model'] = df_test_seasons_alt_model['point_diff_predicted']
df_test_seasons_alt_model['correct_alt_model'] = df_test_seasons_alt_model['correct']

# -------
# once run the regular lin regression model:
df_test_seasons['point_diff_predicted_alt_model'] = df_test_seasons_alt_model['point_diff_predicted_alt_model']
df_test_seasons['correct_alt_model'] = df_test_seasons_alt_model['correct_alt_model']

df_test_seasons[['point_diff_predicted', 'point_diff_predicted_alt_model', 'spread']].tail(20)
df_test_seasons[['point_diff_predicted', 'point_diff_predicted_alt_model']].corr()
plt.scatter(df_test_seasons['point_diff_predicted'], df_test_seasons['point_diff_predicted_alt_model'], alpha=.3)
sns.lmplot(x='point_diff_predicted', y='point_diff_predicted_alt_model', data=df_test_seasons, scatter_kws={'alpha':.05})

df_test_seasons[['correct', 'correct_alt_model']].tail(20)
df_test_seasons[['correct', 'correct_alt_model']].corr()


df_test_seasons['point_diff_predicted_two_models'] = df_test_seasons[['point_diff_predicted', 'point_diff_predicted_alt_model']].mean(axis=1)
df_test_seasons[['point_diff_predicted', 'point_diff_predicted_alt_model', 'point_diff_predicted_two_models']].tail(20)
df_test_seasons['absolute_error'] = np.abs(df_test_seasons['point_diff_predicted_two_models'] - df_test_seasons['point_difference'])

def create_correct_metric_mean_prediction(df):
    df['correct_two_models'] = np.nan
    df.loc[(df['point_diff_predicted_two_models'] > df['spread']*-1) &
                                       (df['ats_win'] == '1'), 'correct_two_models'] = 1
    df.loc[(df['point_diff_predicted_two_models'] < df['spread']*-1) &
                                       (df['ats_win'] == '0'), 'correct_two_models'] = 1
    df.loc[(df['point_diff_predicted_two_models'] > df['spread']*-1) &
                                       (df['ats_win'] == '0'), 'correct_two_models'] = 0
    df.loc[(df['point_diff_predicted_two_models'] < df['spread']*-1) &
                                       (df['ats_win'] == '1'), 'correct_two_models'] = 0
    # create var to say how much my prediction deviates from actual spread:
    df['predicted_spread_deviation'] = np.abs(df['spread'] + df['point_diff_predicted_two_models'])
    return df

df_test_seasons = create_correct_metric_mean_prediction(df_test_seasons)
df_test_seasons[['correct_two_models', 'point_diff_predicted_two_models', 'spread', 'point_difference']].tail(20)
df_test_seasons[['correct', 'correct_alt_model', 'correct_two_models']].corr()


#-------------------------------------------
# CAN I MAKE THICKNESS OF LINE CORRESPOND TO SAMPLE SIZE AT THAT POINT?
# THAT WOULD BE A REALLY HELPFUL THING GENERALLY
for season in seasons[2:]:
    df_test_seasons['predicted_spread_deviation'][df_test_seasons['season_start']==season].hist(alpha=.1, color='green')
    plt.xlim(0,4)

sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_test_seasons[df_test_seasons['season_start']>2009], hue='season_start', lowess=True, line_kws={'alpha':.6})
plt.ylim(.4, .7)
plt.xlim(0,4)
plt.ylabel('percent correct', fontsize=15)
plt.grid(axis='y', linestyle='--', alpha=.15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.75)


bin_number = 20
n, bins, patches = plt.hist(df_test_seasons['predicted_spread_deviation'], bin_number, color='white')
#plt.bar(bins[:-1], n, width=bins[1]-bins[0], alpha=.1, color='white')

sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.5, 'color':'blue'})
plt.ylim(.45, .6)
plt.xlim(0,4)
plt.ylabel('percent correct', fontsize=15)
#plt.grid(axis='y', linestyle='--', alpha=.75)
max_n = n.max()
plt.title('accuracy as a function of prediction strength')
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.xlabel('degree the predicted point difference \n deviated from the spread', fontsize=15)

for i in range(bin_number):
    a = 1 - n[i]/max_n
    print(a)
    plt.bar(bins[:-1][i], n[i], width=bins[1]-bins[0], alpha=a, color='b', linewidth=0)


# i'm getting there. can i figure out how to plot a white hist so that with alpha=1
# it will obscure whatever is behind it?

df_test_seasons['correct'][df_test_seasons['season_start']==2015].mean()
df_test_seasons['correct'][df_test_seasons['season_start']==2015].count()
df_test_seasons[['correct', 'point_diff_predicted', 'point_difference', 'spread']]
sns.barplot(x='correct', y='point_diff_predicted', data=df_test_seasons)
sns.countplot(y='correct', data=df_test_seasons)


sns.barplot(x='correct', y='spread_abs_diff', data=df_test_seasons)
plt.ylim(.05,.12)
sns.despine()

sns.lmplot(x='spread_abs_diff', y='correct', data=df_test_seasons, logistic=True)



# this suggests i skip betting early in season. it's harder to beat chance here
sns.lmplot(x='game', y='absolute_error', data=df_test_seasons, lowess=True, scatter_kws={'alpha':.06})
plt.ylim(7,9)
results = smf.ols(formula = 'absolute_error ~ game', data=df_test_seasons).fit()
print(results.summary())  # p = .49

sns.lmplot(x='game', y='correct', data=df_test_seasons[df_test_seasons['season_start']>2011], hue='season_start', lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.65)

sns.lmplot(x='game', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.65)
# does this dip during trade period???

df_test_seasons['spread'].hist(alpha=.5)
plt.grid()
sns.despine()

# no real pattern
# except -- for svr looks like svr predicts when home team is favored???!!!
# ah, but for svr with diff parameters, it predicts underdog teams better.
# so think it makes sense to do a few diff svr models and combo them 
# except that maybe games in the middle are easier to model
sns.lmplot(x='spread', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)

# what does this look like if omit last two seasons? same pattern?
sns.lmplot(x='spread_to_bet', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)

sns.lmplot(x='totals', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)

#------------------------------------------------------------------------------
# look at winning % with diff selection criteria
print('betting on all games:')
[print(str(year)+':', accuracy) for year, accuracy in accuracy_list]
print (df_test_seasons.groupby('season_start')['correct'].count())

#df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .5) & (df_test_seasons['predicted_spread_deviation'] < 3)]
df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .25)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['game'] > -1.5)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .25) & (df_test_seasons['game'] > -1.5)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

# no restrictions:
df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > 0) & (df_test_seasons['game'] > 0)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .1) & (df_test_seasons['game'] > -1.5)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct_alt_model'].mean())
print()
print (df_truncated.groupby('season_start')['correct_alt_model'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .1) & (df_test_seasons['game'] > -1.5)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct_two_models'].mean())
print()
print (df_truncated.groupby('season_start')['correct_two_models'].count())

#df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .1) & (df_test_seasons['game'] > 5)]
#df_truncated = df_truncated[(df_truncated['correct'] == df_truncated['correct_alt_model'])]
#df_truncated['season_start'].unique()
#print (df_truncated.groupby('season_start')['correct'].mean())
#print()
#print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .01) & (df_test_seasons['game'] > -2)]
df_truncated['pt_diff_side_of_spread'] = np.nan
df_truncated.loc[df_truncated['point_diff_predicted']>-1*df_truncated['spread'], 'pt_diff_side_of_spread'] = 1
df_truncated.loc[df_truncated['point_diff_predicted']<-1*df_truncated['spread'], 'pt_diff_side_of_spread'] = -1
df_truncated['pt_diff_alt_side_of_spread'] = np.nan
df_truncated.loc[df_truncated['point_diff_predicted_alt_model']>-1*df_truncated['spread'], 'pt_diff_alt_side_of_spread'] = 1
df_truncated.loc[df_truncated['point_diff_predicted_alt_model']<-1*df_truncated['spread'], 'pt_diff_alt_side_of_spread'] = -1
df_truncated['models_agree'] = 0
df_truncated.loc[(df_truncated['pt_diff_side_of_spread'] == df_truncated['pt_diff_alt_side_of_spread']), 'models_agree'] = 1
df_truncated['models_agree'].mean()
df_truncated_agree = df_truncated[df_truncated['models_agree']==1]
df_truncated_disagree = df_truncated[df_truncated['models_agree']==0]
print(len(df_truncated_agree), len(df_truncated_disagree))
print (df_truncated_agree.groupby('season_start')['correct'].mean())
print (df_truncated_disagree.groupby('season_start')['correct'].mean())

print()
print (df_truncated_agree.groupby('season_start')['correct'].count())
df_truncated[['point_diff_predicted', 'point_diff_predicted_alt_model', 'spread', 'pt_diff_side_of_spread', 'models_agree']].tail(20)

# compute var to say whether the two models agree or not
df_test_seasons
df_test_seasons['pt_diff_side_of_spread'] = np.nan
df_test_seasons.loc[df_test_seasons['point_diff_predicted']>-1*df_test_seasons['spread'], 'pt_diff_side_of_spread'] = 1
df_test_seasons.loc[df_test_seasons['point_diff_predicted']<-1*df_test_seasons['spread'], 'pt_diff_side_of_spread'] = -1
df_test_seasons['pt_diff_alt_side_of_spread'] = np.nan
df_test_seasons.loc[df_test_seasons['point_diff_predicted_alt_model']>-1*df_test_seasons['spread'], 'pt_diff_alt_side_of_spread'] = 1
df_test_seasons.loc[df_test_seasons['point_diff_predicted_alt_model']<-1*df_test_seasons['spread'], 'pt_diff_alt_side_of_spread'] = -1
df_test_seasons['models_agree'] = 0
df_test_seasons.loc[(df_test_seasons['pt_diff_side_of_spread'] == df_test_seasons['pt_diff_alt_side_of_spread']), 'models_agree'] = 1
df_test_seasons['models_agree'].mean()


# interesting -- when do normal reguression with svr linear, they're in pretty
# good agreement, so leaves the n to bet in high. so even though it doesn't result
# in a better win % than using normal svr, it makes more because leaves more games
# to bet on. ALTER CODE SO CAN INCORPORATE 3-4 ALGOS INSTEAD OF JUST TWO.


#----------------------------------------------------------
# plot winnings

df_season = df_test_seasons[(df_test_seasons['season_start']>2014) & (df_test_seasons['season_start']<2016)]
df_season = df_season.reset_index(drop=True)
len(df_season)

cumulative_money_list = []
df_season_truncated = df_season[(df_season['game'] > -500)]
df_season_truncated = df_season_truncated[df_season_truncated['predicted_spread_deviation'] > .0001]
len(df_season_truncated)
# un-comment below if using multiple algos
df_season_truncated = df_season_truncated[(df_season_truncated['models_agree'] == 1)]
len(df_season_truncated)

df_season_truncated = df_season_truncated.reset_index(drop=True)
for i in range(len(df_season_truncated)):
    outcome = df_season_truncated[['correct']][df_season_truncated.index == i].values
    cumulative_money_list.append(outcome)

cumulative_money_list = [o[0][0] for o in cumulative_money_list]
len(cumulative_money_list)
df_wins = pd.DataFrame(cumulative_money_list)
df_wins = df_wins.reset_index()
df_wins.rename(columns={0:'wins'}, inplace=True)
len(df_wins[df_wins['wins'].isnull()])
actual_win_pct = str(round(df_wins['wins'].mean()*100, 3))


# i like using the kelly formula here -- it mitigates the troughs? less fluctuation

# regular appraoch:
win_probability = .52 # put pct I think we can realistically win at.
# kelly formula says that if i can put the actual pct, i'll maximize the winnings
# but the toal pot gets more and more volatilse the higher it goes, i.e., betting more and more
# so makes some sense to put a more conservative estimate, under what I think we'll get
kelly_criteria = (win_probability * .94 - (1 - win_probability)) / .94
money = 10000
bet = money * kelly_criteria
total_pot = 0
total_winnings_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot += bet*.94
        total_winnings_list.append(total_pot)
        money += bet*.94
        bet = bet
    if game == 0:
        total_pot += -1*bet
        total_winnings_list.append(total_pot)
        money += -1*bet
        bet = bet

# kelly approach:
win_probability = .52
kelly_criteria = (win_probability * .94 - (1 - win_probability)) / .94
money_kelly = 10000
bet_kelly = money_kelly * kelly_criteria
total_pot_kelly = 0
total_winnings_kelly_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot_kelly += bet_kelly*.94
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += bet_kelly*.94
        bet_kelly = money_kelly * kelly_criteria
    if game == 0:
        total_pot_kelly += -1*bet_kelly
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += -1*bet_kelly
        bet_kelly = money_kelly * kelly_criteria


# plot winnings
plt.plot(total_winnings_list, alpha=.4, color='purple', linewidth=2)
plt.plot(total_winnings_kelly_list, alpha=.4, color='green', linewidth=2)
plt.xlabel('\n games', fontsize=15)
plt.ylabel('winnings', fontsize=15)
plt.xlim(0,len(total_winnings_kelly_list)+100)
#plt.xlim(0,500)
plt.ylim(min(total_winnings_kelly_list)-5000,max(total_winnings_kelly_list)+5000)
#plt.ylim(min(total_winnings_kelly_list)-5000,70000)
plt.axhline(.5, linestyle='--', color='black', linewidth=1, alpha=.5)
plt.grid(axis='y', alpha=.2)
plt.title('win percentage: '+actual_win_pct+'\n\n' + 'winnings regular: $' + str(int(total_winnings_list[-1]))+ '\n winnings kelly: $' + str(int(total_winnings_kelly_list[-1])), fontsize=15)
#plt.title('total pot: $' + str(int(money)))
sns.despine()



# seems approach may be to start with a win_probability that'll get the pot
# up fairly quickly while still being conservative. and then will have to 
# shrink the win_probability, i.e., to shrink the percentage of the pot we're 
# gambing each game. Because it'll get way to high too quickly. betting more
# than 8,000 at pinnacle is harder. or at leaset you have to make two bets 
# and then the odds change a bit. BUT if we can bet totals too, then we can 
# increase the number of times we're betting, and so still stay under 8,000
# per bet while making a lot.


#df_test_seasons.to_excel('metis_presentation_data.xlsx')

#------------------------------------------------------------------------------
# with all vars in, what's signif?
string_for_regression = ''
for var in iv_variables:
    print(var)
    string_for_regression += ' + ' + var

string_for_regression = string_for_regression[3:]    
    
model_plot = smf.ols(formula = 'point_difference ~ ' + string_for_regression,
                data=df_covers_bball_ref_home_train).fit()  
print(model_plot.summary()) 

# what happens if i take all vars out of model with ps > .5?

# biggest effect is:
# difference_current_spread_vs_spread_ewma         
# what does this mean? can i capitalize more on this somehow? expand on this?
# it's saying that if a teams has been favored by a lot in the past, but they're
# favored by even more this game, then the point diff will be prett big.
# but I think i meant do do something diff here: to compute the diff between
# past spreads between team and oppt. and then compare this to the spread. right?


for var in iv_variables:
    print(var)


# do grid search for svr,  to see what C and gamma combo are best
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from matplotlib.colors import Normalize

df_covers_bball_ref_home_train['season_start'].unique()
df_covers_bball_ref_home_train_early_years = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start']<2013]

X = df_covers_bball_ref_home_train_early_years[iv_variables].values
y = df_covers_bball_ref_home_train_early_years['point_difference'].values

C_range = np.logspace(-3, 3, 10)
gamma_range = np.logspace(-5, 5, 10)
param_grid = dict(gamma=gamma_range, C=C_range)

grid = GridSearchCV(svm.SVR(), param_grid=param_grid, cv=3)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

grid.scorer_
grid.best_score_  # .218 (i assume this is r sq.)
grid.best_params_  # {'C': 1000.0, 'gamma': 1.0000000000000001e-05}
grid.best_estimator_
# SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
# gamma=1.0000000000000001e-05, kernel='rbf', max_iter=-1, shrinking=True,
# tol=0.001, verbose=False)

grid_scores_data = grid.grid_scores_

df_grid_search = pd.DataFrame(grid_scores_data)
df_grid_search.head()
df_grid_search['C'] = df_grid_search.loc[:, 'parameters']

c_and_gamma = df_grid_search['C'].values

c_list = []
gamma_list = []
for params in c_and_gamma:
    #print(params)
    c = [round(params['C'],6)]
    gamma = [round(params['gamma'],6)]
    c_list.append(c)
    gamma_list.append(gamma)

df_grid_search.loc[:,'C'] = c_list
df_grid_search.loc[:,'gamma'] = gamma_list

df_grid_search[['C', 'gamma']].head(10)
df_grid_search.pop('parameters')
df_grid_search.pop('cv_validation_scores')

df_grid_search.sort_values(by = 'mean_validation_score').tail(15)
#    mean_validation_score            C     gamma
#51               0.177017     2.154435  0.000129
#82               0.184623   215.443469  0.001668
#43               0.188713     0.464159  0.021544
#53               0.190451     2.154435  0.021544
#70               0.191133    46.415888  0.000010
#42               0.197926     0.464159  0.001668
#72               0.204222    46.415888  0.001668
#61               0.206812    10.000000  0.000129
#80               0.210361   215.443469  0.000010
#91               0.210417  1000.000000  0.000129
#62               0.210430    10.000000  0.001668
#52               0.211031     2.154435  0.001668
#81               0.211988   215.443469  0.000129
#71               0.212319    46.415888  0.000129
#90               0.212340  1000.000000  0.000010




# ----------------------------------------------------------------------------
win_pct = .525
bet_amount = 100
win_pct*1000*bet_amount*.95 - (1-win_pct)*1000*bet_amount

print(df_season_2015[['correct']][3:4])
print(df_season_2015[['correct']][df_season_2015.index==3])

#df_season_2015[['correct']].head()


# if the lineup is diff than the game before, does that make it harder to guess?

#df_test_seasons[['date', 'starters_same_as_last_g']][df_test_seasons['team']=='San Antonio Spurs'].tail(5)



print df_test_seasons['starters_same_as_last_g'].value_counts()
print df_test_seasons['x_starters_same_as_last_g'].value_counts()



# this pattern doesn't make any sense to me. why would the gs in which the home team had the same lineup but
# the away team had a diff lineup be the one's that i predict the best??
sns.barplot(x='starters_same_as_last_g', y='correct', hue='x_starters_same_as_last_g', data=df_test_seasons)
plt.ylim(.45,.62)



# this also seem to suggest those games with a lineup that's diff than the last g are erratic.
sns.barplot(x='starters_same_as_last_g', y='correct', hue='lineup_count', data=df_test_seasons[df_test_seasons['lineup_count']<25])
#plt.ylim(.45,.60)



# next severa graphs abuot whether it's hard to predict when it's the first time that lineup has played together
# or they haven't played much. 
print df_test_seasons['lineup_count'].hist(alpha=.7, bins=20)



# don't really think the graphs below suggest thinking about this var.
df_test_seasons[['point_diff_predicted', 'spread', 'predicted_spread_deviation', 'lineup_count']].tail()

#print df_test_seasons['lineup_count'].hist(alpha=.7, bins=20)
sns.barplot(x='lineup_count', y='correct', data=df_test_seasons[df_test_seasons['lineup_count']<10])
sns.lmplot(x='lineup_count', y='correct', data=df_test_seasons, lowess=True)
plt.ylim(.45,.65)
plt.xlim(0,150)










# ##Investigate the errors


df_year = df_test_seasons[df_test_seasons['season_start']==2014]
df_year = df_year.reset_index()
print len(df_year)
print len(df_year.columns)
df_year[['correct', 'date', 'team', 'opponent', 'spread', 'point_difference', 'point_diff_predicted', 'predicted_spread_deviation']].head(10)



# could predict spread with a model and see when my predicted spread deviates a lot from the actual spread? and stay away from those?
# see if there are any outliers of a var in past 10 games. or count up the outliers in past 10 games? for that team? or in general?



team1 = df_year['team'].unique()[4]
df_team1 = df_year[df_year['team']==team1]
df_team1.loc[:,'correct_moving_avg'] = pd.rolling_mean(df_team1.loc[:,'correct'], window=20, min_periods=15)
df_team1.loc[:,'win_moving_avg'] = pd.rolling_mean(df_team1.loc[:,'win'], window=20, min_periods=15)

plt.plot(df_team1['date'], df_team1['correct_moving_avg'], 'green')
plt.plot(df_team1['date'], df_team1['win_moving_avg'], 'black')



df_year['predict_vs_spread_moving_avg'] = pd.rolling_mean(df_year.loc[:,'predicted_spread_deviation'], window=200, min_periods=3)
plt.plot(df_year['date'], df_year['predict_vs_spread_moving_avg'])
# if can trust this, looks like maybe takes a dip towards end of season?
# would indicated that my predicitons of the point diff and the spread are getting closer at the end of the season
# but doesn't seem to mean i'm getting less accurate?


df_year['correct_moving_avg'] = pd.rolling_mean(df_year.loc[:,'correct'], window=200, min_periods=3)
plt.plot(df_year['date'], df_year['correct_moving_avg'])
plt.axhline(y=.52, linestyle='--', linewidth=.5, color='red')


plt.plot(df_year['date'], df_year['predict_vs_spread_moving_avg'], label='predicted point spread vs spread')
plt.plot(df_year['date'], df_year['correct_moving_avg'], label='correct')
plt.axhline(y=.52, linestyle='--', linewidth=.5, color='red')
# if look at ea year, these two curves don't really seem related


# are there certain teams that i can't predict from 2015 or 2014?
teams = df_year['team'].unique()
for t in list(teams)[:]:
    df_team = df_year[df_year['team']==t]
    df_team = df_team.reset_index(drop=True)
    df_team['correct_moving_avg'] = pd.rolling_mean(df_team.loc[:,'correct'], window=40, min_periods=3)
    plt.plot(df_team.index, df_team['correct_moving_avg'], label=t)
    plt.legend(bbox_to_anchor=(1.4, 1.0))
plt.axhline(y=.52, linestyle='--', linewidth=.5, color='black')

# on thing see right away, teams convert towards mean as season goes on. the super discripant happen earlyin season.
# also see that there are a couple teams who i really can't predict. and there are no teams that i'm amazing at predicting
# to the same extent that i'm terrible at predicting these. 

# in 2015
# the teams i sucked at predicting early in season: nets, 76ers, timberwolves, pacers
# the teams i sucked at predicting most of the season: knicks, spurs, denver (a lot), cavs (a little)
# maybe these teams either won or lost by a ton -- e.g., lots of 20+ point victories or losses?

# in 2014
# suck at pacers, warriors, hawks, wolves (a little), spurs


teams = df_year.groupby('team')
teams['point_difference'].mean().sort_values()  
# interesting -- the teams i had trouble predicting are in the bottom or top handful
# kind of makes sense -- if they're prone to blow outs (either wins or losses, then can get weird stuff happending
# like taking out startes more, etc. also suggests that i shouldn't bet in games in which teh spread is large???


# how to count how many big wins or losses ea team had? get abs val of pt diff
teams = df_year.groupby('team')
teams['point_difference'].std().sort_values()  



df_year['point_difference'].hist(bins=15, alpha=.7)



df_year['wins_greater_20'] = 0
df_year.loc[df_year['point_difference'] > 20, 'wins_greater_20'] = 1

teams = df_year.groupby('team')
teams['wins_greater_20'].sum().sort_values()  



df_year['losses_less_15'] = 0
df_year.loc[df_year['point_difference'] < -15, 'losses_less_15'] = 1

teams = df_year.groupby('team')
teams['losses_less_15'].sum().sort_values()  



df_year['big_point_diffs'] = df_year['losses_less_15'] + df_year['wins_greater_20']
teams = df_year.groupby('team')
teams['big_point_diffs'].sum().sort_values()  



# could be the above is helpful -- but need to see if this is predictive
# i.e., if in season up to this point how many gs have they lost by more than 15 or win by more than 20
df_test_seasons['wins_greater_20'] = 0
df_test_seasons.loc[df_test_seasons['point_difference'] > 20, 'wins_greater_20'] = 1

df_test_seasons['losses_less_15'] = 0
df_test_seasons.loc[df_test_seasons['point_difference'] < -15, 'losses_less_15'] = 1

df_test_seasons['big_point_diffs'] = df_test_seasons['losses_less_15'] + df_test_seasons['wins_greater_20']

df_test_seasons['big_point_diffs_expanding'] = df_test_seasons.groupby(['season_start', 'team'])['big_point_diffs'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
df_test_seasons['point_diff_expanding'] = df_test_seasons.groupby(['season_start', 'team'])['point_difference'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))



df_test_seasons['big_point_diffs_expanding'].hist(alpha=.7, bins=20)



df_test_seasons['point_diff_expanding'].hist(alpha=.7, bins=20)



sns.lmplot(x='big_point_diffs_expanding', y='correct', data=df_test_seasons, lowess=True)
plt.ylim(.4,.6)

results = smf.logit(formula = 'correct ~ big_point_diffs_expanding', data=df_test_seasons).fit()
print results.summary()

# kind of does look like the teams who have had a lot of big point diffs in past games are harder to predict 



sns.lmplot(x='point_diff_expanding', y='correct', data=df_test_seasons, lowess=True)
plt.ylim(.4,.6)

results = smf.logit(formula = 'correct ~ point_diff_expanding + I(point_diff_expanding**2)', data=df_test_seasons).fit()
print results.summary()

# looks here like maybe the teams with the biggest and worse avg pt diffs in prior games are harder to predict or get wrong
# suggesting, along with above, anys, that ... not sure i should incorp point diff into orig model if this is just saying
# that it's hard to predict them? but maybe i make a second model to predict correct? so running two ml processes?
# or maybe i can incorporate these with interactions? such that if the team has a big prior point spread, then it reduces
# the prediction towards the mean or the spread? so taht my confidence is less??








# with all vars in, what's signif?
model_plot = smf.ols(formula = 'point_difference ~ difference_lineup_count + difference_beat_spread_rolling_mean_11 +                      difference_team_3PAr_ewma_15 + difference_team_ASTpct_ewma_15 + difference_team_BLKpct_ewma_15 +                      difference_team_DRBpct_ewma_15 + difference_team_DRtg_ewma_15 + difference_team_FTr_ewma_15 +                      difference_team_TOVpct_ewma_15 + difference_team_TRBpct_ewma_15 + difference_team_TSpct_ewma_15 +                      difference_team_eFGpct_ewma_15 + difference_team_fg3_pct_ewma_15 + difference_team_fg_pct_ewma_15 +                      difference_team_ft_pct_ewma_15 + difference_team_pf_ewma_15 + difference_opponent_3PAr_ewma_15 +                      difference_opponent_ASTpct_ewma_15 + difference_opponent_BLKpct_ewma_15 + difference_opponent_DRBpct_ewma_15 +                      difference_opponent_DRtg_ewma_15 + difference_opponent_FTr_ewma_15 + difference_opponent_ORBpct_ewma_15 +                      difference_opponent_ORtg_ewma_15 + difference_opponent_STLpct_ewma_15 + difference_opponent_TOVpct_ewma_15 +                      difference_opponent_TRBpct_ewma_15 + difference_opponent_TSpct_ewma_15 + difference_opponent_eFGpct_ewma_15 +                      difference_opponent_pf_ewma_15 + totals + spread + difference_days_rest + zone_distance +                      difference_distance_playoffs_abs + difference_starters_same_as_last_g + difference_lineup_count*difference_days_rest ', 
                data=df_covers_bball_ref_home_train).fit()  #  + spread
print(model_plot.summary()) 

# these are signif < .01:
# difference_team_ASTpct_ewma_15 - difference_team_pf_ewma_15  - 

# < .05:
# difference_opponent_ASTpct_ewma_15 - difference_opponent_FTr_ewma_15 - difference_opponent_pf_ewma_15 - spread 



#df_covers_bball_ref_home_train['difference_starters_same_as_last_g'].value_counts()
sns.barplot(x='difference_starters_same_as_last_g', y='point_difference', data=df_covers_bball_ref_home_train)



# use pca / factor anys to reduce vars?
# http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py



# see which teams i can predict the best and worst in 2015 (and 2014)
df_test_seasons_2015 = df_test_seasons[df_test_seasons['season_start']==2015]
df_test_seasons_2014 = df_test_seasons[df_test_seasons['season_start']==2014]
df_test_seasons_2013 = df_test_seasons[df_test_seasons['season_start']==2013]


def f(df_test_seasons, year):
    df_test_season_x = df_test_seasons[df_test_seasons['season_start']==year]
    df_group_teams_season_x = pd.DataFrame()
    teams = df_test_season_x['team'].unique()
    for team in teams:
        df_team = df_test_season_x[(df_test_season_x['team']==team) | (df_test_season_x['opponent']==team)]    
        df_team['the_team'] = team
        df_team = df_team[['the_team', 'correct', 'starters_same_as_last_g']]
        df_group_teams_season_x = pd.concat([df_group_teams_season_x, df_team], ignore_index=True)
    return df_group_teams_season_x

    
    
df_group_teams_season_2015 = f(df_test_seasons[df_test_seasons['game']>15], 2015)
group_teams_2015 = df_group_teams_season_2015.groupby('the_team')['correct'].mean()
group_teams_2015.sort()
group_teams_2015.plot(kind='barh', alpha=.8, colormap='Spectral')
plt.xlabel('percent_correct', fontsize=15)
plt.title('2015', fontsize=15)
plt.ylabel('')

df_group_teams_season_2014 = f(df_test_seasons[df_test_seasons['game']>15], 2014)
group_teams_2014 = df_group_teams_season_2014.groupby('the_team')['correct'].mean()
group_teams_2014.sort()
group_teams_2014.plot(kind='barh', alpha=.8, colormap='Spectral')
plt.xlabel('percent_correct', fontsize=15)
plt.title('2014', fontsize=15)
plt.ylabel('')

df_group_teams_season_2013 = f(df_test_seasons[df_test_seasons['game']>15], 2013)
group_teams_2013 = df_group_teams_season_2013.groupby('the_team')['correct'].mean()
group_teams_2013.sort()
group_teams_2013.plot(kind='barh', alpha=.8, colormap='Spectral')
plt.xlabel('percent_correct', fontsize=15)
plt.title('2013', fontsize=15)
plt.ylabel('')


# do the teams that are harder to predict have more changing lineups from night to night?
df_group_team_starters_same_as_last_2015 = df_group_teams_season_2015.groupby('the_team')['starters_same_as_last_g'].mean()
df_group_team_starters_same_as_last_2015 = df_group_team_starters_same_as_last_2015.reset_index()
group_teams_2015 = group_teams_2015.reset_index()
df_merge_2015 = pd.merge(df_group_team_starters_same_as_last_2015, group_teams_2015, on='the_team', how='inner')
plt.scatter(df_merge_2015['starters_same_as_last_g'], df_merge_2015['correct'])

df_group_team_starters_same_as_last_2014 = df_group_teams_season_2014.groupby('the_team')['starters_same_as_last_g'].mean()
df_group_team_starters_same_as_last_2014 = df_group_team_starters_same_as_last_2014.reset_index()
group_teams_2014 = group_teams_2014.reset_index()
df_merge_2014 = pd.merge(df_group_team_starters_same_as_last_2014, group_teams_2014, on='the_team', how='inner')

df_group_team_starters_same_as_last_2013 = df_group_teams_season_2013.groupby('the_team')['starters_same_as_last_g'].mean()
df_group_team_starters_same_as_last_2013 = df_group_team_starters_same_as_last_2013.reset_index()
group_teams_2013 = group_teams_2013.reset_index()
df_merge_2013 = pd.merge(df_group_team_starters_same_as_last_2013, group_teams_2013, on='the_team', how='inner')

df_multi_years = pd.concat([df_merge_2015, df_merge_2014, df_merge_2013], ignore_index=True)
plt.scatter(df_multi_years['starters_same_as_last_g'], df_multi_years['correct'])
sns.lmplot(x='starters_same_as_last_g', y='correct', data=df_multi_years)
# no relationship




# this is pretty clearly showing that i shouldn't bet anything early than game 20
# but bins for games higher than that pretty much show a nuce curve where get the best
# accuracy after about .5 of predicted_spred_deviation and then it dips later in pred
df_test_seasons.game.describe() #will give 25, 50, and 75 percentiles. Use those numbers to bin
bins = np.array([22, 42, 62]) #these three numbers are the 25 th , 50 th , and 75 th percentiles.
#this will give 4 bins, 1 below -2.8, 1 btwn -2.8 and .9, etc.
binned = np.digitize(df_test_seasons.game, bins)
df_test_seasons['game_binned'] = binned #create new variable that’s comprised of the bins
sns.lmplot(x='predicted_spread_deviation', y='correct', hue='game_binned', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.ylim(.35,.65)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)

df_test_seasons.spread.describe() #will give 25, 50, and 75 percentiles. Use those numbers to bin
bins = np.array([-8, -4, 2.5]) #these three numbers are the 25 th , 50 th , and 75 th percentiles.
binned = np.digitize(df_test_seasons.spread, bins)
df_test_seasons['spread_binned'] = binned #create new variable that’s comprised of the bins
sns.lmplot(x='predicted_spread_deviation', y='correct', hue='spread_binned', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.ylim(.35,.65)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)

df_test_seasons['spread_favored'] = np.nan
df_test_seasons.loc[df_test_seasons['spread'] < 0, 'spread_favored'] = 1
df_test_seasons.loc[df_test_seasons['spread'] >= 0, 'spread_favored'] = 0

results = smf.logit(formula = 'correct ~ predicted_spread_deviation + I(predicted_spread_deviation**2) + game_binned', data=df_test_seasons).fit()
print(results.summary())  # p = .49


# knd of suggesting that i guess correctly when the spread is in the middle,
# between say, -15 and 15. though only for games in the last 3/4 of the season
# don't bet at all on games in first 1/4 of season or on those with spreads
# on the fringes. if i'm not betting on games with spreads on tehe fringes
# or on games in first 1/4 of season, then i probably shoudln't train model on 
# these gaes either??? 
sns.lmplot(x='spread', y='correct', hue='game_binned',  data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35,.65)

sns.lmplot(x='lineup_count', y='correct', hue='game_binned',  data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35,.65)

df_test_seasons['spread_ewma_15'].hist(alpha=.7)
plt.grid()
sns.lmplot(x='difference_spread_ewma_15', y='correct', hue='game_binned',  data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35,.65)
results = smf.logit(formula = 'correct ~ difference_spread_ewma_15', data=df_test_seasons).fit()
print(results.summary())  # p = .13

# pretty much saying that i'm better at predicting this variable around the middle. 
# fits theme of a lot of these vars
df_test_seasons['difference_beat_spread_ewma_15'].hist(alpha=.7)
plt.grid()
sns.despine()
sns.lmplot(x='difference_beat_spread_ewma_15', y='correct',  data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35,.65)


# think this is saying that the low std devs are the games i can predic tbetter
df_test_seasons['beat_spread_std_ewma_15'].hist(alpha=.7)
plt.grid()
sns.lmplot(x='beat_spread_std_ewma_15', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35,.65)
# looks like early in season, high std predict worse performance
# but in later 3/4 of season it predict better behavior. why? based on more data?
# when based on a lot of data, it shows a teams potential. when based on little data
# it shows ...?
sns.lmplot(x='beat_spread_std_ewma_15', y='point_difference', hue='game_binned',  data=df_test_seasons, scatter_kws={'alpha':.1})
#plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(-10,10)
results = smf.ols(formula = 'point_difference ~ beat_spread_std_ewma_15 * game', data=df_test_seasons).fit()
print(results.summary())  # p = .253

# when supposed to win (neg spread), high std predict performaing better
# when supposed to lose, high std predicts performing worse
sns.lmplot(x='beat_spread_std_ewma_15', y='correct', hue='spread_favored',  data=df_test_seasons, lowess=True, scatter_kws={'alpha':.3})
plt.ylim(-10,15)
results = smf.logit(formula = 'correct ~ beat_spread_std_ewma_15', data=df_test_seasons).fit()
print(results.summary())  # p = .815




# starters_the_same doesn't seem to say anything consistent
df_test_seasons.loc[:, 'starters_the_same'] = df_test_seasons[['starters_same_as_last_g', 'x_starters_same_as_last_g']].sum(axis=1)
for season in [2012,2013,2014,2015]:
    season = 2015
    sns.barplot(x='starters_the_same', y='correct', data=df_test_seasons[df_test_seasons['season_start']==season])
    plt.ylim(.4,.6)
    plt.title(str(season), fontsize=15)
    plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)

results = smf.logit(formula = 'correct ~ starters_the_same', data=df_test_seasons).fit()
print(results.summary())

results = smf.ols(formula = 'point_difference ~ game * difference_starters_same_as_last_g', data=df_test_seasons).fit()
print(results.summary())

# suggests really laying off the games at the beginning of the season IF
# both (or even one) of the teams have a diff lineup than the game before
sns.lmplot(x='game', y='correct', hue='starters_the_same', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
#plt.title('starters the same: ' + str(number) + ' year: ' + str(season))
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.35, .65)

for season in [2012,2013,2014,2015]:
    sns.lmplot(x='game', y='correct', hue='starters_the_same', data=df_test_seasons[df_test_seasons['season_start']==season], lowess=True, line_kws={'alpha':.6})
    plt.title('year: ' + str(season))
    plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
    plt.ylim(.35, .65)


# interesting that the games in which both teams have the same lineup --
# games you'd think are the most predictable, have the highest error
# and the graph of my correct guesses, just above, is consistent here too
# it does consistnely ok at guessing, but at no point is it particuarly
# good at guessing. maybe this is saying that when we know the lineups
# the spread is really good, and the additional variables in the model 
# aren't doing much to change the prediction. and so my guess is correct
# closer to 50% of the time than i'd like.
sns.lmplot(x='game', y='absolute_error', hue='starters_the_same', data=df_test_seasons, lowess=True, scatter_kws={'alpha':.04})
plt.title('starters the same: ' + str(number) + ' year: ' + str(season))
plt.ylim(5, 10)
