
# coding: utf-8

# ##To Do: 
# 

cd /Users/charlesmartens/Documents/projects/bet_bball


# technically, need to beat 51.3% to win money. though the juice seems a bit erractic, 
# so probably better to think about it at 51.5%


win_pct = .525
bet_amount = 100
win_pct*1000*bet_amount*.95 - (1-win_pct)*1000*bet_amount

# when to increase the bet? after how many games and having made how much?
# is there a point that the win% generally stabilizes for a season? 
# i.e., after 200 games? and at that point, increase the bet to 1.5% of my new pot.


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
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
#from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.learning_curve import learning_curve
import matplotlib.cm as cm
#from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.ensemble.partial_dependence import plot_partial_dependence
#from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
# try tpot
#from tpot import TPOT

sns.set_style('white')


df_covers_bball_ref = pd.read_csv('df_covers_bball_ref_2004_to_2015.csv')
df_covers_bball_ref.pop('Unnamed: 0')
df_covers_bball_ref['date'] = pd.to_datetime(df_covers_bball_ref['date'])


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

variables_for_team_metrics = ['team_3PAr', 'team_AST%', 'team_BLK%', 'team_DRB%', 'team_DRtg', 
    'team_FTr', 'team_ORB%', 'team_ORtg', 'team_STL%', 'team_TOV%', 'team_TRB%', 
    'team_TS%', 'team_eFG%', 'team_fg3_pct', 'team_fg_pct', 'team_ft_pct', 'team_pf', 
    'opponent_3PAr', 'opponent_AST%', 'opponent_BLK%', 'opponent_DRB%', 'opponent_DRtg', 'opponent_FTr', 
    'opponent_ORB%', 'opponent_ORtg', 'opponent_STL%', 'opponent_TOV%', 'opponent_TRB%', 'opponent_TS%',
    'opponent_eFG%', 'opponent_fg3_pct', 'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf']           


# Choose one of next set of cells to create metrics for ea team/lineup


# group by season and by team:

df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team', 'date'])

def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
    for var in variables_for_team_metrics:
        var_ewma = var + '_ewma_15'
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby(['season_start', 'team'])[var].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref['count'] = df_covers_bball_ref.groupby(['season_start', 'team'])['team_3PAr'].transform(lambda x: pd.expanding_count(x))



# alt - group just by team, not by season too. need to do weighted mean or a rolling mean here.

def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
    for var in variables_for_team_metrics:
        var_ewma = var + '_ewma_15'
        # i have played around with span=x in line below
        # 50 is worse than 20; 25 is worse than 20; 15 is better than 20; 10 is worse than 15; 16 is worse than 15; 14 is worse than 15
        # stick w 15. but 12 is second best. 
        # (12 looks better if want to use cutoffs and bet on fewer gs. but looks like 15 will get about same pct -- 53+ -- wih 1,000 gs)
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=15))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))
#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))
df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))



# alt - group by lineup (not team) 

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
df_covers_bball_ref[['team_AST%_ewma_15', 'team_ORtg_ewma_15', 'team_pf_ewma_15', 'team_count',
                     'team_AST%_lineups_15', 'team_ORtg_lineups_15', 'team_pf_lineups_15', 'lineup_count']].tail()



def loop_through_vars_to_create_compsite_metrics(df_covers_bball_ref, variables_for_team_metrics):
    for var in variables_for_team_metrics:
        var_ewma_orig = var + '_ewma_15'
        var_ewma_lineup = var + '_lineups_15'
        var_ewma_comp = var + '_ewma_15_comp'
        df_covers_bball_ref[var_ewma_comp] = df_covers_bball_ref[var_ewma_orig]
        df_covers_bball_ref.loc[df_covers_bball_ref['lineup_count'] > 10, var_ewma_comp] = df_covers_bball_ref[[var_ewma_orig, var_ewma_lineup]].mean(axis=1)
    return df_covers_bball_ref
        
df_covers_bball_ref = loop_through_vars_to_create_compsite_metrics(df_covers_bball_ref, variables_for_team_metrics)


df_covers_bball_ref[['team_AST%_ewma_15', 'team_ORtg_ewma_15', 'team_AST%_lineups_15', 'team_ORtg_lineups_15', 
                     'team_AST%_ewma_15_comp', 'team_ORtg_ewma_15_comp', 'lineup_count']].tail()


def loop_through_vars_to_change_names(df_covers_bball_ref, variables_for_team_metrics):
    for var in variables_for_team_metrics:
        var_ewma_orig = var + '_ewma_15'
        var_ewma_comp = var + '_ewma_15_comp'
        df_covers_bball_ref[var_ewma_orig] = df_covers_bball_ref[var_ewma_comp]
    return df_covers_bball_ref
        
df_covers_bball_ref = loop_through_vars_to_change_names(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref[['team_AST%_ewma_15', 'team_ORtg_ewma_15', 'team_AST%_lineups_15', 'team_ORtg_lineups_15', 
                     'team_AST%_ewma_15_comp', 'team_ORtg_ewma_15_comp', 'lineup_count']].tail()


# ##Continue:


# compute days rest
df_covers_bball_ref = df_covers_bball_ref.sort_values('date')
df_covers_bball_ref['date_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(1))
df_covers_bball_ref['days_rest'] = (df_covers_bball_ref['date'] - df_covers_bball_ref['date_prior_game']) / np.timedelta64(1, 'D')
df_covers_bball_ref.loc[df_covers_bball_ref['days_rest']>1, 'days_rest'] = 2
#df_covers_bball_ref['days_rest'].replace(np.nan, df_covers_bball_ref['days_rest'].mean(), inplace=True)

df_covers_bball_ref['point_difference'] = df_covers_bball_ref['score_team'] - df_covers_bball_ref['score_oppt']
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
df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone']
#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'zone_distance'].abs()
df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'zone_distance']].head(10)



#df_covers_bball_ref['zone_distance'].hist()
sns.barplot(x='zone_distance', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0])
sns.lmplot(x='zone_distance', y='point_difference', data=df_covers_bball_ref[df_covers_bball_ref['venue_x']==0], y_partial='team_ORtg_ewma_15', x_partial='team_ORtg_ewma_15')



# compute sort sort of distance from the playoffs
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

df_2015 = df_covers_bball_ref[df_covers_bball_ref['season_start']==2015]
df_2015 = df_2015[['date', 'team', 'opponent', 'team_win', 'team_win_pct', 'conference']]
df_2015.tail()



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

df_covers_bball_ref['distance_playoffs_abs'] = df_covers_bball_ref['distance_from_playoffs'].abs()

df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][500:520]
#df_w_distance_from_playoffs_all_years[df_w_distance_from_playoffs_all_years['date']=='2004-12-06']
#df_w_distance_from_playoffs_all_years['date'].dtypes


# ##Graphs Exploring Distance From Playoffs
# ##--SKIP--


df_covers_bball_ref['distance_from_playoffs'].hist(alpha=.7)
plt.grid(axis='x')
sns.despine()



# kind of cool -- think this is showing a hump for the teams who are on the cusp -- right around 0
# and there's also a little uptick in the top teams, maybe vying for home court advantage and so more motivated
sns.lmplot(x='distance_from_playoffs', y='point_difference', 
           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], 
           lowess=True, scatter_kws={'alpha':.05}, x_partial='team_win_pct_x', y_partial='team_win_pct_x')
plt.ylim(-0,10)
plt.xlim(-.4,.4)
# distance from playoffs is essentially acting as how good the team is, rather than
# how far from the playoffs the team is. abs val of distance doesn't show anything

# but adding this distance_from_playoffs var to the model makes it worse. 
# vs. adding the abs val of it -- distance_playoffs_abs -- which makes it better



# gengerally, think should stay away from the beat spread anys. unreliable and influence by who knows what
#sns.lmplot(x='distance_from_playoffs', y='beat_spread', 
#           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], 
#           lowess=True, scatter_kws={'alpha':.01})
#plt.ylim(-1,2)



sns.barplot(x='conference_rank', y='point_difference', 
            data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)]) 



df_covers_bball_ref['distance_playoffs_abs'].hist(alpha=.7)
plt.grid(axis='x')



sns.lmplot(x='distance_playoffs_abs', y='point_difference', 
           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], 
           scatter_kws={'alpha':.01}, x_partial='team_win_pct_x', 
           y_partial='team_win_pct_x')
plt.ylim(0, 7)
plt.xlim(0, .6)



sns.lmplot(x='distance_playoffs_abs', y='point_difference', 
           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], 
           lowess=True, scatter_kws={'alpha':.01}, x_partial='team_win_pct_x', 
           y_partial='team_win_pct_x', x_bins=6)
#plt.ylim(-20, 20)



model_plot = smf.ols(formula = 'point_difference ~ distance_playoffs_abs + I(distance_playoffs_abs**2) + team_win_pct_x', data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)]).fit()
print(model_plot.summary()) 
# controlling for team win pct actually makes this signif
# and quadratic is even more signif



# i tried including this quadratic but it seemed to hurt a bit. and didn't help. maybe getting too
# fancy since there are so few teams after about 3.5 on distance_playoffs_abs, the point at which the
# pattern reverses itself and see a positive curve. 

df_covers_bball_ref['distance_playoffs_abs_sq'] = df_covers_bball_ref['distance_playoffs_abs'] * df_covers_bball_ref['distance_playoffs_abs'] 
#df_covers_bball_ref[['distance_playoffs_abs', 'distance_playoffs_abs_sq']].tail()



#sns.lmplot(x='distance_playoffs_abs', y='beat_spread', 
#           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], 
#           lowess=True, scatter_kws={'alpha':.01}, x_bins=5)



#sns.lmplot(x='team_win_pct_y', y='beat_spread', 
#           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], 
#           lowess=True, scatter_kws={'alpha':.01}, x_bins=5)
#plt.xlim(.2,.8)



# next: take abs val of distance from playoffs and make it a var, so gettting the diff between team and oppt
# then see if it helps predict



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

sns.barplot(x='starters_same_as_last_g', y='point_difference', data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)])
#sns.barplot(x='starters_same_as_two_g', y='point_difference', data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)])



sns.barplot(x='starters_same_as_last_g', y='point_difference', hue='starters_same_as_two_g', data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)])
sns.lmplot(x='starters_same_as_last_g', y='point_difference',
           data=df_covers_bball_ref[(df_covers_bball_ref['venue_x']==0)], hue='starters_same_as_two_g',  
           x_partial='team_ORtg_ewma_15', y_partial='team_ORtg_ewma_15')
plt.ylim(-1,7)


# ##Continue


df_covers_bball_ref = df_covers_bball_ref[['date', 'team', 'opponent', 'venue_x', 'lineup_count', 
       'starters_team', 'starters_opponent', 'team_3PAr', 'team_AST%', 'team_BLK%', 'team_DRB%',
       'team_DRtg', 'team_FTr', 'team_ORB%', 'team_ORtg', 'team_STL%',
       'team_TOV%', 'team_TRB%', 'team_TS%', 'team_eFG%', 'team_fg3_pct',
       'team_fg_pct', 'team_ft_pct', 'team_pf', 'opponent_3PAr',
       'opponent_AST%', 'opponent_BLK%', 'opponent_DRB%', 'opponent_DRtg',
       'opponent_FTr', 'opponent_ORB%', 'opponent_ORtg', 'opponent_STL%',
       'opponent_TOV%', 'opponent_TRB%', 'opponent_TS%', 'opponent_eFG%',
       'opponent_fg3_pct', 'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf',
       'spread', 'totals', 'venue_y', 'score_team', 'score_oppt',
       'team_predicted_points', 'oppt_predicted_points',
       'spread_expanding_mean', 'current_spread_vs_spread_expanding',
       'beat_spread', 'beat_spread_rolling_mean_11', 'beat_spread_rolling_std_11',
       'beat_spread_last_g', 'season_start', 'team_3PAr_ewma_15',
       'team_AST%_ewma_15', 'team_BLK%_ewma_15', 'team_DRB%_ewma_15',
       'team_DRtg_ewma_15', 'team_FTr_ewma_15', 'team_ORB%_ewma_15',
       'team_ORtg_ewma_15', 'team_STL%_ewma_15', 'team_TOV%_ewma_15',
       'team_TRB%_ewma_15', 'team_TS%_ewma_15', 'team_eFG%_ewma_15',
       'team_fg3_pct_ewma_15', 'team_fg_pct_ewma_15', 'team_ft_pct_ewma_15',
       'team_pf_ewma_15', 'opponent_3PAr_ewma_15', 'opponent_AST%_ewma_15',
       'opponent_BLK%_ewma_15', 'opponent_DRB%_ewma_15',
       'opponent_DRtg_ewma_15', 'opponent_FTr_ewma_15',
       'opponent_ORB%_ewma_15', 'opponent_ORtg_ewma_15',
       'opponent_STL%_ewma_15', 'opponent_TOV%_ewma_15',
       'opponent_TRB%_ewma_15', 'opponent_TS%_ewma_15',
       'opponent_eFG%_ewma_15', 'opponent_fg3_pct_ewma_15',
       'opponent_fg_pct_ewma_15', 'opponent_ft_pct_ewma_15',
       'opponent_pf_ewma_15', 'days_rest', 'zone_distance', 'distance_playoffs_abs', 'starters_same_as_last_g']]



def create_switched_df(df_all_teams):
    # create df with team and opponent swithced (so can then merge the team's 
    # weighted/rolling metrics onto the original df but as the opponents)
    df_all_teams_swtiched = df_all_teams.copy(deep=True)
    df_all_teams_swtiched.rename(columns={'team':'opponent_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent':'team_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent_hold':'opponent'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'team_hold':'team'}, inplace=True)
    df_all_teams_swtiched = df_all_teams_swtiched[['date', 'opponent', 'team', 'lineup_count',
        'spread_expanding_mean', 'current_spread_vs_spread_expanding',
       'beat_spread_rolling_mean_11', 'beat_spread_rolling_std_11', 
       'beat_spread_last_g', 'team_3PAr_ewma_15',
       'team_AST%_ewma_15', 'team_BLK%_ewma_15', 'team_DRB%_ewma_15',
       'team_DRtg_ewma_15', 'team_FTr_ewma_15', 'team_ORB%_ewma_15',
       'team_ORtg_ewma_15', 'team_STL%_ewma_15', 'team_TOV%_ewma_15',
       'team_TRB%_ewma_15', 'team_TS%_ewma_15', 'team_eFG%_ewma_15',
       'team_fg3_pct_ewma_15', 'team_fg_pct_ewma_15', 'team_ft_pct_ewma_15',
       'team_pf_ewma_15', 'opponent_3PAr_ewma_15', 'opponent_AST%_ewma_15',
       'opponent_BLK%_ewma_15', 'opponent_DRB%_ewma_15',
       'opponent_DRtg_ewma_15', 'opponent_FTr_ewma_15',
       'opponent_ORB%_ewma_15', 'opponent_ORtg_ewma_15',
       'opponent_STL%_ewma_15', 'opponent_TOV%_ewma_15',
       'opponent_TRB%_ewma_15', 'opponent_TS%_ewma_15',
       'opponent_eFG%_ewma_15', 'opponent_fg3_pct_ewma_15',
       'opponent_fg_pct_ewma_15', 'opponent_ft_pct_ewma_15',
       'opponent_pf_ewma_15', 'days_rest', 'zone_distance', 'distance_playoffs_abs', 'starters_same_as_last_g']]
    return df_all_teams_swtiched

df_covers_bball_ref_switched = create_switched_df(df_covers_bball_ref)



def preface_oppt_stats_in_switched_df(df_all_teams_swtiched):
    # preface all these stats -- they belong to the team in this df but to
    # the opponent in the orig df -- with an x_. then when merge back onto original df
    # these stats will be for the opponent in that df. and that's what i'll use to predict ats
    df_all_teams_swtiched.columns=['date', 'opponent', 'team', 'x_lineup_count',
            'x_spread_expanding_mean', 'x_current_spread_vs_spread_expanding',
           'x_beat_spread_rolling_mean_11', 'x_beat_spread_rolling_std_11', 
           'x_beat_spread_last_g', 'x_team_3PAr_ewma_15',
           'x_team_AST%_ewma_15', 'x_team_BLK%_ewma_15', 'x_team_DRB%_ewma_15',
           'x_team_DRtg_ewma_15', 'x_team_FTr_ewma_15', 'x_team_ORB%_ewma_15',
           'x_team_ORtg_ewma_15', 'x_team_STL%_ewma_15', 'x_team_TOV%_ewma_15',
           'x_team_TRB%_ewma_15', 'x_team_TS%_ewma_15', 'x_team_eFG%_ewma_15',
           'x_team_fg3_pct_ewma_15', 'x_team_fg_pct_ewma_15', 'x_team_ft_pct_ewma_15',
           'x_team_pf_ewma_15', 'x_opponent_3PAr_ewma_15', 'x_opponent_AST%_ewma_15',
           'x_opponent_BLK%_ewma_15', 'x_opponent_DRB%_ewma_15',
           'x_opponent_DRtg_ewma_15', 'x_opponent_FTr_ewma_15',
           'x_opponent_ORB%_ewma_15', 'x_opponent_ORtg_ewma_15',
           'x_opponent_STL%_ewma_15', 'x_opponent_TOV%_ewma_15',
           'x_opponent_TRB%_ewma_15', 'x_opponent_TS%_ewma_15',
           'x_opponent_eFG%_ewma_15', 'x_opponent_fg3_pct_ewma_15',
           'x_opponent_fg_pct_ewma_15', 'x_opponent_ft_pct_ewma_15',
           'x_opponent_pf_ewma_15', 'x_days_rest', 'x_zone_distance', 'x_distance_playoffs_abs', 'x_starters_same_as_last_g']    
    return df_all_teams_swtiched   
           
df_covers_bball_ref_switched = preface_oppt_stats_in_switched_df(df_covers_bball_ref_switched)



def merge_regular_df_w_switched_df(df_all_teams, df_all_teams_swtiched):    
    df_all_teams_w_ivs = df_all_teams.merge(df_all_teams_swtiched, on=['date', 'team', 'opponent'], how='left')
    df_all_teams_w_ivs.head(50)
    return df_all_teams_w_ivs

df_covers_bball_ref = merge_regular_df_w_switched_df(df_covers_bball_ref, df_covers_bball_ref_switched)    



variables = ['spread_expanding_mean', 'current_spread_vs_spread_expanding', 'lineup_count',
       'beat_spread_rolling_mean_11', 'beat_spread_rolling_std_11', 
       'beat_spread_last_g', 'team_3PAr_ewma_15',
       'team_AST%_ewma_15', 'team_BLK%_ewma_15', 'team_DRB%_ewma_15',
       'team_DRtg_ewma_15', 'team_FTr_ewma_15', 'team_ORB%_ewma_15',
       'team_ORtg_ewma_15', 'team_STL%_ewma_15', 'team_TOV%_ewma_15',
       'team_TRB%_ewma_15', 'team_TS%_ewma_15', 'team_eFG%_ewma_15',
       'team_fg3_pct_ewma_15', 'team_fg_pct_ewma_15', 'team_ft_pct_ewma_15',
       'team_pf_ewma_15', 'opponent_3PAr_ewma_15', 'opponent_AST%_ewma_15',
       'opponent_BLK%_ewma_15', 'opponent_DRB%_ewma_15',
       'opponent_DRtg_ewma_15', 'opponent_FTr_ewma_15',
       'opponent_ORB%_ewma_15', 'opponent_ORtg_ewma_15',
       'opponent_STL%_ewma_15', 'opponent_TOV%_ewma_15',
       'opponent_TRB%_ewma_15', 'opponent_TS%_ewma_15',
       'opponent_eFG%_ewma_15', 'opponent_fg3_pct_ewma_15',
       'opponent_fg_pct_ewma_15', 'opponent_ft_pct_ewma_15',
       'opponent_pf_ewma_15', 'days_rest', 'zone_distance', 'distance_playoffs_abs', 'starters_same_as_last_g']

def create_team_opponent_difference_variables(df_all_teams_w_ivs, variables):
    for var in variables:
        new_difference_variable = 'difference_'+var
        df_all_teams_w_ivs[new_difference_variable] = df_all_teams_w_ivs[var] - df_all_teams_w_ivs['x_'+var]
    return df_all_teams_w_ivs

df_covers_bball_ref = create_team_opponent_difference_variables(df_covers_bball_ref, variables)



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



def create_home_df(df_covers_bball_ref):
    df_covers_bball_ref_home = df_covers_bball_ref[df_covers_bball_ref['venue_x'] == 0]
    df_covers_bball_ref_home = df_covers_bball_ref_home.reset_index()
    len(df_covers_bball_ref_home)
    return df_covers_bball_ref_home

df_covers_bball_ref_home = create_home_df(df_covers_bball_ref)



all_iv_vars = ['difference_spread_expanding_mean', 'difference_lineup_count', #'difference_current_spread_vs_spread_expanding',
       'difference_beat_spread_rolling_mean_11', 'difference_beat_spread_rolling_std_11', 
       'difference_beat_spread_last_g', 'difference_team_3PAr_ewma_15',
       'difference_team_AST%_ewma_15', 'difference_team_BLK%_ewma_15', 'difference_team_DRB%_ewma_15',
       'difference_team_DRtg_ewma_15', 'difference_team_FTr_ewma_15', 'difference_team_ORB%_ewma_15',
       'difference_team_ORtg_ewma_15', 'difference_team_STL%_ewma_15', 'difference_team_TOV%_ewma_15',
       'difference_team_TRB%_ewma_15', 'difference_team_TS%_ewma_15', 'difference_team_eFG%_ewma_15',
       'difference_team_fg3_pct_ewma_15', 'difference_team_fg_pct_ewma_15', 'difference_team_ft_pct_ewma_15',
       'difference_team_pf_ewma_15', 'difference_opponent_3PAr_ewma_15', 'difference_opponent_AST%_ewma_15',
       'difference_opponent_BLK%_ewma_15', 'difference_opponent_DRB%_ewma_15',
       'difference_opponent_DRtg_ewma_15', 'difference_opponent_FTr_ewma_15',
       'difference_opponent_ORB%_ewma_15', 'difference_opponent_ORtg_ewma_15',
       'difference_opponent_STL%_ewma_15', 'difference_opponent_TOV%_ewma_15',
       'difference_opponent_TRB%_ewma_15', 'difference_opponent_TS%_ewma_15',
       'difference_opponent_eFG%_ewma_15', 'difference_opponent_fg3_pct_ewma_15',
       'difference_opponent_fg_pct_ewma_15', 'difference_opponent_ft_pct_ewma_15',
       'difference_opponent_pf_ewma_15', 'spread', 'totals', 'difference_days_rest', 'zone_distance', 
       'difference_distance_playoffs_abs', 'difference_starters_same_as_last_g']  #* using


df_covers_bball_ref_home.rename(columns={'difference_team_AST%_ewma_15':'difference_team_ASTpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_BLK%_ewma_15':'difference_team_BLKpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_DRB%_ewma_15':'difference_team_DRBpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_ORB%_ewma_15':'difference_team_ORBpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_STL%_ewma_15':'difference_team_STLpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_TOV%_ewma_15':'difference_team_TOVpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_TRB%_ewma_15':'difference_team_TRBpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_TS%_ewma_15':'difference_team_TSpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_team_eFG%_ewma_15':'difference_team_eFGpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_AST%_ewma_15':'difference_opponent_ASTpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_BLK%_ewma_15':'difference_opponent_BLKpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_DRB%_ewma_15':'difference_opponent_DRBpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_ORB%_ewma_15':'difference_opponent_ORBpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_STL%_ewma_15':'difference_opponent_STLpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_TOV%_ewma_15':'difference_opponent_TOVpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_TRB%_ewma_15':'difference_opponent_TRBpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_TS%_ewma_15':'difference_opponent_TSpct_ewma_15'}, inplace=True)
df_covers_bball_ref_home.rename(columns={'difference_opponent_eFG%_ewma_15':'difference_opponent_eFGpct_ewma_15'}, inplace=True)



all_iv_vars = ['difference_lineup_count', #'difference_current_spread_vs_spread_expanding', 'difference_spread_expanding_mean', 
       'difference_beat_spread_rolling_mean_11', 'difference_beat_spread_rolling_std_11', 
       'difference_beat_spread_last_g', 'difference_team_3PAr_ewma_15',
       'difference_team_ASTpct_ewma_15', 'difference_team_BLKpct_ewma_15', 'difference_team_DRBpct_ewma_15',
       'difference_team_DRtg_ewma_15', 'difference_team_FTr_ewma_15', 'difference_team_ORBpct_ewma_15',
       'difference_team_ORtg_ewma_15', 'difference_team_STLpct_ewma_15', 'difference_team_TOVpct_ewma_15',
       'difference_team_TRBpct_ewma_15', 'difference_team_TSpct_ewma_15', 'difference_team_eFGpct_ewma_15',
       'difference_team_fg3_pct_ewma_15', 'difference_team_fg_pct_ewma_15', 'difference_team_ft_pct_ewma_15',
       'difference_team_pf_ewma_15', 'difference_opponent_3PAr_ewma_15', 'difference_opponent_ASTpct_ewma_15',
       'difference_opponent_BLKpct_ewma_15', 'difference_opponent_DRBpct_ewma_15',
       'difference_opponent_DRtg_ewma_15', 'difference_opponent_FTr_ewma_15',
       'difference_opponent_ORBpct_ewma_15', 'difference_opponent_ORtg_ewma_15',
       'difference_opponent_STLpct_ewma_15', 'difference_opponent_TOVpct_ewma_15',
       'difference_opponent_TRBpct_ewma_15', 'difference_opponent_TSpct_ewma_15',
       'difference_opponent_eFGpct_ewma_15', 'difference_opponent_fg3_pct_ewma_15',
       'difference_opponent_fg_pct_ewma_15', 'difference_opponent_ft_pct_ewma_15',
       'difference_opponent_pf_ewma_15', 'totals', 'spread', 'difference_days_rest', 'zone_distance', 
       'difference_distance_playoffs_abs', 'difference_starters_same_as_last_g']  #  using



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



#dv_var = 'ats_win'  
#dv_var = 'win'
#dv_var = 'spread'
dv_var = 'point_difference'

iv_and_dv_vars = all_iv_vars + [dv_var] + ['team', 'opponent', 'date']
df_covers_bball_ref_home[iv_and_dv_vars].head(10)
print(len(df_covers_bball_ref_home))

df_covers_bball_ref__dropna_home = df_covers_bball_ref_home.dropna()
print(len(df_covers_bball_ref__dropna_home))



def scale_variables(df, all_iv_vars):
    #df[all_iv_vars] = scale(df[all_iv_vars])
    for var in variables:
        df.loc[:,(var)] = (df.loc[:,var] - df.loc[:,var].mean()) / df.loc[:,var].std()
    return df


df_covers_bball_ref__dropna_home = scale_variables(df_covers_bball_ref__dropna_home, all_iv_vars)


# ## examine accuracy of spread ea year


df_covers_bball_ref__dropna_home.loc[:,'spread_accuracy'] = np.abs(df_covers_bball_ref__dropna_home.loc[:,'point_difference'] + df_covers_bball_ref__dropna_home.loc[:,'spread'])
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2004].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2005].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2006].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2007].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2008].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2009].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2010].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2011].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2012].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2013].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2014].mean())
print(df_covers_bball_ref__dropna_home[['spread_accuracy']][df_covers_bball_ref__dropna_home['season_start'] == 2015].mean())
#df_covers_bball_ref__dropna_home[['spread', 'point_difference', 'spread_accuracy']].head()



for year in [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]:
    model_plot = smf.ols(formula = 'point_difference ~ spread', data=df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] == year]).fit() 
    print(np.round(model_plot.rsquared, 3), np.round(model_plot.fvalue, 3))
# spread seems to be getting more consistently good
# i looked at 2007, really accuracy spread year early on. if i bet on that year after training on previous 3 seasons
# i do terrible that. year. suggests that the more accurate the spread, the worse this model will do?



#df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] != 2015)]

def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, test_year):
    #df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < test_year) &
    #                                                                  (df_covers_bball_ref__dropna_home['season_start'] > test_year-7)]
    df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[(df_covers_bball_ref__dropna_home['season_start'] < test_year) &
                                                                      (df_covers_bball_ref__dropna_home['season_start'] > 2004)]
    print ('training n:', len(df_covers_bball_ref_home_train))
    df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] == test_year]
    print ('test n:', len(df_covers_bball_ref_home_test))
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test

#df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, 2010)



def create_df_weight_recent_seasons_more(df_covers_bball_ref_home_train):
    seasons = sorted(df_covers_bball_ref_home_train['season_start'].unique())
    df_training_weighted = pd.DataFrame()
    for i in range(len(seasons)):
        df_season = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start'] == seasons[i]]
        df_season_multiple = pd.DataFrame()
        for k in range(5+i):
            df_season_multiple = pd.concat([df_season_multiple,df_season], ignore_index=True)
        df_training_weighted = pd.concat([df_training_weighted, df_season_multiple], ignore_index=True)
    return df_training_weighted



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

def mse_in_training_set(df_covers_bball_ref_home_train, algorithm):
    model = algorithm
    mses = cross_val_score(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], 
                           scoring = 'mean_squared_error', cv=10)
    print ('cross validated mse training:', -1*mses.mean())

#mse_in_training_set(df_covers_bball_ref_home_train, linear_model.LinearRegression())


# ##Translate point diff model into ats in test season(s)


# create correct metric:
def create_correct_metric(df):
    df['correct'] = np.nan
    df.loc[(df['point_diff_predicted'] > df['spread']*-1) &
                                       (df['ats_win'] == '1'), 'correct'] = 1
    df.loc[(df['point_diff_predicted'] < df['spread']*-1) &
                                       (df['ats_win'] == '0'), 'correct'] = 1
    df.loc[(df['point_diff_predicted'] > df['spread']*-1) &
                                       (df['ats_win'] == '0'), 'correct'] = 0
    df.loc[(df['point_diff_predicted'] < df['spread']*-1) &
                                       (df['ats_win'] == '1'), 'correct'] = 0
    # create var to say how much my prediction deviates from actual spread:
    df['predicted_spread_deviation'] = np.abs(df['spread'] + df['point_diff_predicted'])
    return df



# fit model on data up through 2014 season
# when i use alpha > 1 and scale, seems to be doing better. presumably not overfitting.
# the predicted spread deviation looks more variable. but the n is small here, so think it should
# and actually, it does look pretty similar to the one above in the training set -- peaks about 3 at 55%. cool.

def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm):
    model = algorithm
    model.fit(df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var])
    predictions_test_set = model.predict(df_covers_bball_ref_home_test[all_iv_vars])
    df_covers_bball_ref_home_test.loc[:,'point_diff_predicted'] = predictions_test_set
    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
    return df_covers_bball_ref_home_test

#df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, linear_model.LinearRegression())



#df_covers_bball_ref_home_test[['date', 'team', 'spread', 'point_difference', 'point_diff_predicted', 'beat_spread', 'ats_win', 'correct', 'predicted_spread_deviation']].head()



#print 'mean accuracy:', np.round(df_covers_bball_ref_home_test['correct'].mean(), 3)*100, 'percent'
#df_covers_bball_ref_home_test['predicted_spread_deviation'].hist(alpha=.8, bins=15);

#sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_covers_bball_ref_home_test, lowess=True)
#plt.ylim(.4, .7)
#plt.xlim(0,4)


# ##Compute multiple seasons training on prior seasons


from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


import inspect
from sklearn.utils.testing import all_estimators
for name, clf in all_estimators(type_filter='regressor'):
    if 'sample_weight' in inspect.getargspec(clf().fit)[0]:
       print(name)

# IS THERE ANY REASON THAT THIS SHOULD PRODUCE HIGHER ACCURACY WHEN JUST DOING
# 2015 VS. SETTING THE SEASONS VARIABLE TO A LIST OF YEARS?


# general approach idea: run regular regression and get predictions and run
# 4-5 adaboost regressions and avg those, and then avg the regular regression
# point diff and the adaboost avg pt diffs. and then make deicsions. do these
# corr better with actually beating spread? if not, could only bet on those 
# games that both regular and boosted agree? that should weed out a few games and
# hopefully improve odds a little.


#seasons = [2010, 2011, 2012, 2013, 2014, 2015]
seasons = [2010, 2012, 2013, 2014, 2015]  # omit 2011
seasons = [2015]
#seasons = [2014]

#model = linear_model.Ridge(alpha=2)
model = linear_model.LinearRegression()
#model = KNeighborsRegressor(n_neighbors=800, weights='distance')
#model = RandomForestRegressor(n_estimators=750)  # , min_samples_leaf = 10, min_samples_split = 50)
#model = ensemble.GradientBoostingRegressor(n_estimators=750, max_depth=3, learning_rate=.01, subsample=.5) # this sucked
#model = tree.DecisionTreeRegressor()

# holy shit, this adaboost ups the 2015 accuracy to 55%. but took about 1/2 hr to run w 200 estimators
# I think shrinking the learning rate below 1 helps. need bigger n_estimators, though?
# using loss='square' may also be better than 'linear?'
#model = AdaBoostRegressor(tree.DecisionTreeRegressor(), n_estimators=200, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
# this is predicting well for 2015 and maybe 2014. but not predicting for all. why?

# but the i did it again w 200 estimators and only got 50%?!
# nearest neighbors didn't seem good for adaboost:
#model = AdaBoostRegressor(KNeighborsRegressor(n_neighbors=500), n_estimators=100)  # this decision tree regressor is the default
# yeah -- this boosting model below is giving gains on the regular regression

#THIS IS THE ONE:
#model = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default

#model = AdaBoostRegressor(linear_model.Ridge(alpha=.01), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
#model = AdaBoostRegressor(linear_model.KernelRidge(alpha=.01), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
#model = AdaBoostRegressor(AdaBoostRegressor(), n_estimators=50, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default

#The parameter learning_rate strongly interacts with the parameter n_estimators, the number 
#of weak learners to fit. Smaller values of learning_rate require larger numbers of weak 
#learners to maintain a constant training error. Empirical evidence suggests that small 
#values of learning_rate favor better test error. [HTF2009] recommend to set the learning 
#rate to a small constant (e.g. learning_rate <= 0.1) and choose n_estimators by early 
#stopping. For a more detailed discussion of the interaction between learning_rate and 
#n_estimators see [R2007].

def analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, algorithm):
    accuracy_list = []
    df_test_seasons = pd.DataFrame()
    #df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start']!=2011]
    for season in seasons:
        print(season)
        df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, season)
        #df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start'] != 2011]  #shortened season. helps a bit.
        #df_covers_bball_ref_home_train = create_df_weight_recent_seasons_more(df_covers_bball_ref_home_train)
        df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm)
        df_test_seasons = pd.concat([df_test_seasons, df_covers_bball_ref_home_test])
        accuracy_list.append((season, np.round(df_covers_bball_ref_home_test['correct'].mean(), 4)*100))
        #print season, 'mean accuracy:', np.round(df_covers_bball_ref_home_test['correct'].mean(), 3)*100, 'percent'
        #print sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_covers_bball_ref_home_test, lowess=True)
        print()
    return accuracy_list, df_test_seasons, df_covers_bball_ref_home_train

#accuracy_list, df_test_seasons = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, linear_model.Ridge(alpha=.01))
#accuracy_list, df_test_seasons = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, linear_model.Lasso(alpha=.01))
accuracy_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model)

df = pd.DataFrame(accuracy_list, columns=['season', 'accuracy'])
#sns.barplot(x='season', y='accuracy', data=df)
#plt.ylim(45,60)


for season in range(len(seasons)):
    plt.plot([accuracy_list[season][1], accuracy_list[season][1]], label=str(seasons[season]));
plt.ylim(48, 58)
#plt.ylim(38, 68)

#plt.grid(axis='y', linestyle='--', alpha=.5)
plt.xticks([])
plt.legend()
plt.ylabel('percent correct')
plt.axhline(51.5, linestyle='--', color='grey', linewidth=1, alpha=.5)
sns.despine()


# nice that this graph is showing a lot of stability -- when i'm taking just the 6 years prior to the test year
# but if i take all years back to 2005 before the test year, more spread out so that the more recent the year 
# and in turn the more the trianing data, the better the accuracy, except the last 2015 season. why?

# when i weight by most recent seasons, the lowess curves look nicer in that they increase as the predicted_spread_deviation
# increases, at least til about 2. makes sense. BUT the overall accuracy percent is lower. hmmm. looks like when i
# weight it's making it look more like i trained on teh same amount of prior seasons, rather than an expeanding amount
# depending on how recent the year. saftest bet now seems not to weight them.



accuracy_list



for season in seasons:
    df_test_seasons['predicted_spread_deviation'][df_test_seasons['season_start']==season].hist(alpha=.1, color='green')

sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_test_seasons, hue='season_start', lowess=True, line_kws={'alpha':.6})
#plt.ylim(.3, .7)
plt.xlim(0,4)
plt.grid(axis='y', linestyle='--', alpha=.5)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)



sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.ylim(.3, .7)
plt.xlim(0,4)
#plt.grid(axis='y', linestyle='--', alpha=.75)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)



#df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .5) & (df_test_seasons['predicted_spread_deviation'] < 3)]
df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .5)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_season_2015_alt = df_test_seasons[df_test_seasons['season_start']==2015]
len(df_season_2015_alt)

print('games in season:', len(df_season_2015))
print('games in season:', len(df_season_2015_alt))

df_season_2015['correct_alt'] = df_season_2015_alt['correct'].values
df_season_2015[['correct', 'correct_alt']].tail(10)

df_season_2015['correct_both_agree'] = 0
df_season_2015.loc[df_season_2015['correct'] == df_season_2015['correct_alt'], 'correct_both_agree'] = 1

df_season_2015[['correct', 'correct_alt', 'correct_both_agree']].tail(15)
df_season_2015['correct_both_agree'].mean()  # agree on 96%


print()
print(df_season_2015[['correct']].mean())
print(df_season_2015[['correct_alt']].mean())
print(df_season_2015[['correct']][df_season_2015['correct_both_agree']==1].mean())
print(df_season_2015[['correct']][df_season_2015['correct_both_agree']==1].count())




#df_season_2015 = df_test_seasons[df_test_seasons['season_start']==2015]
#df_season_2015 = df_season_2015.reset_index()
print()
print('games in season:', len(df_season_2015))
print('games in season:', len(df_season_2015_alt))


df_season = df_season_2015[df_season_2015['correct_both_agree']==1].copy(deep=True)
df_season = df_season.reset_index(drop=True)

cumulative_money_list = []
for i in range(len(df_season)):
    outcome = df_season[['correct']][df_season.index == i].values
    cumulative_money_list.append(outcome)

cumulative_money_list = [o[0][0] for o in cumulative_money_list]

#len(cumulative_money_list)
#print(cumulative_money_list[:10])

bet = 100
total_pot = 0
total_winnings_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot += bet*.95
        total_winnings_list.append(total_pot)
    if game == 0:
        total_pot += -1*bet
        total_winnings_list.append(total_pot)


plt.plot(total_winnings_list, alpha=.5, color='green')
plt.xlabel('\n games', fontsize=15)
plt.ylabel('winnings', fontsize=15)
plt.xlim(0,1100)
#plt.ylim(-2500,4000)
plt.axhline(.5, linestyle='--', color='red', linewidth=1, alpha=.5)
sns.despine()





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





