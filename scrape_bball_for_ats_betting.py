# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:57:58 2016

@author: charlesmartens
"""


# plan: should put all data, home and away teams offensive and defensive
# stats into the same row. and just represent ea game w one row
# then run the ml model. so can do it for ea date (rather than ea date
# for ea team). this will dramatically reduce time to test it.

# second, try fitting/training on past two years of data and testing
# on the current year (same as above -- indlucing home and away team's
# off and def stats for ea game).



cd /Users/charlesmartens/Documents/projects/bet fantasy/current_data

import urllib.request   
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import statsmodels.formula.api as smf 
from statsmodels.formula.api import *

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
#from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.learning_curve import learning_curve
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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
from sklearn import tree
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# try tpot
from tpot import TPOT




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# 1. get data up to yesterday (bball-ref, rotoguru, covers)
# 2. model/fit the data 
# 3. get today's fanduel and spread data
# 4. predict today's scores


#-----------------------------------------------------------------------------
# get spread /totals info
# does it nicely and quickly

def scrape_team_gamelog(link):
    on_webpage_team_gamelog = urllib.request.urlopen(link)
    html_contents_team_gamelog = on_webpage_team_gamelog.read()
    on_webpage_team_gamelog.close()        
    soupObject_team_gamelog = BeautifulSoup(html_contents_team_gamelog)
    team_name = soupObject_team_gamelog.find('h1', class_='teams').text.strip()
    gamelog = soupObject_team_gamelog.find_all('tr', class_='datarow')
    print(len(gamelog))
    return gamelog, team_name


def create_team_gamelog(team_gamelog, team_name, team_dict):
    i = 1
    for game in team_gamelog:    
        if "Regular Season" in str(game):
            game_data = game.find_all('td')
            date = game_data[0].text.strip()
            opponent = game_data[1].text.strip()
            if opponent[:1] == '@':
                opponent = opponent[2:]
                venue = 'away'
            else:
                opponent = opponent
                venue = 'home'
            score = game_data[2].text.strip()
            regular_season = game_data[3].text.strip()
            spread = game_data[4].text.strip()
            ats, spread = spread.split()
            totals = game_data[5].text.strip()
            over, totals = totals.split()
            #print(i)
            team_dict['team'].append(team_name)
            team_dict['game_number'].append(i)
            team_dict['date'].append(date)
            team_dict['opponent'].append(opponent)
            team_dict['score'].append(score)
            team_dict['regular_season'].append(regular_season)
            team_dict['spread'].append(spread)
            team_dict['totals'].append(totals)
            team_dict['over'].append(over)
            team_dict['venue'].append(venue)
            team_dict['ats'].append(ats)
            i = i+1
    return team_dict

covers_dict = {'brooklyn':404117, 'new york':404288, 'boston':404169, 'cleveland':404213, 
'philadelphia':404083, 'toronto':404330, 'chicago':404198, 'detroit':404153, 
'indiana':404155, 'milwaukee':404011, 'atlanta':404085, 'charlotte':664421, 
'miami':404171, 'orlando':404013, 'washington':404067, 'denver':404065, 'minnesota':403995, 
'oklahoma city':404316, 'portland':403993, 'utah':404031, 'golden state':404119,
'los angeles clippers':404135, 'los angeles lakers':403977, 'phoenix':404029, 
'sacramento':403975, 'dallas':404047, 'houston':404137, 'memphis':404049, 
'new orleans':404101, 'san antonio':404302, 'dallas':404047, 'los angeles clippers':404135}


def get_covers_info(season_years):
    league_df = pd.DataFrame()
    team_dict = defaultdict(list)
    for team in covers_dict.keys():    
        team_dict = defaultdict(list)
        print(team)
        covers_link = 'http://www.covers.com/pageLoader/pageLoader.aspx?page=/data/nba/teams/pastresults/'+season_years+'/team'+str(covers_dict[team])+'.html'
        team_gamelog, team_name = scrape_team_gamelog(covers_link)
        team_dict = create_team_gamelog(team_gamelog, team_name, team_dict)
        team_df = pd.DataFrame(team_dict)
        league_df = pd.concat([league_df, team_df], ignore_index=True)
    df_teams_covers = league_df.copy()
    return df_teams_covers

df_teams_covers = get_covers_info('2004-2005')


def fix_variables_covers(df_teams_covers):
    df_teams_covers.dtypes
    df_teams_covers.columns
    df_teams_covers['date'] = pd.to_datetime(df_teams_covers['date'])
    df_teams_covers['spread'].replace('PK', 0, inplace=True)
    df_teams_covers['spread'] = df_teams_covers['spread'].astype(float)
    df_teams_covers['totals'] = df_teams_covers['totals'].astype(float)
    len(df_teams_covers['team'].unique())
    df_teams_covers['score'] = df_teams_covers['score'].str.strip('(OT)').str.strip()
    df_teams_covers['score'] = df_teams_covers['score'].str.split().str[-1]
    df_teams_covers['score_team'] = df_teams_covers['score'].str.split('-').str[0].astype(float)
    df_teams_covers['score_oppt'] = df_teams_covers['score'].str.split('-').str[1].astype(float)
    return df_teams_covers

df_teams_covers = fix_variables_covers(df_teams_covers)


def replace_team_names(df_teams_covers):
    # replace team names so can merge with updated file below
    teams_covers = list(df_teams_covers['team'].unique())
    teams_covers = sorted(teams_covers)
    teams_acronyms = ['ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN',
                      'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 
                      'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO',
                      'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']                            
    df_teams_covers['team'].replace(teams_covers, teams_acronyms, inplace=True)
    df_teams_covers = df_teams_covers[['date', 'regular_season', 'spread', 'team', 
    'opponent', 'totals', 'venue', 'score_team', 'score_oppt']]
    return df_teams_covers

df_teams_covers = replace_team_names(df_teams_covers)


def create_variables_covers(df_teams_covers):
    # compute predicted team and oppt points, and other metrics
    df_teams_covers['team_predicted_points'] = (df_teams_covers['totals']/2) - (df_teams_covers['spread']/2)
    df_teams_covers['oppt_predicted_points'] = (df_teams_covers['totals']/2) + (df_teams_covers['spread']/2)
    df_teams_covers = df_teams_covers.sort(['team', 'date'])
    df_teams_covers['score_team_expanding'] = df_teams_covers.groupby('team')['score_team'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    df_teams_covers['score_oppt_expanding'] = df_teams_covers.groupby('team')['score_oppt'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    df_teams_covers['team_expanding_vs_pred_points'] = df_teams_covers['team_predicted_points'] - df_teams_covers['score_team_expanding']
    df_teams_covers['oppt_expanding_vs_pred_points'] = df_teams_covers['oppt_predicted_points'] - df_teams_covers['score_oppt_expanding']
    df_teams_covers['spread_expanding_mean'] = df_teams_covers.groupby('team')['spread'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    df_teams_covers['current_spread_vs_spread_expanding'] = df_teams_covers['spread'] - df_teams_covers['spread_expanding_mean']    
    # how much beating the spread by of late -- to get sense of under/over performance of late
    df_teams_covers['beat_spread'] = df_teams_covers['spread'] + (df_teams_covers['score_team'] - df_teams_covers['score_oppt'])    
    df_teams_covers['beat_spread_rolling_mean_11'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_mean(x.shift(1), 11, min_periods=5))
    df_teams_covers['beat_spread_rolling_std_11'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: pd.rolling_std(x.shift(1), 11, min_periods=5))
    df_teams_covers['beat_spread_last_g'] = df_teams_covers.groupby('team')['beat_spread'].transform(lambda x: x.shift(1))
    return df_teams_covers

df_teams_covers = create_variables_covers(df_teams_covers)
df_teams_covers[['team_predicted_points', 'beat_spread_rolling_std_11']]

#---------------------------------------------------------------
# save the file -- make sure to name it w correct year of season:
df_teams_covers.to_pickle('df_teams_covers_2004_05.pkl')
df_teams_covers_2004 = pd.read_pickle('df_teams_covers_2004_05.pkl')
df_teams_covers_2004[df_teams_covers_2004['spread'].isnull()]

df_teams_covers.to_pickle('df_teams_covers_2005_06.pkl')
df_teams_covers_2005 = pd.read_pickle('df_teams_covers_2005_06.pkl')
df_teams_covers_2005[df_teams_covers_2005['spread'].isnull()]

df_teams_covers.to_pickle('df_teams_covers_2006_07.pkl')
df_teams_covers_2006 = pd.read_pickle('df_teams_covers_2006_07.pkl')
df_teams_covers_2006[df_teams_covers_2006['spread'].isnull()]

df_teams_covers.to_pickle('df_teams_covers_2007_08.pkl')
df_teams_covers_2007 = pd.read_pickle('df_teams_covers_2007_08.pkl')

df_teams_covers.to_pickle('df_teams_covers_2008_09.pkl')
df_teams_covers_2008 = pd.read_pickle('df_teams_covers_2008_09.pkl')

df_teams_covers.to_pickle('df_teams_covers_2009_10.pkl')
df_teams_covers_2009 = pd.read_pickle('df_teams_covers_2009_10.pkl')

df_teams_covers.to_pickle('df_teams_covers_2010_11.pkl')
df_teams_covers_2010 = pd.read_pickle('df_teams_covers_2010_11.pkl')

df_teams_covers.to_pickle('df_teams_covers_2011_12.pkl')
df_teams_covers_2011 = pd.read_pickle('df_teams_covers_2011_12.pkl')

df_teams_covers.to_pickle('df_teams_covers_2012_13.pkl')
df_teams_covers_2012 = pd.read_pickle('df_teams_covers_2012_13.pkl')

df_teams_covers.to_pickle('df_teams_covers_2013_14.pkl')
df_teams_covers_2013 = pd.read_pickle('df_teams_covers_2013_14.pkl')

df_teams_covers.to_pickle('df_teams_covers_2014_15.pkl')
df_teams_covers_2014 = pd.read_pickle('df_teams_covers_2014_15.pkl')

df_teams_covers.to_pickle('df_teams_covers_2015_16.pkl')
df_teams_covers_2015 = pd.read_pickle('df_teams_covers_2015_16.pkl')
#-----------------------------------------------------------------
del [df_teams_covers]

#df_teams_covers = pd.concat([df_teams_covers_2013, df_teams_covers_2014, df_teams_covers_2015], ignore_index=True)
#df_teams_covers = df_teams_covers_2007.copy(deep=True)
#df_teams_covers = df_teams_covers_2008.copy(deep=True)
#df_teams_covers = df_teams_covers_2009.copy(deep=True)
#df_teams_covers = df_teams_covers_2010.copy(deep=True)
#df_teams_covers = df_teams_covers_2011.copy(deep=True)
#df_teams_covers = df_teams_covers_2012.copy(deep=True)
#df_teams_covers = df_teams_covers_2013.copy(deep=True)
#df_teams_covers = df_teams_covers_2014.copy(deep=True)
#df_teams_covers = df_teams_covers_2015.copy(deep=True)
#len(df_teams_covers)
#df_teams_covers.head()

def fix_names_covers_2(df_teams_covers):
    print(df_teams_covers['team'].unique())
    print(df_teams_covers['opponent'].unique()) 
    old_names = ['ATL', 'CHI', 'GSW', 'BOS', 'BRK', 'DET', 'HOU', 'LAL', 'MEM',
                 'MIA', 'MIL', 'OKC', 'ORL', 'PHO', 'POR', 'SAC', 'TOR', 'IND', 
                 'LAC', 'NYK', 'CLE', 'DEN', 'PHI', 'SAS', 'NOP', 'WAS', 'CHO', 
                 'MIN', 'DAL', 'UTA'] 
    old_names_opponent = ['Atlanta', 'Chicago', 'Golden State', 'Boston', 'Brooklyn',
                          'Detroit', 'Houston', 'L.A. Lakers', 'Memphis', 'Miami', 
                          'Milwaukee', 'Oklahoma City', 'Orlando', 'Phoenix', 'Portland',
                          'Sacramento', 'Toronto', 'Indiana',  'L.A. Clippers', 'New York', 
                          'Cleveland', 'Denver', 'Philadelphia', 'San Antonio', 'New Orleans', 
                          'Washington', 'Charlotte', 'Minnesota', 'Dallas', 'Utah'] 
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
    df_teams_covers['team'].replace(old_names, new_names, inplace=True)
    df_teams_covers['opponent'].replace(old_names_opponent, new_names, inplace=True)
    df_teams_covers_filtered = df_teams_covers[['date', 'team', 'opponent', 'spread',
    'totals', 'venue', 'score_team', 'score_oppt', 'team_predicted_points', 'oppt_predicted_points', 
    'spread_expanding_mean', 'current_spread_vs_spread_expanding', 'beat_spread', 
    'beat_spread_rolling_mean_11', 'beat_spread_rolling_std_11', 'beat_spread_last_g']]
    return df_teams_covers_filtered

df_teams_covers_filtered_2015 = fix_names_covers_2(df_teams_covers_2015)
len(df_teams_covers_filtered_2015['team'].unique())

df_teams_covers_filtered_2014 = fix_names_covers_2(df_teams_covers_2014)
len(df_teams_covers_filtered_2014['team'].unique())

df_teams_covers_filtered_2013 = fix_names_covers_2(df_teams_covers_2013)
len(df_teams_covers_filtered_2013['team'].unique())

df_teams_covers_filtered_2012 = fix_names_covers_2(df_teams_covers_2012)
len(df_teams_covers_filtered_2012['team'].unique())

df_teams_covers_filtered_2011 = fix_names_covers_2(df_teams_covers_2011)
len(df_teams_covers_filtered_2011['team'].unique())

df_teams_covers_filtered_2010 = fix_names_covers_2(df_teams_covers_2010)
len(df_teams_covers_filtered_2010['team'].unique())

df_teams_covers_filtered_2009 = fix_names_covers_2(df_teams_covers_2009)
len(df_teams_covers_filtered_2009['team'].unique())

df_teams_covers_filtered_2008 = fix_names_covers_2(df_teams_covers_2008)
len(df_teams_covers_filtered_2008['team'].unique())

df_teams_covers_filtered_2007 = fix_names_covers_2(df_teams_covers_2007)
len(df_teams_covers_filtered_2007['team'].unique())

df_teams_covers_filtered_2006 = fix_names_covers_2(df_teams_covers_2006)
len(df_teams_covers_filtered_2006['team'].unique())

df_teams_covers_filtered_2005 = fix_names_covers_2(df_teams_covers_2005)
len(df_teams_covers_filtered_2005['team'].unique())

df_teams_covers_filtered_2004 = fix_names_covers_2(df_teams_covers_2004)
len(df_teams_covers_filtered_2004['team'].unique())


# create season variable
for year, df in zip(['2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005', '2004'], 
                    [df_teams_covers_filtered_2015, df_teams_covers_filtered_2014, df_teams_covers_filtered_2013, 
                     df_teams_covers_filtered_2012, df_teams_covers_filtered_2011,
                     df_teams_covers_filtered_2010, df_teams_covers_filtered_2009, df_teams_covers_filtered_2008, 
                     df_teams_covers_filtered_2007, df_teams_covers_filtered_2006, df_teams_covers_filtered_2005,
                     df_teams_covers_filtered_2004]):
    df['season_start'] = year
#df_teams_covers_filtered_2007[['team', 'date', 'season_start']].tail(10)

df_covers_2004_2015 = pd.concat([df_teams_covers_filtered_2015, df_teams_covers_filtered_2014, df_teams_covers_filtered_2013, 
                                 df_teams_covers_filtered_2012, df_teams_covers_filtered_2011,
                                 df_teams_covers_filtered_2010, df_teams_covers_filtered_2009, df_teams_covers_filtered_2008, 
                                 df_teams_covers_filtered_2007, df_teams_covers_filtered_2006, df_teams_covers_filtered_2005,
                                 df_teams_covers_filtered_2004],
                                 ignore_index=True)

len(df_covers_2004_2015)  # 29038


#------------------------------------------------------------------------------
# scrape bball reference 

# go to http://www.basketball-reference.com/leagues/NBA_2016_games.html
# when view souruce can see the box score link for ea game
# go to that, then there are four 'bold_text stat_total' and for each one, get stats

#teamWebpage = 'http://www.basketball-reference.com/teams/ATL/2016.html?lid=header_teams'
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2016_games.html'
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2015_games.html'
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2014_games.html'
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2013_games.html'  # same as the 2012-13 season in covers
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2012_games.html'  # same as the 2011-12 season in covers
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2011_games.html' 
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2010_games.html' 
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2009_games.html' 
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2008_games.html' 
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2007_games.html' 
#season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2006_games.html' 
season_webpage = 'http://www.basketball-reference.com/leagues/NBA_2005_games.html' 

def create_gamelog_links_and_player_list_for_team(season_webpage):
    on_webpage = urllib.request.urlopen(season_webpage)
    html_contents = on_webpage.read()
    on_webpage.close()        
    soupObject = BeautifulSoup(html_contents, 'html.parser')  
    a_tags = soupObject.find_all('a')
    box_score_links_list = []
    for link in a_tags:
        if link.text == 'Box Score':
            a_tag_link = link.get('href')
            box_score_links_list.append(a_tag_link)
        else:
            None
    print(len(box_score_links_list))
    print(box_score_links_list[:5])
    return box_score_links_list
    
box_score_links_list = create_gamelog_links_and_player_list_for_team(season_webpage)


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
  

def get_starters_for_box_score(soupObject):
    # get starters for ea team:
    table_bodies = soupObject.find_all('tbody')
    th_tag_team1 = soupObject.find_all('th', text='Starters')[0]
    player_rows1 = th_tag_team1.parent.next_sibling.next_element.next_element.find_all('tr')
    starters_list_team1 = [player_rows1[i].td.text for i in range(0,5)]
    starters_list_team1.sort()
    starters_string_team1 = ' '.join(starters_list_team1)
    th_tag_team2 = soupObject.find_all('th', text='Starters')[2]
    player_rows2 = th_tag_team2.parent.next_sibling.next_element.next_element.find_all('tr')
    starters_list_team2 = [player_rows2[i].td.text for i in range(0,5)]
    starters_list_team2.sort()
    starters_string_team2 = ' '.join(starters_list_team2)
    return starters_string_team1, starters_string_team2

#starters_string_team1, starters_string_team2 = get_starters_for_box_score(soupObject)


def give_game_stats(soupObject):
    # team 1 
    table_bodies = soupObject.find_all('tfoot')
    team1_basic_stats = table_bodies[0]
    table_data = team1_basic_stats.find_all('td')
    team1_basic_stats_list = [float(stat.text) for stat in table_data[1:-1]]
    team1_adv_stats = table_bodies[1]
    table_data = team1_adv_stats.find_all('td')
    team1_adv_stats_list = [float(stat.text) for stat in table_data[1:]]
    # team 2 
    team2_basic_stats = table_bodies[2]
    table_data = team2_basic_stats.find_all('td')
    team2_basic_stats_list = [float(stat.text) for stat in table_data[1:-1]]
    team2_adv_stats = table_bodies[3]
    table_data = team2_adv_stats.find_all('td')
    team2_adv_stats_list = [float(stat.text) for stat in table_data[1:]]
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
        'home_team_tov', 'home_team_pf', 'home_team_pts', 'home_team_MP', 'home_team_TS%', 
        'home_team_eFG%', 'home_team_3PAr', 'home_team_FTr', 'home_team_ORB%', 
        'home_team_DRB%', 'home_team_TRB%', 'home_team_AST%', 'home_team_STL%', 
        'home_team_BLK%', 'home_team_TOV%', 'home_team_USG%', 'home_team_ORtg', 'home_team_DRtg',
        'away_team_mp', 'away_team_fg', 'away_team_fga', 'away_team_fg_pct', 
        'away_team_fg3', 'away_team_fg3a', 'away_team_fg3_pct', 'away_team_ft', 
        'away_team_fta', 'away_team_ft_pct', 'away_team_orb', 'away_team_drb', 
        'away_team_trb', 'away_team_ast', 'away_team_stl', 'away_team_blk',
        'away_team_tov', 'away_team_pf', 'away_team_pts', 'away_team_MP', 
        'away_team_TS%', 'away_team_eFG%', 'away_team_3PAr', 'away_team_FTr', 
        'away_team_ORB%', 'away_team_DRB%', 'away_team_TRB%', 'away_team_AST%', 
        'away_team_STL%', 'away_team_BLK%', 'away_team_TOV%', 'away_team_USG%', 
        'away_team_ORtg', 'away_team_DRtg', 'starters_away_team', 'starters_home_team']

#dftest = pd.DataFrame([[date] + [home_team] + [away_team] + team1_basic_stats_list +
#team1_adv_stats_list + team2_basic_stats_list + team2_adv_stats_list +
#[starters_string_team1] + [starters_string_team2]], columns = cols)


def get_data_from_all_box_scores(box_score_links_list):
    df_all_games = pd.DataFrame()
    counter = 0
    for link in box_score_links_list[:]:
        counter = counter + 1
        print(counter)
        soupObject = get_box_score_soup_object(link)
        starters_string_team1, starters_string_team2 = get_starters_for_box_score(soupObject)
        team1_basic_stats_list, team1_adv_stats_list, team2_basic_stats_list, team2_adv_stats_list = give_game_stats(soupObject)
        date, home_team, away_team = get_teams_and_date(soupObject)
        df_game = pd.DataFrame([[date] + [home_team] + [away_team] + team1_basic_stats_list +
                                team1_adv_stats_list + team2_basic_stats_list + team2_adv_stats_list +
                                [starters_string_team1] + [starters_string_team2]], columns = cols)
        df_all_games = pd.concat([df_all_games, df_game], ignore_index=True)
    return df_all_games
    
df_all_games = get_data_from_all_box_scores(box_score_links_list)
  
# format date:
df_all_games['date'] = pd.to_datetime(df_all_games['date'])


#--------------------------------------------------------------
# save bball ref dfs -- make sure to name w correct season year
df_all_games.to_pickle('df_bball_ref_2004_05.pkl')
df_all_games_2004 = pd.read_pickle('df_bball_ref_2004_05.pkl')
df_all_games_2004[df_all_games_2004['date'] == '2005-04-10']

df_all_games.to_pickle('df_bball_ref_2005_06.pkl')
df_all_games_2005 = pd.read_pickle('df_bball_ref_2005_06.pkl')
df_all_games_2005[df_all_games_2005['date'] == '2006-04-11']
df_all_games_2005['date'] = pd.to_datetime(df_all_games_2005['date'])
df_all_games_2005.to_pickle('df_bball_ref_2005_06.pkl')
# MIGHT HAVE TO USE T0_DATETIME ON ANY YEARS PRIOR TO THIS

df_all_games.to_pickle('df_bball_ref_2006_07.pkl')
df_all_games_2006 = pd.read_pickle('df_bball_ref_2006_07.pkl')
df_all_games_2006['date']

df_all_games.to_pickle('df_bball_ref_2007_08.pkl')
df_all_games_2007 = pd.read_pickle('df_bball_ref_2007_08.pkl')

df_all_games.to_pickle('df_bball_ref_2008_09.pkl')
df_all_games_2008 = pd.read_pickle('df_bball_ref_2008_09.pkl')

df_all_games.to_pickle('df_bball_ref_2009_10.pkl')
df_all_games_2009 = pd.read_pickle('df_bball_ref_2009_10.pkl')

df_all_games.to_pickle('df_bball_ref_2010_11.pkl')
df_all_games_2010 = pd.read_pickle('df_bball_ref_2010_11.pkl')

df_all_games.to_pickle('df_bball_ref_2011_12.pkl')
df_all_games_2011 = pd.read_pickle('df_bball_ref_2011_12.pkl')

df_all_games.to_pickle('df_bball_ref_2012_13.pkl')
df_all_games_2012 = pd.read_pickle('df_bball_ref_2012_13.pkl')
  
df_all_games.to_pickle('df_bball_ref_2013_14.pkl')
df_all_games_2013 = pd.read_pickle('df_bball_ref_2013_14.pkl')
#
df_all_games.to_pickle('df_bball_ref_2014_15.pkl')
df_all_games_2014 = pd.read_pickle('df_bball_ref_2014_15.pkl')

df_all_games.to_pickle('df_bball_ref_2015_16.pkl')
df_all_games_2015 = pd.read_pickle('df_bball_ref_2015_16.pkl')
#--------------------------------------------------------------

# filter to regular season games. for 2008-09:
df_all_games_2004 = df_all_games_2004[df_all_games_2004['date'] < '2005-04-21']
# filter to regular season games. for 2008-09:
df_all_games_2005 = df_all_games_2005[df_all_games_2005['date'] < '2006-04-20']
# filter to regular season games. for 2008-09:
df_all_games_2006 = df_all_games_2006[df_all_games_2006['date'] < '2007-04-19']
# filter to regular season games. for 2008-09:
df_all_games_2007 = df_all_games_2007[df_all_games_2007['date'] < '2008-04-17']
# filter to regular season games. for 2008-09:
df_all_games_2008 = df_all_games_2008[df_all_games_2008['date'] < '2009-04-16']
# filter to regular season games. for 2009-10:
df_all_games_2009 = df_all_games_2009[df_all_games_2009['date'] < '2010-04-15']
# filter to regular season games. for 2010-11:
df_all_games_2010 = df_all_games_2010[df_all_games_2010['date'] < '2011-04-15']
# filter to regular season games. for 2011-12:
df_all_games_2011 = df_all_games_2011[df_all_games_2011['date'] < '2012-04-27']
# filter to regular season games. for 2012-13:
df_all_games_2012 = df_all_games_2012[df_all_games_2012['date'] < '2013-04-18']
# filter to regular season games. for 2013-14:
df_all_games_2013 = df_all_games_2013[df_all_games_2013['date'] < '2014-04-17']
# filter to regular season games. for 2014-15:
df_all_games_2014 = df_all_games_2014[df_all_games_2014['date'] < '2015-04-16']
# filter to regular season games. for 2015-16:
df_all_games_2015 = df_all_games_2015[df_all_games_2015['date'] < '2016-04-14']
#df_all_games = df_all_games.rename(columns={'home_team':'team'})
#df_all_games = df_all_games.rename(columns={'away_team':'opponent'})


df_bball_ref_2004_2015 = pd.concat([df_all_games_2004, df_all_games_2005, df_all_games_2006,
                                    df_all_games_2007, df_all_games_2008, 
                                    df_all_games_2009, df_all_games_2010, 
                                    df_all_games_2011, df_all_games_2012,
                                    df_all_games_2013, df_all_games_2014,
                                    df_all_games_2015], ignore_index=True)
len(df_bball_ref_2004_2015)  # 14519


def change_columns(df_all_games):
    # change columns to team and oppt
    df_all_games.columns = ['date', 'team', 'opponent', 'team_mp', 'team_fg',
           'team_fga', 'team_fg_pct', 'team_fg3', 'team_fg3a',
           'team_fg3_pct', 'team_ft', 'team_fta',
           'team_ft_pct', 'team_orb', 'team_drb', 'team_trb',
           'team_ast', 'team_stl', 'team_blk', 'team_tov',
           'team_pf', 'team_pts', 'team_MP', 'team_TS%',
           'team_eFG%', 'team_3PAr', 'team_FTr', 'team_ORB%',
           'team_DRB%', 'team_TRB%', 'team_AST%', 'team_STL%',
           'team_BLK%', 'team_TOV%', 'team_USG%', 'team_ORtg',
           'team_DRtg', 'opponent_mp', 'opponent_fg', 'opponent_fga',
           'opponent_fg_pct', 'opponent_fg3', 'opponent_fg3a',
           'opponent_fg3_pct', 'opponent_ft', 'opponent_fta',
           'opponent_ft_pct', 'opponent_orb', 'opponent_drb', 'opponent_trb',
           'opponent_ast', 'opponent_stl', 'opponent_blk', 'opponent_tov',
           'opponent_pf', 'opponent_pts', 'opponent_MP', 'opponent_TS%',
           'opponent_eFG%', 'opponent_3PAr', 'opponent_FTr', 'opponent_ORB%',
           'opponent_DRB%', 'opponent_TRB%', 'opponent_AST%', 'opponent_STL%',
           'opponent_BLK%', 'opponent_TOV%', 'opponent_USG%', 'opponent_ORtg',
           'opponent_DRtg', 'starters_opponent', 'starters_team']
    df_all_games['venue'] = 0
    return df_all_games

df_bball_ref_2004_2015 = change_columns(df_bball_ref_2004_2015)


def create_df_w_team_oppt_switched(df_all_games):
    df_all_games_switched = df_all_games.copy(deep=True)
    df_all_games_switched['venue'] = 1
    df_all_games_switched.columns = ['date', 'opponent', 'team', 'opponent_mp', 'opponent_fg',
           'opponent_fga', 'opponent_fg_pct', 'opponent_fg3', 'opponent_fg3a',
           'opponent_fg3_pct', 'opponent_ft', 'opponent_fta',
           'opponent_ft_pct', 'opponent_orb', 'opponent_drb', 'opponent_trb',
           'opponent_ast', 'opponent_stl', 'opponent_blk', 'opponent_tov',
           'opponent_pf', 'opponent_pts', 'opponent_MP', 'opponent_TS%',
           'opponent_eFG%', 'opponent_3PAr', 'opponent_FTr', 'opponent_ORB%',
           'opponent_DRB%', 'opponent_TRB%', 'opponent_AST%', 'opponent_STL%',
           'opponent_BLK%', 'opponent_TOV%', 'opponent_USG%', 'opponent_ORtg',
           'opponent_DRtg', 'team_mp', 'team_fg', 'team_fga',
           'team_fg_pct', 'team_fg3', 'team_fg3a',
           'team_fg3_pct', 'team_ft', 'team_fta',
           'team_ft_pct', 'team_orb', 'team_drb', 'team_trb',
           'team_ast', 'team_stl', 'team_blk', 'team_tov',
           'team_pf', 'team_pts', 'team_MP', 'team_TS%',
           'team_eFG%', 'team_3PAr', 'team_FTr', 'team_ORB%',
           'team_DRB%', 'team_TRB%', 'team_AST%', 'team_STL%',
           'team_BLK%', 'team_TOV%', 'team_USG%', 'team_ORtg',
           'team_DRtg', 'starters_team', 'starters_opponent', 'venue']
    return df_all_games_switched

df_bball_ref_2004_2015_switched = create_df_w_team_oppt_switched(df_bball_ref_2004_2015)


def double_games(df_all_games, df_all_games_switched):
    df_games_doubled = pd.concat([df_all_games, df_all_games_switched], ignore_index=True)
    df_games_doubled_filtered = df_games_doubled[['date', 'team', 'opponent', 'venue', 
    'starters_team', 'starters_opponent', 'team_3PAr', 'team_AST%', 'team_BLK%', 'team_DRB%', 
    'team_DRtg', 'team_FTr', 'team_ORB%', 'team_ORtg', 'team_STL%', 'team_TOV%', 'team_TRB%', 
    'team_TS%', 'team_eFG%', 'team_fg3_pct', 'team_fg_pct', 'team_ft_pct', 'team_pf', 
    'opponent_3PAr', 'opponent_AST%', 'opponent_BLK%', 'opponent_DRB%', 'opponent_DRtg', 'opponent_FTr', 
    'opponent_ORB%', 'opponent_ORtg', 'opponent_STL%', 'opponent_TOV%', 'opponent_TRB%', 'opponent_TS%',
    'opponent_eFG%', 'opponent_fg3_pct', 'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf']]
    print(len(df_games_doubled_filtered))
    print(len(df_games_doubled_filtered['team'].unique()))
    print(len(df_games_doubled_filtered['opponent'].unique()))
    print(len(df_games_doubled_filtered['starters_team'].unique()))
    print(len(df_games_doubled_filtered['starters_opponent'].unique()))
    return df_games_doubled_filtered

df_bball_ref_2004_2015_doubled = double_games(df_bball_ref_2004_2015, df_bball_ref_2004_2015_switched)
df_bball_ref_2004_2015_doubled.tail()

df_bball_ref_2004_2015_doubled['team'].unique()

# merge bbal ref stats df w spread covers df
# first, replace old team names in bball ref w current ones (used by covers)
def replace_bball_ref_names(df_bball_ref):
    df_bball_ref['team'].replace(['New Orleans Hornets', 'Charlotte Bobcats', 'New Jersey Nets', 'Seattle SuperSonics'],
    ['New Orleans Pelicans', 'Charlotte Hornets', 'Brooklyn Nets', 'Oklahoma City Thunder'], inplace=True)
    df_bball_ref['opponent'].replace(['New Orleans Hornets', 'Charlotte Bobcats', 'New Jersey Nets', 'Seattle SuperSonics'],
    ['New Orleans Pelicans', 'Charlotte Hornets', 'Brooklyn Nets', 'Oklahoma City Thunder'], inplace=True)
    df_bball_ref['team'].replace('New Orleans/Oklahoma City Hornets', 'New Orleans Pelicans', inplace=True)
    df_bball_ref['opponent'].replace('New Orleans/Oklahoma City Hornets', 'New Orleans Pelicans', inplace=True)
    return df_bball_ref

df_bball_ref_2004_2015_doubled = replace_bball_ref_names(df_bball_ref_2004_2015_doubled)
len(df_bball_ref_2004_2015_doubled['team'].unique())

len(df_covers_2004_2015)  # 29038
len(df_bball_ref_2004_2015_doubled)  # 29038
df_bball_ref_2004_2015_doubled.columns
len(df_bball_ref_2004_2015_doubled['team'].unique())  # 30

#df_merge_test = pd.merge(df_covers_2005_2015, df_bball_ref_2005_2015_doubled, on=['team', 'date'], how='outer')
#len(df_merge_test)  # 26578
#df_merge_test[['team', 'date', 'opponent_x', 'opponent_y', 'spread', 'team_3PAr']].tail()


df_covers_bball_ref = df_bball_ref_2004_2015_doubled.merge(df_covers_2004_2015, on=['date', 'opponent', 'team'], how='left')
df_covers_bball_ref[['team', 'opponent', 'date','score_team', 'spread']].head(20)
len(df_covers_bball_ref)  # 29038

# create weighted or rolling or expanding metrics to use for prediction
                 
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

variables = ['team_3PAr', 'team_AST%', 'team_BLK%', 'team_DRB%', 'team_DRtg', 
    'team_FTr', 'team_ORB%', 'team_ORtg', 'team_STL%', 'team_TOV%', 'team_TRB%', 
    'team_TS%', 'team_eFG%', 'team_fg3_pct', 'team_fg_pct', 'team_ft_pct', 'team_pf', 
    'opponent_3PAr', 'opponent_AST%', 'opponent_BLK%', 'opponent_DRB%', 'opponent_DRtg', 'opponent_FTr', 
    'opponent_ORB%', 'opponent_ORtg', 'opponent_STL%', 'opponent_TOV%', 'opponent_TRB%', 'opponent_TS%',
    'opponent_eFG%', 'opponent_fg3_pct', 'opponent_fg_pct', 'opponent_ft_pct', 'opponent_pf']           
 

# save an upload these to wikari so can use them there -- try pickled one first
df_covers_bball_ref.to_csv('df_covers_bball_ref_2004_to_2015.csv')
df_covers_bball_ref.to_pickle('df_covers_bball_ref_2004_to_2015.pkl')





def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables):
    df_covers_bball_ref = df_covers_bball_ref.sort('date')
    for var in variables:
        var_ewma = var + '_ewma_15'
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby(['season_start', 'team'])[var].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables)


def loop_through_lineups_to_create_rolling_metrics(df_covers_bball_ref, variables):
    df_covers_bball_ref = df_covers_bball_ref.sort('date')
    for var in variables:
        var_ewma = var + '_ewma_15'
        #df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('starters_team')[var].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=10))
        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('starters_team')[var].transform(lambda x: pd.ewma(x.shift(1), span=8))
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_lineups_to_create_rolling_metrics(df_covers_bball_ref, variables)

     
#df_covers_bball_ref.to_pickle('df_covers_bball_ref_metrics_for_lineups.csv')
#df_covers_bball_ref = pd.read_pickle('df_covers_bball_ref_metrics_for_lineups.csv')

df_covers_bball_ref.columns[100:]
df_covers_bball_ref = df_covers_bball_ref[['date', 'team', 'opponent', 'venue_x', 'starters_team',
       'starters_opponent', 'team_3PAr', 'team_AST%', 'team_BLK%', 'team_DRB%',
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
       'opponent_pf_ewma_15']]

# change name of starters_team and starters_opponent to team and opponent
# so that all subsequent code will treat these as the teams and opponent
def substitute_linups_for_team_names(df_covers_bball_ref):
    df_covers_bball_ref['team_actual'] = df_covers_bball_ref['team']
    df_covers_bball_ref['opponent_actual'] = df_covers_bball_ref['opponent']
    df_covers_bball_ref['team'] = df_covers_bball_ref['starters_team']
    df_covers_bball_ref['opponent'] = df_covers_bball_ref['starters_opponent']
#    df_covers_bball_ref.rename(columns={'starters_team':'team'}, inplace=True)
#    df_covers_bball_ref.rename(columns={'starters_opponent':'opponent'}, inplace=True)
    df_covers_bball_ref[['team_actual', 'opponent_actual', 'team', 'opponent', 'starters_team']].head()
    return df_covers_bball_ref
    
df_covers_bball_ref = substitute_linups_for_team_names(df_covers_bball_ref)
for col in df_covers_bball_ref.columns:
    print(col)
    
def create_switched_df(df_all_teams):
    # create df with team and opponent swithced (so can then merge the team's 
    # weighted/rolling metrics onto the original df but as the opponents)
    df_all_teams_swtiched = df_all_teams.copy(deep=True)
    df_all_teams_swtiched.rename(columns={'team':'opponent_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent':'team_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent_hold':'opponent'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'team_hold':'team'}, inplace=True)
    df_all_teams_swtiched = df_all_teams_swtiched[['date', 'opponent', 'team', 
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
       'opponent_pf_ewma_15']]
    return df_all_teams_swtiched

df_covers_bball_ref_switched = create_switched_df(df_covers_bball_ref)
for col in df_covers_bball_ref_switched:
    print(col)

def preface_oppt_stats_in_switched_df(df_all_teams_swtiched):
    # preface all these stats -- they belong to the team in this df but to
    # the opponent in the orig df -- with an x_. then when merge back onto original df
    # these stats will be for the opponent in that df. and that's what i'll use to predict ats
    df_all_teams_swtiched.columns=['date', 'opponent', 'team', 
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
           'x_opponent_pf_ewma_15']    
    return df_all_teams_swtiched   
           
df_covers_bball_ref_switched = preface_oppt_stats_in_switched_df(df_covers_bball_ref_switched)
df_covers_bball_ref_switched['x_spread_expanding_mean']

def merge_regular_df_w_switched_df(df_all_teams, df_all_teams_swtiched):    
    df_all_teams_w_ivs = df_all_teams.merge(df_all_teams_swtiched, on=['date', 'team', 'opponent'], how='left')
    df_all_teams_w_ivs.head(50)
    return df_all_teams_w_ivs

df_covers_bball_ref = merge_regular_df_w_switched_df(df_covers_bball_ref, df_covers_bball_ref_switched)    

for col in df_covers_bball_ref.columns:
    print(col)


# save file
#df_covers_bball_ref.to_pickle('df_covers_bball_ref_2005_2014.pkl')
#df_covers_bball_ref = pd.read_pickle('df_covers_bball_ref_2005_2014.pkl')

# create substraction vars
variables = ['spread_expanding_mean', 'current_spread_vs_spread_expanding',
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
       'opponent_pf_ewma_15']

def create_team_opponent_difference_variables(df_all_teams_w_ivs, variables):
    for var in variables:
        new_difference_variable = 'difference_'+var
        df_all_teams_w_ivs[new_difference_variable] = df_all_teams_w_ivs[var] - df_all_teams_w_ivs['x_'+var]
    return df_all_teams_w_ivs

df_covers_bball_ref = create_team_opponent_difference_variables(df_covers_bball_ref, variables)
df_covers_bball_ref.head()


def create_basic_variables(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
    df_all_teams_w_ivs['ats_win'] = np.nan
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] > 0, 'ats_win'] = 1
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] < 0, 'ats_win'] = 0
    df_all_teams_w_ivs[['date', 'team', 'opponent', 'beat_spread', 'ats_win']]
    df_all_teams_w_ivs['win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'win'] = 1
    df_all_teams_w_ivs[['score_team', 'score_oppt', 'win']]
    df_all_teams_w_ivs['point_difference'] = df_all_teams_w_ivs['score_team'] - df_all_teams_w_ivs['score_oppt']
    return df_all_teams_w_ivs

df_covers_bball_ref = create_basic_variables(df_covers_bball_ref)
len(df_covers_bball_ref)


def create_home_df(df_covers_bball_ref):
    df_covers_bball_ref_home = df_covers_bball_ref[df_covers_bball_ref['venue_x'] == 0]
    df_covers_bball_ref_home = df_covers_bball_ref_home.reset_index()
    len(df_covers_bball_ref_home)
    return df_covers_bball_ref_home

df_covers_bball_ref_home = create_home_df(df_covers_bball_ref)
len(df_covers_bball_ref_home)
df_covers_bball_ref_home.head()

#dates = df_all_teams_w_ivs_home[['date']].sort('date')
#dates = dates.drop_duplicates(subset='date')
#rank_order = range(len(dates))
#date_to_rank_dict = dict(zip(dates['date'], rank_order))
#df_all_teams_w_ivs_home['date_rank'] = df_all_teams_w_ivs_home['date'].map(date_to_rank_dict)
#df_all_teams_w_ivs_home[['date', 'date_rank']] 

#plt.scatter(df_all_teams_w_ivs['spread'], df_all_teams_w_ivs['point_difference'], alpha=.3)
#plt.ylim(-30, 30)
#plt.xlim(-30, 30)
#
#sns.lmplot('spread', 'point_difference', data=df_all_teams_w_ivs)
#plt.ylim(-30, 30)
#plt.xlim(-30, 30)

df_covers_bball_ref_home['difference_team_AST%_ewma_15']

#------------------------------------------------------------------------------
# difference variables
# ml to predict -- ea row is one game w both team's stats
all_iv_vars = ['difference_spread_expanding_mean', #'difference_current_spread_vs_spread_expanding',
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
       'difference_opponent_pf_ewma_15', 'spread', 'totals']  #* using


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


all_iv_vars = ['difference_spread_expanding_mean', #'difference_current_spread_vs_spread_expanding',
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
       'difference_opponent_pf_ewma_15', 'spread', 'totals']  #* using

all_iv_vars = ['difference_team_ASTpct_ewma_15', 'difference_team_FTr_ewma_15',
       'difference_team_pf_ewma_15', 'difference_opponent_FTr_ewma_15',
       'difference_opponent_pf_ewma_15', 'spread']  #* using


# regular variables
# ml to predict -- ea row is one game w both team's stats
#opponent_iv_vars = ['x_spread_expanding_mean', 'x_beat_spread_rolling_mean_11',
#        'x_team_3PAr_ewma_15', 'x_team_AST%_ewma_15', 'x_team_BLK%_ewma_15', 'x_team_DRB%_ewma_15',
#       'x_team_DRtg_ewma_15', 'x_team_FTr_ewma_15', 'x_team_ORB%_ewma_15',
#       'x_team_ORtg_ewma_15', 'x_team_STL%_ewma_15', 'x_team_TOV%_ewma_15',
#       'x_team_TRB%_ewma_15', 'x_team_TS%_ewma_15', 'x_team_eFG%_ewma_15',
#       'x_team_fg3_pct_ewma_15', 'x_team_fg_pct_ewma_15', 'x_team_ft_pct_ewma_15',
#       'x_team_pf_ewma_15', 'x_opponent_3PAr_ewma_15', 'x_opponent_AST%_ewma_15',
#       'x_opponent_BLK%_ewma_15', 'x_opponent_DRB%_ewma_15', 'x_opponent_DRtg_ewma_15', 
#       'x_opponent_FTr_ewma_15', 'x_opponent_ORB%_ewma_15', 'x_opponent_ORtg_ewma_15',
#       'x_opponent_STL%_ewma_15', 'x_opponent_TOV%_ewma_15', 'x_opponent_TRB%_ewma_15', 
#       'x_opponent_TS%_ewma_15', 'x_opponent_eFG%_ewma_15', 'x_opponent_fg3_pct_ewma_15',
#       'x_opponent_fg_pct_ewma_15', 'x_opponent_ft_pct_ewma_15', 'x_opponent_pf_ewma_15'] 
#
#team_iv_vars = ['date_rank', 'spread_expanding_mean', 'beat_spread_rolling_mean_11', 
#        'team_3PAr_ewma_15', 'team_AST%_ewma_15', 'team_BLK%_ewma_15', 'team_DRB%_ewma_15',
#       'team_DRtg_ewma_15', 'team_FTr_ewma_15', 'team_ORB%_ewma_15',
#       'team_ORtg_ewma_15', 'team_STL%_ewma_15', 'team_TOV%_ewma_15',
#       'team_TRB%_ewma_15', 'team_TS%_ewma_15', 'team_eFG%_ewma_15',
#       'team_fg3_pct_ewma_15', 'team_fg_pct_ewma_15', 'team_ft_pct_ewma_15',
#       'team_pf_ewma_15', 'opponent_3PAr_ewma_15', 'opponent_AST%_ewma_15',
#       'opponent_BLK%_ewma_15', 'opponent_DRB%_ewma_15', 'opponent_DRtg_ewma_15', 
#       'opponent_FTr_ewma_15', 'opponent_ORB%_ewma_15', 'opponent_ORtg_ewma_15',
#       'opponent_STL%_ewma_15', 'opponent_TOV%_ewma_15', 'opponent_TRB%_ewma_15', 
#       'opponent_TS%_ewma_15', 'opponent_eFG%_ewma_15', 'opponent_fg3_pct_ewma_15',
#       'opponent_fg_pct_ewma_15', 'opponent_ft_pct_ewma_15', 'opponent_pf_ewma_15'] 

#all_iv_vars = team_iv_vars + opponent_iv_vars
len(all_iv_vars)

dv_var = 'ats_win'  #* using
dv_var = 'win'
dv_var = 'point_difference'

iv_and_dv_vars = all_iv_vars + [dv_var] + ['team', 'opponent', 'date']
df_covers_bball_ref_home[iv_and_dv_vars].head(10)
len(df_covers_bball_ref_home)

df_covers_bball_ref__dropna_home = df_covers_bball_ref_home.dropna()
len(df_covers_bball_ref__dropna_home)

sns.set_style('white')

# plan: 
# use cross val and build best model on 2007 to 2011, tweak random forest 
# parameters, etc. then test on 2012 season.
df_covers_bball_ref_home_train = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] != '2015']
len(df_covers_bball_ref_home_train)
df_covers_bball_ref_home_test = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start'] == '2015']
len(df_covers_bball_ref_home_test)

params = {'n_estimators': 500, 'min_samples_split': 50,
          'learning_rate': 0.01, 'subsample': .25}

model = ensemble.GradientBoostingRegressor(**params)

model = ensemble.GradientBoostingClassifier(**params)

# gradient boosting above seems to out-perform random forest below.
# also, seem to get just about same mse whether using all the vars or just a handful of good ones
# i think that setting min split size to 50 means that when there's an n of below 50, it wont attemp to split it again. so limits the size of the tree
# and min sampes leaf means that it won't split a node too unevenly, i.e., won't make a decision that sends 2 people down one branch, and 150 down the other (if set to 5, would only send 5 or more down the branch)
model = RandomForestRegressor(n_estimators = 500, #oob_score = True,  # see if faster wout oob_score
    max_features = (len(all_iv_vars) - 1), min_samples_leaf = 5, min_samples_split = 50, max_depth=5)  

model = RandomForestClassifier(n_estimators = 500, #oob_score = True,  # see if faster wout oob_score
    max_features = (len(all_iv_vars) - 1), min_samples_leaf = 5, min_samples_split = 50, max_depth = 5)   # when i didn't set max depth, it took forever. 

model = tree.DecisionTreeClassifier(max_features = (len(all_iv_vars) - 1), min_samples_leaf = 5, 
                                    min_samples_split = 50, max_depth = 5)  # here, max depth really helps

model = tree.DecisionTreeClassifier(max_features=min(1602, len(all_iv_vars) - 1), max_depth=1)

model = linear_model.Ridge(alpha=.001, normalize=True)

model = linear_model.LinearRegression()

model = LogisticRegression(C=10) #the smaller the C, the more regularization


# to get cross val scoring metric:

mean_sq_error = cross_val_score(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], cv=10, scoring='mean_squared_error')

plt.plot([np.abs(mean_sq_error)[0]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[1]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[2]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[3]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[4]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[5]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[6]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[7]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[8]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.abs(mean_sq_error)[9]]*5, color='aqua', linewidth=1, alpha=.4)
plt.plot([np.mean(np.abs(mean_sq_error))]*5, linewidth=5, color='aqua', alpha=.8)
plt.ylim(120, 150)
plt.grid(axis='y', linestyle='--')
plt.ylabel('mean square error', fontsize=15)
plt.xticks([])
sns.despine()

# see how it looks for 2015 season
# fit model on all seasons prior to 2015
model.fit(df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var])
# get mse using that model to predict pt diff in 2015 games
mse = mean_squared_error(df_covers_bball_ref_home_test[dv_var], model.predict(df_covers_bball_ref_home_test[all_iv_vars]))
# resulted in 133.93. Nice!!!  w ridge regression: 133.15!!
mae = mean_absolute_error(df_covers_bball_ref_home_test[dv_var], model.predict(df_covers_bball_ref_home_test[all_iv_vars]))
# resulted in 9.13. w ridge regression: 9.10!!



# look at scatterplots and corrs between ivs? too much, just need to do factor reduction?
my_vars = ['point_difference', 'difference_spread_expanding_mean',
 'difference_beat_spread_last_g',
 'difference_team_3PAr_ewma_15',
 'difference_team_ASTpct_ewma_15',
 'difference_team_BLKpct_ewma_15',
 'difference_team_DRBpct_ewma_15',
 'difference_team_DRtg_ewma_15',
 'difference_team_FTr_ewma_15',
 'difference_team_ORBpct_ewma_15',
 'difference_team_ORtg_ewma_15',
 'difference_team_STLpct_ewma_15',
 'difference_team_TOVpct_ewma_15',
 'difference_team_TRBpct_ewma_15',
 'difference_team_TSpct_ewma_15',
 'difference_team_eFGpct_ewma_15',
 'difference_team_fg3_pct_ewma_15',
 'difference_team_fg_pct_ewma_15',
 'difference_team_ft_pct_ewma_15',
 'difference_team_pf_ewma_15',
 'difference_opponent_3PAr_ewma_15',
 'difference_opponent_ASTpct_ewma_15',
 'difference_opponent_BLKpct_ewma_15',
 'difference_opponent_DRBpct_ewma_15',
 'difference_opponent_DRtg_ewma_15',
 'difference_opponent_FTr_ewma_15',
 'difference_opponent_ORBpct_ewma_15',
 'difference_opponent_ORtg_ewma_15',
 'difference_opponent_STLpct_ewma_15',
 'difference_opponent_TOVpct_ewma_15',
 'difference_opponent_TRBpct_ewma_15',
 'difference_opponent_TSpct_ewma_15',
 'difference_opponent_eFGpct_ewma_15',
 'difference_opponent_fg3_pct_ewma_15',
 'difference_opponent_fg_pct_ewma_15',
 'difference_opponent_ft_pct_ewma_15',
 'difference_opponent_pf_ewma_15',
 'spread', 'totals']

df_covers_bball_ref_home_train[my_vars[:10]].corr()


pd.scatter_matrix(df_covers_bball_ref_home_train[my_vars[:3]], alpha=0.025)
plt.scatter(df_covers_bball_ref_home_train['spread_expanding_mean'], df_covers_bball_ref_home_train['point_difference'], alpha=.05)
# fit a linear, quadratic, and cubic through this -- which fits best?


df_covers_bball_ref_home_train['spread_expanding_squared'] = df_covers_bball_ref_home_train['spread_expanding_mean'] * df_covers_bball_ref_home_train['spread_expanding_mean'] 
df_covers_bball_ref_home_test['spread_expanding_squared'] = df_covers_bball_ref_home_test['spread_expanding_mean'] * df_covers_bball_ref_home_test['spread_expanding_mean'] 
df_covers_bball_ref_home_train['spread_expanding_log'] = np.log(df_covers_bball_ref_home_train['spread_expanding_mean'])

df_covers_bball_ref_home_train[['spread_expanding_mean', 'spread_expanding_squared', 'spread_expanding_log']]
model = linear_model.Ridge()
model.fit(df_covers_bball_ref_home_train[['spread_expanding_mean', 'spread_expanding_squared']], df_covers_bball_ref_home_train['point_difference'])
plt.plot(df_covers_bball_ref_home_train[['spread_expanding_mean']], model.predict(df_covers_bball_ref_home_train[['spread_expanding_mean', 'spread_expanding_squared']].values), linewidth=2)
plt.scatter(df_covers_bball_ref_home_train['spread_expanding_mean'], df_covers_bball_ref_home_train['point_difference'], alpha=.05, color='orange')
# weird -- the predicted curve is filled in instead of a regular line. why??

mse_train = mean_squared_error(df_covers_bball_ref_home_train['point_difference'], model.predict(df_covers_bball_ref_home_train[['spread_expanding_mean', 'spread_expanding_squared']]))
mse_test = mean_squared_error(df_covers_bball_ref_home_test['point_difference'], model.predict(df_covers_bball_ref_home_test[['spread_expanding_mean', 'spread_expanding_squared']]))
# weird -- the test mse is lower



m, b = np.polyfit(df_covers_bball_ref_home_train['spread_expanding_mean'].values, df_covers_bball_ref_home_train['point_difference'].values, 1)
plt.plot(df_covers_bball_ref_home_train[['spread_expanding_mean']], df_covers_bball_ref_home_train['point_difference'], '.', alpha=.15)
plt.plot(df_covers_bball_ref_home_train[['spread_expanding_mean']], m*df_covers_bball_ref_home_train[['spread_expanding_mean']] + b, '-')

m, b = np.polyfit(df_covers_bball_ref_home_train[['spread_expanding_mean', 'spread_expanding_squared']], df_covers_bball_ref_home_train['point_difference'].values, 2)
plt.plot(df_covers_bball_ref_home_train[['spread_expanding_mean']], df_covers_bball_ref_home_train['point_difference'], '.', alpha=.15)
plt.plot(df_covers_bball_ref_home_train[['spread_expanding_mean']], m*df_covers_bball_ref_home_train[['spread_expanding_mean']] + b, '-')



degree = 2
mse_train_list = []
for degree in [1, 2, 3, 4, 5, 6, 7]:
    model = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge())
    model.fit(df_covers_bball_ref_home_train[['spread_expanding_mean']], df_covers_bball_ref_home_train['point_difference'])
    mse = mean_squared_error(df_covers_bball_ref_home_train['point_difference'], model.predict(df_covers_bball_ref_home_train[['spread_expanding_mean']]))
    mse_train_list.append(mse)
plt.plot(mse_train_list, label='train')

mse_test_list = []
for degree in [1, 2, 3, 4, 5, 6, 7]:
    model = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge())
    model.fit(df_covers_bball_ref_home_train[['spread_expanding_mean']], df_covers_bball_ref_home_train['point_difference'])
    mse = mean_squared_error(df_covers_bball_ref_home_test['point_difference'], model.predict(df_covers_bball_ref_home_test[['spread_expanding_mean']]))
    mse_test_list.append(mse)
plt.plot(mse_test_list, label='test')
plt.legend()




model_lin = linear_model.LinearRegression()
model_lin.fit(df_covers_bball_ref_home_train[['spread_expanding_mean']].values, df_covers_bball_ref_home_train['point_difference'].values)
mse = mean_squared_error(df_covers_bball_ref_home_train[dv_var], model_lin.predict(df_covers_bball_ref_home_train[['spread_expanding_mean']]))

model_log = linear_model.LinearRegression()
df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['spread_expanding_mean'] != 0]
df_covers_bball_ref_home_train[['date','spread_expanding_mean']][df_covers_bball_ref_home_train['spread_expanding_mean'] == 0]
model_log.fit(np.log(df_covers_bball_ref_home_train[['spread_expanding_mean']].values), df_covers_bball_ref_home_train['point_difference'].values)
mse = mean_squared_error(df_covers_bball_ref_home_train[dv_var], model_log.predict(np.log(df_covers_bball_ref_home_train[['spread_expanding_mean']])))

# look at the corr between ivs and dv. for the strongest predictors, is the
# relationship linear? logarithmic?







# for learning curves:

# for random forest regressors, that don't need scaling
# for regression:
m, train_errors, test_errors = learning_curve(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], train_sizes=[.05, .35, .65, .95], scoring = 'mean_squared_error', cv=5)
# for classification:
m, train_errors, test_errors = learning_curve(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], train_sizes=[.05, .35, .65, .95], scoring = 'accuracy', cv=5)

# for linear regression, use scale vars:
m, train_errors, test_errors = learning_curve(model, scale(df_covers_bball_ref_home_train[all_iv_vars]), df_covers_bball_ref_home_train[dv_var], train_sizes=[.05, .35, .65, .95], scoring = 'mean_squared_error', cv=5)
# for classification
m, train_errors, test_errors = learning_curve(model, scale(df_covers_bball_ref_home_train[all_iv_vars]), df_covers_bball_ref_home_train[dv_var], train_sizes=[.05, .35, .65, .95], scoring = 'accuracy', cv=5)

#m, train_errors, test_errors = learning_curve(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], train_sizes=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], scoring = 'r2', cv=10)
# 'r2' is the default for scroring because it's the default for random forest regressor -- it's the r squared

# plot the error/accuracy:
train_cv_err = np.mean(train_errors,axis=1)
test_cv_err = np.mean(test_errors,axis=1)
train_cv_err = np.abs(train_cv_err)
test_cv_err = np.abs(test_cv_err)
tr, = plt.plot(m,train_cv_err)
ts, = plt.plot(m,test_cv_err)
plt.legend((tr,ts),('training','test'),loc='best')
plt.title('Learning Curve')
plt.ylabel('mean squared error or accuracy')
plt.ylim(.40, .60)

# interesting - for predicting wins, using just a few key vars works better
# gives an extra .5% accuracy. 

# so far the logistic regression wihtout a lot of regularization (c=100)
# works best. it's about 51.5% on ATS. try cutting down to essential vars next.
# then can try some interactions too. but should probably stick with predicting
# point diff and convert that to accuracy by comparing with teh spread. 
# yeah, focus on that for now. interesting that random forest isn't doing any
# better for prediction, even w classification.

# actually, can predict just over 52% w ewma from lineups and logistic on ats





df_covers_bball_ref_home_train[all_iv_vars]
df_covers_bball_ref_home_train[dv_var]

my_tpot = TPOT(generations=30, verbosity=2)  
my_tpot.fit(df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var])  

print(my_tpot.score(df_covers_bball_ref_home_test[all_iv_vars], df_covers_bball_ref_home_test[dv_var]))  
my_tpot.export('tpot_exported_pipeline_win.py')


















# get coeff and p-vals
model = linear_model.Ridge(alpha=10)
model.fit(scale(df_covers_bball_ref_home_train[all_iv_vars]), df_covers_bball_ref_home_train[dv_var])
coefficients = model.coef_
len(coefficients)

for variable, coefficient in zip(all_iv_vars, coefficients):
    print(variable, '\t', coefficient)


results = smf.logit(formula = """ ats_win ~ difference_spread_expanding_mean + difference_beat_spread_rolling_mean_11 + 
difference_beat_spread_rolling_std_11 + difference_beat_spread_last_g + difference_team_3PAr_ewma_15 + 
difference_team_ASTpct_ewma_15 + difference_team_BLKpct_ewma_15 + difference_team_DRBpct_ewma_15 + difference_team_DRtg_ewma_15 + 
difference_team_FTr_ewma_15 + difference_team_ORBpct_ewma_15 + difference_team_ORtg_ewma_15 + difference_team_STLpct_ewma_15 + 
difference_team_TOVpct_ewma_15 + difference_team_TRBpct_ewma_15 + difference_team_TSpct_ewma_15 + difference_team_eFGpct_ewma_15 + 
difference_team_fg3_pct_ewma_15 + difference_team_fg_pct_ewma_15 + difference_team_ft_pct_ewma_15 + difference_team_pf_ewma_15 + 
difference_opponent_3PAr_ewma_15 + difference_opponent_ASTpct_ewma_15 + difference_opponent_BLKpct_ewma_15 + difference_opponent_DRBpct_ewma_15 + 
difference_opponent_DRtg_ewma_15 + difference_opponent_FTr_ewma_15 + difference_opponent_ORBpct_ewma_15 + difference_opponent_ORtg_ewma_15 + 
difference_opponent_STLpct_ewma_15 + difference_opponent_TOVpct_ewma_15 + difference_opponent_TRBpct_ewma_15 + difference_opponent_TSpct_ewma_15 + 
difference_opponent_eFGpct_ewma_15 + difference_opponent_fg3_pct_ewma_15 + difference_opponent_fg_pct_ewma_15 + difference_opponent_ft_pct_ewma_15 + 
difference_opponent_pf_ewma_15 + spread + totals """, data=df_covers_bball_ref_home_train).fit()
print(results.summary())  # 

df_covers_bball_ref_home_train[all_iv_vars].columns


# w no limit on max_depth (i.e., not including parameter in model) this this 
# suggests i'm massively overfitting. to help -- more n, fewer features, smaller 
# trees, etc.

# i then set the max_depth to 3 -- cool, this looks like what i see in class examples
# but now it's suggesting high bias -- that need better features

# and increasing max_depth to 5 creates more space between training and learning
# suggesting i'm just overfitting when i do that. max_depth = 3 or 4 is probably optimal

# how to get better features?
# use lineup instead of team to predict. what else? 
# see what looks like with gradient boosting and play with that learning
# rate parameter. that's about preventing over-fitting.  see if can get 
# mean sq error below about 140. i can't seem to beat that w random forest.

# could also try extremely randomized trees? could also try and reduce features 
# if getting at some latent variables?


# to me this looks like we're seeing a big gap between 
# training and test r2. suggests i'm overfitting -- i.e.,
# that it's fitting the training set a lot better than the 
# test set. and kind of looks like the r2 is still moving up,
# though not sure, maybe plateud. so might suggests i could use
# more data, and simplify the model.


mean_sq_error = cross_val_score(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], cv=5, scoring='mean_squared_error')
np.mean(mean_sq_error)

r2 = cross_val_score(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], cv=5, scoring='r2')


scoring_metric = 'mean_squared_error'
scoring_metric = 'r2'

r2_list = []
number_estimators_list = [100, 500, 900, 1300, 1700, 2100, 2500]
for number in number_estimators_list:
    model = RandomForestRegressor(n_estimators = number, max_features = 'sqrt', min_samples_leaf = 50)  
    r2 = cross_val_score(model, df_covers_bball_ref_home_train[all_iv_vars], df_covers_bball_ref_home_train[dv_var], cv=5, scoring=scoring_metric)
    r2_list.append(r2)

r2_means_list = []
for array in r2_list:
    r2_mean = np.mean(array)
    r2_means_list.append(r2_mean)

plt.plot(number_estimators_list, r2_means_list)
# this is showing that after 500 estimators, r2 doesn't go up anymore, and actually goes down

# to plot mean sq error, need to multiply by -1
mean_sq_err_means_list = []
for array in r2_list:
    mean_sq_error = np.mean(array) * -1
    mean_sq_err_means_list.append(mean_sq_error)

plt.plot(number_estimators_list, mean_sq_err_means_list)
# this is showing that the mse is lowest at about 1500 estimators
# but are they all actually super similar? jeez, yes, super similar
[138.27091771611768,
 138.1964448532072,
 138.12419108780495,
 138.16057923483655,
 138.17801212733713,
 138.1045851697892,
 138.13048204750197]


# next - try this w ats -- so can have a classifier


# try my own learning curve. i'd leave the final year out as the test set
# the prior years are training. then i take different amount of the training
# set to predict the test set. record the predictions for each training n.
# compute the mean sq error for ea training n. then plot.






# predict ea date of games based only on preceding dates
# for date in dates
# df = df < date
# train on data up to day before this day
# use current days games stats to predict the games on that day w the model fitted on training data
# record proba, etc. how long does this take for seaons? would like to try cross validating 
# first on training data
df_all_teams_w_ivs_home = df_all_teams_w_ivs_home.dropna()
df_dates = df_all_teams_w_ivs_home.sort('date')
dates = df_dates['date']
dates = dates.drop_duplicates()
dates = dates.reset_index(drop=True)
len(dates)
df_all_dates = pd.DataFrame()
for date in dates[10:]:
    print()
    print(date)
    #date = dates[200]
    df_date = df_all_teams_w_ivs_home[df_all_teams_w_ivs_home['date'] == date]  # map predictions on to this and concat all of them to re-create df
    df_part_train = df_all_teams_w_ivs_home[df_all_teams_w_ivs_home['date'] < date]
    df_part_x_train = df_part_train[all_iv_vars]
    df_part_y_train = df_part_train[dv_var]
    df_part_x_test = df_all_teams_w_ivs_home[all_iv_vars][df_all_teams_w_ivs_home['date'] == date]
    # model
    params = {'n_estimators': 750, 'max_depth': 3, 'min_samples_split': 1,
              'learning_rate': 0.01, 'loss': 'ls'}
    grad_boost_reg = ensemble.GradientBoostingRegressor(**params)
    grad_boost_reg.fit(df_part_x_train, df_part_y_train)
    predicted_current_games = grad_boost_reg.predict(df_part_x_test)
#    model = RandomForestClassifier(n_estimators = 800, # oob_score = True,  # see if faster wout oob_score
#        max_features = 'auto') # but try a higher # too  min_samples_leaf = 15
#    model.fit(df_part_x_train, df_part_y_train)
#    predicted_current_games = model.predict(df_part_x_test)
#    probas = model.predict_proba(df_part_x_test)  
#    probas = probas[:,1]
    df_date['predicted'] = predicted_current_games
#    df_date['probability'] = probas
    df_all_dates = pd.concat([df_all_dates, df_date], ignore_index=True)

df_all_dates.to_pickle('df_2015.pkl')
df_all_dates.to_pickle('df_2014.pkl')
df_all_dates.to_pickle('df_2013.pkl')


len(df_all_dates)
for col in df_all_dates.columns:
    print(col)

df_all_dates_truncated = df_all_dates[['date', 'team', 'opponent', 'score_team', 'score_oppt', 
'point_difference', 'spread', 'beat_spread', 'ats_win', 'predicted', 'win', 'date_rank']]

df_all_dates_truncated = df_all_dates[['date', 'team', 'opponent', 'score_team', 'score_oppt', 
'point_difference', 'spread', 'beat_spread', 'ats_win', 'predicted', 'probability', 'win']]

plt.scatter(df_all_dates_truncated['predicted'], df_all_dates_truncated['point_difference'], alpha=.3)
plt.scatter(df_all_dates_truncated['predicted'], df_all_dates_truncated['spread'], alpha=.3)


# to compute correct for predicted point diff dv
df_all_dates_truncated['predicted_points_over_spread'] = df_all_dates_truncated['predicted'] + df_all_dates_truncated['spread']
df_all_dates_truncated[['predicted', 'spread', 'point_difference', 'predicted_points_over_spread']]

df_all_dates_truncated['correct'] = np.nan
df_all_dates_truncated.loc[(df_all_dates_truncated['predicted_points_over_spread'] > 0) & (df_all_dates_truncated['ats_win'] == 1), 'correct'] = 1
df_all_dates_truncated.loc[(df_all_dates_truncated['predicted_points_over_spread'] < 0) & (df_all_dates_truncated['ats_win'] == 1), 'correct'] = 0
df_all_dates_truncated.loc[(df_all_dates_truncated['predicted_points_over_spread'] > 0) & (df_all_dates_truncated['ats_win'] == 0), 'correct'] = 0
df_all_dates_truncated.loc[(df_all_dates_truncated['predicted_points_over_spread'] < 0) & (df_all_dates_truncated['ats_win'] == 0), 'correct'] = 1
df_all_dates_truncated[['predicted', 'spread', 'point_difference', 'predicted_points_over_spread', 'correct']]
df_all_dates_truncated['correct'].mean()

df_all_dates_truncated['predicted_confidence'] = df_all_dates_truncated['predicted_points_over_spread'].abs()
df_all_dates_truncated['predicted_confidence'].hist()
df_all_dates_truncated = df_all_dates_truncated[df_all_dates_truncated['predicted_confidence'] < 20]
df_all_dates_truncated[['predicted_points_over_spread', 'predicted_confidence']]

sns.lmplot('date_rank', 'correct', data=df_all_dates_truncated, lowess=True)
sns.lmplot('predicted_confidence', 'correct', data=df_all_dates_truncated, lowess=True)
sns.lmplot('predicted_confidence', 'correct', data=df_all_dates_truncated, logistic=True)
sns.interactplot("predicted_confidence", "date_rank", "correct", df_all_dates_truncated, cmap="coolwarm", filled=True)
# seems to fit what would expect -- high confidence and later in season = more likely to predict correctly

df_all_dates_truncated_2 = df_all_dates_truncated[df_all_dates_truncated['date_rank'] > 60]
sns.lmplot('date_rank', 'correct', data=df_all_dates_truncated_2, lowess=True)
sns.lmplot('predicted_confidence', 'correct', data=df_all_dates_truncated_2, lowess=True)

len(df_all_dates_truncated_2['correct'][df_all_dates_truncated_2['predicted_confidence'] > 5])
df_all_dates_truncated_2['correct'][df_all_dates_truncated_2['predicted_confidence'] > 5].mean()
# holy shit, this is hitting at .6! for 120 games! but only for 2015. sucks for 2014! why?!?!


df_all_dates_truncated['correct'] = np.nan
df_all_dates_truncated.loc[(df_all_dates_truncated['beat_spread'] > 0) & (df_all_dates_truncated['probability'] > .5), 'correct'] = 1
df_all_dates_truncated.loc[(df_all_dates_truncated['beat_spread'] < 0) & (df_all_dates_truncated['probability'] < .5), 'correct'] = 1
df_all_dates_truncated.loc[(df_all_dates_truncated['beat_spread'] > 0) & (df_all_dates_truncated['probability'] < .5), 'correct'] = 0
df_all_dates_truncated.loc[(df_all_dates_truncated['beat_spread'] < 0) & (df_all_dates_truncated['probability'] > .5), 'correct'] = 0

df_all_dates_truncated[['probability', 'predicted', 'beat_spread', 'ats_win', 'correct']]

df_all_dates_truncated['correct'] = np.nan
df_all_dates_truncated.loc[df_all_dates_truncated['ats_win'] == df_all_dates_truncated['predicted'], 'correct'] = 1
df_all_dates_truncated.loc[df_all_dates_truncated['ats_win'] != df_all_dates_truncated['predicted'], 'correct'] = 0

df_all_dates_truncated['correct'] = np.nan
df_all_dates_truncated.loc[df_all_dates_truncated['win'] == df_all_dates_truncated['predicted'], 'correct'] = 1
df_all_dates_truncated.loc[df_all_dates_truncated['win'] != df_all_dates_truncated['predicted'], 'correct'] = 0

df_all_dates_truncated['proba_absolute_value'] = .5 - df_all_dates_truncated['probability']
df_all_dates_truncated['proba_absolute_value'] = df_all_dates_truncated['proba_absolute_value'].abs()
df_all_dates_truncated[['probability', 'proba_absolute_value']] 

df_all_dates_truncated.head(10)
df_all_dates_truncated['correct'].mean()

dates = df_all_dates_truncated['date'].unique()
rank_order = range(len(dates))
date_to_rank_dict = dict(zip(dates, rank_order))
df_all_dates_truncated['date_rank'] = df_all_dates_truncated['date'].map(date_to_rank_dict)
df_all_dates_truncated[['date', 'date_rank']] 

# huh, predictions aren't getting better over time. should i expect more data to help, then?
sns.lmplot(x = 'date_rank', y = 'correct', data = df_all_dates_truncated, lowess=True)


df_all_dates_truncated[['proba_absolute_value']].sort('proba_absolute_value')
df_all_dates_truncated = df_all_dates_truncated[df_all_dates_truncated['proba_absolute_value'] < .25]

# nice, the further the proba from .5, the more likely it is to be right
sns.lmplot(x = 'proba_absolute_value', y = 'correct', data = df_all_dates_truncated, lowess=True)
sns.lmplot(x = 'proba_absolute_value', y = 'correct', data = df_all_dates_truncated, logistic=True)

sns.lmplot(x = 'proba_absolute_value', y = 'correct', data = df_all_dates_truncated, lowess=True)
sns.lmplot(x = 'probability', y = 'correct', data = df_all_dates_truncated, lowess=True)
sns.lmplot(x = 'probability', y = 'spread', data = df_all_dates_truncated, lowess=True)
sns.lmplot(x = 'probability', y = 'spread', data = df_all_dates_truncated)

plt.scatter(df_all_dates_truncated['probability'], df_all_dates_truncated['spread'], alpha=.25)


sns.set_style('whitegrid')
plt.scatter(df_all_dates_truncated['probability'][df_all_dates_truncated['ats_win'] == 0], df_all_dates_truncated['spread'][df_all_dates_truncated['ats_win'] == 0], color='red', s=75, alpha=.2)
plt.scatter(df_all_dates_truncated['probability'][df_all_dates_truncated['ats_win'] == 1], df_all_dates_truncated['spread'][df_all_dates_truncated['ats_win'] == 1], color='green', s=75, alpha=.2)
plt.scatter(df_all_dates_truncated['probability'][df_all_dates_truncated['ats_win'] == 0], df_all_dates_truncated['spread'][df_all_dates_truncated['ats_win'] == 0], color='red', s=75, alpha=.2)
# kind of looks like: if ml proba says will lose (proba>.5 but the spread is sayiing differenlty -- that either not supposed to lose by
# as much or even supposed to win, then i should follow the spread.)
# also just generally says bet at the extremes -- bet on those who have the lowest probas and against those w the highest probas



df_all_dates_truncated_proba10 = df_all_dates_truncated[df_all_dates_truncated['proba_absolute_value'] > .101]
len(df_all_dates_truncated_proba10)
df_all_dates_truncated_proba10['correct'].mean()  # .57 -- .52 w out the team off effic and def effic and tot reb

results = smf.logit(formula = 'correct ~ proba_absolute_value', data=df_all_dates_truncated).fit()
print(results.summary())  # 


# look at multicolinearity
df_all_dates[all_iv_vars[:20]].corr()

# next:
# do feature importances graph
# read about and play w tuning knobs. try and reduce over-fitting
# do some cross val to figure out which vars aren't important. 
# do this stuff on a earlier year? or do it on training data, somehow

# could do lineup stats instead of team stats

# try 3 diff models cross-val using data prior to ea date
# then take best model and use to predict that date of gs


# feature importances ---------------------------------------------------------
df_part_train = df_all_teams_w_ivs_home.copy(deep=True)
df_part_x_train = df_part_train[all_iv_vars]
df_part_y_train = df_part_train[dv_var]
# model
model = RandomForestClassifier(n_estimators = 800, # oob_score = True,  # see if faster wout oob_score
    max_features = 'auto', min_samples_leaf = 15) # but try a higher # too
model.fit(df_part_x_train, df_part_y_train)

feature_importance = model.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(df.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# 2nd set of code and plot (ive been using this one)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(df_part_x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(df_part_x_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(df_part_x_train.shape[1]), indices)
plt.xlim([-1, df_part_x_train.shape[1]])
plt.show()

print("Feature ranking:")
for f in range(df_part_x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# top 10:
print(df_part_x_train.columns[22])
print(df_part_x_train.columns[54])
print(df_part_x_train.columns[21])
print(df_part_x_train.columns[23])
print(df_part_x_train.columns[24])
print(df_part_x_train.columns[20])
print(df_part_x_train.columns[30])
print(df_part_x_train.columns[68])
print(df_part_x_train.columns[50])
print(df_part_x_train.columns[51])

# bottom 10:
print(df_part_x_train.columns[70])
print(df_part_x_train.columns[55])
print(df_part_x_train.columns[34])
print(df_part_x_train.columns[36])
print(df_part_x_train.columns[62])
print(df_part_x_train.columns[25])
print(df_part_x_train.columns[1])
print(df_part_x_train.columns[48])
print(df_part_x_train.columns[72])
print(df_part_x_train.columns[2])
# this last one is spread. saying not important. seem surprising?
# maybe i need to think about spread in conjuction w another var.
# really i want to know if the spread is set too high or too low.
# maybe since beating the spread is part of the dv, the spread isn't 
# relevant? don't know. oh yeah, should also try predicting amount
# that beat the spread, w forest regression or gradient boosting reg.

# try subtracting all the team's vars from the corresponding oppt vars (i.e., x_...)



#------------------------------------------------------------------------------
# predict this years gs fro last two seasons

# first take 2013 and 2014 seasons and do cross val to build model
df_dates = df_all_teams_w_ivs_home.sort('date')
dates = df_dates['date']
dates = dates.drop_duplicates()
dates = dates.reset_index(drop=True)
len(dates)
date_season_separator = dates[315]

df_2013_and_2014_train = df_all_teams_w_ivs_home[df_all_teams_w_ivs_home['date'] < date_season_separator]  # map predictions on to this and concat all of them to re-create df
df_part_x_train = df_2013_and_2014_train[all_iv_vars]
df_part_y_train = df_2013_and_2014_train[dv_var]
df_part_x_test = df_all_teams_w_ivs_home[all_iv_vars][df_all_teams_w_ivs_home['date'] >= date]
# model
model = RandomForestClassifier(n_estimators = 500, # oob_score = True,  # see if faster wout oob_score
    max_features = 'auto', min_samples_leaf = 2) # but try a higher # too  min_samples_leaf = 15

cross_val_accuracy = cross_val_score(model, df_part_x_train, df_part_y_train, cv=5, scoring='accuracy')
cross_val_accuracy_mean = cross_val_accuracy.mean()
cross_val_accuracy_stdev = cross_val_accuracy.std()
cross_val_accuracy_min = cross_val_accuracy.min()
cross_val_accuracy_max = cross_val_accuracy.max()

sk_gradient_boost_classifier = GradientBoostingClassifier(n_estimators=800, max_depth=2, learning_rate=.01, subsample=.25) 
cross_val_accuracy = cross_val_score(sk_gradient_boost_classifier, df_part_x_train, df_part_y_train, cv=5, scoring='accuracy')
cross_val_accuracy_mean = cross_val_accuracy.mean()




# model
model = RandomForestClassifier(n_estimators = 500, # oob_score = True,  # see if faster wout oob_score
    max_features = 'auto') # but try a higher # too  min_samples_leaf = 15
model.fit(df_part_x_train, df_part_y_train)
predicted_current_games = model.predict(df_part_x_test)
probas = model.predict_proba(df_part_x_test)  
probas = probas[:,1]






#------------------------------------------------------------------------------
# ml to predict -- just opponent stats to predict ea team separately

# first, use the x_ variables -- these are for the opponent. and ats metrics for opponent
# and use the venue and i guess use the team's ats metrics. 
# oh, should i use the spread? i think so. and use the totals. hmmmm.
# (eventually, look at multicolineariy -- can i merger or eliminate certain vars?
# but wouldnt i wan to do this differently for ea team?)

# predict a certain date
iv_vars = [ 'x_team_3PAr_ewma_15',
       'x_team_AST%_ewma_15', 'x_team_BLK%_ewma_15', 'x_team_DRB%_ewma_15',
       'x_team_DRtg_ewma_15', 'x_team_FTr_ewma_15', 'x_team_ORB%_ewma_15',
       'x_team_ORtg_ewma_15', 'x_team_STL%_ewma_15', 'x_team_TOV%_ewma_15',
       'x_team_TRB%_ewma_15', 'x_team_TS%_ewma_15', 'x_team_eFG%_ewma_15',
       'x_team_fg3_pct_ewma_15', 'x_team_fg_pct_ewma_15', 'x_team_ft_pct_ewma_15',
       'x_team_pf_ewma_15', 'x_opponent_3PAr_ewma_15', 'x_opponent_AST%_ewma_15',
       'x_opponent_BLK%_ewma_15', 'x_opponent_DRB%_ewma_15', 'x_opponent_DRtg_ewma_15', 
       'x_opponent_FTr_ewma_15', 'x_opponent_ORB%_ewma_15', 'x_opponent_ORtg_ewma_15',
       'x_opponent_STL%_ewma_15', 'x_opponent_TOV%_ewma_15', 'x_opponent_TRB%_ewma_15', 
       'x_opponent_TS%_ewma_15', 'x_opponent_eFG%_ewma_15', 'x_opponent_fg3_pct_ewma_15',
       'x_opponent_fg_pct_ewma_15', 'x_opponent_ft_pct_ewma_15', 'x_opponent_pf_ewma_15', 
       'venue_x'] 
#, 'beat_spread_rolling_mean_11', 'beat_spread_rolling_std_11', 'beat_spread_last_g', 'spread', 'totals']
#, 'x_spread_expanding_mean', 'x_current_spread_vs_spread_expanding', 'x_beat_spread_rolling_mean_11', 'x_beat_spread_rolling_std_11', 'x_beat_spread_last_g',
dv_var = 'ats_win'
iv_and_dv_vars = iv_vars + [dv_var] + ['team', 'opponent', 'date']

df_all_teams_w_ivs[iv_and_dv_vars].head(10)


#model = RandomForestRegressor(n_estimator = 300, oob_score = TRUE,  # see if faster wout oob_score
#                              max_features = 'auto', min_samples_leaf = 3) # but try a higher # too

#roc_auc_score(y,model.oob_prediction)  # might not work for classifier

# ( second approach: train on 2012-2014 and use that fitted model to predict 2015. )
# ( i'd enter the team's off and def stats up to that point in season and the oppt's 
# ( off and def stats up to that point in season. )
# ( or could try this several times, e.g., train on 2010-2012 and predict 2013, )
# ( and repeat for next few years. see http://www.seas.upenn.edu/~gberta/uploads/3/1/4/8/31486883/)
# ( or just do this for current season. may as well try that first. )

#team = 'Atlanta Hawks'
def get_df_w_prediction_metrics(df_all_teams_w_ivs, team, game_cutoff):
    cross_val_accuracy_mean_list = []
    cross_val_accuracy_stdev_list = []
    cross_val_accuracy_min_list = []
    cross_val_accuracy_max_list = []
    cross_val_log_loss_mean_list = []
    cross_val_log_loss_stdev_list = []
    cross_val_log_loss_min_list = []
    cross_val_log_loss_max_list = []
    proba_list = []
    predicted_current_g_list = []
    # select the team:
    df_team = df_all_teams_w_ivs[df_all_teams_w_ivs['team'] == team]
    df_team = df_team[iv_and_dv_vars]
    df_team = df_team.dropna()
    df_team = df_team.reset_index()
    dates = df_team['date']
    for date in dates[game_cutoff:]:  # try making this 10 and making cross val score kfold = 10
        print(date)    
        #date = dates[65]    
        df_team_date = df_team[df_team['date'] <= date]
        df_team_date_x = df_team_date[iv_vars]
        #df_team_date_x = scale(df_team_date_x)  
        df_team_date_y = df_team_date[dv_var].values  
        df_team_date_x_train = df_team_date_x[:-1]
        df_team_date_y_train = df_team_date_y[:-1]
        df_team_date_x_test = df_team_date_x[-1:]
        df_team_date_y_test = df_team_date_y[-1:]
        # logistic regression classifier:
#        sk_logistic_model = LogisticRegression(C=.05) #the smaller the C, the more regularization
#        sk_logistic_model.fit(df_team_x_train, df_team_y_train)
#        predicted_current_g = sk_logistic_model.predict(df_team_x_test)
#        probas = sk_logistic_model.predict_proba(df_team_x_test)   
        # get cross validated scores/metrics:
        model = RandomForestClassifier(n_estimators = 300, oob_score = True,  # see if faster wout oob_score
                              max_features = 'auto', min_samples_leaf = 10) # but try a higher # too
        cross_val_accuracy = cross_val_score(model, df_team_date_x_train, df_team_date_y_train, cv=5, scoring='accuracy')
        cross_val_log_loss = cross_val_score(model, df_team_date_x_train, df_team_date_y_train, cv=5, scoring='log_loss')

#        sk_gradient_boost_classifier = GradientBoostingClassifier(n_estimators=500, max_depth=2, learning_rate=.01, subsample=.33) 
#        cross_val_accuracy = cross_val_score(sk_gradient_boost_classifier, df_team_date_x_train, df_team_date_y_train, cv=5, scoring='accuracy')
#        cross_val_log_loss = cross_val_score(sk_gradient_boost_classifier, df_team_date_x_train, df_team_date_y_train, cv=5, scoring='log_loss')
        cross_val_accuracy_mean = cross_val_accuracy.mean()
        cross_val_accuracy_stdev = cross_val_accuracy.std()
        cross_val_accuracy_min = cross_val_accuracy.min()
        cross_val_accuracy_max = cross_val_accuracy.max()
        cross_val_log_loss_mean = cross_val_log_loss.mean()
        cross_val_log_loss_stdev = cross_val_log_loss.std()
        cross_val_log_loss_min = cross_val_log_loss.min()
        cross_val_log_loss_max = cross_val_log_loss.max()
        cross_val_accuracy_mean_list = cross_val_accuracy_mean_list + [cross_val_accuracy_mean]
        cross_val_accuracy_stdev_list = cross_val_accuracy_stdev_list + [cross_val_accuracy_stdev]
        cross_val_accuracy_min_list = cross_val_accuracy_min_list + [cross_val_accuracy_min]
        cross_val_accuracy_max_list = cross_val_accuracy_max_list + [cross_val_accuracy_max]
        cross_val_log_loss_mean_list = cross_val_log_loss_mean_list + [cross_val_log_loss_mean]
        cross_val_log_loss_min_list = cross_val_log_loss_min_list + [cross_val_log_loss_min]
        cross_val_log_loss_max_list = cross_val_log_loss_max_list + [cross_val_log_loss_max]
        cross_val_log_loss_stdev_list = cross_val_log_loss_stdev_list + [cross_val_log_loss_stdev]
        # get actual prediction metrics:
        model = RandomForestClassifier(n_estimators = 500, oob_score = True,  # see if faster wout oob_score
                              max_features = 'auto', min_samples_leaf = 10) # but try a higher # too
        model.fit(df_team_date_x_train, df_team_date_y_train)
        predicted_current_g = model.predict(df_team_date_x_test)
        probas = model.predict_proba(df_team_date_x_test)  
        proba_list = proba_list + [probas[:,1]]
        predicted_current_g_list = predicted_current_g_list + [predicted_current_g]
#        sk_gradient_boost_classifier = GradientBoostingClassifier(n_estimators=500, max_depth=2, learning_rate=.01, subsample=.33) 
#        sk_gradient_boost_classifier.fit(df_team_date_x_train, df_team_date_y_train)
#        predicted_current_g = sk_gradient_boost_classifier.predict(df_team_date_x_test)
#        probas = sk_gradient_boost_classifier.predict_proba(df_team_date_x_test)   
#        proba_list = proba_list + [probas[:,1]]
#        predicted_current_g_list = predicted_current_g_list + [predicted_current_g]
    proba_list = [prob for proba in proba_list for prob in proba]
    predicted_current_g_list = [pred_g for predicted_g in predicted_current_g_list for pred_g in predicted_g]
    # map prediction metrics onto df
    # truncate df to same size as metric lists
    df_team = df_team[df_team.index >= game_cutoff]
    df_team['proba'] = proba_list
    df_team['predicted_g'] = predicted_current_g_list
    df_team['cross_val_acccuracy_mean'] = cross_val_accuracy_mean_list
    df_team['cross_val_acccuracy_stdev'] = cross_val_accuracy_stdev_list
    df_team['cross_val_acccuracy_min'] = cross_val_accuracy_min_list
    df_team['cross_val_acccuracy_max'] = cross_val_accuracy_max_list
    df_team['cross_val_log_loss_mean'] = cross_val_log_loss_mean_list
    df_team['cross_val_log_loss_stdev'] = cross_val_log_loss_stdev_list
    df_team['cross_val_log_loss_min'] = cross_val_log_loss_min_list
    df_team['cross_val_log_loss_max'] = cross_val_log_loss_max_list
    df_team = compute_correct(df_team)
    return df_team


def compute_correct(df_team):
    df_team['correct'] = 0
    df_team.loc[df_team['predicted_g'] != df_team['ats_win'], 'correct'] = 0
    df_team.loc[df_team['predicted_g'] == df_team['ats_win'], 'correct'] = 1
    return df_team


def create_df_all_teams_w_predictions(df_all_teams_w_ivs):
    team_names = df_all_teams_w_ivs['team'].unique()
    df_all_teams_w_prediction_metrics = pd.DataFrame()
    for team in team_names[:]:    
        print()
        print()
        print(team)
        df_team = get_df_w_prediction_metrics(df_all_teams_w_ivs, team, 30)
        df_all_teams_w_prediction_metrics = pd.concat([df_all_teams_w_prediction_metrics, df_team], ignore_index=True)
    return df_all_teams_w_prediction_metrics


df_all_teams_w_prediction_metrics = create_df_all_teams_w_predictions(df_all_teams_w_ivs)

# can start w this -- it has all prediction metrics -- took a couple hours to loop through
# quicker way to do this to test ideas? one thought -- start at game 50 and stop at game 75
# can set that above: for date in dates[10:]:
# things to try: take out venue from model
df_all_teams_w_prediction_metrics.to_pickle('df_all_teams_w_prediction_metrics_2015_16.pkl')
df_all_teams_w_prediction_metrics.to_pickle('df_all_teams_w_random_forest_prediction_metrics_2015_16.pkl')

#------------------------------------------------------------------------
# explore the metrics df
df_all_teams_w_prediction_metrics_truncated_gs = df_all_teams_w_prediction_metrics[df_all_teams_w_prediction_metrics['date'] > '2016-01-01']
#df_home = df_all_teams_w_prediction_metrics[df_all_teams_w_prediction_metrics['venue_x'] == 0]
#df_away = df_all_teams_w_prediction_metrics[df_all_teams_w_prediction_metrics['venue_x'] == 1]
#df_home_truncated_gs = df_home[df_home['date'] > '2016-02-01']
#df_away_truncated_gs = df_away[df_away['date'] > '2016-02-01']
df_all_teams_w_prediction_metrics['correct'].mean()
df_all_teams_w_prediction_metrics_truncated_gs['correct'].mean()


# examine relation between proba and correct. to do, compute abs val of proba - .5
df_all_teams_w_prediction_metrics_truncated_gs['abs_val_proba'] = np.abs(.5 - df_all_teams_w_prediction_metrics_truncated_gs['proba'])
sns.lmplot(x='abs_val_proba', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
# i think this makes sense -- as abs_val_proba gets to high, above .2, it's probably
# from analyses with small n, so not reliable.

# examine relation between cross val metrics and correct (or proba)
sns.lmplot(x='cross_val_acccuracy_mean', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
# it's true that with the whole df i'm predicing ea g twice. but i'm predicting it
# with difference variables. 

sns.lmplot(x='cross_val_acccuracy_stdev', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
sns.lmplot(x='cross_val_acccuracy_min', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
# ** take home -- when the mis sucks, don't bet on game
# suggesting that really don't want to gamble w min cross val accuracy below  .3
# this is nice, because sort of takes into account small saples. if small sample,
# will probably get at least one very low accuracy score
# is it interesting that with a very low accuracy score, if i bet OPPOSTITE the 
# proba/prediction spit out by gradient boosting algo, i'd do well??!! but seems risky, right?
sns.lmplot(x='cross_val_acccuracy_max', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
# don't think this max is particularly meaningful. because can get very high max
# because small sample size, but still wouldn't want to bet on those.
# again, every time the max accuracy is below .5, i could bet against??!!
# yeah, maybe this is what makes the most sense here:
# every time min accuracy is above .5, bet on the game and follow the prediction/proba
# every time the max accruacy is below .5, bet on the game but bet oppostive of the prediction/proba

# look at max - min (i.e., range)
df_all_teams_w_prediction_metrics_truncated_gs['cross_val_accuracy_range'] = df_all_teams_w_prediction_metrics_truncated_gs['cross_val_acccuracy_max'] - df_all_teams_w_prediction_metrics_truncated_gs['cross_val_acccuracy_min'] 
sns.lmplot(x='cross_val_accuracy_range', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
# eh

sns.lmplot(x='cross_val_log_loss_mean', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
sns.lmplot(x='cross_val_log_loss_stdev', y='correct', data=df_all_teams_w_prediction_metrics_truncated_gs, lowess=True)
results = smf.logit(formula = 'correct ~ cross_val_log_loss_mean * cross_val_log_loss_stdev', data=df_all_teams_w_prediction_metrics_truncated_gs).fit()
print(results.summary())  # no X
# ** take home here -- don't bet if cross_val_log_loss_mean is less than -.8 or -.9
# i set it up so when run stats again, will compute log_loss_min. that might be better than mean


# what's the mean accuracy if only bet on games with log loss greater than -.9
# only bet on games with cross_val_accuracy_min greater than .5
df_all_teams_w_prediction_metrics_truncated_gs['correct'][df_all_teams_w_prediction_metrics_truncated_gs['cross_val_acccuracy_min']>.5].mean()
len(df_all_teams_w_prediction_metrics_truncated_gs['correct'][df_all_teams_w_prediction_metrics_truncated_gs['cross_val_acccuracy_min']>.5])

df_all_teams_w_prediction_metrics_truncated_gs['correct'][df_all_teams_w_prediction_metrics_truncated_gs['cross_val_log_loss_mean']>-.6].mean()

results = smf.logit(formula = 'correct ~ cross_val_acccuracy_max * cross_val_acccuracy_stdev', data=df_all_teams_w_prediction_metrics_truncated_gs).fit()
print(results.summary())  # no X

# this combining the probas below -- it seems like it should be good, but is pretty iffy
# above, i'm betting on ea game twice, but when i apply filters, e.g., if i take out probas
# with below some minium or with a log_loss above some number, i must be taking out one of
# the predictions for a game and leaving the other one? so maybe this approach above is fine?
# initially, every game is represented twice, but in the end maybe only one of them suggests
# i should be in a particular way?

#------------------------------------------------------------------------------
# only bet when both probas agree -- i.e., i have two probas for ea g, one for ea team
# need to think this through. ea g also has two cross val accuracy scores, and two
# of all other cross val metrics. so should take mean of all of these, right?
df_all_teams_w_prediction_metrics_switched = df_all_teams_w_prediction_metrics[['date', 
'team', 'opponent', 'proba', 'ats_win', 'correct', 'cross_val_acccuracy_mean',
'cross_val_acccuracy_min', 'cross_val_acccuracy_max', 'cross_val_log_loss_mean',
'cross_val_log_loss_stdev', 'predicted_g']]
df_all_teams_w_prediction_metrics_switched.columns
df_all_teams_w_prediction_metrics_switched.columns=['date', 'opponent', 'team', 
'proba_opponent', 'ats_win_opponent', 'correct_opponent', 'cross_val_acccuracy_mean_opponent', 
'cross_val_acccuracy_min_opponent', 'cross_val_acccuracy_max_opponent', 
'cross_val_log_loss_mean_opponent', 'cross_val_log_loss_stdev_opponent', 'predicted_g_opponent']

# need to reverse teh probability variables so can fit when merge in a moment
def reverse_opponent_variables(df_all_teams_w_prediction_metrics_switched):
    for var in ['proba_opponent', 'ats_win_opponent', 'correct_opponent', 'cross_val_acccuracy_mean_opponent', 
                'cross_val_acccuracy_min_opponent', 'cross_val_acccuracy_max_opponent', 
                'cross_val_log_loss_mean_opponent', 'cross_val_log_loss_stdev_opponent', 'predicted_g_opponent']:
        df_all_teams_w_prediction_metrics_switched[var] = 1 - df_all_teams_w_prediction_metrics_switched[var] 
    return df_all_teams_w_prediction_metrics_switched

df_all_teams_w_prediction_metrics_switched = reverse_opponent_variables(df_all_teams_w_prediction_metrics_switched)
df_all_teams_w_prediction_metrics_switched.head()

df_all_teams_w_two_prediction_metrics = df_all_teams_w_prediction_metrics.merge(df_all_teams_w_prediction_metrics_switched,
                                                                                on=['date', 'team', 'opponent'], how='inner')

df_all_teams_w_two_prediction_metrics[['date', 'team', 'opponent', 'proba', 'proba_opponent', 'venue_x']].head()
df_all_teams_w_two_prediction_metrics[['date', 'team', 'opponent', 'proba', 'proba_opponent', 'venue_x']][df_all_teams_w_two_prediction_metrics['team'] == 'Orlando Magic'].head()

# now should take avg of all cross val metrics. and also mean of the two probas 
# and compte new correct based on that proba mean
df_two_predictions = df_all_teams_w_two_prediction_metrics[df_all_teams_w_two_prediction_metrics['venue_x'] == 0] 

df_two_predictions['proba_team_oppt_mean'] = (df_two_predictions['proba'] + df_two_predictions['proba_opponent']) / 2
df_two_predictions['proba_team_oppt_mean'].hist() 

# compute new correct score based on two probas
df_two_predictions['correct_w_two_probas'] = np.nan
df_two_predictions.loc[(df_two_predictions['proba_team_oppt_mean'] > .52) & (df_two_predictions['ats_win'] == 1), 'correct_w_two_probas'] = 1
df_two_predictions.loc[(df_two_predictions['proba_team_oppt_mean'] < .48) & (df_two_predictions['ats_win'] == 0), 'correct_w_two_probas'] = 1
df_two_predictions.loc[(df_two_predictions['proba_team_oppt_mean'] > .52) & (df_two_predictions['ats_win'] == 0), 'correct_w_two_probas'] = 0
df_two_predictions.loc[(df_two_predictions['proba_team_oppt_mean'] < .48) & (df_two_predictions['ats_win'] == 1), 'correct_w_two_probas'] = 0
df_two_predictions[['correct', 'correct_w_two_probas']].corr()
df_two_predictions[['correct', 'correct_opponent', 'correct_w_two_probas']].mean()

df_two_predictions[['date', 'team', 'opponent', 'proba', 'proba_opponent', 'proba_team_oppt_mean', 'ats_win', 'spread', 'correct_w_two_probas']]

# compute if two probas agree
df_two_predictions[['predicted_g', 'predicted_g_opponent']].corr()
df_two_predictions['two_probas_agree'] = np.nan
df_two_predictions.loc[df_two_predictions['predicted_g'] == df_two_predictions['predicted_g_opponent'], 'two_probas_agree'] = 1
df_two_predictions.loc[df_two_predictions['predicted_g'] != df_two_predictions['predicted_g_opponent'], 'two_probas_agree'] = 0

df_two_predictions[['predicted_g', 'predicted_g_opponent', 'two_probas_agree']]

df_two_predictions[['cross_val_acccuracy_mean', 'cross_val_acccuracy_mean_opponent']].corr()
df_two_predictions['cross_val_accuracy_two_means'] = (df_two_predictions['cross_val_acccuracy_mean'] + df_two_predictions['cross_val_acccuracy_mean_opponent']) / 2
df_two_predictions['cross_val_accuracy_two_mins'] = (df_two_predictions['cross_val_acccuracy_min'] + df_two_predictions['cross_val_acccuracy_min_opponent']) / 2
df_two_predictions['cross_val_accuracy_two_maxs'] = (df_two_predictions['cross_val_acccuracy_max'] + df_two_predictions['cross_val_acccuracy_max_opponent']) / 2
df_two_predictions['cross_val_log_loss_two_means'] = (df_two_predictions['cross_val_log_loss_mean'] + df_two_predictions['cross_val_log_loss_mean_opponent']) / 2
df_two_predictions['cross_val_log_loss_two_stdevs'] = (df_two_predictions['cross_val_log_loss_stdev'] + df_two_predictions['cross_val_log_loss_stdev_opponent']) / 2

# truncate so not looking at early games
df_two_predictions_truncated_gs = df_two_predictions[df_two_predictions['date'] > '2016-01-01']
df_two_predictions_truncated_gs[['correct', 'correct_opponent', 'correct_w_two_probas']].mean()
len(df_two_predictions_truncated_gs[df_two_predictions_truncated_gs['correct_w_two_probas'].notnull()])

df_two_predictions_truncated_gs[['correct', 'correct_opponent', 'correct_w_two_probas']][df_two_predictions_truncated_gs['two_probas_agree'] == 1].mean()
# uhg -- why would the accuracy be shitty when the two probas don't agree!? and good when they do agree!?
# actually, not terrible when set cutoff for proba a bit higher

df_two_predictions_truncated_gs['abs_val_proba_team_oppt'] = np.abs(.5 - df_two_predictions_truncated_gs['proba_team_oppt_mean'])
sns.lmplot(x='abs_val_proba_team_oppt', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, lowess=True)
# this looks good w random forest
# but i can't find an error. alternatively, it's saying that the models is somewhat
# accurate if the predicted prob is around .5, but when it gets too far away, then
# it's not to be trusted anymore, and get's pretty innacurate. but is it curious that
# it actually seems to get innacurate, not just random (i.e., around .5 correct)?

sns.lmplot(x='cross_val_accuracy_two_means', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, lowess=True)
sns.lmplot(x='cross_val_accuracy_two_means', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, logistic=True)
sns.lmplot(x='cross_val_accuracy_two_mins', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, lowess=True)
# i like this again -- bet if the min > .5 (thouth seems very few gs)
sns.lmplot(x='cross_val_accuracy_two_maxs', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, lowess=True)

sns.lmplot(x='cross_val_log_loss_two_means', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, lowess=True)
sns.lmplot(x='cross_val_log_loss_two_stdevs', y='correct_w_two_probas', data=df_two_predictions_truncated_gs, lowess=True)

# yikes, none of this is looking particulary good

results = smf.logit(formula = 'correct_w_two_probas ~ abs_val_proba_team_oppt * cross_val_accuracy_two_means', data=df_two_predictions_truncated_gs).fit()
print(results.summary())  # no X


df_two_predictions_truncated_gs['correct_w_two_probas'][df_two_predictions_truncated_gs['cross_val_acccuracy_min']>.60].mean()
# there is a neg slope from .5 onwards. makes no sense at all. why should it be neg???
# something is fishy in the brining together of the two cross vals metrics and probas
# but can't figure out what...
# maybe the team and the oppt stats aren't starting in the same place somehow?
# like one is shifted up or down? could that be? 





#------------------------------------------------------------------------
df_hawks_prediction = get_proba_and_game_lists(df_all_teams_w_ivs, team)
df_hawks_prediction.head()
df_hawks_prediction[['ats_win', 'proba', 'predicted_g', 'correct']].head()
df_hawks_prediction['correct'].mean()

# get ea team towrads end of season. and get predictions and probas
# and get the feature important from below
# and put into a regression and see which vars are very signif
# and get corr matrics to see which are corr a lot
# do again and again for many teams. same vars generally?
# if i take out the less important vars (by feature importance
# and regression), does the prediction/accuracy get better?


# for when just looking at predicting one game for one team
# and want to really explore the model:
# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(df.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()




