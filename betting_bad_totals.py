
# coding: utf-8

# ##To Do: 
# 

cd /Users/charlesmartens/Documents/projects/bet_bball


# think something is weird -- when used opening spread -- am i using closing
# totals? and when using opening totals, am i useing opening spread? yeah, shit.
# fix.


# opening totals is far away the best. keep doube checking.
# what about opening spread? as long as bet on opening spread, should be ok?
# or if not, why would totals be diff? look at how accuracy opening spread vs.
# closing spread is, and opening total vs. closing total. does spread get
# more accuracy as get closer to the game, and opening spread doesn't?

# look at opening spread again. think just as good as others? what if i
# use the code below for predicing each team's points and then using that
# to try and beat 



# think maybe the ewma is doing something funky???
# but why would it process differently depending on diff sorts when the
# sort within team is the same??? don't get it.
# plot the ewmas?


# try: 

# i'm getting somewhere. use sq root. 
# i use span=20 to create oppt's ewma metrics and then adjust the raw
# metrics for these oppt metrics. then take the ewma w span=8 of these
# new adjusted metrics. try diff spans. try span=25 or 30 for inital stage
# and try span = 10 or 15 for next stage. and try, in 2nd stage, creating
# span = 8 and span = 15 and averaging them. or do them both separate and
# only bet if they both agree. or if they're both similar.


# try diff spans for ewma
# i like cube root transformation the best (vs sq root or log)
# can see in the histogram too that the skew goes away
# quantile regression
# why can't predict favored teams at home (or, to a lesser extent, games with high totals)


# HOW DID I GET THINGS GOOD BEFORE? I THOUGHT I WAS USING A SPAN OF 20
# THEN ADJUSTING W RESIDUALIZING OUT OPPT AND THEN SPAN OF 8. BUT SOMETHING
# ELSE MIGHT HAVE BEEN GOING ON? TRY GOING BACK TO ADJ FOR X_...EWMA_15 RATHER
# THAN TURNING IT INTO AN OPPT. IT DOESTN'T MAKE AS MUCH SENSE BUT STILL.
# HMMMM. ESSENTIALLY I WAS SEEING THAT THOUGH I COULDN'T GET 2013 GOOD WITH 
# THE REGULAR SPAN=15 MODEL, A SPAN=8 MODEL WORKED WELL IN 2013, THOUGH NOT IN OTHER
# YAEARS. SO THEY COULD COMPENSATE FOR EACH OTHER. COULD JUST TRY AND RUN TWO MODELS
# ONE WITH SPAN=15 AND ONE WITH SPAN=8 AND ONLY BET WHEN THEY AGREE?




# technically, need to beat 51.3% to win. though the juice seems a bit erractic, 
# so probably better to think about it at 51.5%

# there are some really high outlier totals
# this may be pulling my regression line up and making me guess 
# over more than i should??? how can i take into account this 
# skewed distribution? some kind of transformation?

# got this from quora (in a nutshell, try quantile regression):
#However, while having a highly skewed dependent variable does not violate an 
#assumption, it may make OLS regression rather inapporpriate. OLS regression 
#models the mean and the mean is (usually) not a good measure of central 
#tendency in a skewed distribution.  The median is often better and it can be 
#modeled with quantile regression. In addition, when the DV is highly skewed 
#the interest may be in modeling the tails of the distribution - this can also 
#be done with quantile regression.  

# how to do quantile regression in statsmodels
# http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/quantile_regression.html
# http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.quantile_regression.QuantReg.html
# http://stackoverflow.com/questions/33050636/python-statsmodels-quantreg-intercept

# also look at think stats -- what was that type of correlattion that 
# was uesful for skewed data? is there a regression equiv? maybe that
# was about rank ordering. but that might not be good for actually predicting
# a y value?


# change wording so oppt is team_def. and then make x_ into oppt_
# think this will help make more sense of things

# to predict totals, starting to have some success using the exact code i 
# was using when predicing point diff with difference variables. but instead
# of creating difference variables, just sum the variables of team and oppt.
# that approach is in 'betting_bad_totals_like_spread.py' file. but try using
# here except use all the raw varaibles instead of the percentage varas.

# current plan: predict the teams score. that's the dv (score_team). gear the ivs
# towards this dv -- things that would affect how man ponits the teams scores
# i.e., their offense and the opponent's defense. (but not the team's defense?)
# then i'll have a predicted team's score for ea game. and ea game will be
# represented twice. so i'll have to create a new df with the games only each
# represented once and then simply add the home team's predicted score to the 
# away team's predicted score, and that's the predicted total. (can also use 
# this approach to predict the spread by subtracting the two predicted scores.)

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.ensemble import GradientBoostingRegressor
# try tpot
#from tpot import TPOT
#from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
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

for col in df_covers_bball_ref.columns:
    print(col)

# merge existing df w moneline df
df_covers_bball_ref = pd.merge(df_covers_bball_ref, df_moneyline_full, on=['date','team', 'opponent'], how='left')
df_covers_bball_ref.head()
df_covers_bball_ref[['date','team', 'spread', 'team_close_spread', 'team_open_spread']]
df_covers_bball_ref[['spread','team_close_spread', 'team_open_spread']].corr()
plt.scatter(df_covers_bball_ref['spread'], df_covers_bball_ref['team_close_spread'], alpha=.4)
plt.scatter(df_covers_bball_ref['spread'], df_covers_bball_ref['team_open_spread'], alpha=.1)
plt.scatter(df_covers_bball_ref['team_close_spread'], df_covers_bball_ref['team_open_spread'], alpha=.1)

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

df_covers_bball_ref[['team_predicted_points', 'score_team']].head()

# compute pace
df_covers_bball_ref.loc[:, 'team_possessions'] = ((df_covers_bball_ref.loc[:, 'team_fga'] + .44*df_covers_bball_ref.loc[:, 'team_fta'] - df_covers_bball_ref.loc[:, 'team_orb'] + df_covers_bball_ref.loc[:, 'team_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'opponent_possessions'] = ((df_covers_bball_ref.loc[:, 'opponent_fga'] + .44*df_covers_bball_ref.loc[:, 'opponent_fta'] - df_covers_bball_ref.loc[:, 'opponent_orb'] + df_covers_bball_ref.loc[:, 'opponent_tov']) / (df_covers_bball_ref['team_MP']/5)) * 48
df_covers_bball_ref.loc[:, 'the_team_pace'] = df_covers_bball_ref[['team_possessions', 'opponent_possessions']].mean(axis=1)
df_covers_bball_ref[['team_possessions', 'opponent_possessions', 'the_team_pace']].head(20)



# assign the totals and spread variables
def assigne_spread_and_totals(df_covers_bball_ref, spread_to_use, totals_to_use):
    df_covers_bball_ref.loc[:, 'totals_covers'] = df_covers_bball_ref.loc[:, 'totals']
    df_covers_bball_ref.loc[:, 'spread_covers'] = df_covers_bball_ref.loc[:, 'spread']
    
    df_covers_bball_ref.loc[:, 'totals'] = df_covers_bball_ref.loc[:, totals_to_use]
    #df_covers_bball_ref.loc[:, 'totals'] = df_covers_bball_ref.loc[:, 'Closing Total']
    
    df_covers_bball_ref.loc[:, 'spread'] = df_covers_bball_ref.loc[:, spread_to_use]
    #df_covers_bball_ref.loc[:, 'spread'] = df_covers_bball_ref.loc[:, 'team_close_spread']
    return df_covers_bball_ref

df_covers_bball_ref = assigne_spread_and_totals(df_covers_bball_ref, 'team_open_spread', 'Opening Total')
#df_covers_bball_ref = assigne_spread_and_totals(df_covers_bball_ref, 'team_close_spread', 'Closing Total')
#df_covers_bball_ref = assigne_spread_and_totals(df_covers_bball_ref, 'spread', 'totals')


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





# don't mess w venue, the_team_predicted_points, the_oppt_predicted_points,
# beat_spread, spread, score_team, score_oppt, totals


#df_covers_bball_ref['opponent_ft_pct']

#------------------------------------------------------------------------------
# OPTIONAL - add noise to the spread:
# try again, but randomly add .5 or subtract .5 from spread. more akin to what will actually see?
#df_covers_bball_ref['random_number'] = np.random.normal(0, .2, len(df_covers_bball_ref))
#df_covers_bball_ref['random_number'].hist(alpha=.7)
#df_covers_bball_ref['spread'] = df_covers_bball_ref.loc[:, 'spread'] + df_covers_bball_ref.loc[:, 'random_number']
#
## use ending spread or starting spread?
## ending spread is likely more accurate
## beginning spread may provide my predicted score with more room to move?
#
#
#df_covers_bball_ref.iloc[:,:20].head()
#df_covers_bball_ref.iloc[:,20:40].head()
#df_covers_bball_ref.iloc[:,40:60].head()
#df_covers_bball_ref.iloc[:,60:80].head()
#df_covers_bball_ref.iloc[:,80:100].head()
#
#for var in variables_for_team_metrics:
#    print(len(df_covers_bball_ref[df_covers_bball_ref[var].isnull()]))


###########  create new variables to try out in this section  #############
# Choose one of next set of cells to create metrics for ea team/lineup

# WHY IF I DO THIS LOOP TWICE THE RESULTS ARE DIFF (for the better)??? AND DIFF YET AGAIN IF DO THREE TIMES (for the worse)!
# I THINK SOMETHING ABOUT THE SORTING. WHAT WAS IT???
# group just by team, not by season too. need to do weighted mean or a rolling mean here.
#df_covers_bball_ref[['date', 'team']]
#df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
#df_covers_bball_ref = df_covers_bball_ref.reset_index(drop=True)
#df_covers_bball_ref_2 = df_covers_bball_ref.sort_values(by='date')

#df_covers_bball_ref[df_covers_bball_ref['date'].isnull()]
#
## what is sort_values doing??? it's changing?!
#df_covers_bball_ref['date'].head()
#
#df_covers_bball_ref[['date', 'team']][df_covers_bball_ref['team']=='Denver Nuggets']
#
#group_team = df_covers_bball_ref.groupby('team')
#for team, key in group_team:
#    print(team, len(key))
#

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
    df_covers_bball_ref.loc[:, 'lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))
    df_covers_bball_ref = df_covers_bball_ref.sort_values(['team','date'])
    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)

df_covers_bball_ref[['date','team','lineup_count']]  # still in team-date order. nice.

#df_covers_bball_ref_1 = df_covers_bball_ref.copy(deep=True)
#df_covers_bball_ref_2 = df_covers_bball_ref.copy(deep=True)
#
#df_covers_bball_ref_1 = df_covers_bball_ref_1.sort_values(by=['team', 'date'])
#df_covers_bball_ref_2 = df_covers_bball_ref_2.sort_values(by=['team', 'date'])
#
#df_covers_bball_ref_1[['date', 'team', 'opponent']].head(20)
#df_covers_bball_ref_2[['date', 'team', 'opponent']].head(20)

#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))
#df_covers_bball_ref['team_count'] = df_covers_bball_ref.groupby('team')['team_3PAr'].transform(lambda x: pd.expanding_count(x))
#df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))


# to test if below comparison works:
#df_covers_bball_ref_2['team_DRtg'] = 5
#df_covers_bball_ref_1['team_DRtg']
#
## code to compare two dfs.  use to compare dfs at end of this process
## but should sort both by team and then date before doing this
#for var in list(df_covers_bball_ref_1.columns[:30]):
#    print(var)
#    one = df_covers_bball_ref_1[var]
#    two = df_covers_bball_ref_2[var]
#    for i in range(len(one[:])): 
#        if df_covers_bball_ref_1[var].dtypes == 'float':
#            item_one = one[i].astype(str)
#            item_two = two[i].astype(str)
#            #print(item_one, item_two)
#            if item_one != item_two:
#                print(item_one, item_two)
#            else:
#                None
#        else:
#            item_one = one[i]
#            item_two = two[i]
#            #print(item_one, item_two)
#            if item_one != item_two:
#                print(item_one, item_two)
#            else:
#                None
#        
        


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






# ================================
# ================================
# try simple adjustment -- just adjust each var by the spread and venue
# SKP THIS -- NOT DOING CORRCTLY BECAUSE TAKING INFO FROM THE FUTURE TO MODEL

# test out getting oppt's ewma_15 scores and then reside out of team's raw score and 
# then run ewma_15 again
def create_switched_df(df_all_teams, variables_for_team_metrics):
    # create df with team and opponent swithced (so can then merge the team's 
    # weighted/rolling metrics onto the original df but as the opponents)
    df_all_teams_swtiched = df_all_teams.copy(deep=True)
    df_all_teams_swtiched.rename(columns={'team':'opponent_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent':'team_hold'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'opponent_hold':'opponent'}, inplace=True)
    df_all_teams_swtiched.rename(columns={'team_hold':'team'}, inplace=True)
    # take out all the raw vars from df switched
    for variable in variables_for_team_metrics:
        df_all_teams_swtiched.pop(variable)        
    return df_all_teams_swtiched

df_covers_bball_ref_switched = create_switched_df(df_covers_bball_ref, variables_for_team_metrics)


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
       'opponent_fta_ewma_15','opponent_orb_ewma_15','opponent_pts_ewma_15','opponent_stl_ewma_15',
       'opponent_tov_ewma_15','venue_x_ewma_15','beat_spread_ewma_15',
       'spread_ewma_15','score_team_ewma_15','score_oppt_ewma_15','current_spread_vs_spread_ewma',
       'the_oppt_predicted_points_ewma_15', 'team_possessions', 'opponent_possessions', 
       'team_possessions_ewma_15', 'opponent_possessions_ewma_15'] 


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
        if variable in df_all_teams_swtiched.columns:
            df_all_teams_swtiched.rename(columns={variable:'x_'+variable}, inplace=True)
        else:
            print(variable)
    return df_all_teams_swtiched   
           
df_covers_bball_ref_switched = preface_oppt_stats_in_switched_df(df_covers_bball_ref_switched, variables_for_df)
df_covers_bball_ref_switched.columns


def merge_regular_df_w_switched_df(df_all_teams, df_all_teams_swtiched):    
    df_all_teams_w_ivs = df_all_teams.merge(df_all_teams_swtiched, on=['date', 'team', 'opponent'], how='left')
    df_all_teams_w_ivs.head(50)
    df_all_teams_w_ivs = df_all_teams_w_ivs.sort_values(by=['team','date'])
    df_all_teams_w_ivs = df_all_teams_w_ivs.reset_index(drop=True)
    return df_all_teams_w_ivs

# the vars with x at end are team's, the vars with ys are oppt's
df_covers_bball_ref = merge_regular_df_w_switched_df(df_covers_bball_ref, df_covers_bball_ref_switched)    
len(df_covers_bball_ref)
#df_covers_bball_ref[['team','date']]


# don't mess w venue, the_team_predicted_points, the_oppt_predicted_points,
# beat_spread, spread, score_team, score_oppt, totals


# the results if only main effects are exactly the same whether add back in the mean  or not
# but what about when i include X terms?
def adjust_metrics_for_opponent_quality(df_covers_bball_ref, variables_for_team_metrics):
    #variable = variables_for_team_metrics[2]
    # preserve the actual score of the team and oppt. because score_team is the dv so don't want to adjust     
    df_covers_bball_ref.loc[:,'actual_score_team'] = df_covers_bball_ref.loc[:,'score_team']
    df_covers_bball_ref.loc[:,'actual_score_oppt'] = df_covers_bball_ref.loc[:,'score_oppt']

#variable = variables_for_team_metrics[9]

    # residualize out oppt's ewma_15 stats
    for variable in variables_for_team_metrics[:]:
        #print(variable, len(df_covers_bball_ref[df_covers_bball_ref['x_' + variable + '_ewma_15'].isnull()]))      
        #print(variable, len(df_covers_bball_ref[df_covers_bball_ref[variable + '_ewma_15'].isnull()]))      
        if variable[:4] == 'team':
            print(variable)
            oppt_variable = 'x_opponent_' + variable[5:] + '_ewma_15'
            df_covers_bball_ref[oppt_variable].replace(np.nan, df_covers_bball_ref[oppt_variable].mean(), inplace=True)
            mod = smf.ols(formula = variable + ' ~ ' + oppt_variable + ' + spread',
                          data=df_covers_bball_ref).fit()  
#mod.summary()
#mod.resid.mean()
#df_covers_bball_ref[variable].mean()
#var_adj_for_oppt.mean()
#df_covers_bball_ref['team_TOVpct']
            var_adj_for_oppt = mod.resid + df_covers_bball_ref.loc[:,variable].mean()
            #var_adj_for_oppt = [(score + df_covers_bball_ref[variable].mean()) for score in mod.resid]
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
        elif variable[:4] == 'oppo':
            print(variable)
            oppt_variable = 'x_team_' + variable[9:] + '_ewma_15'  
            df_covers_bball_ref[oppt_variable].replace(np.nan, df_covers_bball_ref[oppt_variable].mean(), inplace=True)
            mod = smf.ols(formula = variable + ' ~ ' + oppt_variable + ' + spread',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid + df_covers_bball_ref.loc[:,variable].mean()
            #var_adj_for_oppt = [(score + df_covers_bball_ref[variable].mean()) for score in mod.resid]            
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
    # need to comment this out below. doesn't make sense. only want to ajd metrics
    # that i want to increase if they play a good teamo lower if they play a bad team
    # obviously venue doesn't make sense to do this with. spread and totals don't make
    # sense? if team is supposed to win by 5, hmmmm, i could adj by the spread of the
    # oppt. BUT it's not anything the team did. neither is totals. so skip. what about
    # score_team -- should probably do this one? if the typical defense of oppt is good,
    # then i want to take this into account.
        elif variable == 'score_team':
            oppt_variable = 'x_score_oppt_ewma_15'
            df_covers_bball_ref[oppt_variable].replace(np.nan, df_covers_bball_ref[oppt_variable].mean(), inplace=True)
            mod = smf.ols(formula = variable + ' ~ ' + oppt_variable + ' + spread',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid + df_covers_bball_ref.loc[:,variable].mean()
            #var_adj_for_oppt = [(score + df_covers_bball_ref[variable].mean()) for score in mod.resid]            
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
        elif variable == 'score_oppt':
            oppt_variable = 'x_score_team_ewma_15'
            df_covers_bball_ref[oppt_variable].replace(np.nan, df_covers_bball_ref[oppt_variable].mean(), inplace=True)
            mod = smf.ols(formula = variable + ' ~ ' + oppt_variable + ' + spread',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid + df_covers_bball_ref.loc[:,variable].mean()
            #var_adj_for_oppt = [(score + df_covers_bball_ref[variable].mean()) for score in mod.resid]
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
        else:
            None
    return df_covers_bball_ref      
        
df_covers_bball_ref = adjust_metrics_for_opponent_quality(df_covers_bball_ref, variables_for_team_metrics)



# ------ same as above except just adj each var for spread
def adjust_metrics_for_opponent_quality(df_covers_bball_ref, variables_for_team_metrics):
    #variable = variables_for_team_metrics[2]
    # preserve the actual score of the team and oppt. because score_team is the dv so don't want to adjust     
    df_covers_bball_ref.loc[:,'actual_score_team'] = df_covers_bball_ref.loc[:,'score_team']
    df_covers_bball_ref.loc[:,'actual_score_oppt'] = df_covers_bball_ref.loc[:,'score_oppt']

    # residualize out oppt's ewma_15 stats
    for variable in variables_for_team_metrics[:]:
        if variable[:4] == 'team':
            print(variable)
            mod = smf.ols(formula = variable + ' ~ spread + venue_x',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid #+ df_covers_bball_ref[variable].mean()
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
        if variable[:4] == 'oppo':
            print(variable)
            mod = smf.ols(formula = variable + ' ~ spread + venue_x',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid # + df_covers_bball_ref[variable].mean()
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
    # need to comment this out below. doesn't make sense. only want to ajd metrics
    # that i want to increase if they play a good teamo lower if they play a bad team
    # obviously venue doesn't make sense to do this with. spread and totals don't make
    # sense? if team is supposed to win by 5, hmmmm, i could adj by the spread of the
    # oppt. BUT it's not anything the team did. neither is totals. so skip. what about
    # score_team -- should probably do this one? if the typical defense of oppt is good,
    # then i want to take this into account.
        if variable == 'score_team':
            mod = smf.ols(formula = variable + ' ~ spread + venue_x',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid # + df_covers_bball_ref[variable].mean()
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
        if variable == 'score_oppt':
            mod = smf.ols(formula = variable + ' ~ spread + venue_x',
                          data=df_covers_bball_ref).fit()  
            var_adj_for_oppt = mod.resid # + df_covers_bball_ref[variable].mean()
            df_covers_bball_ref.loc[:,variable] = var_adj_for_oppt  # REMOVE test_
    
    return df_covers_bball_ref      
        
df_covers_bball_ref = adjust_metrics_for_opponent_quality(df_covers_bball_ref, variables_for_team_metrics)


# tests
df_covers_bball_ref[variables_for_team_metrics].head()
df_covers_bball_ref[['team_ORBpct', 'test_team_ORBpct']].head(10)
df_covers_bball_ref[['team_ORBpct', 'test_team_ORBpct']].tail(10)
df_covers_bball_ref[['team_ORBpct', 'test_team_ORBpct']].corr()
df_covers_bball_ref[['team_ORBpct', 'test_team_ORBpct']].mean()
plt.scatter(df_covers_bball_ref['team_ORBpct'] , df_covers_bball_ref['test_team_ORBpct'], alpha=.1)
plt.scatter(df_covers_bball_ref['opponent_ORBpct'] , df_covers_bball_ref['test_opponent_ORBpct'], alpha=.1)
plt.scatter(df_covers_bball_ref['score_team'] , df_covers_bball_ref['test_score_team'], alpha=.1)

df_covers_bball_ref[['score_team','actual_score_team','score_team_ewma_15']]
df_hawks = df_covers_bball_ref[df_covers_bball_ref['team']=='Atlanta Hawks']
plt.plot(df_hawks['score_team_ewma_15'][df_hawks['season_start']==2004])
plt.plot(df_hawks['score_team_ewma_15'][df_hawks['season_start']==2005])
df_covers_bball_ref[['score_team','actual_score_team','score_team_ewma_15']].corr()
plt.scatter(df_covers_bball_ref['actual_score_team'], df_covers_bball_ref['score_team'], alpha=.1)


# test to see if adj metric is highly corr wtih raw metric
#df_covers_bball_ref.columns
#sns.lmplot(x='opponent_possessions', y='test_opponent_possessions', data=df_covers_bball_ref)


# so now go back to teh first loop and just re-do (but skip this 2nd time around)
# group just by team, not by season too. need to do weighted mean or a rolling mean here.
def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')    
    for var in variables_for_team_metrics:
        var_ewma_15 = var + '_ewma_15'
        # i have played around with span=x in line below
        # 50 is worse than 20; 25 is worse than 20; 15 is better than 20; 10 is worse than 15; 16 is worse than 15; 14 is worse than 15
        # stick w 15. but 12 is second best. 
        # (12 looks better if want to use cutoffs and bet on fewer gs. but looks like 15 will get about same pct -- 53+ -- wih 1,000 gs)
        #df_covers_bball_ref[var_ewma_15] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=15))  # change back to 15

        # compute ewma w 5
        #var_ewma_5 = var + '_ewma_5'
        # i have played around with span=x in line below
        # 50 is worse than 20; 25 is worse than 20; 15 is better than 20; 10 is worse than 15; 16 is worse than 15; 14 is worse than 15
        # stick w 15. but 12 is second best. 
        # (12 looks better if want to use cutoffs and bet on fewer gs. but looks like 15 will get about same pct -- 53+ -- wih 1,000 gs)
        df_covers_bball_ref[var_ewma_15] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=6))  # change back to 15

        # avg the ewma_15 and ewma_5
        #df_covers_bball_ref[var_ewma_15] = df_covers_bball_ref[[var_ewma_15, var_ewma_5]].mean(axis=1)

    df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))
    df_covers_bball_ref['current_spread_vs_spread_ewma'] = df_covers_bball_ref.loc[:, 'spread'] - df_covers_bball_ref.loc[:, 'spread_ewma_15']
    df_covers_bball_ref['current_totals_vs_totals_ewma'] = df_covers_bball_ref.loc[:, 'totals'] - df_covers_bball_ref.loc[:, 'totals_ewma_15']
    # revert score_Team back to actual score of the team, so can use it as the dv
    df_covers_bball_ref = df_covers_bball_ref.sort_values(by=['team','date'])
    df_covers_bball_ref.loc[:,'score_team'] = df_covers_bball_ref.loc[:,'actual_score_team']
    df_covers_bball_ref.loc[:,'score_oppt'] = df_covers_bball_ref.loc[:,'actual_score_oppt']

    return df_covers_bball_ref

df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)





#def loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics):
#    df_covers_bball_ref = df_covers_bball_ref.sort_values(by='date')
#    for var in variables_for_team_metrics:
#        var_ewma = var + '_ewma_15'
#        df_covers_bball_ref[var_ewma] = df_covers_bball_ref.groupby('team')[var].transform(lambda x: pd.ewma(x.shift(1), span=8))  # change back to 15
#    df_covers_bball_ref['current_spread_vs_spread_ewma'] = df_covers_bball_ref.loc[:, 'spread'] - df_covers_bball_ref.loc[:, 'spread_ewma_15']
#    df_covers_bball_ref['current_totals_vs_totals_ewma'] = df_covers_bball_ref.loc[:, 'totals'] - df_covers_bball_ref.loc[:, 'totals_ewma_15']
#    df_covers_bball_ref['lineup_count'] = df_covers_bball_ref.groupby('starters_team')['team_3PAr'].transform(lambda x: pd.expanding_count(x.shift(1)))
#    return df_covers_bball_ref
#
#df_covers_bball_ref = loop_through_teams_to_create_rolling_metrics(df_covers_bball_ref, variables_for_team_metrics)
#


# ===============================
# ===============================



#------------------------------------------------------------------------------
# compute new variables in this section


# compute days rest
#df_covers_bball_ref[['date','team']].head(20)
#df_covers_bball_ref[['date','team']][4000:4050]
#df_covers_bball_ref = df_covers_bball_ref.sort_values('date')
df_covers_bball_ref.loc[:,'date_prior_game'] = df_covers_bball_ref.groupby('team')['date'].transform(lambda x: x.shift(1))
df_covers_bball_ref.loc[:,'days_rest'] = (df_covers_bball_ref.loc[:,'date'] - df_covers_bball_ref.loc[:,'date_prior_game']) / np.timedelta64(1, 'D')
df_covers_bball_ref.loc[df_covers_bball_ref['days_rest']>1, 'days_rest'] = 2
df_covers_bball_ref.loc[:,'point_difference'] = df_covers_bball_ref.loc[:,'score_team'] - df_covers_bball_ref.loc[:,'score_oppt']

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

df_covers_bball_ref.loc[:, 'team_zone'] = df_covers_bball_ref.loc[:,'team'].map(team_to_zone_dict)
df_covers_bball_ref.loc[:, 'opponent_zone'] = df_covers_bball_ref.loc[:,'opponent'].map(team_to_zone_dict)
df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'team_zone'] - df_covers_bball_ref.loc[:, 'opponent_zone']
#df_covers_bball_ref.loc[:, 'zone_distance'] = df_covers_bball_ref.loc[:, 'zone_distance'].abs()
#df_covers_bball_ref[['date', 'team','team_zone','opponent', 'opponent_zone', 'zone_distance']][4000:4030]

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

df_covers_bball_ref.loc[:,'conference'] = df_covers_bball_ref.loc[:,'team'].map(team_to_conference_dict)


def create_wins(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
    df_all_teams_w_ivs['team_win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'team_win'] = 1
    df_all_teams_w_ivs.loc[:,'team_win_pct'] = df_all_teams_w_ivs.groupby(['team', 'season_start'])['team_win'].transform(lambda x: pd.expanding_mean(x.shift(1), min_periods=1))
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
    df_standings_date.loc[:,'conference_rank'] = df_standings_date.groupby('conference')['team_win_pct'].rank(ascending=False)
    df_standings_date.loc[:,'top_win_pct_in_conference'] = df_standings_date.groupby('conference')['team_win_pct'].transform(lambda x: x.max())
    df_standings_date.loc[:,'distance_from_first_win_pct'] = df_standings_date.loc[:,'top_win_pct_in_conference'] - df_standings_date.loc[:,'team_win_pct']
    df_standings_date.loc[:,'eigth_pct_in_conference'] = df_standings_date.groupby('conference')['team_win_pct'].transform(lambda x: x.values[7])
    df_standings_date.loc[:,'ninth_pct_in_conference'] = df_standings_date.groupby('conference')['team_win_pct'].transform(lambda x: x.values[8])
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

df_covers_bball_ref.loc[:,'game'] = df_covers_bball_ref.groupby(['team','season_start'])['spread'].transform(lambda x: pd.expanding_count(x))
df_covers_bball_ref[['date', 'team', 'game']][df_covers_bball_ref['team']== 'Detroit Pistons'].head(100)

df_covers_bball_ref.loc[:,'distance_playoffs_abs'] = df_covers_bball_ref.loc[:,'distance_from_playoffs'].abs()
#df_covers_bball_ref[['distance_from_playoffs','distance_playoffs_abs']][5000:5050]
#df_covers_bball_ref[['distance_from_playoffs','distance_playoffs_abs']].head(20)
# should weight these so that takes the mean and gradually a team-specific number overtakes
# it as the n increases. how?

df_covers_bball_ref.loc[df_covers_bball_ref['distance_playoffs_abs'].isnull(), 'distance_playoffs_abs'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['game'] < 20, 'distance_playoffs_abs'] = df_covers_bball_ref.loc[:,'distance_playoffs_abs']/2
#df_covers_bball_ref[['date', 'game', 'distance_playoffs_abs', 'team']][df_covers_bball_ref['team']=='Atlanta Hawks']
#df_covers_bball_ref[['date', 'team', 'opponent', 'distance_from_playoffs', 'distance_playoffs_abs']][500:520]
#df_w_distance_from_playoffs_all_years[df_w_distance_from_playoffs_all_years['date']=='2004-12-06']
#df_w_distance_from_playoffs_all_years['date'].dtypes


# compute if last g has a diff lineup
# if the lineup is diff than the game before, does that make it harder to guess?
# need compute whether lineup is diff than prior g here
df_covers_bball_ref.loc[:,'starters_team_last_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(1))
df_covers_bball_ref.loc[:,'starters_team_two_g'] = df_covers_bball_ref.groupby('team')['starters_team'].transform(lambda x: x.shift(2))

df_covers_bball_ref.loc[:,'starters_same_as_last_g'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['starters_team']==df_covers_bball_ref['starters_team_last_g'], 'starters_same_as_last_g'] = 1

df_covers_bball_ref.loc[:,'starters_same_as_two_g'] = 0
df_covers_bball_ref.loc[df_covers_bball_ref['starters_team']==df_covers_bball_ref['starters_team_two_g'], 'starters_same_as_two_g'] = 1

#df_covers_bball_ref[['date', 'starters_team', 'starters_team_last_g', 'starters_same_as_last_g']][df_covers_bball_ref['team']=='San Antonio Spurs'].tail(10)
# but doesn't seem to help in model, at last at this point


# if i looped twice (to adj for oppt):
if 'venue_y' not in df_covers_bball_ref.columns:
    print('true')    
    df_covers_bball_ref['venue_y'] = df_covers_bball_ref['venue_y_y']

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

#g = df_covers_bball_ref.groupby('team')['home_court_advantage'].mean()
#g.sort()
#g.plot(kind='barh', sort_columns=True)
#plt.xlabel('home court advantage')
# makes sense -- when you play at home, you get half of this home court adv.
# and when you play away, you loose half of it.

df_covers_bball_ref.loc[:,'spread_abs_val'] = df_covers_bball_ref.loc[:,'spread'].abs()

#------------------------------------------------------------------------------
df_covers_bball_ref_save = df_covers_bball_ref.copy(deep=True)

# run this again when want to add a new var and re-run below code:
df_covers_bball_ref = df_covers_bball_ref_save.copy(deep=True)
#------------------------------------------------------------------------------
df_covers_bball_ref.rename(columns={'venue_x':'venue'}, inplace=True)

#for var in df_covers_bball_ref.columns:
#    print(var)

#--------------

# add or subtract vars from two lists below. should only have to do that once, here.

# include all vars here that i'll want to use:
#variables_for_df = ['date', 'team', 'opponent', 'venue', 'lineup_count', 
#       'starters_team', 'starters_opponent', 'game',
#       'spread', 'totals', 'score_team', 'score_oppt', 'beat_spread', 
#       'over', 'beat_spread_last_g', 'season_start', 'days_rest', 'zone_distance', 
#       'distance_playoffs_abs', 'starters_same_as_last_g', 'point_total', 'team_predicted_points', 
#       'oppt_predicted_points', 'home_team_score_advantage', 'home_oppt_score_advantage',
#       'team_3PAr_ewma_15','team_ASTpct_ewma_15','team_BLKpct_ewma_15','team_DRBpct_ewma_15',
#       'team_DRtg_ewma_15','team_FTr_ewma_15','team_ORBpct_ewma_15','team_ORtg_ewma_15',
#       'team_STLpct_ewma_15','team_TOVpct_ewma_15','team_TRBpct_ewma_15','team_TSpct_ewma_15',
#       'team_eFGpct_ewma_15','team_fg3_pct_ewma_15','team_fg_pct_ewma_15','team_ft_pct_ewma_15',
#       'team_pf_ewma_15','team_predicted_points_ewma_15','opponent_3PAr_ewma_15',
#       'opponent_ASTpct_ewma_15','opponent_BLKpct_ewma_15','opponent_DRBpct_ewma_15',
#       'opponent_DRtg_ewma_15','opponent_FTr_ewma_15','opponent_ORBpct_ewma_15','opponent_ORtg_ewma_15',
#       'opponent_STLpct_ewma_15','opponent_TOVpct_ewma_15','opponent_TRBpct_ewma_15',
#       'opponent_TSpct_ewma_15','opponent_eFGpct_ewma_15','opponent_fg3_pct_ewma_15',
#       'opponent_fg_pct_ewma_15','opponent_ft_pct_ewma_15','opponent_pf_ewma_15',
#       'team_ast_ewma_15','team_blk_ewma_15','team_drb_ewma_15','team_fg_ewma_15',
#       'team_fg3_ewma_15','team_fg3a_ewma_15','team_fga_ewma_15','team_ft_ewma_15',
#       'team_fta_ewma_15','team_orb_ewma_15','team_stl_ewma_15','team_tov_ewma_15',
#       'opponent_ast_ewma_15','opponent_blk_ewma_15','opponent_drb_ewma_15','opponent_fg_ewma_15',
#       'opponent_fg3_ewma_15','opponent_fg3a_ewma_15','opponent_fga_ewma_15','opponent_ft_ewma_15',
#       'opponent_fta_ewma_15','opponent_orb_ewma_15','opponent_pts_ewma_15','opponent_stl_ewma_15',
#       'opponent_tov_ewma_15','venue_x_ewma_15','oppt_predicted_points_ewma_15','beat_spread_ewma_15',
#       'spread_ewma_15','score_team_ewma_15','score_oppt_ewma_15','current_spread_vs_spread_ewma',
#       'lineup_count'] 

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
       'team_possessions_ewma_15', 'opponent_possessions_ewma_15', 'Opening Total', 'Closing Total']


# include all vars i want to precict with
#iv_variables = ['venue', 'lineup_count', 'game','spread', 'totals', 'beat_spread_last_g', 
#'days_rest', 'zone_distance', 'distance_playoffs_abs', 'team_predicted_points',  # starters_same_as_last_g
#'oppt_predicted_points', 'home_team_score_advantage', 'home_oppt_score_advantage',
#'team_3PAr_ewma_15','team_ASTpct_ewma_15','team_BLKpct_ewma_15','team_DRBpct_ewma_15',
#'team_DRtg_ewma_15','team_FTr_ewma_15','team_ORBpct_ewma_15','team_ORtg_ewma_15',
#'team_STLpct_ewma_15','team_TOVpct_ewma_15','team_TRBpct_ewma_15','team_TSpct_ewma_15',
#'team_eFGpct_ewma_15','team_fg3_pct_ewma_15','team_fg_pct_ewma_15','team_ft_pct_ewma_15',
#'team_pf_ewma_15','team_predicted_points_ewma_15','opponent_3PAr_ewma_15', 'opponent_ASTpct_ewma_15',
#'opponent_BLKpct_ewma_15','opponent_DRBpct_ewma_15','opponent_DRtg_ewma_15','opponent_FTr_ewma_15',
#'opponent_ORBpct_ewma_15','opponent_ORtg_ewma_15','opponent_STLpct_ewma_15','opponent_TOVpct_ewma_15',
#'opponent_TRBpct_ewma_15','opponent_TSpct_ewma_15','opponent_eFGpct_ewma_15','opponent_fg3_pct_ewma_15',
#'opponent_fg_pct_ewma_15','opponent_ft_pct_ewma_15','opponent_pf_ewma_15','team_ast_ewma_15',
#'team_blk_ewma_15','team_drb_ewma_15','team_fg_ewma_15','team_fg3_ewma_15','team_fg3a_ewma_15',
#'team_fga_ewma_15','team_ft_ewma_15','team_fta_ewma_15','team_orb_ewma_15','team_stl_ewma_15',
#'team_tov_ewma_15','opponent_ast_ewma_15','opponent_blk_ewma_15','opponent_drb_ewma_15',
#'opponent_fg_ewma_15','opponent_fg3_ewma_15','opponent_fg3a_ewma_15','opponent_fga_ewma_15',
#'opponent_ft_ewma_15','opponent_fta_ewma_15','opponent_orb_ewma_15','opponent_pts_ewma_15',
#'opponent_stl_ewma_15','opponent_tov_ewma_15','venue_x_ewma_15','oppt_predicted_points_ewma_15',
#'beat_spread_ewma_15','spread_ewma_15','score_team_ewma_15','score_oppt_ewma_15',
#'current_spread_vs_spread_ewma'] 
     
#iv_variables = ['venue','game','beat_spread_last_g', 
#'days_rest', 'zone_distance', 'distance_playoffs_abs', 'team_predicted_points',  # starters_same_as_last_g
#'oppt_predicted_points', 'home_team_score_advantage', 'home_oppt_score_advantage',
#'team_3PAr_ewma_15','team_DRtg_ewma_15','team_FTr_ewma_15','team_ORtg_ewma_15',
#'team_pf_ewma_15','team_predicted_points_ewma_15','opponent_3PAr_ewma_15',
#'opponent_DRtg_ewma_15','opponent_FTr_ewma_15','opponent_ORtg_ewma_15','opponent_pf_ewma_15',
#'team_ast_ewma_15','team_blk_ewma_15','team_drb_ewma_15','team_fg_ewma_15','team_fg3_ewma_15',
#'team_fg3a_ewma_15','team_fga_ewma_15','team_ft_ewma_15','team_fta_ewma_15','team_orb_ewma_15',
#'team_stl_ewma_15','team_tov_ewma_15','opponent_ast_ewma_15','opponent_blk_ewma_15',
#'opponent_drb_ewma_15','opponent_fg_ewma_15','opponent_fg3_ewma_15','opponent_fg3a_ewma_15',
#'opponent_fga_ewma_15','opponent_ft_ewma_15','opponent_fta_ewma_15','opponent_orb_ewma_15',
#'opponent_pts_ewma_15','opponent_stl_ewma_15','opponent_tov_ewma_15','venue_x_ewma_15',
#'oppt_predicted_points_ewma_15','score_team_ewma_15','score_oppt_ewma_15',
#'current_spread_vs_spread_ewma'] 
  

iv_variables = ['game', 'beat_spread_last_g', 'lineup_count', 'spread', 'spread_abs_val', #'totals',
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
'team_possessions_ewma_15', 'the_oppt_predicted_points_ewma_15', # 
'opponent_possessions_ewma_15', 'score_team_ewma_15', 'score_oppt_ewma_15'] 
#'current_totals_vs_totals_ewma', #'current_spread_vs_spread_ewma', 'beat_spread_ewma_15',

   
df_covers_bball_ref.loc[:,'point_total'] = df_covers_bball_ref[['score_team', 'score_oppt']].sum(axis=1)
df_covers_bball_ref[['score_team', 'score_oppt', 'point_total']].head()

df_covers_bball_ref['score_team'].hist(bins=20, alpha=.7)

# log transform the dv
df_covers_bball_ref.loc[:, 'score_team_log'] = np.log(df_covers_bball_ref.loc[:, 'score_team'])
df_covers_bball_ref[['score_team', 'score_team_log']].head()
df_covers_bball_ref['score_team_log'].hist(bins=20, alpha=.6)
stats.probplot(df_covers_bball_ref['score_team'], dist="norm", plot=plt)
stats.probplot(df_covers_bball_ref['score_team_log'], dist="norm", plot=plt)

df_covers_bball_ref.loc[:, 'score_team_sq_rt'] = np.sqrt(df_covers_bball_ref.loc[:, 'score_team'])
df_covers_bball_ref['score_team_sq_rt'].hist(bins=20, alpha=.6)
stats.probplot(df_covers_bball_ref['score_team_sq_rt'], dist="norm", plot=plt)
# teh square root may be better than cube root in as far as sq root has equal tails

df_covers_bball_ref.loc[:, 'score_team_cu_rt'] = np.cbrt(df_covers_bball_ref.loc[:, 'score_team'])
df_covers_bball_ref['score_team_cu_rt'].hist(bins=20, alpha=.6)
stats.probplot(df_covers_bball_ref['score_team_cu_rt'], dist="norm", plot=plt)

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
    df_all_teams_w_ivs = df_all_teams_w_ivs.sort_values(by=['team','date'])
    df_all_teams_w_ivs = df_all_teams_w_ivs.reset_index(drop=True)
    return df_all_teams_w_ivs

df_covers_bball_ref = merge_regular_df_w_switched_df(df_covers_bball_ref, df_covers_bball_ref_switched)    
len(df_covers_bball_ref)


def create_basic_variables(df_all_teams_w_ivs):
    # create variables -- maybe should put elsewhere, earlier
    df_all_teams_w_ivs['over_win'] = 'push'
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['over'] > 0, 'over_win'] = '1'
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['over'] < 0, 'over_win'] = '0'
    #df_all_teams_w_ivs[['date', 'team', 'opponent', 'beat_spread', 'over_win']]
    df_all_teams_w_ivs['win'] = 0
    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'win'] = 1
    #df_all_teams_w_ivs[['score_team', 'score_oppt', 'win']]
    df_all_teams_w_ivs.loc[:,'point_difference'] = df_all_teams_w_ivs.loc[:,'score_team'] - df_all_teams_w_ivs.loc[:,'score_oppt']
    return df_all_teams_w_ivs

df_covers_bball_ref = create_basic_variables(df_covers_bball_ref)



# get ewma of x(oppt's) up to that point. so get
# if iv_var has ewma_15 at end and x_ at beginning -- all the oppt ewma 15
# vars, both their offensive and defenseive (i.e., x_team...ewma_15 and 
# x_opponent...ewma_15). include these in ivs. see what happens there. 
# should be better, right? taking into account the opponent. or was my
# thinking that i should only take into account the team because then i'll
# have doubled stats? shouldn't matter i don't think. anyway, next, 
# create ewma x_ (opponent) vars from the ewma vars. i.e., this way i'm
# getting a sense for how good the recent opponents were and that should
# help adjust the team ewma vars because these will be correlated with the
# oppt over the last 6 games.


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

# this below code seemed to make worse. figure out which var(s) made worse
# why don't i have oppt day's rest in here? and all other x_vars! distance from playoffs, etc.
iv_variables_more = ['days_rest' ,'spread_ewma_15', 'score_team_ewma_15',
                     'score_oppt_ewma_15']  # distance_playoffs_abs seems to screw it up
                     # 'the_team_predicted_points_ewma_15', 'the_oppt_predicted_points_ewma_15' also screw it up
                     # should i remove these from the team side? try it

#iv_variables_more = ['days_rest' ,'spread_ewma_15', 'score_team_ewma_15',
#                     'score_oppt_ewma_15', 'distance_playoffs_abs', 'the_team_predicted_points_ewma_15', 
#                     'the_oppt_predicted_points_ewma_15']

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



# skip for totals prediction
#def create_team_opponent_difference_variables(df_all_teams_w_ivs, iv_variables):
#    for var in iv_variables:
#        new_difference_variable = 'difference_'+var
#        df_all_teams_w_ivs.loc[:, new_difference_variable] = df_all_teams_w_ivs.loc[:, var] - df_all_teams_w_ivs.loc[:, 'x_'+var]
#    return df_all_teams_w_ivs
#
#df_covers_bball_ref = create_team_opponent_difference_variables(df_covers_bball_ref, iv_variables)


#def create_basic_variables(df_all_teams_w_ivs):
#    # create variables -- maybe should put elsewhere, earlier
#    df_all_teams_w_ivs['ats_win'] = 'push'
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] > 0, 'ats_win'] = '1'
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['beat_spread'] < 0, 'ats_win'] = '0'
#    df_all_teams_w_ivs[['date', 'team', 'opponent', 'beat_spread', 'ats_win']]
#    df_all_teams_w_ivs['win'] = 0
#    df_all_teams_w_ivs.loc[df_all_teams_w_ivs['score_team'] > df_all_teams_w_ivs['score_oppt'], 'win'] = 1
#    df_all_teams_w_ivs[['score_team', 'score_oppt', 'win']]
#    df_all_teams_w_ivs['point_difference'] = df_all_teams_w_ivs['score_team'] - df_all_teams_w_ivs['score_oppt']
#    return df_all_teams_w_ivs
#
#df_covers_bball_ref = create_basic_variables(df_covers_bball_ref)




#df_covers_bball_ref.loc[:, 'spread_x_the_team_predicted_points_x_venue'] = df_covers_bball_ref.loc[:, 'the_team_predicted_points'] * df_covers_bball_ref.loc[:, 'spread'] * df_covers_bball_ref.loc[:, 'venue']
#iv_variables = iv_variables + ['spread_x_the_team_predicted_points_x_venue']        

#df_covers_bball_ref.loc[:, 'spread_x_the_team_offense_15'] = df_covers_bball_ref.loc[:, 'team_ORtg_ewma_15'] * df_covers_bball_ref.loc[:, 'spread'] 
#iv_variables = iv_variables + ['spread_x_the_team_offense_15']        





# SKIP BECAUSE THIS IS TO ADD DIFFERENCE VARS TO THE IV LIST------------------
#def create_iv_list(iv_variables, variables_without_difference_score_list):
#    """Put all variables that don't want to compute a difference score on."""
#    spread_and_totals_list = variables_without_difference_score_list
#    for i in range(len(spread_and_totals_list)):
#        print(i)
#        iv_variables.remove(spread_and_totals_list[i])
#    x_iv_variables = ['x_'+iv_var for iv_var in iv_variables]
#    team_iv_variables = iv_variables
#    iv_variables = team_iv_variables + x_iv_variables + spread_and_totals_list
#    return iv_variables
#
#iv_variables = create_iv_list(iv_variables, ['game', 'venue', 'team_predicted_points', 'oppt_predicted_points'])  # spread


# not sure about this, but can re-visit:
# THINK I SHOULD TAKE OUT ALL THE OPPONENT METRICS AND ALL THE X_TEAM METRICS. 
# THAT WAY I'LL JUST BE PREDICING THE SCORE BASED ON THE TEAMS OFFENSE
# AND THE OPPONENT'S DEFENSE. ALTHOUGH COULD MAKE ARGUMENT THAT THE TEAM'S
# DEFENSE COULD SLOW THINGS DOWN AND SO CHANGE THE SCORE?


# skip
#def create_home_df(df_covers_bball_ref):
#    df_covers_bball_ref_home = df_covers_bball_ref[df_covers_bball_ref['venue'] == 0]
#    df_covers_bball_ref_home = df_covers_bball_ref_home.reset_index()
#    len(df_covers_bball_ref_home)
#    return df_covers_bball_ref_home
#
#df_covers_bball_ref_home = create_home_df(df_covers_bball_ref)


#--------------------
# create new vars

# home court advantage - not working. how else to compute? subtract home win% from oppt away win%
#df_covers_bball_ref.loc[:, 'home_advantage_added'] = df_covers_bball_ref.loc[:, 'home_point_diff_ewma'] - df_covers_bball_ref.loc[:, 'x_away_point_diff_ewma']
#iv_variables = iv_variables + ['home_advantage_added', 'home_point_diff_ewma', 'away_point_diff_ewma', 'x_home_point_diff_ewma', 'x_away_point_diff_ewma']
#variables_for_df = variables_for_df + ['home_advantage_added', 'x_away_point_diff_ewma']

#df_covers_bball_ref_home.loc[:, 'home_court_advantage_difference'] = df_covers_bball_ref_home.loc[:, 'home_court_advantage'] - df_covers_bball_ref_home.loc[:, 'x_home_court_advantage']*-1
#iv_variables = iv_variables + ['home_court_advantage_difference']
#variables_for_df = variables_for_df + ['home_court_advantage_difference']


# ----------------------------
# call the main file df_covers_bball_ref__dropna_home even though it's not just
# home games. but it'll let me use that file name in code below.
df_covers_bball_ref__dropna_home = df_covers_bball_ref.copy(deep=True)
# drop nans
print('\n number of games with nans:', len(df_covers_bball_ref))
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.dropna()
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.sort_values(by=['team','date'])
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.reset_index(drop=True)
print('\n number of games without nans:', (len(df_covers_bball_ref__dropna_home)))
#df_covers_bball_ref__dropna_home[['date','team','spread']]

#for col in df_covers_bball_ref.columns:
#    if len(df_covers_bball_ref[df_covers_bball_ref[col].notnull()]) < 15000:
#        print(col, len(df_covers_bball_ref[df_covers_bball_ref[col].notnull()]))

# ----------------------------
# scale vars - skp this here. right way to do it is to standardize the training
# set, then take those params and apply to the test set. but then what about Xs?
# should then compute all the Xs at that point? or, if standarsize the training 
# set Xs, can i apply those params to test set X terms? so could move teh below
# standardizing code to the part where i split the train and test set.

#variable = iv_variables[0]
#iv_variables_z = []
#for i, variable in enumerate(iv_variables[:]):
#    iv_variables_z.append(variable+'_z')
#    #df_covers_bball_ref__dropna_home.loc[:,variable +'_z'] = (df_covers_bball_ref__dropna_home.loc[:,variable] - df_covers_bball_ref__dropna_home.loc[:,variable].mean()) / df_covers_bball_ref__dropna_home.loc[:,variable].std()
#    df_covers_bball_ref__dropna_home.loc[:,variable +'_z'] = StandardScaler().fit_transform(df_covers_bball_ref__dropna_home[[variable]])
#    model_plot = smf.ols(formula = 'score_team ~ ' + variable,
#                data=df_covers_bball_ref__dropna_home).fit()  
#    t = round(model_plot.tvalues[1], 3)
#    p = round(model_plot.pvalues[1], 3)
#    model_plot = smf.ols(formula = 'score_team ~ ' + variable+'_z',
#                data=df_covers_bball_ref__dropna_home).fit()  
#    t_z = round(model_plot.tvalues[1], 3)
#    p_z = round(model_plot.pvalues[1], 3)
#    if t == t_z and p == p_z:
#        print(i, 'ok', p_z)
#    else:
#        print(i, variable)


#iv_variables_original = iv_variables
# CHANGE THIS TO iv_variables_to_analyze? I THINK SO. AND THEN DON'T SCALE IN FINAL FUNCTION
#iv_variables = iv_variables_z


#df_covers_bball_ref__dropna_home[[variable, variable+'_z']].hist()


## scale 0 to 1
##(x - minimum) / (maximum - minimum)
#iv_variables_z = []
#for i, variable in enumerate(iv_variables[:]):
#    iv_variables_z.append(variable+'_z')
#    df_covers_bball_ref__dropna_home.loc[:,variable +'_z'] = (df_covers_bball_ref__dropna_home.loc[:, variable] - df_covers_bball_ref__dropna_home[variable].min()) / (df_covers_bball_ref__dropna_home[variable].max() - df_covers_bball_ref__dropna_home[variable].min())
#    print(variable)
#    #df_covers_bball_ref__dropna_home.loc[:,variable +'_z'] = StandardScaler().fit_transform(df_covers_bball_ref__dropna_home[[variable]])
##    model_plot = smf.ols(formula = 'score_team ~ ' + variable,
##                data=df_covers_bball_ref__dropna_home).fit()  
##    t = round(model_plot.tvalues[1], 3)
##    p = round(model_plot.pvalues[1], 3)
##    model_plot = smf.ols(formula = 'score_team ~ ' + variable+'_z',
##                data=df_covers_bball_ref__dropna_home).fit()  
##    t_z = round(model_plot.tvalues[1], 3)
##    p_z = round(model_plot.pvalues[1], 3)
##    if t == t_z and p == p_z:
##        print(i, 'ok', p_z)
##    else:
##        print(i, variable)
#
#iv_variables_original = iv_variables
## CHANGE THIS TO iv_variables_to_analyze? I THINK SO. AND THEN DON'T SCALE IN FINAL FUNCTION
##iv_variables = iv_variables_z
##iv_variables_to_analyze = iv_variables_z
#iv_variables = iv_variables_z
#


# do regression right here w and without standardizing vars. should be same.
string_for_regression = ''
for var in iv_variables:
    print(var)
    string_for_regression += ' + ' + var
string_for_regression = string_for_regression[3:]        
model_plot_2 = smf.ols(formula = 'score_team ~ ' + string_for_regression,
                data=df_covers_bball_ref__dropna_home).fit()  
# this works:
#model_plot = smf.glm(formula = 'score_team ~ ' + string_for_regression,
#                data=df_covers_bball_ref__dropna_home, family=sm.families.Poisson()).fit()  
print(model_plot.summary()) 
print(model_plot_2.summary()) 
# exactly the same! so why are ml results differing based on z vs not? 
# 

R-squared: 0.266
Adj. R-squared: 0.262
AIC: 2.152e+05

# this doesn't work -- because need to enter dv and iv differently.
#model_plot = sm.Poisson(formula = 'score_team ~ ' + string_for_regression,
#                data=df_covers_bball_ref__dropna_home).fit()  




# SKIP Xs FOR NOW. NOT SURE HELPING.
# --------------------------------------------------------------
# if standardizing, need to create Xs after, after standardizing

# need to set this in order to do any of the interactions below
iv_variables_original = iv_variables


# this made thinkgs worse:
#
## multiply team's posesessions_ewma_15 by all vars. and do same for oppt
## logic: the team has these vars that say how good it is, but they're mostly
## standardize per 100 possessions or a percentage. so multiplying the by
## the number of posessions we'd expect should translate that percentage into
## an actual value, and that should match up and better predict actual points
#def create_team_possessions_multiplicative_variables(df_all_teams_w_ivs, iv_variables):
#    for var in iv_variables:
#        #if var[:4] == 'team':
#        new_team_possessions_X_variable = 'team_possessions_by_'+var
#        new_oppt_possessions_X_variable = 'opponent_possessions_by_'+var
#        # possession vars need to be z if using z-scored ivs. i think. can add
#        # a switch to the params and do if statement here.
#        df_all_teams_w_ivs.loc[:, new_team_possessions_X_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'team_possessions_ewma_15']
#        df_all_teams_w_ivs.loc[:, new_oppt_possessions_X_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'opponent_possessions_ewma_15']
#        iv_variables = iv_variables + [new_team_possessions_X_variable, new_oppt_possessions_X_variable]        
#    return df_all_teams_w_ivs, iv_variables
#
#df_covers_bball_ref, iv_variables = create_team_possessions_multiplicative_variables(df_covers_bball_ref__dropna_home, iv_variables_original)



# doesn't help
#def create_spread_multiplicative_variables(df_all_teams_w_ivs, iv_variables):
#    for var in iv_variables:
#        #if var[:4] == 'team':
#        new_spread_X_variable = 'spread_by_'+var
#        df_all_teams_w_ivs.loc[:, new_spread_X_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'spread']
#        iv_variables = iv_variables + [new_spread_X_variable]        
#    return df_all_teams_w_ivs, iv_variables
#
#df_covers_bball_ref, iv_variables = create_spread_multiplicative_variables(df_covers_bball_ref__dropna_home, iv_variables_original)



# so...multiply the team vs the x_oppt vars here? and then in below function
# keep the indiv vars but add the mutiplicative terms too? 
# *****didn't seem to help much
#def create_team_opponent_multiplicative_variables(df_all_teams_w_ivs, iv_variables):
#    for var in iv_variables:
#        if var[:4] == 'team':
#            new_off_def_multiplicative_variable = 'multiply_team_off_by_oppt_def_'+var
#            df_all_teams_w_ivs.loc[:, new_off_def_multiplicative_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'x_opponent'+var[4:]]
#            iv_variables = iv_variables + [new_off_def_multiplicative_variable]
#        if var[:4] == 'oppo':
#            new_def_off_multiplicative_variable = 'multiply_team_def_by_oppt_off_'+var
#            df_all_teams_w_ivs.loc[:, new_def_off_multiplicative_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'x_team'+var[8:]]                        
#            iv_variables = iv_variables + [new_def_off_multiplicative_variable]
#    return df_all_teams_w_ivs, iv_variables
#
#df_covers_bball_ref, iv_variables = create_team_opponent_multiplicative_variables(df_covers_bball_ref__dropna_home, iv_variables_original)



# *****didn't seem to help much
#def create_team_opponent_multiplicative_variables(df_all_teams_w_ivs, iv_variables):
#    for var in iv_variables:
#        if var[:4] == 'team':
#            new_off_def_multiplicative_variable = 'multiply_team_off_by_oppt_off_'+var
#            df_all_teams_w_ivs.loc[:, new_off_def_multiplicative_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'x_team'+var[4:]]
#            iv_variables = iv_variables + [new_off_def_multiplicative_variable]
#        if var[:4] == 'oppo':
#            new_def_off_multiplicative_variable = 'multiply_team_def_by_oppt_def_'+var
#            df_all_teams_w_ivs.loc[:, new_def_off_multiplicative_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'x_opponent'+var[8:]]                        
#            iv_variables = iv_variables + [new_def_off_multiplicative_variable]
#    return df_all_teams_w_ivs, iv_variables
#
#df_covers_bball_ref, iv_variables = create_team_opponent_multiplicative_variables(df_covers_bball_ref__dropna_home, iv_variables_original)


## --------------------------------------------------------------
## if standardizing, need to create Xs later, after standardizing
#iv_variables_original = iv_variables
#
## multiply team's posesessions_ewma_15 by all vars. and do same for oppt
## logic: the team has these vars that say how good it is, but they're mostly
## standardize per 100 possessions or a percentage. so multiplying the by
## the number of posessions we'd expect should translate that percentage into
## an actual value, and that should match up and better predict actual points
#def create_team_possessions_multiplicative_variables(df_all_teams_w_ivs, iv_variables):
#    for var in iv_variables:
#        #if var[:4] == 'team':
#        new_team_possessions_X_variable = 'team_possessions_by_'+var
#        new_oppt_possessions_X_variable = 'opponent_possessions_by_'+var
#        df_all_teams_w_ivs.loc[:, new_team_possessions_X_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'team_possessions_ewma_15_z']
#        df_all_teams_w_ivs.loc[:, new_oppt_possessions_X_variable] = df_all_teams_w_ivs.loc[:, var] * df_all_teams_w_ivs.loc[:, 'opponent_possessions_ewma_15_z']
#        iv_variables = iv_variables + [new_team_possessions_X_variable, new_oppt_possessions_X_variable]        
#    return df_all_teams_w_ivs, iv_variables
#
#df_covers_bball_ref, iv_variables = create_team_possessions_multiplicative_variables(df_covers_bball_ref__dropna_home, iv_variables_original)



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
pca_stand = PCA(n_components=60).fit(df_covers_bball_ref__dropna_home[iv_variables])
variable_names = []
for i in range(60):
    variable = '_'+str(i)
    variable_names.append(variable)
df_home_pca =  pd.DataFrame(pca_stand.transform(df_covers_bball_ref__dropna_home[iv_variables]), columns=variable_names)
df_home_pca['_4']
df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home.reset_index()
df_covers_bball_ref__dropna_home = pd.concat([df_covers_bball_ref__dropna_home, df_home_pca], axis=1)
df_covers_bball_ref__dropna_home.head()

iv_variables = variable_names

iv_variables = iv_variables + ['team_predicted_points_z']

# remove team defense vars opponent defense offense vars
#vars_to_remove = ['opponent_3PAr_ewma_15_z','opponent_ASTpct_ewma_15_z',
# 'opponent_BLKpct_ewma_15_z', 'opponent_DRBpct_ewma_15_z', 'opponent_DRtg_ewma_15_z',
# 'opponent_FTr_ewma_15_z', 'opponent_ORBpct_ewma_15_z', 'opponent_ORtg_ewma_15_z',
# 'opponent_STLpct_ewma_15_z', 'opponent_TOVpct_ewma_15_z',
# 'opponent_TRBpct_ewma_15_z', 'opponent_TSpct_ewma_15_z', 'opponent_eFGpct_ewma_15_z',
# 'opponent_fg3_pct_ewma_15_z', 'opponent_fg_pct_ewma_15_z', 'opponent_ft_pct_ewma_15_z',
# 'opponent_pf_ewma_15_z',  'x_team_3PAr_ewma_15_z', 'x_team_ASTpct_ewma_15_z', 
# 'x_team_BLKpct_ewma_15_z', 'x_team_DRBpct_ewma_15_z', 'x_team_DRtg_ewma_15_z', 
# 'x_team_FTr_ewma_15_z', 'x_team_ORBpct_ewma_15_z', 'x_team_ORtg_ewma_15_z',
# 'x_team_STLpct_ewma_15_z', 'x_team_TOVpct_ewma_15_z', 'x_team_TRBpct_ewma_15_z',
# 'x_team_TSpct_ewma_15_z', 'x_team_eFGpct_ewma_15_z', 'x_team_fg3_pct_ewma_15_z',
# 'x_team_fg_pct_ewma_15_z', 'x_team_ft_pct_ewma_15_z', 'x_team_pf_ewma_15_z',]
#
#for var in vars_to_remove:
#    iv_variables.remove(var)
#

# ---------------------------
# add interactions -- do after scaling. skip for now.
df_covers_bball_ref__dropna_home.loc[:, 'venue_x_home_team_score_advantage_z'] = df_covers_bball_ref__dropna_home.loc[:, 'venue_z'] * df_covers_bball_ref__dropna_home.loc[:, 'home_team_score_advantage_z']
df_covers_bball_ref__dropna_home.loc[:, 'venue_x_oppt_team_score_advantage_z'] = df_covers_bball_ref__dropna_home.loc[:, 'venue_z'] * df_covers_bball_ref__dropna_home.loc[:, 'home_oppt_score_advantage_z']
iv_variables = iv_variables + ['venue_x_home_team_score_advantage_z']
iv_variables = iv_variables + ['venue_x_oppt_team_score_advantage_z']
variables_for_df = variables_for_df + ['venue_x_home_team_score_advantage_z']
variables_for_df = variables_for_df + ['venue_x_oppt_team_score_advantage_z']

# add interactions w spread because was predicting poorly for those wtih a high spread
# THIS ISN'T HELPING AT MOMENT. SKIP.
variables_for_spread_interaction = ['beat_spread_last_g_z', 'beat_spread_last_g', 'team_predicted_points',
'oppt_predicted_points','team_3PAr_ewma_15','opponent_3PAr_ewma_15',
'team_ast_ewma_15','team_blk_ewma_15','team_drb_ewma_15', 'team_fg_ewma_15',
'team_fg3_ewma_15','team_fg3a_ewma_15', 'team_fga_ewma_15',]

variables_for_spread_interaction = ['totals', 'days_rest_z', 'zone_distance_z',
'distance_playoffs_abs_z','team_predicted_points_ewma_15_z','oppt_predicted_points_ewma_15_z',
'current_spread_vs_spread_ewma_z','x_team_predicted_points_ewma_15_z','x_current_spread_vs_spread_ewma_z',
'team_predicted_points_z','oppt_predicted_points_z', 'team_ft_ewma_15','team_fta_ewma_15',
'team_orb_ewma_15','team_stl_ewma_15','team_tov_ewma_15']

for variable in variables_for_spread_interaction:
    df_covers_bball_ref__dropna_home.loc[:, variable+'_x_spread'] = df_covers_bball_ref__dropna_home.loc[:,variable] * df_covers_bball_ref__dropna_home.loc[:, 'spread']
    iv_variables = iv_variables + [variable+'_x_spread']
    variables_for_df = variables_for_df + [variable+'_x_spread']
    
for col in df_covers_bball_ref__dropna_home.columns:
    print(col)
#
#df_covers_bball_ref__dropna_home[['team_predicted_points_z', 'spread', 'team_predicted_points_z_x_spread']].head(10)



# SKIP INTERACTIONS BELOW FOR NOW. PROBABLY WANT DIFF ONES FOR PREDICINT TEAM'S POINTS.

# seemed to help, so use. and explore more
df_covers_bball_ref__dropna_home['game_x_playoff_distance'] = df_covers_bball_ref__dropna_home['game'] * df_covers_bball_ref__dropna_home['distance_playoffs_abs']
iv_variables = iv_variables + ['game_x_playoff_distance']
variables_for_df = variables_for_df + ['game_x_playoff_distance']

iv_variables_original = iv_variables_original + ['game_x_playoff_distance']
variables_for_df = variables_for_df + ['game_x_playoff_distance']

df_covers_bball_ref__dropna_home['game_x_playoff_distance_z'] = df_covers_bball_ref__dropna_home['game_z'] * df_covers_bball_ref__dropna_home['distance_playoffs_abs_z']
iv_variables = iv_variables + ['game_x_playoff_distance_z']
variables_for_df = variables_for_df + ['game_x_playoff_distance_z']

# if don't scale here, do this:
#df_covers_bball_ref__dropna_home['game_x_playoff_distance'] = df_covers_bball_ref__dropna_home['game'] * df_covers_bball_ref__dropna_home['difference_distance_playoffs_abs']
#iv_variables = iv_variables + ['game_x_playoff_distance']
#variables_for_df = variables_for_df + ['game_x_playoff_distance']







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


# ----------------------------
# functions for final analysis

def create_train_and_test_dfs(df_covers_bball_ref__dropna_home, test_year):
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
    return df_covers_bball_ref_home_train, df_covers_bball_ref_home_test

#df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, 2010)


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

def mse_in_training_set(df_covers_bball_ref_home_train, algorithm, iv_variables):
    model = algorithm
    mses = cross_val_score(model, df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var], 
                           scoring = 'mean_squared_error', cv=10)
    print ('cross validated mse training:', -1*mses.mean())
# create correct metric:

#mse_in_training_set(df_covers_bball_ref_home_train, linear_model.LinearRegression())


# change point_diff_predicted to point_total_predicted
#def create_correct_metric(df):
#    df['correct'] = np.nan
#    df.loc[(df['point_total_predicted'] > df['spread']*-1) &
#                                       (df['ats_win'] == '1'), 'correct'] = 1
#    df.loc[(df['point_total_predicted'] < df['spread']*-1) &
#                                       (df['ats_win'] == '0'), 'correct'] = 1
#    df.loc[(df['point_total_predicted'] > df['spread']*-1) &
#                                       (df['ats_win'] == '0'), 'correct'] = 0
#    df.loc[(df['point_total_predicted'] < df['spread']*-1) &
#                                       (df['ats_win'] == '1'), 'correct'] = 0
#    # create var to say how much my prediction deviates from actual spread:
#    df['predicted_spread_deviation'] = np.abs(df['spread'] + df['point_diff_predicted'])
#    return df


def create_correct_metric(df):
    df['correct'] = np.nan
    df.loc[(df['point_total_predicted'] > df['totals']) &
                                       (df['over_win'] == '1'), 'correct'] = 1
    df.loc[(df['point_total_predicted'] < df['totals']) &
                                       (df['over_win'] == '0'), 'correct'] = 1
    df.loc[(df['point_total_predicted'] > df['totals']) &
                                       (df['over_win'] == '0'), 'correct'] = 0
    df.loc[(df['point_total_predicted'] < df['totals']) &
                                       (df['over_win'] == '1'), 'correct'] = 0
    # create var to say how much my prediction deviates from actual spread:
    df['predicted_totals_deviation'] = np.abs(df['totals'] - df['point_total_predicted'])
    return df

def create_correct_metric(df):
    df['correct'] = np.nan
    df.loc[(df['point_total_predicted'] > df['totals']) &
                                       (df['point_total'] > df['totals']), 'correct'] = 1
    df.loc[(df['point_total_predicted'] > df['totals']) &
                                       (df['point_total'] < df['totals']), 'correct'] = 0
    df.loc[(df['point_total_predicted'] < df['totals']) &
                                       (df['point_total'] < df['totals']), 'correct'] = 1
    df.loc[(df['point_total_predicted'] < df['totals']) &
                                       (df['point_total'] > df['totals']), 'correct'] = 0
    # create var to say how much my prediction deviates from actual spread:
    df['predicted_totals_deviation'] = np.abs(df['totals'] - df['point_total_predicted'])
    return df

#def create_correct_metric(df):
#    df['correct'] = np.nan
#    df.loc[(df['point_total_predicted'] > df['Closing Total']) &
#                                       (df['point_total'] > df['Closing Total']), 'correct'] = 1
#    df.loc[(df['point_total_predicted'] > df['Closing Total']) &
#                                       (df['point_total'] < df['Closing Total']), 'correct'] = 0
#    df.loc[(df['point_total_predicted'] < df['Closing Total']) &
#                                       (df['point_total'] < df['Closing Total']), 'correct'] = 1
#    df.loc[(df['point_total_predicted'] < df['Closing Total']) &
#                                       (df['point_total'] > df['Closing Total']), 'correct'] = 0
#    # create var to say how much my prediction deviates from actual spread:
#    df['predicted_totals_deviation'] = np.abs(df['Closing Total'] - df['point_total_predicted'])
#    return df
#
#

# fit model on data up through 2014 season
# when i use alpha > 1 and scale, seems to be doing better. presumably not overfitting.
# the predicted spread deviation looks more variable. but the n is small here, so think it should
# and actually, it does look pretty similar to the one above in the training set -- peaks about 3 at 55%. cool.


# play w this to try and fit poisson and quantile regression
#df_covers_bball_ref_home_train
#


string_for_regression = ''
for var in iv_variables[:]:  # breaks at 33
#for var in iv_variables[32:50]:  # breaks at 33
    print(var)
    string_for_regression += ' + ' + var
string_for_regression = string_for_regression[3:]        

iv_variables[32]  # 'opponent_DRtg_ewma_15'
df_covers_bball_ref_home_train['opponent_DRtg_ewma_15'].min()
df_covers_bball_ref_home_train['opponent_DRtg_ewma_15'].max()
df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['opponent_DRtg_ewma_15'].isnull()]
df_covers_bball_ref_home_train['opponent_DRtg_ewma_15'] = 

for col in df_covers_bball_ref_home_train[iv_variables].columns:
    print(df_covers_bball_ref_home_train[col].dtypes)


#model_plot = smf.glm(formula = 'score_team ~ ' + string_for_regression,
#                     data=df_covers_bball_ref_home_train, family=sm.families.Poisson()).fit()  
#print(model_plot.summary())
#predictions_test_set = model_plot.predict(df_covers_bball_ref_home_train[iv_variables])

mod = smf.quantreg('score_team_sq_rt ~ ' + string_for_regression, data=df_covers_bball_ref_home_train).fit(q=.5)
print(mod.summary())
predictions_test_set = mod.predict(df_covers_bball_ref_home_train[iv_variables[:]])


ivs = df_covers_bball_ref_home_train[iv_variables[:]].values
ivs = sm.add_constant(ivs)
dv = df_covers_bball_ref_home_train['score_team_sq_rt'].values
#dv = [[score] for score in dv]
print(len(ivs), len(dv))
#vs = [item for item in ivs]

ivs.shape
ivs.ndim
dv.shape
dv.ndim


mod = smf.QuantReg(dv, ivs).fit(q=.5)
print(mod.summary())

mod = smf.OLS(dv, ivs).fit(q=.5)
mod = sm.OLS(dv, ivs).fit(q=.5)
mod = sm.QuantReg(dv, ivs).fit(q=.5)

#from statsmodels.regression.quantile_regression import QuantReg

len(df_covers_bball_ref_home_train[df_covers_bball_ref_home_train[iv_variables].isnull()])
len(df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['oppt_score_predicted'].isnull()])
for iv in iv_variables[:]:
    print(len(df_covers_bball_ref_home_train[df_covers_bball_ref_home_train[iv].isnull()]))


def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables):

    # trains model using poisson regression or quantile regression.
    # this didn't improve my predictions, still predicting over too often  
    #model = smf.glm(formula = 'score_team_cu_rt ~ ' + string_for_regression,
    #                     data=df_covers_bball_ref_home_train, family=sm.families.Poisson()).fit()  

    # quantile regression. but only works with pca cutting down ivs. and doesn't really help
    #model = smf.quantreg('score_team ~ ' + string_for_regression, data=df_covers_bball_ref_home_train).fit(q=.75)

    # i commented below 2 lines and instead use above statsmodels piosson regression   
    model = algorithm
    model.fit(df_covers_bball_ref_home_train[iv_variables], df_covers_bball_ref_home_train[dv_var])
    predictions_test_set = model.predict(df_covers_bball_ref_home_test[iv_variables])    
    # for the square root or cube root version:    
    predictions_test_set = np.power(predictions_test_set,3)  # set to 2 if cube root version
    #predictions_test_set = np.exp(predictions_test_set) 
    #predictions_test_set = [l[0] for l in predictions_test_set]    
    df_covers_bball_ref_home_test.loc[:,'team_score_predicted'] = predictions_test_set
    # adjust the point total predicted. should take into account how higher predicted
    # points should be even higher than predicted because the more ponts scored, the more
    # possessions, and so even more points can be scored
    predictions_train_set = model.predict(df_covers_bball_ref_home_train[iv_variables])
    # for the square root version:    
    predictions_train_set = np.power(predictions_train_set,3)  # set to 2 if sq root version
    #predictions_train_set = np.exp(predictions_train_set)     
    #predictions_train_set = [l[0] for l in predictions_train_set]
    #
    df_covers_bball_ref_home_train.loc[:,'team_score_predicted'] = predictions_train_set    
    df_opponent_train = df_covers_bball_ref_home_train.copy(deep=True)
    df_opponent_train.loc[:,'opponent'] = df_opponent_train.loc[:,'team']
    df_opponent_train.loc[:,'oppt_score_predicted'] = df_opponent_train.loc[:,'team_score_predicted']
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train.merge(df_opponent_train[['date', 'opponent', 'oppt_score_predicted']], on=['opponent', 'date'], how='outer')
    df_covers_bball_ref_home_train.loc[:, 'point_total_predicted'] = df_covers_bball_ref_home_train.loc[:, 'team_score_predicted'] + df_covers_bball_ref_home_train.loc[:, 'oppt_score_predicted']
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['venue']==0]
    #model_plot = smf.ols(formula = 'point_total ~ point_total_predicted',
    #                data=df_covers_bball_ref_home_train).fit()  
    #print(model_plot.summary()) 
    #intercept = model_plot.params[0]
    #coef = model_plot.params[1]
    #print('intercept:', intercept, 'coef:', coef)    
    
    #df_covers_bball_ref__dropna_home['team_score_predicted'] = 11
    df_opponent = df_covers_bball_ref_home_test.copy(deep=True)
    df_opponent.loc[:,'opponent'] = df_opponent.loc[:,'team']
    df_opponent.loc[:,'oppt_score_predicted'] = df_opponent.loc[:,'team_score_predicted']
    len(df_covers_bball_ref_home_test)  # 28560
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.merge(df_opponent[['date', 'opponent', 'oppt_score_predicted']], on=['opponent', 'date'], how='outer')
    len(df_covers_bball_ref_home_test)
    df_covers_bball_ref_home_test.loc[:, 'point_total_predicted'] = df_covers_bball_ref_home_test.loc[:, 'team_score_predicted'] + df_covers_bball_ref_home_test.loc[:, 'oppt_score_predicted']
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test[df_covers_bball_ref_home_test['venue']==0]
    
    #df_covers_bball_ref_home_test['point_total_predicted'] = intercept + df_covers_bball_ref_home_test['point_total_predicted']*coef
    
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

    # skip
#    df_covers_bball_ref_home_train.loc[:, 'team_x_opp_predicted'] = df_covers_bball_ref_home_train.loc[:, 'team_score_predicted'] * df_covers_bball_ref_home_train.loc[:, 'oppt_score_predicted'] 
#    df_covers_bball_ref_home_train.loc[:, 'spread_x_opp_predicted'] = df_covers_bball_ref_home_train.loc[:, 'spread'] * df_covers_bball_ref_home_train.loc[:, 'oppt_score_predicted'] 
#    df_covers_bball_ref_home_train.loc[:, 'spread_x_team_predicted'] = df_covers_bball_ref_home_train.loc[:, 'spread'] * df_covers_bball_ref_home_train.loc[:, 'team_score_predicted'] 
#    df_covers_bball_ref_home_train.loc[:, 'spread_x_team_x_opp_predicted'] = df_covers_bball_ref_home_train.loc[:, 'spread'] * df_covers_bball_ref_home_train.loc[:, 'team_score_predicted'] * df_covers_bball_ref_home_train.loc[:, 'spread'] 
#    df_covers_bball_ref_home_test.loc[:, 'team_x_opp_predicted'] = df_covers_bball_ref_home_test.loc[:, 'team_score_predicted'] * df_covers_bball_ref_home_test.loc[:, 'oppt_score_predicted'] 
#    df_covers_bball_ref_home_test.loc[:, 'spread_x_opp_predicted'] = df_covers_bball_ref_home_test.loc[:, 'spread'] * df_covers_bball_ref_home_test.loc[:, 'oppt_score_predicted'] 
#    df_covers_bball_ref_home_test.loc[:, 'spread_x_team_predicted'] = df_covers_bball_ref_home_test.loc[:, 'spread'] * df_covers_bball_ref_home_test.loc[:, 'team_score_predicted'] 
#    df_covers_bball_ref_home_test.loc[:, 'spread_x_team_x_opp_predicted'] = df_covers_bball_ref_home_test.loc[:, 'spread'] * df_covers_bball_ref_home_test.loc[:, 'team_score_predicted'] * df_covers_bball_ref_home_test.loc[:, 'spread'] 
#
#    model.fit(df_covers_bball_ref_home_train[['team_score_predicted', 'oppt_score_predicted', 'spread', 'team_x_opp_predicted', 'spread_x_opp_predicted', 'spread_x_team_predicted', 'spread_x_team_x_opp_predicted']], df_covers_bball_ref_home_train['point_total'])
#    predictions_test_set = model.predict(df_covers_bball_ref_home_test[['team_score_predicted', 'oppt_score_predicted', 'spread', 'team_x_opp_predicted', 'spread_x_opp_predicted', 'spread_x_team_predicted', 'spread_x_team_x_opp_predicted']])
#    df_covers_bball_ref_home_test.loc[:,'point_total_predicted'] = predictions_test_set

    return df_covers_bball_ref_home_test, df_covers_bball_ref_home_train

#df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, linear_model.LinearRegression())


#=============================================
# alt version of above to predict point spread.
# so i can predict point diff with these vars but not totals. why?
# something happending wtih those teams who are favored by a lot. it's like 
# the formula goes haywire when the team home team is favored, esp by a lot
# so i could introduce an intearction wtih spread, for most vars???!!!
# (makes me think could also look at interactions with totals in point diff prediction)
# (this works well. use this to make alt prediction to avg with what i get from predicting point diff the normal way)

df_covers_bball_ref__dropna_home['ats_win'] = 'push'
df_covers_bball_ref__dropna_home.loc[df_covers_bball_ref__dropna_home['beat_spread'] > 0, 'ats_win'] = '1'
df_covers_bball_ref__dropna_home.loc[df_covers_bball_ref__dropna_home['beat_spread'] < 0, 'ats_win'] = '0'

def create_correct_metric(df):
    df['correct'] = np.nan
    df.loc[(df['point_total_predicted'] > df['spread']*-1) &
                                       (df['ats_win'] == '1'), 'correct'] = 1
    df.loc[(df['point_total_predicted'] < df['spread']*-1) &
                                       (df['ats_win'] == '0'), 'correct'] = 1
    df.loc[(df['point_total_predicted'] > df['spread']*-1) &
                                       (df['ats_win'] == '0'), 'correct'] = 0
    df.loc[(df['point_total_predicted'] < df['spread']*-1) &
                                       (df['ats_win'] == '1'), 'correct'] = 0
    # create var to say how much my prediction deviates from actual spread:
    df['predicted_totals_deviation'] = np.abs(df['spread'] + df['point_total_predicted'])
    return df


def create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables):
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
    df_covers_bball_ref_home_train.loc[:, 'point_total_predicted'] = df_covers_bball_ref_home_train.loc[:, 'team_score_predicted'] - df_covers_bball_ref_home_train.loc[:, 'oppt_score_predicted']
    df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['venue']==0]
    df_opponent = df_covers_bball_ref_home_test.copy(deep=True)
    df_opponent.loc[:,'opponent'] = df_opponent.loc[:,'team']
    df_opponent.loc[:,'oppt_score_predicted'] = df_opponent.loc[:,'team_score_predicted']
    len(df_covers_bball_ref_home_test)  # 28560
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test.merge(df_opponent[['date', 'opponent', 'oppt_score_predicted']], on=['opponent', 'date'], how='outer')
    len(df_covers_bball_ref_home_test)
    df_covers_bball_ref_home_test.loc[:, 'point_total_predicted'] = df_covers_bball_ref_home_test.loc[:, 'team_score_predicted'] - df_covers_bball_ref_home_test.loc[:, 'oppt_score_predicted']
    df_covers_bball_ref_home_test = df_covers_bball_ref_home_test[df_covers_bball_ref_home_test['venue']==0]
    df_covers_bball_ref_home_test = create_correct_metric(df_covers_bball_ref_home_test)
    return df_covers_bball_ref_home_test, df_covers_bball_ref_home_train


#=======================================


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
seasons = [2011, 2012, 2013, 2014, 2015]
seasons = [2010, 2011, 2012, 2013]
seasons = [2013, 2014, 2015]

model = linear_model.Ridge(alpha=10)  # higher number regularize more, 
# but even .0001 or something like that is adding a decent amount of regularization
# doesn't seem to be helping a lot to predict team score (cube root)

model = linear_model.LinearRegression()

#model = KNeighborsRegressor(n_neighbors=800, weights='distance')
model = RandomForestRegressor(n_estimators=400, max_features='auto')  # , min_samples_leaf = 10, min_samples_split = 50)
#model = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=.01, subsample=.7) # this sucked
#model = tree.DecisionTreeRegressor(min_samples_split = 50)
#model = ExtraTreesRegressor(n_estimators=500)  

# holy shit, this adaboost ups the 2015 accuracy to 55%. but took about 1/2 hr to run w 200 estimators
# I think shrinking the learning rate below 1 helps. need bigger n_estimators, though?
# using loss='square' may also be better than 'linear?'

# used this below in conjuction with regular linear model. and bet the games they both agree. 
# except it's variable and give different answers
# really slow. if use, do something else while it's working
model = AdaBoostRegressor(tree.DecisionTreeRegressor(), n_estimators=200)  # this decision tree regressor is the default
# this is predicting well for 2015 and maybe 2014. but not predicting for all. why?


# but the i did it again w 200 estimators and only got 50%?!
# nearest neighbors didn't seem good for adaboost:
#model = AdaBoostRegressor(KNeighborsRegressor(n_neighbors=500), n_estimators=100)  

# this sometimes worked well
# TRY THIS W REGULAR REGRESSION:
model = AdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default

model = AdaBoostRegressor(linear_model.Ridge(alpha=10), n_estimators=100, learning_rate=.0001)  #, loss='exponential')  # 

model = AdaBoostRegressor(svm.LinearSVR(C=.05), n_estimators=50, learning_rate=.001)  #, loss='exponential')  # this decision tree regressor is the default

#model = AdaBoostRegressor(linear_model.Ridge(alpha=.01), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
#model = AdaBoostRegressor(linear_model.KernelRidge(alpha=.01), n_estimators=100, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default
#model = AdaBoostRegressor(AdaBoostRegressor(), n_estimators=50, learning_rate=.01, loss='exponential')  # this decision tree regressor is the default

# might try this in conjuction with regular regression
# but needs a lot of tweaking. it's predicting favored teams well and underdog teams poorly
# so something is up, presumably, and maybe tuning will help it
model = svm.SVR(C=1000, gamma=.00001)  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=2, gamma=.005)  # these ar ethe best parameters. found w grid search
model = svm.SVR(C=50, gamma=.0001) 
model = svm.SVR(C=200, gamma=.0001) 
# cache_size doesn't make a diff for accuracy. leave as default
#model = svm.SVR(C=10, kernel='poly', degree=3)  # takes loonger and doesn't work as well. but considers interactions so might want to try more, esp after pca?
#model = svm.SVR(C=.1, kernel='sigmoid')  # takes loonger and doesn't work as well. but considers interactions so might want to try more, esp after pca?
#model = svm.SVR(C=10, kernel='poly', gamma=0)  # takes loonger and doesn't work as well. but considers interactions so might want to try more, esp after pca?


# TRY THIS W REGULAE REGRESSION
# C = The bigger this parameter, the less regularization is used.
# low C regularizees more
model = svm.LinearSVR()  #   #  
model = svm.LinearSVR(C=10)  #  
model = svm.LinearSVR(C=.05)  #  best according to grid search
model = svm.LinearSVR(C=.1)  #  
model = svm.LinearSVR(C=.01)  

#model = svm.LinearSVR(C=1, intercept_scaling=.1)  
#model = svm.LinearSVR(C=.05, intercept_scaling=10)  


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


def analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, algorithm, iv_variables):
    accuracy_list = []
    mean_sq_error_list = []
    df_test_seasons = pd.DataFrame()
    #df_covers_bball_ref__dropna_home = df_covers_bball_ref__dropna_home[df_covers_bball_ref__dropna_home['season_start']!=2011]
    for season in seasons:
        print(season)
        df_covers_bball_ref_home_train, df_covers_bball_ref_home_test = create_train_and_test_dfs(df_covers_bball_ref__dropna_home, season)
        #df_covers_bball_ref_home_train = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start'] != 2011]  #shortened season. helps a bit.
        #df_covers_bball_ref_home_train = create_df_weight_recent_seasons_more(df_covers_bball_ref_home_train)
        # AT MOMENT, THIS TOTALLY RUNIS IT:        
        #df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables_to_analyze = standardize_ivs(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, iv_variables)
        #df_covers_bball_ref_home_test = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables_to_analyze)
        df_covers_bball_ref_home_test, df_covers_bball_ref_home_train = create_predictions_and_ats_in_test_df(df_covers_bball_ref_home_train, df_covers_bball_ref_home_test, algorithm, iv_variables)

        df_covers_bball_ref_home_test.loc[:,'absolute_error'] = np.abs(df_covers_bball_ref_home_test.loc[:,'point_total_predicted'] - df_covers_bball_ref_home_test.loc[:,'point_total'])
        #df_covers_bball_ref_home_test['absolute_error'] = np.abs(df_covers_bball_ref_home_test['point_total_predicted'] + df_covers_bball_ref_home_test['spread'])

        df_test_seasons = pd.concat([df_test_seasons, df_covers_bball_ref_home_test])
        accuracy_list.append((season, np.round(df_covers_bball_ref_home_test['correct'].mean(), 4)*100))
        mean_sq_error_list.append((season, np.round(df_covers_bball_ref_home_test['absolute_error'].mean(), 2)))
        #print season, 'mean accuracy:', np.round(df_covers_bball_ref_home_test['correct'].mean(), 3)*100, 'percent'
        #print sns.lmplot(x='predicted_spread_deviation', y='correct', data=df_covers_bball_ref_home_test, lowess=True)
        print()
    return accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train

#accuracy_list, df_test_seasons = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, linear_model.Ridge(alpha=.01))
#accuracy_list, df_test_seasons = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, linear_model.Lasso(alpha=.01))

# USE THIS IF SCALING IN FUNCITON ABOVE
#accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model, iv_variables)
# USE THIS IF SCALED WAY ABOVE

accuracy_list, mean_sq_error_list, df_test_seasons, df_covers_bball_ref_home_train = analyze_multiple_seasons(seasons, df_covers_bball_ref__dropna_home, model, iv_variables)

df = pd.DataFrame(accuracy_list, columns=['season', 'accuracy'])
df_mse = pd.DataFrame(mean_sq_error_list, columns=['season', 'error'])
df_mse['accuracy'] = df['accuracy']
for season in range(len(seasons)):
    plt.plot([accuracy_list[season][1], accuracy_list[season][1]], label=str(seasons[season]));
plt.ylim(48, 58)
plt.xticks([])
plt.legend()
plt.ylabel('percent correct')
plt.axhline(51.5, linestyle='--', color='grey', linewidth=1, alpha=.5)
sns.despine() 
[print(str(year)+':', accuracy) for year, accuracy in accuracy_list]


df_test_seasons.to_csv('test_totals_seasons.csv')


len(iv_variables)


df_test_seasons_1 = df_test_seasons.copy(deep=True)
df_covers_bball_ref_home_train_1 = df_covers_bball_ref_home_train.copy(deep=True)

df_test_seasons_2 = df_test_seasons.copy(deep=True)
df_covers_bball_ref_home_train_2 = df_covers_bball_ref_home_train.copy(deep=True)


for col in df_test_seasons_1.columns:
    print(col)

df_test_seasons_1[['date','team','opponent','team_ft_ewma_15']].head(20)
df_test_seasons_2[['date','team','opponent','team_ft_ewma_15']].head(20)

df_test_seasons_1 = df_test_seasons_1.sort_values(by=['team', 'date'])
df_test_seasons_2 = df_test_seasons_2.sort_values(by=['team', 'date'])

len(df_test_seasons_1)
len(df_test_seasons_2)


i = 10  # should = 95

for var in list(df_test_seasons_1.columns[290:313]):
    print(var)
    one = df_test_seasons_1[var]
    two = df_test_seasons_2[var]
    for i in range(len(one[:])): 
        if df_test_seasons_1[var].dtypes == 'float':
            item_one = one.iloc[i].astype(str)
            item_two = two.iloc[i].astype(str)
            #print(item_one, item_two)
            if item_one != item_two:
                print(item_one, item_two)
            else:
                None
        else:
            item_one = one.iloc[i]
            item_two = two.iloc[i]
            #print(item_one, item_two)
            if item_one != item_two:
                print(item_one, item_two)
            else:
                None


df_test_seasons_1.columns[311]

# ok
# opponent_possessions_by_score_oppt_ewma_15

# bad
# team_score_predicted
# oppt_score_predicted
# point_total_predicted
# correct
# predicted_totals_deviation
# absolute_error









model_plot = smf.ols(formula = 'point_total ~ point_total_predicted',
                data=df_test_seasons).fit()  
print(model_plot.summary()) 
intercept = model_plot.params[0]
coef = model_plot.params[1]
df_test_seasons['point_total_predicted_rev'] = intercept + df_test_seasons['point_total_predicted']*coef
model_plot = smf.ols(formula = 'point_total ~ point_total_predicted_rev',
                data=df_test_seasons).fit()  
print(model_plot.summary()) 

sns.lmplot(x='point_total_predicted_rev', y='point_total', data=df_test_seasons, ci=None, lowess=True, scatter_kws={'alpha':0})
plt.grid()



# -------
# if useing multiple algos, run these 3 lines to save the alt one (i.e., not the linear one)
df_test_seasons_alt_model = df_test_seasons.copy(deep=True)
df_test_seasons_alt_model['point_diff_predicted_alt_model'] = df_test_seasons_alt_model['point_total_predicted']
df_test_seasons_alt_model['correct_alt_model'] = df_test_seasons_alt_model['correct']

# -------
# once run the regular lin regression model:
df_test_seasons['point_diff_predicted_alt_model'] = df_test_seasons_alt_model['point_diff_predicted_alt_model']
df_test_seasons['correct_alt_model'] = df_test_seasons_alt_model['correct_alt_model']


df_test_seasons[['point_total_predicted', 'point_diff_predicted_alt_model']].tail(20)
df_test_seasons[['point_total_predicted', 'point_diff_predicted_alt_model']].corr()
plt.scatter(df_test_seasons['point_total_predicted'], df_test_seasons['point_diff_predicted_alt_model'], alpha=.3)
sns.lmplot(x='point_total_predicted', y='point_diff_predicted_alt_model', data=df_test_seasons, scatter_kws={'alpha':.05})

df_test_seasons[['correct', 'correct_alt_model']].tail(20)
df_test_seasons[['correct', 'correct_alt_model']].corr()


df_test_seasons['point_diff_predicted_two_models'] = df_test_seasons[['point_total_predicted', 'point_diff_predicted_alt_model']].mean(axis=1)
df_test_seasons[['point_total_predicted', 'point_diff_predicted_alt_model', 'point_diff_predicted_two_models']].tail(20)
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
    df['predicted_spread_deviation'] = np.abs(df['total'] + df['point_diff_predicted_two_models'])
    return df

df_test_seasons = create_correct_metric_mean_prediction(df_test_seasons)
df_test_seasons[['correct_two_models', 'point_diff_predicted_two_models', 'spread', 'point_difference']].tail(20)
df_test_seasons[['correct', 'correct_alt_model', 'correct_two_models']].corr()



#-------------------------------------------
# CAN I MAKE THICKNESS OF LINE CORRESPOND TO SAMPLE SIZE AT THAT POINT?
# THAT WOULD BE A REALLY HELPFUL THING GENERALLY
for season in seasons[2:]:
    df_test_seasons['predicted_totals_deviation'][df_test_seasons['season_start']==season].hist(alpha=.1, color='green')
    plt.xlim(0,6)

sns.lmplot(x='predicted_totals_deviation', y='correct', data=df_test_seasons[df_test_seasons['season_start']>2011], hue='season_start', lowess=True, line_kws={'alpha':.6})
plt.ylim(.3, .7)
plt.xlim(0,6)
plt.grid(axis='y', linestyle='--', alpha=.15)
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.75)


bin_number = 20
n, bins, patches = plt.hist(df_test_seasons['predicted_totals_deviation'], bin_number, color='white')
#plt.bar(bins[:-1], n, width=bins[1]-bins[0], alpha=.1, color='white')

sns.lmplot(x='predicted_totals_deviation', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.5, 'color':'blue'})
plt.ylim(.4, .6)
plt.xlim(-.1,6)
#plt.grid(axis='y', linestyle='--', alpha=.75)
max_n = n.max()
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.xlabel('degree the predicted total points \n deviated from the vegas totals' )

for i in range(bin_number):
    a = 1 - n[i]/max_n
    print(a)
    plt.bar(bins[:-1][i], n[i], width=bins[1]-bins[0], alpha=a, color='b', linewidth=0)


# i'm getting there. can i figure out how to plot a white hist so that with alpha=1
# it will obscure whatever is behind it?


# this suggests i skip betting early in season. it's harder to beat chance here
sns.lmplot(x='game', y='absolute_error', data=df_test_seasons, lowess=True, scatter_kws={'alpha':.06})
plt.ylim(9,14)
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
# what does this look like if omit last two seasons? same pattern?
sns.lmplot(x='spread', y='correct', data=df_test_seasons[df_test_seasons['season_start']< 2014], lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)

sns.lmplot(x='spread', y='correct', data=df_test_seasons, hue='season_start', lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)

sns.lmplot(x='spread', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)

df_test_seasons['totals'].hist(alpha=.7)
sns.lmplot(x='totals', y='correct', data=df_test_seasons, lowess=True, line_kws={'alpha':.6})
plt.axhline(.515, linestyle='--', color='grey', linewidth=1, alpha=.5)
plt.ylim(.4,.6)
# yeah this fits with theory -- it's predicting well around the leave average of 200
# but not outside that.
df_test_seasons[['point_total_predicted', 'point_total']].head()

sns.lmplot(x='point_total_predicted', y='point_total', data=df_test_seasons, ci=None, scatter_kws={'alpha':0})
plt.grid()

sns.lmplot(x='point_total_predicted', y='point_total', data=df_test_seasons, ci=None, lowess=True, scatter_kws={'alpha':0})
plt.grid()

model_plot = smf.ols(formula = 'point_total ~ point_total_predicted',
                data=df_test_seasons).fit()  
print(model_plot.summary()) 

model_plot = smf.ols(formula = 'totals ~ point_total_predicted',
                data=df_test_seasons).fit()  
print(model_plot.summary()) 
# so the coeff is slightly more than 1. does that make sense?
# and if look at regression line, can see that as get over 200,
# e.g., if i've predicted 230, the actual total would be higher than 230. 


#------------------------------------------------------------------------------
# look at winning % with diff selection criteria
print('betting on all games:')
[print(str(year)+':', accuracy) for year, accuracy in accuracy_list]

#df_truncated = df_test_seasons[(df_test_seasons['predicted_spread_deviation'] > .5) & (df_test_seasons['predicted_spread_deviation'] < 3)]
df_truncated = df_test_seasons[(df_test_seasons['predicted_totals_deviation'] > 1.75)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['game'] > 1)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_totals_deviation'] > .01) & (df_test_seasons['game'] > 1)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

# no restrictions:
df_truncated = df_test_seasons[(df_test_seasons['predicted_totals_deviation'] > 0) & (df_test_seasons['game'] > 0)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_totals_deviation'] > .01) & (df_test_seasons['game'] > 1)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct_alt_model'].mean())
print()
print (df_truncated.groupby('season_start')['correct_alt_model'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_totals_deviation'] > .2) & (df_test_seasons['game'] > 1)]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct_two_models'].mean())
print()
print (df_truncated.groupby('season_start')['correct_two_models'].count())

df_truncated = df_test_seasons[(df_test_seasons['predicted_totals_deviation'] > .01) & (df_test_seasons['game'] > 1)]
df_truncated = df_truncated[(df_truncated['correct'] == df_truncated['correct_alt_model'])]
df_truncated['season_start'].unique()
print (df_truncated.groupby('season_start')['correct'].mean())
print()
print (df_truncated.groupby('season_start')['correct'].count())


# interesting -- when do normal reguression with svr linear, they're in pretty
# good agreement, so leaves the n to bet in high. so even though it doesn't result
# in a better win % than using normal svr, it makes more because leaves more games
# to bet on. ALTER CODE SO CAN INCORPORATE 3-4 ALGOS INSTEAD OF JUST TWO.


#----------------------------------------------------------
# plot winnings

df_season = df_test_seasons[(df_test_seasons['season_start']>2012) & (df_test_seasons['season_start']<2014)]
df_season = df_season.reset_index(drop=True)
len(df_season)

cumulative_money_list = []
df_season_truncated = df_season[(df_season['game'] > 2)]
df_season_truncated = df_season_truncated[df_season_truncated['predicted_totals_deviation'] > .1]
# un-comment below if using multiple algos
#df_season_truncated = df_season_truncated[(df_season_truncated['correct'] == df_season_truncated['correct_alt_model'])]

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
actual_win_pct = str(round(df_wins['wins'].mean(), 3))


# i like using the kelly formula here -- it mitigates the troughs? less fluctuation

# regular appraoch:
win_probability = .53 # put pct I think we can realistically win at.
# kelly formula says that if i can put the actual pct, i'll maximize the winnings
# but the toal pot gets more and more volatilse the higher it goes, i.e., betting more and more
# so makes some sense to put a more conservative estimate, under what I think we'll get
kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
money = 10000
bet = money * kelly_criteria
total_pot = 0
total_winnings_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot += bet*.95
        total_winnings_list.append(total_pot)
        money += bet*.95
        bet = bet
    if game == 0:
        total_pot += -1*bet
        total_winnings_list.append(total_pot)
        money += -1*bet
        bet = bet

# kelly approach:
win_probability = .53
kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
money_kelly = 10000
bet_kelly = money_kelly * kelly_criteria
total_pot_kelly = 0
total_winnings_kelly_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot_kelly += bet_kelly*.95
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += bet_kelly*.95
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




# -------------------------------------------
# plot betting on both spread and totals

df_spread_seasons = pd.read_csv('test_spred_seasons.csv')
df_spread_seasons['type'] = 'spread'
df_totals_seasons = pd.read_csv('test_totals_seasons.csv')
df_totals_seasons['type'] = 'totals'


# is betting on spread corr with betting on totals? no.
df_spread_totals_merged_for_corr = pd.merge(df_spread_seasons, df_totals_seasons, on=['date','team', 'opponent'], how='left')
for col in df_spread_totals_merged_for_corr.columns:
    print(col)
df_spread_totals_merged_for_corr[['correct_x', 'correct_y']].corr()  # -.005
df_2015 = df_spread_totals_merged_for_corr[df_spread_totals_merged_for_corr['season_start_x']==2015]
df_2015[['correct_x', 'correct_y']].corr()  # -.022
# awesome -- these aren't correlated! so should be like making bets on different games.


# show how would look if betting same amnt on bets in the same day
df_spread_totals_merge = pd.concat([df_spread_seasons, df_totals_seasons], ignore_index=True)
len(df_spread_totals_merge)
df_spread_totals_merge = df_spread_totals_merge.sort_values(by='date')
df_spread_totals_merge = df_spread_totals_merge.reset_index(drop=True)

df_num_gs_ea_day = df_spread_totals_merge.groupby('date')['x_totals'].apply(lambda x: x.count())
df_num_gs_ea_day.plot()
df_num_gs_ea_day.hist(bins=15)  # these are saying we'll generally be betting between 10-25 times per day



# plot winnings
df_season = df_spread_totals_merge[(df_spread_totals_merge['season_start']>2011) & (df_spread_totals_merge['season_start']<2013)]
df_season_truncated = df_season[(df_season['game'] > 1)]
df_season_truncated = df_season_truncated.reset_index(drop=True)
#df_season_truncated = df_season_truncated[df_season_truncated['predicted_totals_deviation'] > .1]
# un-comment below if using multiple algos
#df_season_truncated = df_season_truncated[(df_season_truncated['correct'] == df_season_truncated['correct_alt_model'])]

# bet kelly
win_probability = .52
kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
money_kelly = 10000
bet_kelly = money_kelly * kelly_criteria
total_pot_kelly = 0
total_winnings_kelly_list = [0]
actual_win_pct = df_season_truncated['correct'].mean()

for date in df_season_truncated['date'].unique():
    df_season_truncated_date = df_season_truncated[df_season_truncated['date']==date]
    correct_bets = len(df_season_truncated_date[df_season_truncated_date['correct']==1])
    incorrect_bets = len(df_season_truncated_date[df_season_truncated_date['correct']==0])
    winnings_date = correct_bets * bet_kelly * .95
    losings_date = incorrect_bets * bet_kelly * -1
    total_pot_kelly = total_pot_kelly + winnings_date + losings_date
    total_winnings_kelly_list.append(total_pot_kelly)
    money_kelly += winnings_date + losings_date
    bet_kelly = money_kelly * kelly_criteria

# bet regular
win_probability = .52
kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
money_kelly = 10000
bet = money_kelly * kelly_criteria
total_pot = 0
total_winnings_list = [0]

for date in df_season_truncated['date'].unique():
    df_season_truncated_date = df_season_truncated[df_season_truncated['date']==date]
    correct_bets = len(df_season_truncated_date[df_season_truncated_date['correct']==1])
    incorrect_bets = len(df_season_truncated_date[df_season_truncated_date['correct']==0])
    winnings_date = correct_bets * bet * .95
    losings_date = incorrect_bets * bet * -1
    total_pot = total_pot + winnings_date + losings_date
    total_winnings_list.append(total_pot)

# plot winnings
plt.plot(total_winnings_list, alpha=.4, color='purple', linewidth=2)
plt.plot(total_winnings_kelly_list, alpha=.4, color='green', linewidth=2)
plt.xlabel('\n days', fontsize=15)
plt.ylabel('winnings', fontsize=15)
plt.xlim(-1,len(total_winnings_kelly_list)+1)
plt.ylim(min(total_winnings_kelly_list)-5000,max(total_winnings_kelly_list)+5000)
plt.axhline(.5, linestyle='--', color='black', linewidth=1, alpha=.5)
plt.grid(axis='y', alpha=.2)
plt.title('win percentage: '+str(round(actual_win_pct,3))+'\n\n' + 'winnings regular: $' + str(int(total_winnings_list[-1])) + '\n winnings kelly: $' + str(int(total_winnings_kelly_list[-1])), fontsize=15)
sns.despine()




# plot kelly winnings for past x seasons
for season, linecolor in zip([2011,2012,2013,2014,2015], ['red','blue','green','orange','purple']):
    df_season = df_spread_totals_merge[(df_spread_totals_merge['season_start']==season)]
    df_season_truncated = df_season[(df_season['game'] > 1)]
    # change this switch below to totals or spread, or comment out if betting on both   
    #df_season_truncated = df_season_truncated[df_season_truncated['type'] == 'totals']
    df_season_truncated = df_season_truncated.reset_index(drop=True)
    print(len(df_season))

    win_probability = .52
    kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
    money_kelly = 10000
    bet_kelly = money_kelly * kelly_criteria
    total_pot_kelly = 0
    total_winnings_kelly_list = [0]
    actual_win_pct = df_season_truncated['correct'].mean()
    
    for date in df_season_truncated['date'].unique():
        df_season_truncated_date = df_season_truncated[df_season_truncated['date']==date]
        correct_bets = len(df_season_truncated_date[df_season_truncated_date['correct']==1])
        incorrect_bets = len(df_season_truncated_date[df_season_truncated_date['correct']==0])
        winnings_date = correct_bets * bet_kelly * .95
        losings_date = incorrect_bets * bet_kelly * -1
        total_pot_kelly = total_pot_kelly + winnings_date + losings_date
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += winnings_date + losings_date
        bet_kelly = money_kelly * kelly_criteria

    # plot winnings
    plt.plot(total_winnings_kelly_list, alpha=.4, color=linecolor, linewidth=2)
    plt.xlabel('days', fontsize=15)
    plt.ylabel('winnings', fontsize=15)
    plt.xlim(-1,len(total_winnings_kelly_list)+1)
    plt.ylim(-10000,25000)
    plt.axhline(.5, linestyle='--', color='black', linewidth=1, alpha=.5)
    plt.grid(axis='y', alpha=.3)
    sns.despine()














df_season_truncated_date[['totals', 'point_total', 'type', 'correct']]


for i in range(len(df_season_truncated)):
    outcome = df_season_truncated[['correct']][df_season_truncated.index == i].values
    cumulative_money_list.append(outcome)

cumulative_money_list = [o[0][0] for o in cumulative_money_list]
len(cumulative_money_list)
df_wins = pd.DataFrame(cumulative_money_list)
df_wins = df_wins.reset_index()
df_wins.rename(columns={0:'wins'}, inplace=True)
len(df_wins[df_wins['wins'].isnull()])
actual_win_pct = str(round(df_wins['wins'].mean(), 3))


# i like using the kelly formula here -- it mitigates the troughs? less fluctuation

# regular appraoch:
win_probability = .52 # put pct I think we can realistically win at.
# kelly formula says that if i can put the actual pct, i'll maximize the winnings
# but the toal pot gets more and more volatilse the higher it goes, i.e., betting more and more
# so makes some sense to put a more conservative estimate, under what I think we'll get
kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
money = 10000
bet = money * kelly_criteria
total_pot = 0
total_winnings_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot += bet*.95
        total_winnings_list.append(total_pot)
        money += bet*.95
        bet = bet
    if game == 0:
        total_pot += -1*bet
        total_winnings_list.append(total_pot)
        money += -1*bet
        bet = bet

# kelly approach:
win_probability = .52
kelly_criteria = (win_probability * .95 - (1 - win_probability)) / .95
money_kelly = 10000
bet_kelly = money_kelly * kelly_criteria
total_pot_kelly = 0
total_winnings_kelly_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot_kelly += bet_kelly*.95
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += bet_kelly*.95
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

model_plot = smf.ols(formula = 'point_total ~ ' + string_for_regression,
                data=df_covers_bball_ref_home_train).fit()  
print(model_plot.summary()) 

model_plot = smf.ols(formula = 'point_total ~ spread * team_score_predicted * oppt_score_predicted',
                data=df_covers_bball_ref_home_train).fit()  
print(model_plot.summary()) 


# are the residuals normally distributed?
import scipy.stats as stats
stats.probplot(model_plot.resid, dist="norm", plot=plt)
# i think this is saying residuals are normally distributed

stats.probplot(df_covers_bball_ref_home_train['score_team'], dist="norm", plot=plt)

plt.scatter(df_covers_bball_ref_home_train['team_score_predicted'], model_plot.resid, alpha=.05)
plt.axhline(y=0, linewidth=1, linestyle='--', color='grey')




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


# -----
# grid search w simpler svr linear
df_covers_bball_ref_home_train_early_years = df_covers_bball_ref_home_train[df_covers_bball_ref_home_train['season_start']<2013]

X = df_covers_bball_ref_home_train_early_years[iv_variables].values
y = df_covers_bball_ref_home_train_early_years['score_team_cu_rt'].values

C_range = np.logspace(-3, 3, 10)
C_range = np.array([.005, .01, .05, .1, 10])
param_grid = dict(C=C_range)

grid = GridSearchCV(svm.LinearSVR(), param_grid=param_grid, cv=3)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
# The best parameters are {'C': 0.050000000000000003} with a score of 0.24

grid.best_score_  #
grid.best_params_  # 
grid.best_estimator_

grid_scores_data = grid.grid_scores_
df_grid_search = pd.DataFrame(grid_scores_data)
df_grid_search.head()
df_grid_search['C'] = df_grid_search.loc[:, 'parameters']

c = df_grid_search['C'].values

c_list = []
for params in c:
    print(params)
    c = [round(params['C'],6)]
    c_list.append(c)

df_grid_search.loc[:,'C'] = c_list

df_grid_search[['C']].head(10)
df_grid_search.pop('parameters')
df_grid_search.pop('cv_validation_scores')

df_grid_search.sort_values(by = 'mean_validation_score').tail(15)
df_grid_search.sort_values(by = 'C').tail(15)
sns.barplot(x='C', y='mean_validation_score', data=df_grid_search)

#   mean_validation_score            C
#0              -0.468940     0.001000
#1               0.087919     0.004642
#2               0.222588     0.021544
#3               0.244140     0.100000
#4               0.245130     0.464159
#5               0.193334     2.154435
#6              -0.105557    10.000000
#7              -0.251551    46.415888
#8              -0.908410   215.443469
#9              -0.729381  1000.000000

#   mean_validation_score     C
#0               0.239552  0.05
#1               0.243486  0.10
#2               0.240701  0.50
#3               0.243906  1.00
#4               0.211578  5.00


# -----
intercept_scaling_range = np.logspace(-3, 3, 10)
param_grid = dict(intercept_scaling=intercept_scaling_range)

grid = GridSearchCV(svm.LinearSVR(), param_grid=param_grid, cv=3)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

grid_scores_data = grid.grid_scores_
df_grid_search = pd.DataFrame(grid_scores_data)
df_grid_search.head()
df_grid_search['C'] = df_grid_search.loc[:, 'parameters']

c = df_grid_search['C'].values

c_list = []
for params in c:
    print(params)
    c = [round(params['intercept_scaling'],6)]
    c_list.append(c)

df_grid_search.loc[:,'C'] = c_list

df_grid_search[['C']].head(10)
df_grid_search.pop('parameters')
df_grid_search.pop('cv_validation_scores')

df_grid_search.sort_values(by = 'mean_validation_score').tail(15)
df_grid_search.sort_values(by = 'C').tail(15)

# saying best intercept scaling is .1??? what does this mean???



# -----
# grid search for c and intercept_scaling
C_range = np.logspace(-3, 3, 10)
int_scale_range = np.logspace(-5, 5, 10)
param_grid = dict(intercept_scaling=int_scale_range, C=C_range)

grid = GridSearchCV(svm.LinearSVR(), param_grid=param_grid, cv=3)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
# The best parameters are {'intercept_scaling': 100000.0, 'C': 1000.0} with a score of 0.24

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
    gamma = [round(params['intercept_scaling'],6)]
    c_list.append(c)
    gamma_list.append(gamma)

df_grid_search.loc[:,'C'] = c_list
df_grid_search.loc[:,'intercept_scaling'] = gamma_list

df_grid_search[['C', 'intercept_scaling']].head(10)
df_grid_search.pop('parameters')
df_grid_search.pop('cv_validation_scores')

df_grid_search.sort_values(by = 'mean_validation_score').tail(15)
#    mean_validation_score            C  intercept_scaling
#2                0.093555     0.001000           0.001668
#42               0.095323     0.464159           0.001668
#13               0.101044     0.004642           0.021544
#70               0.105896    46.415888           0.000010
#82               0.117258   215.443469           0.001668
#12               0.119304     0.004642           0.001668
#0                0.129395     0.001000           0.000010
#6                0.154385     0.001000          46.415888
#67               0.166132    10.000000         599.484250
#14               0.174409     0.004642           0.278256
#74               0.178789    46.415888           0.278256
#89               0.207948   215.443469      100000.000000
#23               0.218602     0.021544           0.021544
#16               0.224193     0.004642          46.415888
#99               0.235713  1000.000000      100000.000000
#


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






# ----------------------
# Investigate the errors

df_year = df_test_seasons[df_test_seasons['season_start']==2014]
df_year = df_year.reset_index()
print(len(df_year))
print(len(df_year.columns))
df_year[['correct', 'date', 'team', 'opponent', 'spread', 'predicted_spread_deviation']].head(10)

df_test_seasons['over_dichot'] = np.nan
df_test_seasons.loc[df_test_seasons['over']<0,'over_dichot'] = 0
df_test_seasons.loc[df_test_seasons['over']>0,'over_dichot'] = 1

# do my guesses (predicted totals) go under or over the vegas totals? 
# if i predict under the vegas totals and the actual points are under the vegas totals, then i win
df_test_seasons['my_guess_over_dichot'] = np.nan
df_test_seasons.loc[df_test_seasons['point_total_predicted']<df_test_seasons['totals'],'my_guess_over_dichot'] = 0
df_test_seasons.loc[df_test_seasons['point_total_predicted']>df_test_seasons['totals'],'my_guess_over_dichot'] = 1

sns.barplot(x='correct', y='over_dichot', data=df_test_seasons[df_test_seasons['correct'].notnull()])
df_test_seasons['correct'].mean() 
df_test_seasons['over'].mean()  # so, it only goes over 32% of the time
df_test_seasons['my_guess_over_dichot'].mean()  # 66% - i guess over 66% of the time

# so, i guess over 66% of the time, but it actually goes over 32% of the time
# idally i should be guessing it'll go over about 50% of the time

sns.barplot(x='over_dichot', y='correct', data=df_test_seasons[df_test_seasons['correct'].notnull()], hue='my_guess_over_dichot')
sns.barplot(x='my_guess_over_dichot', y='correct', data=df_test_seasons[df_test_seasons['correct'].notnull()])
# should i just trust my under guesses? i.e. only bet if i'm guessing under?
# but better to adjust so i'm guessing under about 50% of the time

plt.scatter(df_test_seasons['point_total_predicted'], df_test_seasons['point_total'], alpha=.3)
sns.lmplot(x='point_total_predicted', y='point_total', data=df_test_seasons)

# maybe the linear reguression isn't fitting as well as it could because there
# are outliers pulling it. 
df_test_seasons['point_total'].hist(bins=20, alpha=.7)
plt.axvline(df_test_seasons['point_total'].mean(), linestyle='--')
# yeah, kind of seems so? there are some really high outlier totals
# so this may be pulling my regression line up and making me guess 
# over more than i should??? how can i take into account this skewed distribution?
# some kind of transformation?

# got this from quora (in a nutshell, try quantile regression):
#However, while having a highly skewed dependent variable does not violate an 
#assumption, it may make OLS regression rather inapporpriate. OLS regression 
#models the mean and the mean is (usually) not a good measure of central 
#tendency in a skewed distribution.  The median is often better and it can be 
#modeled with quantile regression. In addition, when the DV is highly skewed 
#the interest may be in modeling the tails of the distribution - this can also 
#be done with quantile regression.  

# how to do quantile regression in statsmodels
# http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/quantile_regression.html
# http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.quantile_regression.QuantReg.html
# http://stackoverflow.com/questions/33050636/python-statsmodels-quantreg-intercept


df_test_seasons_favored = df_test_seasons[df_test_seasons['spread']<0]
df_test_seasons_favored[['point_total_predicted', 'point_total']].head()

df_test_seasons_incorrect = df_test_seasons[df_test_seasons['correct']==0]
df_test_seasons_incorrect[['point_total_predicted', 'point_total', 'totals', 'over']]

# if neg, then the actual point total went under the vegas totals. if pos then went over
# do these incorrect ones tend to be unders?
df_test_seasons_incorrect['over'].hist(bins=20, alpha=.7)
plt.axvline(df_test_seasons_incorrect['over'].mean(), linestyle='--')
print(len(df_test_seasons_incorrect[df_test_seasons_incorrect['over']<0])) # 2257
print(len(df_test_seasons_incorrect[df_test_seasons_incorrect['over']>0])) # 1095
# there are twice as manny unders! holy shit.



# look at distributions and residuals
import scipy.stats as stats

# is the dv -- score_team -- normally distributed?
# are the residuals normally distributed?
df_test_seasons['score_team']
df_test_seasons['my_residual'] = df_test_seasons['point_total_predicted'] - df_test_seasons['point_total']  
df_test_seasons['my_residual'].min()

df_test_seasons['my_residual'].hist(bins=20, alpha=.7)  # looks like I tend to be under the actual total
stats.probplot(df_test_seasons['my_residual'], dist="norm", plot=plt)

df_test_seasons['my_residual_vegas'] = df_test_seasons['point_total_predicted'] - df_test_seasons['totals']  
df_test_seasons['my_residual_vegas'].hist(bins=20, alpha=.7)  # saying i tend to be over vegas, so bet over
stats.probplot(df_test_seasons['my_residual_vegas'], dist="norm", plot=plt)
# so i tend to be over vegas's prediction even though I tend to be under the actual total points
# so does this mean that vegas really tends to be under the actual points
# but that means that I should be winning, right? because vegas totals are too low
# and i predicdt higher than them? So what's going wrong?
# is there a certain kind of game where i bet too high

df_test_seasons['vegas_residual'] = df_test_seasons['totals'] - df_test_seasons['point_total']  
df_test_seasons['vegas_residual'].hist(bins=20, alpha=.7)  # looks like I tend to be under the actual total

df_test_seasons['my_residual'].hist(bins=20, alpha=.4, color='green')  # looks like I tend to be under the actual total
df_test_seasons['vegas_residual'].hist(bins=20, alpha=.4, color='blue')  # looks like I tend to be under the actual total


plt.scatter(df_test_seasons['point_total'], df_test_seasons['point_total_predicted'], alpha=.4)
plt.plot(df_test_seasons['point_total'], df_test_seasons['point_total'], 'm--', alpha=.6, label='Ideal Prediction')
plt.xlabel('actual point total')
plt.ylabel('predicted point total')
plt.grid()

# residual plot
plt.scatter(df_test_seasons['point_total_predicted'], df_test_seasons['my_residual'], alpha=.15)
plt.axhline(y=0, linewidth=1, linestyle='--', color='grey')

plt.scatter(df_covers_bball_ref_home_train['team_score_predicted'], model_plot.resid, alpha=.1)
plt.axhline(y=0, linewidth=1, linestyle='--', color='grey')

# there are some games i way under-predict, and under-predict more than over predict


# could predict spread with a model and see when my predicted spread deviates a lot from the actual spread? and stay away from those?
# see if there are any outliers of a var in past 10 games. or count up the outliers in past 10 games? for that team? or in general?










#-----------------------------------
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


# 4, 5, 6, 7, 14, 19, 23, 24, 25, 31, 32, 38, 39, *42*
variable = iv_variables[42]
sns.interactplot('game', variable, 'point_difference', data=df_test_seasons)

results = smf.ols(formula = 'point_difference ~ game * difference_distance_playoffs_abs', data=df_test_seasons).fit()
print(results.summary())  # p = ,000

results = smf.ols(formula = 'point_difference ~ game * difference_opponent_3PAr_ewma_15', data=df_test_seasons).fit()
print(results.summary())  # p = .008

results = smf.ols(formula = 'point_difference ~ game * difference_team_3PAr_ewma_15', data=df_test_seasons).fit()
print(results.summary())

