# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:17:05 2016

@author: charlesmartens
"""

cd /Users/charlesmartens/Documents/projects/bet_bball

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set_style('white')


# actual data:
# examine implications of 50%
#df_test_seasons[df_test_seasons['season_start']==2015]['correct'].mean()
#df_test_seasons[df_test_seasons['season_start']==2015]['correct'].count()
#df_2015 = df_test_seasons[df_test_seasons['season_start']==2015]
#df_2015 = df_2015[['date', 'correct']]
#df_2015 = df_2015.sort_values(by='date')
#df_2015 = df_2015.reset_index(drop=True)
#cumulative_money_list = list(df_2015['correct'].values)
#actual_win_pct = str(round(df_2015['correct'].mean(), 3))

df_test_seasons = pd.read_csv('df_test_seasons_model_1.csv')
df_2015 = df_test_seasons[df_test_seasons['season_start']==2015]
df_2015 = df_2015[['date', 'correct']]
df_2015 = df_2015.sort_values(by='date')
df_2015 = df_2015.reset_index(drop=True)
df_2015 = df_2015[df_2015.index<1000]
len(df_2015)
df_2015['correct'].mean()


# ------------------------------------------------------
# get distributions for diff binomial probability distributions
# can see that even if actual distribution is 50%, in 1000 games
# might get a good deal above or below that

means_for_seasons = []
for i in range(10000):
    n = 1
    p = .53
    cumulative_money_list = np.random.binomial(n, p, size=1000)
    actual_win_pct = np.mean(cumulative_money_list)
    means_for_seasons.append(actual_win_pct)

plt.hist(means_for_seasons, bins=15, alpha=.7)
plt.title('from a binomial distribution with mean of 53%', fontsize=15)
plt.xticks(fontsize=15)
sns.despine()

# ----------------------------------------------------
# simulate 1000 games at x percent
n = 1
p = .52
cumulative_money_list = np.random.binomial(n, p, size=1000)
actual_win_pct = str(round(np.mean(cumulative_money_list),3))
print(actual_win_pct)

# can i try and overlap a bunch of lines and make alphas light?
# so can put lots of graphs into one.

# should they all be of a certain win%? or from the same
# probability distribution (and so haveing diff win% but 
# from the same distribution?)


# ----------------------------------------------------
# simulate winnings over a season using kelly criteria

# regular appraoch:
win_probability = .52 # put pct I think we can realistically win at.
# kelly formula says that if i can put the actual pct, i'll maximize the winnings
# but the toal pot gets more and more volatilse the higher it goes, i.e., betting more and more
# so makes some sense to put a more conservative estimate, under what I think we'll get
juice = .945
kelly_criteria = (win_probability * juice - (1 - win_probability)) / juice
money_kelly = 10000
bet_kelly = money_kelly * kelly_criteria
total_pot_kelly = 0
total_winnings_kelly_list = [0]
for game in cumulative_money_list:
    if game == 1:
        total_pot_kelly += bet_kelly*juice
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += bet_kelly*juice
        bet_kelly = money_kelly * kelly_criteria
    if game == 0:
        total_pot_kelly += -1*bet_kelly
        total_winnings_kelly_list.append(total_pot_kelly)
        money_kelly += -1*bet_kelly
        bet_kelly = money_kelly * kelly_criteria


# plot winnings
plt.plot(total_winnings_kelly_list, alpha=.4, color='green', linewidth=2)
plt.xlabel('\n games', fontsize=18)
plt.ylabel('winnings', fontsize=18)
plt.yticks(fontsize=15)
#plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1500)
plt.ylim(-6000,8000)
plt.xlim(-5,1020)
plt.axhline(.5, linestyle='--', color='red', linewidth=1, alpha=.5)
plt.grid(True, axis='y', which='minor', alpha=.4)
plt.grid(True, axis='y', which='major', alpha=.95)
plt.minorticks_on()
plt.title('\nwin percentage: '+actual_win_pct+'' + '\n winnings kelly: $' + str(int(total_winnings_kelly_list[-1])), fontsize=18)
sns.despine()


# ---
# loop

# 50 %
min_percentage = .495
max_percentage = .505

# 52%
min_percentage = .515
max_percentage = .525

# 52.5%
min_percentage = .52
max_percentage = .53

# 53%
min_percentage = .525
max_percentage = .535

# 53.5
min_percentage = .53
max_percentage = .54

juice = .948

i = 0
while i < 100:
    print(i)
    n = 1
    p = .5
    cumulative_money_list = np.random.binomial(n, p, size=1000)
    actual_win_pct = np.mean(cumulative_money_list)
    if (actual_win_pct > min_percentage and actual_win_pct < max_percentage):
        i = i + 1
        win_probability = .52 
        kelly_criteria = (win_probability * juice - (1 - win_probability)) / juice
        money_kelly = 5000
        bet_kelly = money_kelly * kelly_criteria
        total_pot_kelly = 0
        total_winnings_kelly_list = [0]
        for game in cumulative_money_list:
            if game == 1:
                total_pot_kelly += bet_kelly*juice
                total_winnings_kelly_list.append(total_pot_kelly)
                money_kelly += bet_kelly*juice
                bet_kelly = money_kelly * kelly_criteria
            if game == 0:
                total_pot_kelly += -1*bet_kelly
                total_winnings_kelly_list.append(total_pot_kelly)
                money_kelly += -1*bet_kelly
                bet_kelly = money_kelly * kelly_criteria
        
        # plot winnings
        plt.plot(total_winnings_kelly_list, alpha=.05, color='purple', linewidth=2.5)
        plt.xlabel('\n games', fontsize=18)
        plt.ylabel('winnings', fontsize=18)
        plt.yticks(fontsize=15)
        #plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1500)
        plt.ylim(-4000,8000)
        plt.xlim(-5,1020)
        plt.axhline(.5, linestyle='--', color='green', linewidth=2, alpha=.8)
        plt.grid(True, axis='y', which='minor', alpha=.4)
        plt.grid(True, axis='y', which='major', alpha=.95)
        plt.minorticks_on()
        #plt.title('\nwin percentage: '+ str(round(actual_win_pct, 3))+'' + '\n winnings kelly: $' + str(int(total_winnings_kelly_list[-1])), fontsize=18)
        plt.title('simulated seasons between ' + str(min_percentage) + '% and ' + str(max_percentage) + '%', fontsize=18)        
        sns.despine()
    else:
        None




# ---------------------------------------------------
# alt approach simulating by day (what we actuall do)

df_2015['correct'] = cumulative_money_list
df_2015['correct'].mean()

df_2015_by_day_wins = df_2015.groupby('date')['correct'].sum()
df_2015_by_day_wins = df_2015_by_day_wins.reset_index()
df_2015_by_day_games = df_2015.groupby('date')['correct'].count()
df_2015_by_day_games = df_2015_by_day_games.reset_index()

df_2015_merge = pd.merge(df_2015_by_day_wins, df_2015_by_day_games, on='date')
df_2015_merge.rename(columns={'correct_x':'wins'}, inplace=True)
df_2015_merge.rename(columns={'correct_y':'games'}, inplace=True)
df_2015_merge['games'] = df_2015_merge['games'].astype(float)

print(df_2015_merge['wins'].sum() / df_2015_merge['games'].sum())


# 50 %
min_percentage = .495
max_percentage = .505

# 52.5%
min_percentage = .52
max_percentage = .53

juice = .948


i = 0
while i < 100:
    print(i)
    n = 1
    p = .525
    cumulative_money_list = np.random.binomial(n, p, size=1000)
    actual_win_pct = np.mean(cumulative_money_list)
    df_2015['correct'] = cumulative_money_list    
    df_2015_by_day_wins = df_2015.groupby('date')['correct'].sum()
    df_2015_by_day_wins = df_2015_by_day_wins.reset_index()
    df_2015_by_day_games = df_2015.groupby('date')['correct'].count()
    df_2015_by_day_games = df_2015_by_day_games.reset_index()
    df_2015_merge = pd.merge(df_2015_by_day_wins, df_2015_by_day_games, on='date')
    df_2015_merge.rename(columns={'correct_x':'wins'}, inplace=True)
    df_2015_merge.rename(columns={'correct_y':'games'}, inplace=True)
    df_2015_merge['games'] = df_2015_merge['games'].astype(float)
    actual_win_pct = df_2015_merge['wins'].sum() / df_2015_merge['games'].sum()
    if (actual_win_pct > min_percentage and actual_win_pct < max_percentage):
        i = i + 1
        pot = 5000
        win_probability = .52
        shrink_bets = 1
        kelly_criteria = (win_probability * juice - (1 - win_probability)) / juice
        bet_kelly = pot * kelly_criteria
        bet_kelly = bet_kelly*shrink_bets
        total_pot_kelly = 0
        total_winnings_kelly_list = [0]
        actual_win_pct = str(round(df_2015_merge['wins'].sum() / df_2015_merge['games'].sum(), 3))
        
        dates = df_2015_merge['date'].unique()
        for day in dates:
            print(day)
            wins = df_2015_merge[df_2015_merge['date']==day]['wins'].values[0]
            games = df_2015_merge[df_2015_merge['date']==day]['games'].values[0]
            losses = games - wins
            winnings_for_day = wins*juice*bet_kelly
            losses_for_day = losses*bet_kelly*-1
            total_pot_kelly = total_pot_kelly + winnings_for_day + losses_for_day
            total_winnings_kelly_list.append(total_pot_kelly)
            pot = pot + winnings_for_day + losses_for_day
            bet_kelly = pot * kelly_criteria
            bet_kelly = bet_kelly*shrink_bets
            print(bet_kelly)

        plt.plot(total_winnings_kelly_list, alpha=.05, color='purple', linewidth=2.5)
        plt.xlabel('\n games', fontsize=15)
        plt.ylabel('winnings', fontsize=15)
        plt.xlim(0,len(total_winnings_kelly_list)+5)
        #plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1500)
        plt.ylim(-4000,8000)
        plt.axhline(.5, linestyle='--', color='green', linewidth=2, alpha=.8)
        plt.grid(True, axis='y', which='minor', alpha=.4)
        plt.grid(True, axis='y', which='major', alpha=.95)
        plt.minorticks_on()
        plt.title('simulated seasons between ' + str(min_percentage) + '% and ' + str(max_percentage) + '%', fontsize=18)        
        sns.despine()



# ---------
pot = 10000
win_probability = .52
shrink_bets = 1
kelly_criteria = (win_probability * juice - (1 - win_probability)) / juice
bet_kelly = pot * kelly_criteria
bet_kelly = bet_kelly*shrink_bets
total_pot_kelly = 0
total_winnings_kelly_list = [0]
actual_win_pct = str(round(df_2015_merge['wins'].sum() / df_2015_merge['games'].sum(), 3))

dates = df_2015_merge['date'].unique()
for day in dates:
    print(day)
    wins = df_2015_merge[df_2015_merge['date']==day]['wins'].values[0]
    games = df_2015_merge[df_2015_merge['date']==day]['games'].values[0]
    losses = games - wins
    winnings_for_day = wins*juice*bet_kelly
    losses_for_day = losses*bet_kelly*-1
    total_pot_kelly = total_pot_kelly + winnings_for_day + losses_for_day
    total_winnings_kelly_list.append(total_pot_kelly)
    pot = pot + winnings_for_day + losses_for_day
    bet_kelly = pot * kelly_criteria
    bet_kelly = bet_kelly*shrink_bets
    print(bet_kelly)


plt.plot(total_winnings_kelly_list, alpha=.05, color='purple', linewidth=2.5)
plt.xlabel('\n games', fontsize=15)
plt.ylabel('winnings', fontsize=15)
plt.xlim(0,len(total_winnings_kelly_list)+5)
#plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1500)
plt.ylim(-4000,8000)
plt.axhline(.5, linestyle='--', color='green', linewidth=2, alpha=.8)
plt.grid(True, axis='y', which='minor', alpha=.4)
plt.grid(True, axis='y', which='major', alpha=.95)
plt.minorticks_on()
plt.title('simulated seasons between ' + str(min_percentage) + '% and ' + str(max_percentage) + '%', fontsize=18)        
sns.despine()



























