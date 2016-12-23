# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:47:39 2016

@author: charlesmartens
"""


cd /Users/charlesmartens/Documents/projects/bet_bball

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from scipy import stats
import copy
sns.set_style('white')




# Discretise the x-axis into 1000 separate plotting points
x = np.linspace(0, 1, 1000)

# if i set this prior to 1000 trials, them i'm saying the current season is 
# going to take on as much value as the prior. i.e., if we loose this season
# it will likely wipe out any belief that the model works. is this fair?
trials = 1000  
correct_pct = .535  # this is the mean of the past 6 seasons if i bet on all games
correct = trials * correct_pct
incorrect = trials - correct  

# new evidence: this season
new_trials = 150
correct_pct_new = .54
correct_new = new_trials * correct_pct_new
incorrect_new = new_trials - correct_new  

correct_posterior = correct+correct_new
incorrect_posterior = incorrect+incorrect_new

y = stats.beta.pdf(x, correct_posterior, incorrect_posterior)
mean = correct_posterior/(correct_posterior+incorrect_posterior)
stdev = np.sqrt((correct_posterior * incorrect_posterior) / (np.power((correct_posterior + incorrect_posterior), 2) * (correct_posterior + incorrect_posterior + 1)))

plt.plot(x, y)
# plt.axvline(.515, linewidth=1, alpha=.75, color='red')
plt.axvline((mean - stdev), linewidth=.75, alpha=.7, color='red')
plt.axvline((mean - 2*stdev), linewidth=.75, alpha=.7, color='red')
plt.xlim(.4,.68)
plt.title('mean = ' + str(round(100*mean, 2))+' percent \nlower bounds = ' +str(round(100*(mean - stdev), 2))+ ' and ' + str(round(100*(mean - 2*stdev), 2))+ ' percent', fontsize=15)
sns.despine()


# 2 std devs below mean = 2.3% chance of happening
# 1 std dev below mean = 16% chance of happening
# so should feel ok about going with 1 std dev below mean as my win proba!


# i should run the numbers from after game 15 with my new algo and see what 
# the win% would be. This is about nov 22. win pct since then is 54%. so use 
# that as evidence today. but actually, should run this each day. or just 
# know what number of wins and losses to add to what i have now to get that
# pct. so each day can run our pct from when we actually started gambling. 
# then add thes additional wins and losses and compute a new win pct. this 
# is our evidence to add to the prior.

# since nov 22: 81 / 150
# since dec 7: 34 / 60
# difference correct: 81 - 34 = 47  --  so add 47 wins to whatever my current wins are ea day
# difference games: 150 - 60 = 90  --  so add 90 to whatever my current wins are ea day


total_amount = 10400
pot = total_amount/2 - 30
win_probability = .521
kelly_criteria = (win_probability * .945 - (1 - win_probability)) / .945
bet_kelly = pot * kelly_criteria
print(round(bet_kelly))


# if we just go on this years evidence (i.e., no prior)
y = stats.beta.pdf(x, 81, 69)
mean = 81/(81+69)
stdev = np.sqrt((81 * 150) / (np.power((81 + 150), 2) * (81 + 150 + 1)))
plt.plot(x, y)
# plt.axvline(.515, linewidth=1, alpha=.75, color='red')
plt.axvline((mean - stdev), linewidth=.75, alpha=.7, color='red')
plt.axvline((mean - 2*stdev), linewidth=.75, alpha=.7, color='red')
plt.xlim(.4,.68)
plt.title('mean = ' + str(round(100*mean, 2))+' percent \nlower bounds = ' +str(round(100*(mean - stdev), 2))+ ' and ' + str(round(100*(mean - 2*stdev), 2))+ ' percent', fontsize=15)
sns.despine()




# what else is worth predicting? or automating?
# ------------
# TEST -- SEE HOW VOLATILE THE WIN PROBA FOR KELLY WILL BE THROUGHOUT SEASON

# simulate seasons where win% is around 50%, 51%, 52%, 53%, 54%, 55%
# show how much money we make over time using a flat kelly win proba of
# 52% and 52.5%, vs. using a moving win proba based on the mean and 
# 1 std dev below the mean





# 53%


min_percentage = .516
max_percentage = .518
p = .517



min_percentage = .549
max_percentage = .551
p = .55

min_percentage = .539
max_percentage = .541
p = .54

min_percentage = .529
max_percentage = .531
p = .53

min_percentage = .519
max_percentage = .521
p = .52

min_percentage = .509
max_percentage = .511
p = .51

min_percentage = .499
max_percentage = .501
p = .50

min_percentage = .489
max_percentage = .491
p = .49


juice = .948
n = 1

i = 0
while i < 2:
    print(i)
    cumulative_money_list = np.random.binomial(n, p, size=1000)
    actual_win_pct = np.mean(cumulative_money_list)
    if (actual_win_pct > min_percentage and actual_win_pct < max_percentage):
        i = i + 1
   
len(cumulative_money_list)
win_pct = sum(cumulative_money_list) / len(cumulative_money_list)
print(win_pct)


# produce kelly win proba lists:
x = np.linspace(0, 1, 1000)

# prior:
trials = 1000
trials = 2000  
trials = 3000  
trials = 4000  
  
correct_pct = .535  # this is the mean of the past 6 seasons if i bet on all games
correct = trials * correct_pct
incorrect = trials - correct  

# new evidence: this season
lower_bound_list = []
mean_list = []
for i in range(len(cumulative_money_list)):
    games = list(cumulative_money_list[:i+1])
      
    correct_this_season = sum(games)
    incorrect_this_season =  len(games) - correct_this_season
    correct_pct_new = correct_this_season / (correct_this_season + incorrect_this_season)

    correct_posterior = correct+correct_this_season
    incorrect_posterior = incorrect+incorrect_this_season

    y = stats.beta.pdf(x, correct_posterior, incorrect_posterior)
    mean = correct_posterior/(correct_posterior+incorrect_posterior)
    stdev = np.sqrt((correct_posterior * incorrect_posterior) / (np.power((correct_posterior + incorrect_posterior), 2) * (correct_posterior + incorrect_posterior + 1)))
    #print('mean:', round(mean,4)*100, '| lower bound:', round(mean-stdev,4)*100)
    lower_bound_list = lower_bound_list + [round(mean-stdev,4)*100]
    mean_list = mean_list + [round(mean,4)*100]
lower_bound_list = [52] + lower_bound_list


lower_bound_list_1000 = copy.deepcopy(lower_bound_list)
lower_bound_list_2000 = copy.deepcopy(lower_bound_list)
lower_bound_list_3000 = copy.deepcopy(lower_bound_list)
lower_bound_list_4000 = copy.deepcopy(lower_bound_list)

lower_bound_list_1000 = [win_proba if win_proba > 51.42 else 51.42 for win_proba in lower_bound_list_1000]
lower_bound_list_2000 = [win_proba if win_proba > 51.4 else 51.4 for win_proba in lower_bound_list_2000]
#lower_bound_list_3000 = [win_proba if win_proba > 51.4 else 51.4 for win_proba in lower_bound_list_3000]
#lower_bound_list_3000 = [win_proba if win_proba > 51.4 else 51.4 for win_proba in lower_bound_list_4000]



#lower_bound_list = [52.0] + lower_bound_list
#mean_list = [53.0] + mean_list
#lower_bound_list[:5]
#mean_list[:5]
#mean_lower_bound = np.mean(lower_bound_list)

# plot the moving kelly win probas
plt.plot(lower_bound_list_1000, label='moving kelly win probability, prior w 1000 trials', color='grey', alpha=.5)
plt.plot(lower_bound_list_2000, label='moving kelly win probability, prior w 2000 trials', color='blue', alpha=.5)
plt.plot(lower_bound_list_3000, label='moving kelly win probability, prior w 3000 trials', color='green', alpha=.5)
plt.plot(lower_bound_list_4000, label='moving kelly win probability, prior w 4000 trials', color='red',  alpha=.5)
#plt.plot(mean_list, label='kelly win probability, mean')
plt.grid(axis='y', alpha=.4)
plt.ylim(50.5, 54.4)
plt.yticks(fontsize=15)
plt.title('win probabiilties to use with kelly formula, \nactual winning = ' +str(win_pct), fontsize=15)
plt.xlabel('games in season', fontsize=15)
plt.axhline(52.49, color='purple', label='constant kelly win probability at 52.5%', linewidth=2.1, alpha=.5)
plt.legend(fontsize=12, loc='bottom')
sns.despine()


# the final posterior probability distribution
#plt.plot(x, y)
#plt.axvline((mean - stdev), linewidth=.75, alpha=.7, color='red')
#plt.axvline((mean - 2*stdev), linewidth=.75, alpha=.7, color='red')
#plt.xlim(.4,.68)
#plt.title('mean = ' + str(round(100*mean, 2))+' percent \nlower bounds = ' +str(round(100*(mean - stdev), 2))+ ' and ' + str(round(100*(mean - 2*stdev), 2))+ ' percent', fontsize=15)
#sns.despine()

#i = 0
lower_bound_list = copy.deepcopy(lower_bound_list_1000)
lower_bound_list = copy.deepcopy(lower_bound_list_2000)
lower_bound_list = copy.deepcopy(lower_bound_list_3000)
lower_bound_list = copy.deepcopy(lower_bound_list_4000)
lower_bound_list = [52.5]*1000


total_winnings_kelly_list = [0]
total_winnings = 0
total_pot = 5000
for i in range(len(cumulative_money_list)):
    win_probability = lower_bound_list[i]/100
    #win_probability = mean_list[i]/100
    #win_probability = .525
    #win_probability = .52
    #win_probability = mean_lower_bound/100

#    if win_probability < .519:
#        win_probability = .519
    if win_probability < .514:
        win_probability = .514  # may as well do this?

    kelly_criteria = (win_probability * juice - (1 - win_probability)) / juice
    bet_kelly = total_pot * kelly_criteria
    if cumulative_money_list[i] == 1:
        total_winnings += bet_kelly*juice
        total_winnings_kelly_list.append(total_winnings)
        total_pot += bet_kelly*juice
    if cumulative_money_list[i] == 0:
        total_winnings += -1*bet_kelly
        total_winnings_kelly_list.append(total_winnings)
        total_pot += -1*bet_kelly


total_winnings_kelly_list_1000 = copy.deepcopy(total_winnings_kelly_list)
total_winnings_kelly_list_2000 = copy.deepcopy(total_winnings_kelly_list)
total_winnings_kelly_list_3000 = copy.deepcopy(total_winnings_kelly_list)
total_winnings_kelly_list_4000 = copy.deepcopy(total_winnings_kelly_list)
total_winnings_kelly_list_525 = copy.deepcopy(total_winnings_kelly_list)

win_pct_this_season = sum(cumulative_money_list) / len(cumulative_money_list)
print(win_pct_this_season)


# to see betting with regular win proba:
#plt.plot(total_winnings_kelly_list, alpha=.3, color='purple', linewidth=2.5)
plt.plot(total_winnings_kelly_list_1000, alpha=.3, color='grey', linewidth=2.5, label='moving kelly win probability, prior w 1000 trials')
plt.plot(total_winnings_kelly_list_2000, alpha=.3, color='blue', linewidth=2.5, label='moving kelly win probability, prior w 2000 trials')
plt.plot(total_winnings_kelly_list_3000, alpha=.3, color='green', linewidth=2.5, label='moving kelly win probability, prior w 3000 trials')
plt.plot(total_winnings_kelly_list_4000, alpha=.3, color='red', linewidth=2.5, label='moving kelly win probability, prior w 4000 trials')
plt.plot(total_winnings_kelly_list_525, alpha=.3, color='purple', linewidth=2.5, label='constant kelly win probability = 52.5%')
plt.xlabel('\n games in season', fontsize=12)
#plt.ylabel('winnings', fontsize=12)
plt.yticks(fontsize=15)
#plt.ylim(min(total_winnings_kelly_list)-1000,max(total_winnings_kelly_list)+1500)
#plt.ylim(-3000,8000)
plt.ylim(-4000,41000)
plt.ylim(-2500,25000)
plt.xlim(-10,1010)
plt.axhline(.5, linestyle='--', color='green', linewidth=2, alpha=.8)
plt.grid(True, axis='y', which='minor', alpha=.2)
plt.grid(True, axis='y', which='major', alpha=.7)
plt.minorticks_on()
#plt.title('\nwin percentage: '+ str(round(actual_win_pct, 3))+'' + '\n winnings kelly: $' + str(int(total_winnings_kelly_list[-1])), fontsize=18)
#plt.title('simulated winnings: '+str(round(total_winnings,0))[:-2], fontsize=18)        
plt.title('simulated winnings \nactual winning = ' +str(win_pct), fontsize=15)
plt.legend(loc='left', fontsize=12)
sns.despine()






# ----
probas = [.521, .520, .519, .518, .517, .516, .515, .514]
for proba in probas:
    total_amount = 10000
    pot = total_amount/2 - 30
    win_probability = proba
    kelly_criteria = (win_probability * juice - (1 - win_probability)) / juice
    bet_kelly = pot * kelly_criteria
    print(proba, '\t', round(bet_kelly))



# would be good to break this down into chunks of 7 or 10 games, so simulating days rather than changing kelly win proba each dat
# maybe mean list is actually the best for moving kelly proba because it adjusts?
# try setting prior with 2000 and 4000 trials and compare with 3000
















# ----------------------------------------------------------------------------
# stuff from blog ------------------------------------------------------------
# https://www.quantstart.com/articles/Bayesian-Inference-of-a-Binomial-Proportion-The-Analytical-Approach
# Create a list of the number of coin tosses ("Bernoulli trials")
number_of_trials = [0, 2, 10, 20, 50, 500]

# Conduct 500 coin tosses and output into a list of 0s and 1s
# where 0 represents a tail and 1 represents a head
data = stats.bernoulli.rvs(0.5, size=number_of_trials[-1])

# Discretise the x-axis into 100 separate plotting points
x = np.linspace(0, 1, 1000)


for i, N in enumerate(number_of_trials):
    # Accumulate the total number of heads for this 
    # particular Bayesian update
    heads = data[:N].sum()

    # Create an axes subplot for each update 
    ax = plt.subplot(len(number_of_trials) / 2, 2, i + 1)
    ax.set_title("%s trials, %s heads" % (N, heads))

    # Add labels to both axes and hide labels on y-axis
    plt.xlabel("$P(H)$, Probability of Heads")
    plt.ylabel("Density")
    if i == 0:
        plt.ylim([0.0, 2.0])
    plt.setp(ax.get_yticklabels(), visible=False)
            
    # Create and plot a  Beta distribution to represent the 
    # posterior belief in fairness of the coin.
    # x is a list of 100 numbers between 0 ad 1 to make the line with
    # 1 + heads is the number of successes, 1 + N - heads is the number of failures
    y = stats.beta.pdf(x, 1 + heads, 1 + N - heads)
    plt.plot(x, y, label="observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color="#aaaadd", alpha=0.5)

# Expand plot to cover full width/height and show it
plt.tight_layout()
plt.show()

# my own:
y = stats.beta.pdf(x, 54, 46)
plt.plot(x, y)
plt.fill_between(x, 0, y, color="#aaaadd", alpha=0.4)

y = stats.beta.pdf(x, 28, 22)
plt.plot(x, y)
plt.fill_between(x, 0, y, color="#aaaadd", alpha=0.4)
plt.axvline(.515, linewidth=1, alpha=.75, color='red')
sns.despine()

y = stats.beta.pdf(x, 56, 44)
plt.plot(x, y)
plt.fill_between(x, 0, y, color="#aaaadd", alpha=0.4)
plt.axvline(.515, linewidth=1, alpha=.75, color='red')
sns.despine()


np.mean([52.24, 54.28, 52.69, 54.98, 54.11, 54.4])
#np.mean([52.58, 54.70, 53.33, 55.30, 54.43, 55.02])



























