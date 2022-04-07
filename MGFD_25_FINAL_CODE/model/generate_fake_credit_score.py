from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from scipy.stats import skewnorm

# create some random data from a skewnorm
#Generate Period One Data
data = skewnorm.rvs(-2, loc=650, scale=200, size=10000).astype(np.int)
credit_score_data_period_one = [i for i in data if i > 300 and i < 900]
#Generate Period Two Data
credit_score_data_period_two = [m+random.randint(-50, 50) for m in credit_score_data_period_one if m - 50 >=300 and m + 50 <=900]
print(len(credit_score_data_period_one), len(credit_score_data_period_two))

#Should be unequal length due to death or dropouts
df = pd.DataFrame({'credit_score_period_one':pd.Series(credit_score_data_period_one), 'credit_score_period_two':pd.Series(credit_score_data_period_two)})
"""
    [credit_score_data_period_one, credit_score_data_period_two], 
    columns=['credit_score_period_one', 'credit_score_period_two']
"""
df.to_csv("fake_credit_score.csv")



# draw a histogram and kde of the given data
ax = sns.distplot(data, kde_kws={'label':'kde of given data'}, label='histogram')

# find parameters to fit a skewnorm to the data
params = skewnorm.fit(data, 300, loc=900, scale=40)

# draw the pdf of the fitted skewnorm
x = np.linspace(0, 300, 900)
ax.plot(x, skewnorm.pdf(x, *params), label='approximated skewnorm')
plt.legend()
plt.show()