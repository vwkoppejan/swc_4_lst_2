
' ------ Purpose of the code ------ '
# Determine the N2O consumption rate (rN2O) during a batch test with a mixed culture. 
# This is done by fitting the model of N2O dynamics (considering reaction and 
# gas-liquid transfer) to the experimental data.


import pandas
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt


plt.close('all')


' -------- Import the raw data -------- '

# Load the experimental data
# Column 1: t (min)
# Column 2: N2O concentration in the broth (mM)
b = pandas.read_excel('Data_N2O.xlsx')


# Define the experimental data: timepoints and concentrations
t_exp = b['t(min)'].values    # min
c_exp = b['N2O(mM)'].values   # mM

# Remove NaN elements
t_exp = t_exp[~np.isnan(t_exp)]
c_exp = c_exp[~np.isnan(c_exp)]

' ------ Perform the minimization using least squares------ '

# solve the least squares system with inversion

# create a coefficient matrix: first column are t values, second column is ones. 
TA = np.ones((30,2))
TA[:,0] = t_exp

# choose method by using the poor mans method of commenting/uncommenting
# inversion
#X = np.linalg.inv(TA.T@TA)@TA.T@c_exp

# PINV
#X = np.linalg.pinv(TA)@c_exp

# NPLSQ
X = np.linalg.lstsq(TA,c_exp, rcond=None)[0]

# REGRESS 
#X = linregress(t_exp,c_exp)

print(X)

# reconstruct
c_fit = X[0]*t_exp+X[1]

# mean C fit
cfmean = sum(c_exp)/np.size(c_exp)

# calculate the Rsquared
rsq = 1 - sum((c_exp-c_fit)**2)/sum((c_fit-cfmean)**2)

print(rsq)

plt.plot(t_exp,c_exp, 'o', color = 'C0')
plt.plot(t_exp,c_fit, color = 'C1')

plt.xlabel('t (min)')
plt.ylabel('N2O (mM)')
plt.title('Regress')










