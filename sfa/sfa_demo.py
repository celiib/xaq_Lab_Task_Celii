import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from single_Factor_Analyzer import sfa

#create sample data set by combining two different gaussian distributions 
#into 1 matrix 

data1 = torch.randn(300,5);
data2 = torch.add(torch.randn(200,5),2)
data3 = torch.cat((data1,data2))
print(data3)

FL,diag_unique,LL = sfa(data3,3)

#print the parameters chosen
print("Factor Loadings = \n" + str(FL))
print("Diagnoal Uniqueness Matrix = \n" + str(diag_unique))
print(LL)

#plot the log likelihood curve
fig, ax = plt.subplots()

ax.plot(LL)

ax.set(xlabel='iterations',ylabel='log likelihood',title="Demo of Single Factor Analyzer on random model")
ax.grid()

plt.show()