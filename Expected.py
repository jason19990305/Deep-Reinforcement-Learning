import numpy as np

sum = 0
for i in range(6):
    data = i+1
    sum += (1/6)*data
print("Real Expected :",sum)
sum = 0 
num = 300000
sample = np.random.choice(6, num)+1 # sample 1~ 6 100 times
for d in sample:
    #print(d)
    sum += d
sum /= num
print("Sample Expected :",sum)
    