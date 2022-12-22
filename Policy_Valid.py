from math import log
from matplotlib import pyplot as plt
import numpy as np


y_pred = np.array([0.1,0.8])
reward = 5
loss = -reward*np.log(y_pred)
print("--Positive---\nAction pred :",y_pred)
print("Loss : ",loss)

y_pred = np.array([0.9,0.2])
loss = -reward*np.log(y_pred)
print("Action pred :",y_pred)
print("Loss : ",loss)
print("-----")




reward = -5
y_pred = np.array([0.1,0.8])
loss = -reward*np.log(y_pred)
print("---Nagtive--\nAction pred :",y_pred)
print("Loss : ",loss)

y_pred = np.array([0.9,0.2])
loss = -reward*np.log(y_pred)
print("Action pred :",y_pred)
print("Loss : ",loss)
print("-----")
