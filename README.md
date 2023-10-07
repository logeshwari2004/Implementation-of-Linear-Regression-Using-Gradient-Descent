# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by:LOGESHWARI.P 
RegisterNumber:212221230055

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(X,y,theta))
  return theta,j_history  
  
theta,j_history=gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000)s")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions= np.dot(theta.transpose(),x)
    return predictions[0]
    
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:
##Profit Prediction:
![271843574-a291c119-8fe4-46c6-877c-18c0b8e761ab](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/2bbca5cf-bbaa-4d02-bdd6-e872185fda05)
##Compute Cost:
![271843594-c219e692-5cc8-4b76-8b23-f7845cc84149](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/219a6485-5573-4aaa-813c-bfe8dda5076e)
##h(x) Value:
![271843622-e0aa8f1e-8c95-42b8-8024-e116a2ba5bb7](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/6ea32453-e41b-4d5e-9be1-cbf9239d2ab7)
##Cost function using Gradient Descent Graph:
![271843641-cc61d23f-5bf9-4b4d-80d0-3d4f9e5611d0](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/5c504178-7bc0-4a83-8367-dde3a2a13063)
##Profit Prediction:
![271843662-aea93272-9548-447e-b7c9-b71107a5285e](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/596fa99b-c350-490d-b3e6-93dafb473516)
##Profit for the Population 35,000:
![271843718-b2925b27-eb39-4731-9126-83232f711805](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/02bd16fa-d8e7-4241-a818-07db1c03cc4b)
##Profit for the Population 70,000:
![271843729-09755fed-be1f-4b5d-a865-4f411f600f42](https://github.com/logeshwari2004/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94211349/40953989-1c1a-4a21-bc4e-48c6d4efd469)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
