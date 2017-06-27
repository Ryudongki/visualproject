import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import arange
from numpy import meshgrid

w1 = [[], [], []]
w3 = [[], [], []]
 
f = open("C:\\Users\\Ryu\\Desktop\\w1.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    w1[0].append(float(x[0]))
    w1[1].append(float(x[1]))
    w1[2].append(int(x[2]))

f.close()

f = open("C:\\Users\\Ryu\\Desktop\\w3.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    w3[0].append(float(x[0]))
    w3[1].append(float(x[1]))
    w3[2].append(int(x[2]))

f.close()

weightVector = np.zeros((1, 3))

x = np.zeros((1, 3))
y = np.zeros((1, 3))
temp1 = 0
temp2 = 0
cnt1 = 0
cnt2 = 0
#learning_rate = 0.01

rangeX = arange(-10.0, 10.0, 0.05)
rangeY = arange(-10.0, 10.0, 0.05)
X, Y = meshgrid(rangeX, rangeY)

#weightVector[0][0], weightVector[0][1], weightVector[0][2] = 0, 0, 0
weightVector[0][0], weightVector[0][1], weightVector[0][2] = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
learning_rate = 0.01
for k in range(1001):

    for i in range(len(w1[0])):

        x[0][0], x[0][1], x[0][2] = w1[0][i], w1[1][i], w1[2][i]

        temp1 = weightVector[0][0] * x[0][0] + weightVector[0][1] * x[0][1] + weightVector[0][2] * x[0][2]

        if(temp1 >= 0):
            cnt1 += 1
        else:
            weightVector[0][0] += learning_rate * (0.5 - (weightVector[0][0] * x[0][0] + weightVector[0][1] * x[0][1] + weightVector[0][2] * x[0][2])) / pow((x[0][0] * x[0][0] + x[0][1] * x[0][1] + x[0][2] * x[0][2]), 2) * x[0][0]
            weightVector[0][1] += learning_rate * (0.5 - (weightVector[0][0] * x[0][0] + weightVector[0][1] * x[0][1] + weightVector[0][2] * x[0][2])) / pow((x[0][0] * x[0][0] + x[0][1] * x[0][1] + x[0][2] * x[0][2]), 2) * x[0][1]
            weightVector[0][2] += learning_rate * (0.5 - (weightVector[0][0] * x[0][0] + weightVector[0][1] * x[0][1] + weightVector[0][2] * x[0][2])) / pow((x[0][0] * x[0][0] + x[0][1] * x[0][1] + x[0][2] * x[0][2]), 2) * x[0][2]

    for j in range(len(w3[0])):
        y[0][0], y[0][1], y[0][2] = w3[0][j], w3[1][j], w3[2][j]

        temp2 = weightVector[0][0] * y[0][0] + weightVector[0][1] * y[0][1] + weightVector[0][2] * y[0][2]

        if(temp2 < 0):
            cnt2 += -1
        else:
            weightVector[0][0] += learning_rate  * (0.5 - (weightVector[0][0] * y[0][0] + weightVector[0][1] * y[0][1] + weightVector[0][2] * y[0][2])) / pow((y[0][0] * y[0][0] + y[0][1] * y[0][1] + y[0][2] * y[0][2]), 2) * y[0][0]
            weightVector[0][1] += learning_rate  * (0.5 - (weightVector[0][0] * y[0][0] + weightVector[0][1] * y[0][1] + weightVector[0][2] * y[0][2])) / pow((y[0][0] * y[0][0] + y[0][1] * y[0][1] + y[0][2] * y[0][2]), 2) * y[0][1]
            weightVector[0][2] += learning_rate  * (0.5 - (weightVector[0][0] * y[0][0] + weightVector[0][1] * y[0][1] + weightVector[0][2] * y[0][2])) / pow((y[0][0] * y[0][0] + y[0][1] * y[0][1] + y[0][2] * y[0][2]), 2) * y[0][2]

    if(cnt1 == 10 and cnt2 == -10):
        break

w1dot = np.zeros((len(w1[0]), len(w1[1])))
w3dot = np.zeros((len(w3[0]), len(w3[1])))
w1dot[0], w1dot[1] = w1[0], w1[1]
w3dot[0], w3dot[1] = w3[0], w3[1]

x = plt.plot(w1dot[0], w1dot[1], 'ro')
y = plt.plot(w3dot[0], w3dot[1], 'ro')
plt.setp(x, color = 'r')
plt.setp(y, color = 'b')
plt.axis([-8.0, 10.0, -8.0, 10.0])

equation = weightVector[0][0] * X + weightVector[0][1] * Y + weightVector[0][2]

plt.contour(X, Y, equation, [0], colors = 'black')

plt.show()
