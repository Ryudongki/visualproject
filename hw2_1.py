import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import arange
from numpy import meshgrid

w1 = [[], [], []]
w2 = [[], [], []]

f = open("C:\\Users\\Ryu\\Desktop\\w1.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    w1[0].append(float(x[0]))
    w1[1].append(float(x[1]))
    w1[2].append(int(x[2]))

f.close()

f = open("C:\\Users\\Ryu\\Desktop\\w2.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    w2[0].append(float(x[0]))
    w2[1].append(float(x[1]))
    w2[2].append(int(x[2]))

f.close()

weightVector = np.zeros((1, 3))
x = np.zeros((1, 3))
y = np.zeros((1, 3))
temp1 = 0
temp2 = 0

learning_rate = 0.1

while(1):
    weightVector[0][0], weightVector[0][1], weightVector[0][2] = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
    sum1 = 0
    sum2 = 0
    a = 0
    b = 0

    for i in range(len(w1[0])):

        x[0][0], x[0][1], x[0][2] = w1[0][i], w1[1][i], w1[2][i]

        temp1 = weightVector[0][0] * x[0][0] + weightVector[0][1] * x[0][1] + weightVector[0][2] * x[0][2]

        if(temp1 < 0):
            sum1 += -1
            a = -1
        else:
            a = 1

        weightVector[0][0] += learning_rate * x[0][0] * (-1 - a)
        weightVector[0][1] += learning_rate * x[0][1] * (-1 - a)
        weightVector[0][2] += learning_rate * x[0][2] * (-1 - a)

    for j in range(len(w2[0])):
        y[0][0], y[0][1], y[0][2] = w2[0][j], w2[1][j], w2[2][j]

        temp2 = weightVector[0][0] * y[0][0] + weightVector[0][1] * y[0][1] + weightVector[0][2] * y[0][2]

        if(temp2 >= 0):
            sum2 += 1
            b = 1
        else:
            b = -1

        weightVector[0][0] += learning_rate * y[0][0] * (1 - b)
        weightVector[0][1] += learning_rate * y[0][1] * (1 - b)
        weightVector[0][2] += learning_rate * y[0][2] * (1 - b)

    if((sum1 == -10) and (sum2 == 10)):
        break

w1dot = np.zeros((len(w1[0]), len(w1[1])))
w2dot = np.zeros((len(w2[0]), len(w2[1])))
w1dot[0], w1dot[1] = w1[0], w1[1]
w2dot[0], w2dot[1] = w2[0], w2[1]

x = plt.plot(w1dot[0], w1dot[1], 'ro')
y = plt.plot(w2dot[0], w2dot[1], 'ro')
plt.setp(x, color = 'r')
plt.setp(y, color = 'g')
plt.axis([-8.0, 10.0, -8.0, 10.0])

rangeX = arange(-10.0, 10.0, 0.05)
rangeY = arange(-10.0, 10.0, 0.05)
X, Y = meshgrid(rangeX, rangeY)
equation = weightVector[0][0] * X + weightVector[0][1] * Y + weightVector[0][2]

plt.contour(X, Y, equation, [0], colors = 'black')

plt.show()