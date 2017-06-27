import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import meshgrid
from numpy.linalg import inv

trainFeature = [[[], []], [[], []], [[], []]]

f = open("C:\\Users\\Ryu\\Desktop\\Iris_train1.dat", "r")

lines = f.readlines()
 
for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    trainFeature[int(x[-1]) - 1][0].append(float(x[0]))
    trainFeature[int(x[-1]) - 1][1].append(float(x[1]))

f.close()

meanTotal = [[], [], []]

for i in range(3):
    for j in range(2):
        meanTotal[i].append(float(np.mean(trainFeature[i][j])))

covTotal = np.array(trainFeature)
covMatrix1 = np.zeros((2, 2))
covMatrix2 = np.zeros((2, 2))
covMatrix3 = np.zeros((2, 2))

covMatrix1 = np.cov([covTotal[0][0], covTotal[0][1]])
covMatrix2 = np.cov([covTotal[1][0], covTotal[1][1]])
covMatrix3 = np.cov([covTotal[2][0], covTotal[2][1]])

meanMatrix1 = np.zeros((1, 3))
meanMatrix2 = np.zeros((1, 3))
meanMatrix3 = np.zeros((1, 3))

meanMatrix1 = meanTotal[0]
meanMatrix2 = meanTotal[1]
meanMatrix3 = meanTotal[2]

meanMatrix1 = np.array(meanMatrix1)
meanMatrix2 = np.array(meanMatrix2)
meanMatrix3 = np.array(meanMatrix3)

Vi1 = -0.5 * inv(covMatrix1)
Vi2 = -0.5 * inv(covMatrix2)
Vi3 = -0.5 * inv(covMatrix3)

temp = np.dot(meanMatrix1.T, Vi1)
vi0_1 = np.dot(temp, meanMatrix1) - 0.5 * math.log(np.linalg.det(covMatrix1))

temp = np.dot(meanMatrix2.T, Vi2)
vi0_2 = np.dot(temp, meanMatrix2) - 0.5 * math.log(np.linalg.det(covMatrix2))

temp = np.dot(meanMatrix3.T, Vi3)
vi0_3 = np.dot(temp, meanMatrix3) - 0.5 * math.log(np.linalg.det(covMatrix3))

vi1 = np.dot(inv(covMatrix1), meanMatrix1)
vi2 = np.dot(inv(covMatrix2), meanMatrix2)
vi3 = np.dot(inv(covMatrix3), meanMatrix3)

trainX = np.zeros((len(trainFeature[0]), len(trainFeature[0][0])))
trainY = np.zeros((len(trainFeature[1]), len(trainFeature[1][0])))
trainZ = np.zeros((len(trainFeature[2]), len(trainFeature[2][0])))
trainX[0], trainX[1] = trainFeature[0][0], trainFeature[0][1]
trainY[0], trainY[1] = trainFeature[1][0], trainFeature[1][1]
trainZ[0], trainZ[1] = trainFeature[2][0], trainFeature[2][1]

"""
plot the training data
red is iris1
green is iris2
blue is iris3
"""
x = plt.plot(trainX[0], trainX[1], 'ro')
y = plt.plot(trainY[0], trainY[1], 'ro')
z = plt.plot(trainZ[0], trainZ[1], 'ro')
plt.setp(x, color = 'r')
plt.setp(y, color = 'g')
plt.setp(z, color = 'b')
plt.axis([4.0, 9, 0, 5.0])

meanDot = np.zeros((1, 6))
meanDot[0][0], meanDot[0][1], meanDot[0][2], meanDot[0][3], meanDot[0][4], meanDot[0][5] = meanTotal[0][0], meanTotal[0][1], meanTotal[1][0], meanTotal[1][1], meanTotal[2][0], meanTotal[2][1]

dot1 = plt.plot(meanDot[0][0], meanDot[0][1], 'ro')
dot2 = plt.plot(meanDot[0][2], meanDot[0][3], 'ro')
dot3 = plt.plot(meanDot[0][4], meanDot[0][5], 'ro')

# plot the mean
plt.setp(dot1, color = 'c')
plt.setp(dot2, color = 'c')
plt.setp(dot3, color = 'c')

temp1 = np.zeros((2, 2))
temp2 = np.zeros((2, 2))
temp3 = np.zeros((2, 2))
covRev1 = inv(covMatrix1)
covRev2 = inv(covMatrix2)
covRev3 = inv(covMatrix3)
temp1[0][0], temp1[0][1], temp1[1][0], temp1[1][1] = covRev1[0][0], covRev1[0][1], covRev1[1][0], covRev1[1][1]
temp2[0][0], temp2[0][1], temp2[1][0], temp2[1][1] = covRev2[0][0], covRev2[0][1], covRev2[1][0], covRev2[1][1]
temp3[0][0], temp3[0][1], temp3[1][0], temp3[1][1] = covRev3[0][0], covRev3[0][1], covRev3[1][0], covRev3[1][1]

rangeX = np.arange(3.0, 10.0, 0.05)
rangeY = np.arange(0.0, 5.0, 0.05)
X, Y = meshgrid(rangeX, rangeY)

"""
plot the contours for which the Mahalanobis distance = 2
red is iris 1
green is iris 2
blue is iris 3
"""
mahalanobis1 = np.sqrt((temp1[0][0] * (X - meanTotal[0][0]) + temp1[1][0] * (Y - meanTotal[0][1])) * (X - meanTotal[0][0]) + (temp1[0][1] * (X - meanTotal[0][0]) + temp1[1][1] * (Y - meanTotal[0][1])) * (Y - meanTotal[0][1])) - 2
plt.contour(X, Y, mahalanobis1, [0], colors = 'red')
mahalanobis2 = np.sqrt((temp2[0][0] * (X - meanTotal[1][0]) + temp2[1][0] * (Y - meanTotal[1][1])) * (X - meanTotal[1][0]) + (temp2[0][1] * (X - meanTotal[1][0]) + temp2[1][1] * (Y - meanTotal[1][1])) * (Y - meanTotal[1][1])) - 2
plt.contour(X, Y, mahalanobis2, [0], colors = 'green')
mahalanobis3 = np.sqrt((temp3[0][0] * (X - meanTotal[2][0]) + temp3[1][0] * (Y - meanTotal[2][1])) * (X - meanTotal[2][0]) + (temp3[0][1] * (X - meanTotal[2][0]) + temp3[1][1] * (Y - meanTotal[2][1])) * (Y - meanTotal[2][1])) - 2
plt.contour(X, Y, mahalanobis3, [0], colors = 'blue')

db12 = mahalanobis1 - mahalanobis2
db23 = mahalanobis2 - mahalanobis3
db31 = mahalanobis3 - mahalanobis1

"""
plot the decision boundaries
black is decision boundary 1, 2
yellow is decision boundary 2, 3
cyan is decision boundary 3, 1
"""
plt.contour(X, Y, db12, [0], colors = 'black')
plt.contour(X, Y, db23, [0], colors = 'yellow')
plt.contour(X, Y, db31, [0], colors = 'cyan')

testFeature = [[], [], []]

f = open("C:\\Users\\Ryu\\Desktop\\Iris_test1.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    testFeature[0].append(float(x[0]))
    testFeature[1].append(float(x[1]))
    testFeature[2].append(int(x[2]))

f.close()

testX = np.zeros((1, 2))

temp = np.zeros((len(testFeature) - 1, len(testFeature[0])))
temp = testFeature[0:-1]

tempM = np.array(temp)
testM = tempM.T

resultArr = np.zeros((3, 3))
correctP1 = [[], []]
correctP2 = [[], []]
correctP3 = [[], []]
missP1 = [[], []]
missP2 = [[], []]
missP3 = [[], []]

for i in range(len(testM)):
    testX = testM[i]

    g1 = np.dot(np.dot(testX.T, Vi1), testX) + np.dot(vi1.T, testX) + vi0_1
    g2 = np.dot(np.dot(testX.T, Vi2), testX) + np.dot(vi2.T, testX) + vi0_2
    g3 = np.dot(np.dot(testX.T, Vi3), testX) + np.dot(vi3.T, testX) + vi0_3

    list = []
    list.append(g1)
    list.append(g2)
    list.append(g3)

    maxG = max(list)

    if(float(maxG) == float(g1) and testFeature[2][i] == 1):
        resultArr[0][0] += 1
        correctP1[0].append(testX[0])
        correctP1[1].append(testX[1])

    elif(float(maxG) == float(g1) and testFeature[2][i] == 2):
        resultArr[0][1] += 1
        missP1[0].append(testX[0])
        missP1[1].append(testX[1])

    elif(float(maxG) == float(g1) and testFeature[2][i] == 3):
        resultArr[0][2] += 1
        missP1[0].append(testX[0])
        missP1[1].append(testX[1])

    elif(float(maxG) == float(g2) and testFeature[2][i] == 1):
        resultArr[1][0] += 1
        missP2[0].append(testX[0])
        missP2[1].append(testX[1])

    elif(float(maxG) == float(g2) and testFeature[2][i] == 2):
        resultArr[1][1] += 1
        correctP2[0].append(testX[0])
        correctP2[1].append(testX[1])

    elif(float(maxG) == float(g2) and testFeature[2][i] == 3):
        resultArr[1][2] += 1
        missP2[0].append(testX[0])
        missP2[1].append(testX[1])

    elif(float(maxG) == float(g3) and testFeature[2][i] == 1):
        resultArr[2][0] += 1
        missP3[0].append(testX[0])
        missP3[1].append(testX[1])

    elif(float(maxG) == float(g3) and testFeature[2][i] == 2):
        resultArr[2][1] += 1
        missP3[0].append(testX[0])
        missP3[1].append(testX[1])

    elif(float(maxG) == float(g3) and testFeature[2][i] == 3):
        resultArr[2][2] += 1
        correctP3[0].append(testX[0])
        correctP3[1].append(testX[1])

misclass1 = plt.plot(missP1[0], missP1[1], 'ro', color = 'y')
misclass2 = plt.plot(missP2[0], missP2[1], 'ro', color = 'y')
misclass3 = plt.plot(missP3[0], missP3[1], 'ro', color = 'y')

correctP1 = plt.plot(correctP1[0], correctP1[1], 'g^', color = 'r')
correctP2 = plt.plot(correctP2[0], correctP2[1], 'g^', color = 'g')
correctP3 = plt.plot(correctP3[0], correctP3[1], 'g^', color = 'b')

print("result")
print(resultArr.T)

plt.show()
