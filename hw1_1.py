import numpy as np
import math
from numpy.linalg import inv

trainFeature = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]

f = open("C:\\Users\\Ryu\\Desktop\\Iris_train.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    trainFeature[int(x[-1]) - 1][0].append(float(x[0]))
    trainFeature[int(x[-1]) - 1][1].append(float(x[1]))
    trainFeature[int(x[-1]) - 1][2].append(float(x[2]))
    trainFeature[int(x[-1]) - 1][3].append(float(x[3]))

f.close()

meanTotal = [[], [], []]

for i in range (3):
    for j in range(4):
        meanTotal[i].append(float(np.mean(trainFeature[i][j])))

covTotal = np.array(trainFeature)
covMatrix1 = np.zeros((4, 4))
covMatrix2 = np.zeros((4, 4))
covMatrix3 = np.zeros((4, 4))

covMatrix1 = np.cov([covTotal[0][0], covTotal[0][1], covTotal[0][2], covTotal[0][3]])
covMatrix2 = np.cov([covTotal[1][0], covTotal[1][1], covTotal[1][2], covTotal[1][3]])
covMatrix3 = np.cov([covTotal[2][0], covTotal[2][1], covTotal[2][2], covTotal[2][3]])
 
print("covMatrix1")
print(covMatrix1)
print("\n" + "covMatrix2")
print(covMatrix2)
print("\n" + "covMatrix3")
print(covMatrix3)

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

testFeature = [[], [], [], [], []]

f = open("C:\\Users\\Ryu\\Desktop\\Iris_test.dat", "r")

lines = f.readlines()

for line in lines:
    line = line.rstrip()
    x = line.split(' ')

    testFeature[0].append(float(x[0]))
    testFeature[1].append(float(x[1]))
    testFeature[2].append(float(x[2]))
    testFeature[3].append(float(x[3]))
    testFeature[4].append(int(x[4]))

f.close()
 
testX = np.zeros((1, 4))

temp = np.zeros((len(testFeature) - 1, len(testFeature[0])))
temp = testFeature[0:-1]

tempM = np.array(temp)
testM = tempM.T

resultArr = np.zeros((3, 3))

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

    if(float(maxG) == float(g1) and testFeature[4][i] == 1):
        resultArr[0][0] += 1

    elif(float(maxG) == float(g1) and testFeature[4][i] == 2):
        resultArr[0][1] += 1

    elif(float(maxG) == float(g1) and testFeature[4][i] == 3):
        resultArr[0][2] += 1

    elif(float(maxG) == float(g2) and testFeature[4][i] == 1):
        resultArr[1][0] += 1

    elif(float(maxG) == float(g2) and testFeature[4][i] == 2):
        resultArr[1][1] += 1

    elif(float(maxG) == float(g2) and testFeature[4][i] == 3):
        resultArr[1][2] += 1

    elif(float(maxG) == float(g3) and testFeature[4][i] == 1):
        resultArr[2][0] += 1

    elif(float(maxG) == float(g3) and testFeature[4][i] == 2):
        resultArr[2][1] += 1

    elif(float(maxG) == float(g3) and testFeature[4][i] == 3):
        resultArr[2][2] += 1

print("\n" + "result")
print(resultArr)
