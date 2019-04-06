import numpy as np

from keras.models import Sequential
from keras.optimizers import SGD, adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten

from sklearn import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

ans1 = open("sentence_train_ans.txt", 'r')
ans1Handled = open("sentence_train_ans_handled.txt", 'w')
ans2 = open("sentence_test_ans.txt", 'r')
ans2Handled = open("sentence_test_ans_handled.txt", 'w')
while True:
    line = ans1.readline()
    if not line:
        break;
    if "c" in line:
        ans1Handled.write("1\n")
    else:
        ans1Handled.write("0\n")
while True:
    line = ans2.readline()
    if not line:
        break;
    if "c" in line:
        ans2Handled.write("1\n")
    else:
        ans2Handled.write("0\n")
ans1.close()
ans2.close()
ans1Handled.close()
ans2Handled.close()

trainX = np.loadtxt("all_sentence_results_train.txt")
trainX = trainX[:,1:]
trainY = np.loadtxt("sentence_train_ans_handled.txt")

#print(trainX)
#print(trainY)

logRes = LogisticRegression()
logRes.fit(trainX, trainY)

testX = np.loadtxt("all_sentence_results_test.txt")
testX = testX[:,1:]
testY = np.loadtxt("sentence_test_ans_handled.txt")
print('\nTesting ------------')
testRes = logRes.predict(testX)
print("accuracy: " + str(accuracy_score(testY, testRes)))
print("precision: " + str(precision_score(testY, testRes)))
print("recall: " + str(recall_score(testY, testRes)))
print("f1: " + str(f1_score(testY, testRes)))
'''
count = 0
Length = len(testY)
for i in range(Length):
    if logRes.predict(testX)[i] != testY[i]:  # 预测测试样本
        count += 1
        print(testY[i])

print(count)
'''

scores = np.zeros(1024)
switchs = np.zeros((1024, 10))
for i in range(1024):
    icopy = i
    switch = np.zeros((10,))
    for j in range(10):
        (icopy, switch[j]) = np.divmod(icopy,2)
    length = 0
    newTrainX = np.ones((len(trainY), 1))
    newTestX = np.ones((len(testY), 1))
    for j in range(10):
        if switch[j] == 1:
            length = length + 1
            newTrainX = np.c_[newTrainX, trainX[:,j]]
            newTestX = np.c_[newTestX, testX[:,j]]

    logRes = LogisticRegression()
    #logRes.fit(newTrainX, trainY)
    #testRes = logRes.predict(newTestX)
    #score = precision_score(testY, testRes)
    scores[i] = np.mean(cross_val_score(logRes, newTrainX, trainY, cv=10, scoring='f1_macro'))
    switchs[i] = switch
    #print(switch)
    #print(newTrainX.shape[1])
    #print(score)

print("best cases")
maxposi = scores.argsort()
for i in range(10):
    print(switchs[maxposi[1023-i]])
    print(scores[maxposi[1023-i]])

print("sentence level final: ")
sswitch = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
tswitch = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
newTrainX = np.ones((len(trainY), 1))
newTestX = np.ones((len(testY), 1))
for j in range(10):
    if sswitch[j] == 1:
        newTrainX = np.c_[newTrainX, trainX[:, j]]
        newTestX = np.c_[newTestX, testX[:, j]]
logRes = LogisticRegression()
logRes.fit(newTrainX, trainY)
testRes = logRes.predict(newTestX)
print("accuracy: " + str(accuracy_score(testY, testRes)))
print("precision: " + str(precision_score(testY, testRes)))
print("recall: " + str(recall_score(testY, testRes)))
print("f1: " + str(f1_score(testY, testRes)))
