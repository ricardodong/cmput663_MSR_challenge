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

ans1 = open("token_train_ans.txt", 'r')
ans1Handled = open("token_train_ans_handled.txt", 'w')
ans2 = open("token_test_ans.txt", 'r')
ans2Handled = open("token_test_ans_handled.txt", 'w')
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

trainX = np.loadtxt("all_token_results_train.txt")
trainX = trainX[:,1:]
trainY = np.loadtxt("token_train_ans_handled.txt")

#print(trainX)
#print(trainY)

logRes = LogisticRegression()
logRes.fit(trainX, trainY)

testX = np.loadtxt("all_token_results_test.txt")
testX = testX[:,1:]
testY = np.loadtxt("token_test_ans_handled.txt")

#random result
import random

print("*****results for Random Baseline*****")
tag = [0, 1]
random_predict = []
for t_line in testX:
    ans = random.choice(tag)
    random_predict.append(ans)
predict_all = np.array(random_predict)

all_accuracy = accuracy_score(testY, predict_all)
all_precision = precision_score(testY, predict_all, average=None)
all_recall = recall_score(testY, predict_all, average=None)
all_f1 = f1_score(testY, predict_all, average=None)

print("Accuracy: %0.9f" % all_accuracy)

print("Fact Precision: %0.9f" % all_precision[0])
print("Fact Recall: %0.9f" % all_recall[0])
print("Fact F1: %0.9f" % all_f1[0])

print("Question Precision: %0.9f" % all_precision[1])
print("Question Recall: %0.9f" % all_recall[1])
print("Question F1: %0.9f" % all_f1[1])

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
'''
print("token level final: ")
sswitchs =  np.ones((10,10))
sswitchs[0] = [1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
sswitchs[1] = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
sswitchs[2] = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
sswitchs[3] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
sswitchs[4] = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
sswitchs[5] = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
sswitchs[6] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sswitchs[7] = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
sswitchs[8] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
sswitchs[9] = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
tswitchs = np.ones((11,10))
tswitchs[0] = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]
tswitchs[1] = [1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
tswitchs[2] = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
tswitchs[3] = [1, 1, 0, 0, 0, 1, 1, 0, 0, 1]
tswitchs[4] = [1, 1, 0, 0, 0, 0, 1, 0, 0, 1]
tswitchs[5] = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
tswitchs[6] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
tswitchs[7] = [1, 1, 0, 0, 0, 0, 1, 0, 1, 1]
tswitchs[8] = [1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
tswitchs[9] = [1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
tswitchs[10]= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

for i in range(11):
    print(str(i)+"'th best f1 result: ")
    newTrainX = np.ones((len(trainY), 1))
    newTestX = np.ones((len(testY), 1))
    for j in range(10):
        # tswitch should be changed to sswitch when change to sen level
        if tswitchs[i][j] == 1:
            newTrainX = np.c_[newTrainX, trainX[:, j]]
            newTestX = np.c_[newTestX, testX[:, j]]
    logRes = LogisticRegression(solver="lbfgs", max_iter=100000)
    logRes.fit(newTrainX, trainY)
    testRes = logRes.predict(newTestX)
    print(logRes.coef_)
    print("accuracy: " + str(accuracy_score(testY, testRes)))
    print("precision: " + str(precision_score(testY, testRes)))
    print("recall: " + str(recall_score(testY, testRes)))
    print("f1: " + str(f1_score(testY, testRes))+"\n")
