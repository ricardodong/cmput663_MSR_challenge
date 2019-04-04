import numpy as np

from keras.models import Sequential
from keras.optimizers import SGD, adam, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten

# 样本数据集，两个特征列，两个分类二分类不需要onehot编码，直接将类别转换为0和1，分别代表正样本的概率。
#trainXfile = open("all_sent_result_log.txt")
#trainYfile = open("try_train_ans.txt")
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
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
#trainX = np.hstack((trainX[:,1:4],trainX[:,6:9]))
trainX = trainX[:,1:]
trainY = np.loadtxt("token_train_ans_handled.txt")
'''
for i in range(250):
    trainLine = trainXfile.readline()
    if not trainLine:
        break
    trainLine = trainLine.split('\t')[1]
    trainLine = trainLine.split()
    for j in range(10):
        trainX[i,j] = float(trainLine[j])
'''

for i in range(1000):
    trainY[i] = int(trainY[i])

#print(trainX)
#print(trainY)

logRes = MLPClassifier()
logRes.fit(trainX, trainY)


testX = np.loadtxt("all_token_results_test.txt")
testX = testX[:,1:]
testY = np.loadtxt("token_test_ans_handled.txt")
count = 0
Length = len(testY)
print('\nTesting ------------')
testRes = logRes.predict(testX)
print(precision_score(testY, testRes))
'''
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
    print(switch)
    length = 0
    newTrainX = np.ones((len(trainY), 1))
    newTestX = np.ones((len(testY), 1))
    for j in range(10):
        if switch[j] == 1:
            length = length + 1
            newTrainX = np.c_[newTrainX, trainX[:,j]]
            newTestX = np.c_[newTestX, testX[:,j]]
    print(newTrainX.shape[1])

    logRes = LogisticRegression()
    logRes.fit(newTrainX, trainY)
    #testRes = logRes.predict(newTestX)
    #score = precision_score(testY, testRes)
    scores[i] = np.mean(cross_val_score(logRes, newTestX, testY, cv=10, scoring='precision_macro'))
    switchs[i] = switch
    #print(score)

print("best cases")
maxposi = scores.argsort()
for i in range(10):
    print(switchs[maxposi[1023-i]])
    print(scores[maxposi[1023-i]])


'''
#建立假的测试数据集
X,y = datasets.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,n_repeated=0, n_classes=2, n_clusters_per_class=1)
print(X)
print(y)
'''
'''
# 构建神经网络模型
model = Sequential()
model.add(Dense(input_dim=10, units=1))
model.add(Activation('sigmoid'))
opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#opt = SGD(lr = 0.005)
#opt = adam(lr = 0.001)

# 选定loss函数和优化器
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

# 训练过程

print('Training -----------')
for step in range(20000):
    cost = model.train_on_batch(trainX, trainY)
    if step % 50 == 0:
        print(step)
        print(cost)
    if cost[1]>0.82:
        break

# 测试过程
testX = np.loadtxt("all_sent_test_result_log.txt")
testX = testX[:,1:]
testY = np.loadtxt("try_test_ans.txt")
print('\nTesting ------------')
cost = model.evaluate(testX, testY, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


result = model.predict(testX, batch_size=40)
print(result)
'''


'''
# 将训练结果绘出
Y_pred = model.predict(X)
Y_pred = (Y_pred*2).astype('int')  # 将概率转化为类标号，概率在0-0.5时，转为0，概率在0.5-1时转为1
# 绘制散点图 参数：x横轴 y纵轴
plt.subplot(2,1,1).scatter(X[:,0], X[:,1], c=Y_pred)
plt.subplot(2,1,2).scatter(X[:,0], X[:,1], c=y)
plt.show()
'''