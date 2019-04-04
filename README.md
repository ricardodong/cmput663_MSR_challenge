# cmput663_MSR_challenge
Paper for MSR 2019 mining challenge by Shutong Li and Yixing Luan's group
(not used anymore)

# ensemble learning for project
[best parameters.txt](https://github.com/ricardodong/cmput663_MSR_challenge/blob/master/best%20parameters.txt) show the result of searching for parameter of using ngram, cross validation is used. the CV code is:
```
np.mean(cross_val_score(logRes, newTestX, testY, cv=10, scoring='precision_macro'))
```

[KerasLRTest.py](https://github.com/ricardodong/cmput663_MSR_challenge/blob/master/KerasLRTest.py) is the code to get the result of finding best parameter for our method, it does not have a main, but you can directly run it.
