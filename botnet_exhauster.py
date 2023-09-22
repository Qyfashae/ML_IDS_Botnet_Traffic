import pickle 
from sklearn.tree import *

file = open("BT11flowdata.pickle", "rb")
botnet_dataset = pickle.load(file)

x_train, y_train, x_test, y_test = (
	botnet_dataset[0],
	botnet_dataset[1],
	botnet_dataset[3],
	botnet_dataset[4],
	)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
