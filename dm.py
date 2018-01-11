import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

iris_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
resp = requests.get(iris_url)
print(resp.content)

data = BytesIO(resp.content)


print("=================================================")

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df = pd.read_csv(data)
print(df.iloc[0])

print("=================================================")

target = df[df.columns[-1]]
target = target.astype('category')
numeric_data = df._get_numeric_data()
print(numeric_data.head())

print("=================================================")

training_data,testing_data,training_label,testing_label = train_test_split(numeric_data,target.cat.codes)
print(training_data.head())

print("=================================================")

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(training_data,training_label)

print(tree_model)

print("=================================================")

predict_result = tree_model.predict(testing_data)
score_result = tree_model.predict_proba(testing_data)

print(predict_result[0:5])
print(score_result[0:5])

print("=================================================")

matrix = confusion_matrix(testing_label,predict_result)
report = classification_report(testing_label,predict_result,target_names=target.cat.categories)
acc = accuracy_score(testing_label,predict_result)

print(matrix)
print('-----')
print(report)
print('-----')
print(acc)

print("=================================================")




import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(16,9))
plot_confusion_matrix(matrix, classes=target.cat.categories)