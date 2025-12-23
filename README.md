# Random-Forest-Data-Classification
I made this project to learn the basics of random forest algorithm

import pandas as pd
import sklearn
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "/content/bankloan.csv"

data = pd.read_csv(file_path)

print(data.head())

corr_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

X = data.drop(columns=["ID", "ZIP.Code", "Personal.Loan"])
y = data["Personal.Loan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50, stratify=y)

rf_model = RandomForestClassifier(criterion='gini',random_state=50, n_estimators=100)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

first_tree = rf_model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, filled=True)
plt.show()
print(f"Bölünme Ölçütü: {rf_model.criterion}")
print(f"Model Doğruluk Oranı: {accuracy:.2f}")
print("Model Performans Raporu:")
print(report)
