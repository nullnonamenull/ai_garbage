import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# 1.1 Generowanie zbioru danych
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

# 1.2 Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1.3 Tworzenie klasyfikatora głosującego z trzema klasyfikatorami
lr_clf = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)
svc_clf = SVC(probability=False, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('lr', lr_clf),
    ('rf', rf_clf),
    ('svc', svc_clf)
], voting='hard')

# 1.4 Trenowanie klasyfikatora zespołowego
voting_clf.fit(X_train, y_train)

# 1.5 Ocena dokładności poszczególnych klasyfikatorów
for name, clf in voting_clf.named_estimators_.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'{name} accuracy: {accuracy_score(y_test, y_pred):.2f}')

# 1.6 Prognozowanie dla pierwszego przykładu ze zbioru testowego
first_example = X_test[0].reshape(1, -1)
print("\nPredictions for the first test example:")
for name, clf in voting_clf.named_estimators_.items():
    prediction = clf.predict(first_example)
    print(f'{name} prediction: {prediction[0]}')

voting_prediction = voting_clf.predict(first_example)
print(f'VotingClassifier prediction: {voting_prediction[0]}')

# 1.7 Ocena dokładności klasyfikatora zespołowego
y_pred_voting = voting_clf.predict(X_test)
print(f'VotingClassifier accuracy: {accuracy_score(y_test, y_pred_voting):.2f}')

# 1.8 Zmiana trybu głosowania na "soft voting"
svc_clf = SVC(probability=True, random_state=42)

voting_clf_soft = VotingClassifier(estimators=[
    ('lr', lr_clf),
    ('rf', rf_clf),
    ('svc', svc_clf)
], voting='soft')

voting_clf_soft.fit(X_train, y_train)

# Ocena dokładności klasyfikatora zespołowego przy "soft voting"
y_pred_voting_soft = voting_clf_soft.predict(X_test)
print(f'Soft VotingClassifier accuracy: {accuracy_score(y_test, y_pred_voting_soft):.2f}')
