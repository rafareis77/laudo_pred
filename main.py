import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_sql('SELECT * FROM laudos_medicos',
               )

df['tipo_de_exame'] = LabelEncoder().fit_transform(df['tipo_de_exame'])

X_train, X_test, y_train, y_test = train_test_split(df.drop('tipo_de_exame', axis=1),
                                                    df['tipo_de_exame'], test_size=0.25)

clf = MLPClassifier(hidden_layer_sizes=(100, 50),
                    activation='relu', solver='adam', max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'Precisao : {accuracy_score(y_test, y_pred):.2f}')

joblib.dump(clf, 'model.joblib')

clf = joblib.load('model.joblib')

input1 = input('Digite aqui: ')
y_pred = clf.predict([input1])
print(f'O tipo de exame é: {y_pred[0]}')