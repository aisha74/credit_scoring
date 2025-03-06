import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Загрузка данных
data = pd.read_csv(r'c:\Users\khais\Downloads\credit_scoring.csv')  

data.drop(['RealEstateLoansOrLines', 'GroupAge'], axis=1, inplace=True)

# Определите признаки (X) и целевую переменную (y)
X = data.drop(columns=['SeriousDlqin2yrs']) # Признаки
y = data['SeriousDlqin2yrs']  # Целевая переменная

# Обработка пропущенных значений (если есть)
X.fillna(0, inplace=True)  # Замените пропуски на 0. Вы можете выбрать другую стратегию.

# Разделите данные на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте и обучите модель логистической регрессии
model = LogisticRegression(max_iter=1000)  # Увеличение числа итераций для наилучшего сходимости
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("\nТочность модели:", accuracy)
print("Матрица путаницы:\n", confusion)

# Сохраните модель в файл
with open('logistic_regression_model.pickle', 'wb') as f:
    pickle.dump(model, f)

print("\nМодель успешно сохранена!")


