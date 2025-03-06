import streamlit as st
import pickle
import numpy as np
from PIL import Image
# Загрузка изображения
file_path = r'C:\Users\khais\Downloads\Credit_score.jpg' 

try:
    img = Image.open(file_path)
    st.image(img, caption='Credit Score Image', use_column_width=True)
except FileNotFoundError:
    st.error("Файл не найден! Проверьте путь к файлу.")
# Загрузка модели
@st.cache_resource
def load_model():
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

# Загрузите модель
model = load_model()

# Функция для предсказания
def predict_credit_score(features):
    prediction = model.predict([features])
    return prediction[0]

# Интерфейс Streamlit
st.title("Приложение по кредитному скорингу")

# Поля для ввода данных
age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
DebtRatio = st.number_input("Ежемесячные расходы", min_value=0, value=50000)
loan_amount = st.number_input("Сумма кредита (в валюте)", min_value=0, value=10000)
credit_history = st.selectbox("Кредитная история", options=[0, 1], format_func=lambda x: 'Плохая' if x == 0 else 'Хорошая')
MonthlyIncome = st.number_input("Месячный доход",  min_value=0, value=50000)
NumberOfDependents = st.number_input("Количество иждивенцев", min_value=0, value=1)
NumberOfOpenCreditLinesAndLoans = st.number_input("Количество открытых кредитов",min_value=0, value=1)
NumberOfTimes90DaysLate = st.number_input("Сколько раз наблюдалась просрочка", min_value=0, value=1)
sex =st.selectbox ("Пол", options=[0, 1], format_func=lambda x: 'Мужской' if x == 0 else 'Женский' )
# Когда нажата кнопка "Предсказать"
if st.button("Получить кредитный скоринг"):
    features = [age, DebtRatio, loan_amount, credit_history, MonthlyIncome, NumberOfDependents,NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, sex]
    prediction = predict_credit_score(features)

    if prediction == 1:
        st.success("Кредит одобрен!")
    else:
        st.error("Кредит не одобрен.")
