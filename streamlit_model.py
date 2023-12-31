import streamlit as st
import pandas as pd
from PIL import Image
import pickle

file_model = 'model.pkl'
model = pickle.load(open(file_model, 'rb'))

st.title("Классификация пассажиров титаника")
image = Image.open('titanic.jpg')
st.image(image)
st.sidebar.header("Ваши данные:")
sex = st.sidebar.selectbox("Пол", 
                   ('Мужской', 'Женский'))
embarked = st.sidebar.selectbox("Порт посадки", 
                                ("Шербур-Октевиль", "Квинстаун", "Саутгемптон"))
pclass = st.sidebar.selectbox("Класс", 
                              ("Первый", "Второй", "Третий"))
age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=1, step=1)
sibsp = st.sidebar.slider("Количество ваших братьев / сестер / супругов на борту",
        min_value=0, max_value=8, value=0, step=1)
parch = st.sidebar.slider("Количество ваших детей / родителей на борту",
                               min_value=0, max_value=6, value=0, step=1)

if sex == "Мужской": sex = 1 
else: sex = 0

if embarked == "Шербур-Октевиль": embarked = 0
elif embarked == "Квинстаун": embarked = 1
else: embarked = 2

if pclass == "Первый": pclass = 1
elif pclass == "Второй": pclass = 2
else: pclass = 3

data = {"Pclass" : pclass,
        "Sex" : sex,
        "Age" : age,
        "SibSp" : sibsp,
        "Parch": parch,
        "Embarked": embarked}

df = pd.DataFrame(data, index=[0])
pred = model.predict(df)
pred_prob = model.predict_proba(df)

if pred[0] == 1:
    st.markdown("### :red[Ура! Вы выжили!] :grinning::tada:")
    st.write(f"Вероятность: {round(pred_prob[0][1]*100, 1)}%")
else:
    st.markdown("### К сожалению, Вам не повезло :pensive:")
    st.write(f"Вероятность: {round(pred_prob[0][0]*100, 1)}%")