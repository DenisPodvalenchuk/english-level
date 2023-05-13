import streamlit as st
from catboost import CatBoostClassifier, Pool
import pandas as pd
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pysrt
import nltk
nltk.download('wordnet')

st.title('Классификация фильмов по Common European Framework of Reference (CEFR) - системе уровней владения иностранным языком, используемой в Европейском Союз')
st.text('В системе CEFR знания и умения учащихся подразделяются на три крупных категории,')
st.text('которые далее делятся на шесть уровней:')
st.text('A1 - Уровень Beginner (Начальный)')
st.text('A2 - Уровень Elementary (Базовый)')
st.text('B1 - Уровень Pre-Intermediate (Средний)')
st.text('B2 - Уровень Upper-Intermediate (Выше среднего)')
st.text('C1 - Уровень Advanced (Продвинутый)')
st.text('C2 - Уровень Proficiency (Владение в совершенстве)')

df_words = pd.read_csv('app/Oxford_dikt.csv')

# Загрузка модели
model = CatBoostClassifier()
model.load_model('app/english_level_model.cbm')
features = ['film_start', 'film_end', 'film_length', 'num_sentence', 'text_len', 'words_unique_count', 'A1', 'A2', 'B1', 'B2', 'C1']

wnl = WordNetLemmatizer()

def sub_processing(file, df_words):
    """Принимает объект файла субтитров и возвращает список параметров, расчитанных на основе обработки файла."""
    try:
        subs = pysrt.from_string(file.read().decode('latin-1', 'ignore'))
    except UnicodeDecodeError:
        return None
    if len(subs) < 2:
        return None
    #subs = pysrt.from_string(file.read().decode('latin-1'))
    film_start, film_end, film_length = film_times(subs) # 1,2,3
    text = clear_text(subs)
    num_sentence = count_sentences(text)  # 4
    text_len = process_text(text) # 5
    words_unique = get_unique_words(text)  # предполагая, что get_unique_words возвращает список уникальных слов
    words_unique_count = len(words_unique)    # 6
    difficulty_count = calculate_difficulty(text)
    return [film_start, # 1 
            film_end,   # 2
            film_length, # 3
            num_sentence,  # 4
            text_len,   # 5
            words_unique_count, # 6
            (difficulty_count['A1'] / words_unique_count),  # 7
            (difficulty_count['A2'] / words_unique_count),  # 8
            (difficulty_count['B1'] / words_unique_count),  # 9
            (difficulty_count['B2'] / words_unique_count),  # 10
            (difficulty_count['C1'] / words_unique_count)]  # 11 

def make_predict(data, model):
    predict_pool = Pool(data=data)
    predict = model.predict(predict_pool)
    decode = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1'}
    return decode[int(predict[0])]

def process_text(text):
    """Очищает текст, удаляет стоп-слова и проводит лемматизацию."""
    # Удаляем всё что не буквы
    text = re.sub('[^a-z]', ' ', text)
    
    # Удаляем однобуквенные слова
    text = re.sub(' [a-z] ', ' ', text)
        
    # Лемматизация
    text = ' '.join([wnl.lemmatize(word, wordnet.VERB) for word in text.split(' ')])
    text_len = len(text)
    return text_len # Количество символов

def calculate_difficulty(text):
    """
    Расчитывает количество слов разной сложности в тексте.
    """
    difficulty_count = {'A1': 0, 'A2': 0, 'B1': 0, 'B2': 0, 'C1': 0}
    
    words = text.split(' ')
    words_unique = set(words)  # Создаем множество уникальных слов и исключить повторяющиеся слова
    
    for word in words_unique:
        match = df_words[df_words['word'] == word]['diff'].values
        if len(match) > 0:
            for dif in match:
                difficulty_count[dif] += 1
                
    return difficulty_count

def get_unique_words(text):
    """Принимает текст и возвращает список уникальных слов."""
    words = text.split(' ')
    words_unique = set(words)
    return list(words_unique)

def clear_text(subs):
    """Очищает текст, удаляя лишние символы и приводит к нижнему регистру."""
    text = re.sub('\<.*?\>', '', subs.text)      # удаляем то что в скобках <>
    text = re.sub('\n', ' ', text)               # удаляем разделители строк    
    text = re.sub('\(.*?\)', '', text)           # удаляем то что в скобках ()    
    text = re.sub('\[.*?\]', '', text)           # удаляем то что в скобках []
    text = re.sub('[A-Z]+?:', '', text)          # удаляем слова написанные заглавными буквами с двоеточием(это имена тех кто говорит)
    text = re.sub('\.+?:', '\.', text)           # Заменяем троеточия на одну точку
    text = text.lower()
    text = re.sub('[^a-z\.\!\?]', ' ', text)     # удаляем всё что не буквы и не .?!
    text = re.sub(' +', ' ', text)               # удаляем " +"
    return text

def film_times(subs):
    """Вычисляет время начала и окончания фильма, а также его продолжительность."""
    # Время начала фильма
    film_start = subs[0].start.hours*3600 + subs[0].start.minutes*60 + subs[0].start.seconds
    # Время окончания фильма
    if subs[-1].index - subs[-2].index < 2:
        film_end = subs[-1].end.hours*3600 + subs[-1].end.minutes*60 + subs[-1].end.seconds
    else:
        film_end = subs[-2].end.hours*3600 + subs[-2].end.minutes*60 + subs[-2].end.seconds
    # Продолжительность фильма
    film_start, film_end = min(film_start, film_end), max(film_start, film_end)
    film_length = film_end - film_start
    return film_start, film_end, film_length

def count_sentences(text):
    """Считает количество предложений в тексте."""
    num_sentence = len(re.split('[\.\?\!]', text))
    return num_sentence

uploaded_file = st.file_uploader('Откройте файл с субтитрами. Формат файла должен быть .srt', type=['srt'])

if uploaded_file is not None:
    df = sub_processing(uploaded_file, df_words)
    if df is None:
        st.write('Файл субтитров имеет неизвестный формат')
    else:
        #st.header(f'Данный фильм имеет уровень **{make_predict(df[features], model)}** :sunglasses: по классификации CEFR')
        # Создаем новый датафрейм с одной строкой, содержащей результаты обработки субтитров
        df_result = pd.DataFrame([df], columns=features)
        # Получаем предсказание с использованием нового датафрейма
        prediction = make_predict(df_result, model)
        st.header(f'Данный фильм имеет уровень **{prediction}** :sunglasses: по классификации CEFR')
        # Выводим значения признаков для анализа
        st.write("Значения признаков:")
        st.write(df_result)
