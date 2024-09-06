import pandas as pd
import os

os.chdir("D:/Хакатон/Omsk_Hackathon/Dataset")

# Загрузка датасета
data = pd.read_csv('predictions.csv')

# Получение уникальных моделей жестких дисков
unique_models = data['model'].unique()

# Создание словаря с уникальными моделями
encoding_mapping = {model: index for index, model in enumerate(unique_models)}

# Вывод результата
print(encoding_mapping)