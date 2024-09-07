import os
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score

def predict_failures(new_data_path, model_name):
    """
    Предсказывает значения "Failure_*days" для нового датасета.
    """
    # Загружаем новый датасет
    new_data = pd.read_csv(new_data_path)
    # Дропаем целевые переменные
    features = new_data.drop(["Failure_90days"], axis=1)

    model_path1 = os.path.join("D:/Хакатон/Omsk_Hackathon/ExtractedFeatures/", f"{model_name}30days.joblib")
    model1 = load(model_path1)
    # Делаем предсказания
    predictions1 = model1.predict(features)
    # Создаем новый датафрейм с предсказаниями
    predicted_df1 = pd.DataFrame(predictions1, columns=["Failure_30days"])

    # Выводим список объектов с Failure_30days = True
    print(f"\nИдентификаторы дисков, которые вероятно выйдут из строя в течение месяца")
    print(predicted_df1[predicted_df1["Failure_30days"] == True].index.tolist())

    # Загружаем модель
    model_path = os.path.join("D:/Хакатон/Omsk_Hackathon/ExtractedFeatures/", f"{model_name}90days.joblib")
    model = load(model_path)
    # Делаем предсказания
    predictions = model.predict(features)
    # Создаем новый датафрейм с предсказаниями
    predicted_df = pd.DataFrame(predictions, columns=["Failure_90days"])

    # Выводим список объектов с Failure_90days = True
    print(f"\nИдентификаторы дисков, которые вероятно выйдут из строя в течение 3-х месяцев:")
    print(predicted_df[predicted_df["Failure_90days"] == True].index.tolist())

    # Возвращаем датафрейм с предсказаниями
    return predicted_df

while True:
    # Выводим меню
    print("\nМеню:")
    print("0 - Выход")
    print("1 - Загрузить набор дисков для проверки")
    choice = input("Выберите действие: ")

    if choice == "0":
        break
    elif choice == "1":
        # Получаем путь к датасету от пользователя
        new_data_path = input("Введите путь к CSV-файлу: ")

        # Вызываем функцию для RandomForestClassifier_model.joblib
        predict_failures(new_data_path, "RandomForestClassifier_model_")

    else:
        print("Неверный выбор.")