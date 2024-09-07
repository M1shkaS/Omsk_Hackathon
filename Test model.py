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
    features = new_data.drop(["Failure_30days"], axis=1)
    # Загружаем модель
    model_path = os.path.join("D:/Хакатон/Omsk_Hackathon/ExtractedFeatures/", f"{model_name}30days.joblib")
    model = load(model_path)
    # Делаем предсказания
    predictions = model.predict(features)
    # Создаем новый датафрейм с предсказаниями
    predicted_df = pd.DataFrame(predictions, columns=["Failure_30days"])

    # Вычисляем метрики
    accuracy = accuracy_score(new_data[["Failure_30days"]], predicted_df)
    precision = precision_score(new_data[["Failure_30days"]], predicted_df,
                                average='macro')
    recall = recall_score(new_data[["Failure_30days"]], predicted_df,
                          average='macro')

    # Выводим результаты
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Возвращаем датафрейм с предсказаниями
    return predicted_df


# Путь к новому датасету
new_data_path = "D:/Хакатон/Omsk_Hackathon/Dataset/test_30days.csv"

# Вызов функции для RandomForestClassifier_model.joblib
print("\nRandomForestClassifier:")
predict_failures(new_data_path, "RandomForestClassifier_model_")
