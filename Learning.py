import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from decimal import Decimal, ROUND_HALF_UP
from joblib import dump

colors_below_500 = ['#C8E6C9']
colors_above_500 = ['#FFCDD2']

# Установим текущую директорию
os.chdir("D:/Хакатон/Omsk_Hackathon")
current_dir = os.getcwd()
files = os.listdir(current_dir + "/ExtractedFeatures")

# Удаляем предыдущие графики
#for file in files:
#    os.remove(os.path.join(current_dir + "/ExtractedFeatures", file))

# Загрузка нового датасета
new_data = pd.read_csv("Dataset/predictions_30days.csv")

# Подготовка данных для модели
features = new_data.drop("Failure_30days", axis=1)
target = new_data["Failure_30days"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Создание моделей
rf_model = RandomForestClassifier()

models = [rf_model]
results = []

# Обучение моделей и оценка результатов
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cf_matrix = confusion_matrix(y_test, y_pred)
    mask_above_1000 = cf_matrix > 400
    mask_below_1000 = cf_matrix <= 400

    # Создаем subplot так, чтобы слева был график, а справа - место для текста
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.3})
    plt.subplots_adjust(left=0.1, bottom=0.15)  # пустое пространство по бокам фигуры

    # Строим тепловую карту на левой подобласти
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap=ListedColormap(colors_below_500), cbar=False,
                linecolor='white', linewidth=1, annot_kws={"size": 36}, square=True, ax=ax1, mask=mask_below_1000)
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap=ListedColormap(colors_above_500), cbar=False,
                linecolor='white', linewidth=1, annot_kws={"size": 36}, square=True, ax=ax1, mask=mask_above_1000)
    ax1.set_xlabel('Предск.', fontsize=28)
    ax1.set_xticklabels(["Работа", "Сбой"], fontsize=12)
    ax1.tick_params(axis='x', labelsize=28)
    ax1.set_ylabel('Действ.', fontsize=28, rotation=0, labelpad=50)
    ax1.set_yticklabels(["Работа", "Сбой"], fontsize=12)
    ax1.tick_params(axis='y', labelsize=28)
    ax1.set_title(f'Предсказание на месяц', fontsize=28)

    # Добавляем текст на правой подобласти
    text = (
        "Accuracy:  {0:>6}\n"
        "Precision:  {1:>6}\n"
        "Recall:       {2:>6}"
    ).format(f'{Decimal(str(accuracy * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%',
             f'{Decimal(str(precision * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%',
             f'{Decimal(str(recall * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%')
    ax2.text(-0.5, 0.5, text, fontsize=32, ha='left', va='center', wrap=True, linespacing=2.0)
    ax2.axis('off')  # Убираем оси для текстовой области

    plt.savefig(f'ExtractedFeatures/{type(model).__name__}_confusion_matrix_30days.png')
    plt.close()

    model_name = type(model).__name__
    tn, fp, fn, tp = cf_matrix.ravel()
    results.append((model_name, accuracy, precision, recall, f1, tn, fp, fn, tp))
    dump(model, f'ExtractedFeatures/{type(model).__name__}_model_30days.joblib')

results.sort(key=lambda x: x[1], reverse=True)

# Вывод результатов

for model_name, accuracy, precision, recall, f1, tn, fp, fn, tp in results[1:]:
    print(f"Model: {model_name}")
    print(f"True Negatives: {tn}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("----------------------")