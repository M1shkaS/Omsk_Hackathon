import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from decimal import Decimal, ROUND_HALF_UP
from joblib import dump

light_grey = np.array([230 / 255, 230 / 255, 230 / 255, 1])
light_grey_colormap = ListedColormap([light_grey])
colors_below_500 = ['#C8E6C9']
colors_above_500 = ['#FFCDD2']

os.chdir("D:/Хакатон/Omsk_Hackathon")
current_dir = os.getcwd()
files = os.listdir(current_dir + "/ExtractedFeatures")
# удаляем предыдущий результат
for file in files:
    if file.endswith(".png"):
        os.remove(os.path.join(current_dir + "/ExtractedFeatures", file))

# Загружаем данные
data = pd.read_csv("Dataset/predictions.csv")
# Подготовка данных для модели
features = data.drop(["Failure_7days", "Failure_30days", "Failure_90days"], axis=1)
targets = ["Failure_7days", "Failure_30days", "Failure_90days"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, data[targets], test_size=0.3, random_state=42)

# Создание моделей
rf_model = RandomForestClassifier()
#lr_model = LogisticRegression(max_iter=1000)
#knn_model = KNeighborsClassifier()
#nb_model = GaussianNB()
dt_model = DecisionTreeClassifier()

#models = [rf_model, lr_model, knn_model, nb_model, dt_model]
models = [rf_model, dt_model]
results = []

# Обучение моделей и оценка результатов
for model in models:
    model.fit(X_train, y_train)  # Обучаем модель сразу по всем таргетам
    y_pred = model.predict(X_test)

    # Оцениваем модель по всем таргетам одновременно
    for i, target_name in enumerate(targets):
        y_test_target = y_test[target_name]
        y_pred_target = y_pred[:, i]  # Извлекаем предсказания для текущего таргета

        accuracy = accuracy_score(y_test_target, y_pred_target)
        precision = precision_score(y_test_target, y_pred_target)
        recall = recall_score(y_test_target, y_pred_target)
        f1 = f1_score(y_test_target, y_pred_target)

        # График матрицы ошибок
        cf_matrix = confusion_matrix(y_test_target, y_pred_target)
        mask_above_1000 = cf_matrix > 1000
        mask_below_1000 = cf_matrix <= 1000

        # Создаем subplot так, чтобы слева был график, а справа - место для текста
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.3})
        plt.subplots_adjust(left=0.1, bottom=0.15)  # пустое пространство по бокам фигуры

        # Строим тепловую карту на левой подобласти
        sns.heatmap(cf_matrix, annot=True, fmt='d', cmap=ListedColormap(colors_below_500), cbar=False,
                    linecolor='white', linewidth=1, annot_kws={"size": 36}, square=True, ax=ax1, mask=mask_below_1000)
        sns.heatmap(cf_matrix, annot=True, fmt='d', cmap=ListedColormap(colors_above_500), cbar=False,
                    linecolor='white', linewidth=1, annot_kws={"size": 36}, square=True, ax=ax1, mask=mask_above_1000)
        ax1.set_xlabel('Predicted', fontsize=28)
        ax1.set_xticklabels(["Несбой", "Сбой"], fontsize=12)
        ax1.tick_params(axis='x', labelsize=28)
        ax1.set_ylabel('Actual', fontsize=28, rotation=0, labelpad=50)
        ax1.set_yticklabels(["Несбой", "Сбой"], fontsize=12)
        ax1.tick_params(axis='y', labelsize=28)
        ax1.set_title(f'{type(model).__name__} ({target_name})', fontsize=28)

        # Добавляем текст на правой подобласти
        text = (
            "Accuracy:  {0:>6}n"
            "Precision:  {1:>6}n"
            "Recall:       {2:>6}"
        ).format(f'{Decimal(str(accuracy * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%',
                 f'{Decimal(str(precision * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%',
                 f'{Decimal(str(recall * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)}%')
        ax2.text(-0.5, 0.5, text, fontsize=32, ha='left', va='center', wrap=True, linespacing=2.0)
        ax2.axis('off')  # Убираем оси для текстовой области

        plt.savefig(f'ExtractedFeatures/{type(model).__name__}_{target_name}_confusion_matrix.png')
        plt.close()

        model_name = type(model).__name__
        tn, fp, fn, tp = cf_matrix.ravel()
        results.append((model_name, target_name, accuracy, precision, recall, f1, tn, fp, fn, tp))

    # Сохранение модели в файл
    dump(model, f'ExtractedFeatures/{type(model).__name__}_model.joblib')

    # Вывод результатов
    for model_name, target_name, accuracy, precision, recall, f1, tn, fp, fn, tp in results:
        print(f"Model: {model_name} ({target_name})")
        print(f"True Negatives: {tn}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print("----------------------")