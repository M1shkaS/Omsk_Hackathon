from datetime import datetime, timedelta
import pandas as pd
import os

os.chdir("D:/Хакатон/Hackathon/Dataset")

new_file = 'predictions.csv'
if os.path.isfile(new_file):
    os.remove(new_file)
    print(f"Файл '{new_file}' успешно удалён.")
else:
    print(f"Файл '{new_file}' не существует.")

def add_failure_predictions(file_name, prediction_horizons):
    # Загрузка данных
    df = pd.read_csv(file_name)

    # Создание столбцов с предсказаниями
    horizon_counts = {horizon: 0 for horizon in prediction_horizons}
    for horizon in prediction_horizons:
        # Создаем новый столбец для предсказания
        df[f'Failure_{horizon}days'] = False

        # Получаем имя файла с данными за i-й период
        date_string = "2023-12-31"
        # Преобразуем строку в объект даты
        date_object = datetime.strptime(date_string, "%Y-%m-%d")
        next_data_path = f"data_Q1_2024/{(date_object + timedelta(days=horizon)).strftime('%Y-%m-%d')}.csv"
        print(next_data_path)

        # Если файл с данными за i-й период существует
        if os.path.exists(next_data_path):
            next_df = pd.read_csv(next_data_path)
            # Обновляем столбец с предсказанием
            df.loc[df['serial_number'].isin(next_df[next_df['failure'] == True]['serial_number']) |
                   ~df['serial_number'].isin(next_df['serial_number']), f'Failure_{horizon}days'] = True
            horizon_counts[horizon] = df[f'Failure_{horizon}days'].sum()
    for horizon, count in horizon_counts.items():
        print(f"Количество значений True для {horizon} дней: {count}")
    return df

file_name = 'data_Q4_2023/2023-12-31.csv'
prediction_horizons = [7, 30, 90]  # Неделя, месяц, 3 месяца

# Добавляем предсказания в DataFrame
df = add_failure_predictions(file_name, prediction_horizons)

# Выводим результат
df.to_csv('predictions.csv', index=False, encoding='utf-8')