import pandas as pd

def crop_dataset(dataset_path, start_index=890, num_objects=40):
  try:
    df = pd.read_csv(dataset_path)
    cropped_df = df[start_index : start_index + num_objects]
    cropped_df.to_csv(dataset_path + "_40", index=False)
    print(f"Датасет обрезан, начиная с объекта {start_index} и взяв {num_objects} объектов, "
          f"и сохранен в {dataset_path}.")
  except FileNotFoundError:
    print(f"Файл {dataset_path} не найден.")
  except Exception as e:
    print(f"Ошибка: {e}")

# Пример использования:
dataset_path = "D:/Хакатон/Omsk_Hackathon/Dataset/test_90days.csv"
crop_dataset(dataset_path)
