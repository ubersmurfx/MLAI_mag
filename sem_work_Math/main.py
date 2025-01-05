import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
import config
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

def load_image_dataset(data_dir, img_size=(config.image_size, config.image_size)):
    """
    Загружает датасет изображений из директории, организованной по папкам.

    Args:
        data_dir: Путь к директории с данными (dataset).
        img_size: Размер, до которого будут изменены изображения.

    Returns:
        Кортеж (images, labels, class_names), где images - массив NumPy с изображениями, 
        labels - массив NumPy с метками классов (целочисленными),
        class_names - список названий классов (строки).  Возвращает None, если возникла ошибка.
    """
    images = []
    labels = []
    class_names = []

    try:
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                for filename in tqdm(os.listdir(class_dir), desc=f"Загрузка класса {class_name}"):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        img = cv2.imread(filepath)
                        if img is not None:
                            img = cv2.resize(img, img_size)
                            images.append(img)
                            labels.append(class_name)
                    except Exception as e:
                        print(f"Ошибка при загрузке изображения {filepath}: {e}")

        # Преобразование меток в числовые значения (one-hot encoding не нужен в данном случае, если используем sparse_categorical_crossentropy)
        label_encoder = {class_name: i for i, class_name in enumerate(class_names)}
        numerical_labels = [label_encoder[label] for label in labels]

        return np.array(images), np.array(numerical_labels), class_names

    except FileNotFoundError:
        print(f"Ошибка: Директория {data_dir} не найдена.")
        return None
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")
        return None

if __name__=="__main__":
    debug = False
    result = load_image_dataset(config.data_dir)
    if result:
        images, labels, class_names = result

        if debug:    
            print(f"Загружено {len(images)} изображений.")
            print(f"Классы: {class_names}")
            print(f"Количество классов: {len(class_names)}")
            print(f"Размер массива images: {images.shape}")
            class_index = np.where(np.array(class_names) == "dew")[0][0]  #находим индекс класса "dew"
            images_dew = images[labels == class_index]
            labels_dew = labels[labels == class_index]

            print(f"Количество изображений класса 'dew': {len(images_dew)}")
            print(f"Форма массива images_dew: {images_dew.shape}")

    n_clusters = 11
    X = images.reshape(images.shape[0], -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    print("Kmeans done ")

    # 2. Обучение SVM-классификатора
    kmeans_features = kmeans.predict(X).reshape(-1,1) # новые признаки из меток кластеров
    print(kmeans_features)
    scaler = StandardScaler()
    kmeans_features = scaler.fit_transform(kmeans_features)
    svm = SVC(kernel='poly', probability=True)  # probability=True для получения вероятностей
    svm.fit(kmeans_features, labels)
    print("SVM done")

    # 3. Классификация нового изображения
    new_image_path = "./TRAIN_DATASET/6091.jpg"  # Замените на путь к вашему изображению
    new_image = cv2.imread(new_image_path)
    cv2.imshow("dsad", new_image)
    new_image = cv2.resize(new_image, (config.image_size, config.image_size))  # Измените размер, чтобы соответствовать тренировочным данным
    new_image_vector = new_image.reshape(1, -1)
    new_image_kmeans_features = kmeans.predict(new_image_vector).reshape(-1,1)
    new_image_kmeans_features = scaler.transform(new_image_kmeans_features)
    predicted_label = svm.predict(new_image_kmeans_features)[0]
    predicted_probability = svm.predict_proba(new_image_kmeans_features)[0]
    print(f"Предсказанный класс: {class_names[int(predicted_label)]}")
    print(f"Вероятности: {(class_names, predicted_probability)}")