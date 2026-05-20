# # Лабораторная работа 2
# ## Классификация отзывов к фильмам
# 
# **Вариант:** 9  
# **Тема:** бинарная классификация отзывов IMDB: положительный / отрицательный отзыв.
# 
# В работе выполняется:
# 1. загрузка датасета IMDB;
# 2. подготовка данных;
# 3. обучение базовой модели;
# 4. сравнение моделей с разным количеством слоёв;
# 5. сравнение моделей с разным количеством нейронов;
# 6. проверка функции потерь `mse` вместо `binary_crossentropy`;
# 7. проверка функции активации `tanh` вместо `relu`;
# 8. классификация двух собственных отзывов на фильм.

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # меньше служебных сообщений TensorFlow

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# Фиксируем случайность, чтобы результаты были более воспроизводимыми
SEED = 9
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

NUM_WORDS = 10000
EPOCHS = 8
BATCH_SIZE = 512


# ## 1. Загрузка набора данных IMDB
# 
# Набор IMDB содержит отзывы к фильмам. Метка `1` означает положительный отзыв, метка `0` — отрицательный отзыв.  
# Параметр `num_words=10000` ограничивает словарь десятью тысячами наиболее частотных слов.

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

print("Количество обучающих отзывов:", len(train_data))
print("Количество тестовых отзывов:", len(test_data))
print("Пример метки первого отзыва:", train_labels[0])
print("Длина первого отзыва в индексах:", len(train_data[0]))


# ## 2. Декодирование отзыва обратно в текст
# 
# В датасете отзывы уже закодированы числами. Для проверки можно восстановить примерный текст отзыва.

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return " ".join(reverse_word_index.get(i - 3, "?") for i in encoded_review)

print("Положительный/отрицательный пример из обучающей выборки:")
print("Метка:", train_labels[0])
print(decode_review(train_data[0])[:1000])


# ## 3. Векторизация данных
# 
# Модель будет получать не последовательность слов, а бинарный вектор длиной `10000`.  
# Если слово встречается в отзыве, соответствующая позиция вектора получает значение `1`.

def vectorize_sequences(sequences, dimension=NUM_WORDS):
    results = np.zeros((len(sequences), dimension), dtype="float32")
    for i, sequence in enumerate(sequences):
        for j in sequence:
            if j < dimension:
                results[i, j] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

print("Размер x_train:", x_train.shape)
print("Размер x_test:", x_test.shape)
print("Размер y_train:", y_train.shape)
print("Размер y_test:", y_test.shape)


# ## 4. Создание проверочного набора
# 
# Первые 10000 отзывов обучающей выборки используются как проверочный набор, остальные — для обучения.

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("partial_x_train:", partial_x_train.shape)
print("x_val:", x_val.shape)


# ## 5. Функции для построения, обучения и сравнения моделей
# 
# Чтобы не переписывать код вручную для каждого эксперимента, создаются универсальные функции.

def build_model(hidden_layers=2, units=16, activation="relu", loss="binary_crossentropy"):
    model = keras.Sequential(name=f"{hidden_layers}_layers_{units}_units_{activation}_{loss}")

    for _ in range(hidden_layers):
        model.add(layers.Dense(units, activation=activation))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="rmsprop",
        loss=loss,
        metrics=["accuracy"]
    )
    return model


def train_experiment(name, hidden_layers=2, units=16, activation="relu", loss="binary_crossentropy", epochs=EPOCHS):
    print("=" * 80)
    print("Эксперимент:", name)
    print("Слои:", hidden_layers, "| Нейроны:", units, "| Активация:", activation, "| Loss:", loss)

    model = build_model(
        hidden_layers=hidden_layers,
        units=units,
        activation=activation,
        loss=loss
    )

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=1
    )

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)

    row = {
        "Модель": name,
        "Скрытые слои": hidden_layers,
        "Нейронов в слое": units,
        "Активация": activation,
        "Loss": loss,
        "Лучшая эпоха": best_epoch,
        "Лучшая val_accuracy": round(best_val_acc, 4),
        "Финальная train_accuracy": round(history.history["accuracy"][-1], 4),
        "Финальная val_accuracy": round(history.history["val_accuracy"][-1], 4),
        "Финальная train_loss": round(history.history["loss"][-1], 4),
        "Финальная val_loss": round(history.history["val_loss"][-1], 4),
    }
    return model, history, row


# ## 6. Базовая модель
# 
# Базовая модель содержит два скрытых полносвязных слоя по 16 нейронов с функцией активации `relu`.  
# На выходе используется один нейрон с `sigmoid`, потому что решается задача бинарной классификации.

all_histories = {}
all_results = []
trained_models = {}

model, history, row = train_experiment(
    name="Базовая модель: 2 слоя по 16 relu, binary_crossentropy",
    hidden_layers=2,
    units=16,
    activation="relu",
    loss="binary_crossentropy"
)

all_histories["baseline"] = history
all_results.append(row)
trained_models["baseline"] = model


# ## 7. Графики потерь и точности для базовой модели

def plot_history(history, title_prefix="Модель"):
    history_dict = history.history
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_dict["loss"], "bo", label="Потери на обучении")
    plt.plot(epochs_range, history_dict["val_loss"], "b", label="Потери на проверке")
    plt.title(f"{title_prefix}: потери")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_loss_variant9.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_dict["accuracy"], "bo", label="Точность на обучении")
    plt.plot(epochs_range, history_dict["val_accuracy"], "b", label="Точность на проверке")
    plt.title(f"{title_prefix}: точность")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_accuracy_variant9.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_history(history, "Базовая модель")


# ## 8. Эксперимент: один и три скрытых слоя
# 
# В задании требуется проверить, как изменится качество модели при использовании одного или трёх скрытых слоёв вместо двух.

experiments_layers = [
    ("1 скрытый слой: 16 relu", 1, 16, "relu", "binary_crossentropy"),
    ("3 скрытых слоя: 16 relu", 3, 16, "relu", "binary_crossentropy"),
]

for name, hidden_layers, units, activation, loss in experiments_layers:
    model, history, row = train_experiment(name, hidden_layers, units, activation, loss)
    key = f"layers_{hidden_layers}"
    all_histories[key] = history
    all_results.append(row)
    trained_models[key] = model


# ## 9. Эксперимент: 32 и 64 нейрона
# 
# Проверяется влияние увеличения числа нейронов в скрытых слоях.

experiments_units = [
    ("2 слоя по 32 нейрона, relu", 2, 32, "relu", "binary_crossentropy"),
    ("2 слоя по 64 нейрона, relu", 2, 64, "relu", "binary_crossentropy"),
]

for name, hidden_layers, units, activation, loss in experiments_units:
    model, history, row = train_experiment(name, hidden_layers, units, activation, loss)
    key = f"units_{units}"
    all_histories[key] = history
    all_results.append(row)
    trained_models[key] = model


# ## 10. Эксперимент: `mse` вместо `binary_crossentropy`
# 
# Для бинарной классификации обычно лучше подходит `binary_crossentropy`, но по заданию проверяется вариант с `mse`.

model, history, row = train_experiment(
    name="2 слоя по 16 relu, mse",
    hidden_layers=2,
    units=16,
    activation="relu",
    loss="mse"
)

all_histories["mse"] = history
all_results.append(row)
trained_models["mse"] = model


# ## 11. Эксперимент: `tanh` вместо `relu`
# 
# В этом эксперименте функция активации `relu` заменяется на `tanh`.

model, history, row = train_experiment(
    name="2 слоя по 16 tanh, binary_crossentropy",
    hidden_layers=2,
    units=16,
    activation="tanh",
    loss="binary_crossentropy"
)

all_histories["tanh"] = history
all_results.append(row)
trained_models["tanh"] = model


# ## 12. Сравнительная таблица результатов
# 
# После выполнения всех экспериментов таблица покажет, какая конфигурация дала лучшую точность на проверочном наборе.

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("Лучшая val_accuracy", ascending=False).reset_index(drop=True)

print("\nСРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
print(results_df.to_string(index=False))

results_df.to_csv("imdb_results_variant9.csv", index=False, encoding="utf-8-sig")
print("\nТаблица также сохранена в файл: imdb_results_variant9.csv")

best_row = results_df.iloc[0]
print("\nЛучшая модель по val_accuracy:")
print(best_row.to_string())


# ## 13. Финальное обучение выбранной модели и проверка на тестовой выборке
# 
# Обычно для IMDB-модели достаточно 4 эпох: после этого часто начинается переобучение, когда точность на обучении растёт, а качество на проверке перестаёт улучшаться.
# 
# Ниже используется классическая базовая архитектура из двух слоёв по 16 нейронов. При желании можно заменить параметры на те, которые оказались лучшими в таблице выше.

final_model = build_model(
    hidden_layers=2,
    units=16,
    activation="relu",
    loss="binary_crossentropy"
)

final_model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=BATCH_SIZE,
    verbose=1
)

test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))


# ## 14. Проверка модели на двух собственных отзывах
# 
# В качестве фильма выбран **Fight Club**. Ниже приведены два отзыва на английском языке: один положительный и один отрицательный.
# 
# Важно: модель обучалась на словаре IMDB, поэтому новые отзывы нужно преобразовать в такой же бинарный вектор длиной `10000`, как и обучающие данные.

positive_review = """
Fight Club is one of the most memorable films I have ever watched. At first it looks like a dark story about insomnia,
anger and underground fights, but the film slowly turns into a much deeper reflection on identity, loneliness and the
emptiness of consumer culture. The acting is extremely strong: Edward Norton creates a believable confused narrator,
while Brad Pitt brings dangerous energy and charisma to Tyler Durden. I also liked the visual style, the editing and
the way the story hides important details until the final twist. The film is intense, sometimes uncomfortable, but it
never feels boring or random. Every scene seems to have a purpose, and after the ending many earlier moments become
more meaningful. I would recommend this movie to viewers who enjoy psychological dramas with strong atmosphere,
sharp dialogue and an unusual plot. It is not a simple entertainment film, but it leaves a powerful impression.
"""

negative_review = """
I did not enjoy Fight Club as much as many other viewers seem to. The film tries to be clever and provocative, but for
me it often felt exaggerated, noisy and unpleasant. The story is built around anger, violence and strange philosophical
statements, yet I did not find the main characters sympathetic or convincing. Brad Pitt and Edward Norton act with
energy, but the characters they play are difficult to care about, so the emotional impact was weak. The movie also feels
too long in the middle, and several scenes repeat the same idea without adding much depth. The famous twist may surprise
some people, but I found it more artificial than meaningful. I understand that the film criticizes consumer culture, but
its message is delivered in a chaotic and aggressive way. I would not watch it again, because the atmosphere is too dark
and the story did not give me satisfaction.
"""

custom_reviews = [positive_review, negative_review]
expected_labels = ["положительный отзыв", "отрицательный отзыв"]


def review_to_vector(review, dimension=NUM_WORDS):
    """Преобразует новый текстовый отзыв в бинарный вектор формата IMDB."""
    tokens = re.findall(r"[a-z']+", review.lower())
    vector = np.zeros((1, dimension), dtype="float32")

    for token in tokens:
        index = word_index.get(token)
        if index is not None:
            imdb_index = index + 3
            if imdb_index < dimension:
                vector[0, imdb_index] = 1.0

    return vector


def classify_review(review, model=final_model):
    vector = review_to_vector(review)
    probability = float(model.predict(vector, verbose=0)[0][0])
    predicted_class = "положительный" if probability >= 0.5 else "отрицательный"
    return probability, predicted_class

for i, review in enumerate(custom_reviews, start=1):
    probability, predicted_class = classify_review(review)
    print(f"Отзыв {i}: ожидается — {expected_labels[i-1]}")
    print(f"Вероятность положительного класса: {probability:.4f}")
    print(f"Классификация модели: {predicted_class}")
    print("-" * 80)


# ## 15. Вывод
# 
# В лабораторной работе была обучена нейронная сеть для бинарной классификации отзывов к фильмам.  
# Были проверены разные варианты архитектуры: один, два и три скрытых слоя; 16, 32 и 64 нейрона; функция потерь `mse` вместо `binary_crossentropy`; функция активации `tanh` вместо `relu`.
# 
# Обычно увеличение числа нейронов повышает точность на обучающей выборке, но может ускорить переобучение. Увеличение числа слоёв также не всегда улучшает проверочную точность. Для бинарной классификации отзывов функция потерь `binary_crossentropy` логически подходит лучше, чем `mse`, потому что задача сводится к вероятности принадлежности отзыва к положительному классу. Функция `tanh` может работать, но в современных полносвязных сетях для таких задач часто используется `relu`.
# 
# Финальная модель была проверена на двух собственных англоязычных отзывах к фильму **Fight Club**: положительном и отрицательном. Результаты классификации выводятся в последней кодовой ячейке.
