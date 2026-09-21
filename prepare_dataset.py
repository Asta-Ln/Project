from pathlib import Path
from collections import Counter
import hashlib
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

# Исходные данные
DATASET_DIR = Path("dataset")

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

# Куда положить подготовленный датасет
OUTPUT_DIR = Path("prepared_dataset")

# Поддерживаемые изображения
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# Размер validation
VALIDATION_SIZE = 0.2

# Фиксируем random_state,
# чтобы результат можно было воспроизвести
RANDOM_STATE = 42

# Проверять ли, что координаты разметки находятся
# внутри изображения
CHECK_BOUNDS = True


# ============================================================
# 2. ПОДГОТОВКА ПАПОК
# ============================================================

TRAIN_IMAGES_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMAGES_DIR = OUTPUT_DIR / "images" / "val"

TRAIN_LABELS_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LABELS_DIR = OUTPUT_DIR / "labels" / "val"

REPORT_DIR = OUTPUT_DIR / "reports"


for directory in [
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. MD5
# ============================================================

def calculate_md5(file_path):
    """
    Вычисляет MD5-хеш файла.
    Нужен для поиска полных дубликатов изображений.
    """

    md5 = hashlib.md5()

    with file_path.open("rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================
# 4. ПРОВЕРКА ОДНОЙ YOLO-РАЗМЕТКИ
# ============================================================

def validate_label(label_path):
    """
    Проверяет YOLO-разметку.

    Ожидаемый формат строки:

    class_id x_center y_center width height

    Возвращает:
        valid       - корректна ли разметка
        classes     - классы, которые найдены
        errors      - список ошибок
    """

    errors = []
    classes = []

    try:

        with label_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

    except Exception as error:

        return False, classes, [
            f"Не удалось прочитать файл: {error}"
        ]

    # Пустой файл
    if not lines:

        return False, classes, [
            "пустой файл разметки"
        ]

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        # Пустая строка
        if not line:
            continue

        parts = line.split()

        # Для YOLO detection должно быть 5 значений
        if len(parts) != 5:

            errors.append(
                f"строка {line_number}: "
                f"ожидалось 5 значений, "
                f"получено {len(parts)}"
            )

            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            errors.append(
                f"строка {line_number}: "
                f"обнаружено нечисловое значение"
            )

            continue

        # ----------------------------------------------------
        # Проверяем класс
        # ----------------------------------------------------

        if class_id < 0:

            errors.append(
                f"строка {line_number}: "
                f"отрицательный class_id"
            )

        classes.append(class_id)

        # ----------------------------------------------------
        # Проверяем размеры
        # ----------------------------------------------------

        if width <= 0:

            errors.append(
                f"строка {line_number}: "
                f"width <= 0"
            )

        if height <= 0:

            errors.append(
                f"строка {line_number}: "
                f"height <= 0"
            )

        # ----------------------------------------------------
        # Проверяем нормализованные координаты
        # ----------------------------------------------------

        values = [
            x_center,
            y_center,
            width,
            height,
        ]

        for value in values:

            if not 0 <= value <= 1:

                errors.append(
                    f"строка {line_number}: "
                    f"значение {value} "
                    f"не находится в диапазоне [0, 1]"
                )

        # ----------------------------------------------------
        # Проверяем, не выходит ли bounding box
        # за границы изображения
        # ----------------------------------------------------

        if CHECK_BOUNDS:

            x_min = x_center - width / 2
            x_max = x_center + width / 2

            y_min = y_center - height / 2
            y_max = y_center + height / 2

            if x_min < 0 or x_max > 1:

                errors.append(
                    f"строка {line_number}: "
                    f"bounding box выходит "
                    f"за границы по X"
                )

            if y_min < 0 or y_max > 1:

                errors.append(
                    f"строка {line_number}: "
                    f"bounding box выходит "
                    f"за границы по Y"
                )

    return (
        len(errors) == 0,
        classes,
        errors
    )


# ============================================================
# 5. ПОИСК ИЗОБРАЖЕНИЙ
# ============================================================

def find_images():

    images = []

    for path in IMAGE_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)

    return sorted(images)


# ============================================================
# 6. ПРОВЕРКА И СОПОСТАВЛЕНИЕ
#    ФОТОГРАФИЯ + LABEL
# ============================================================

def analyze_dataset():

    image_files = find_images()

    print(
        f"Найдено изображений: "
        f"{len(image_files)}"
    )

    if not image_files:

        return pd.DataFrame()

    rows = []

    # Для поиска дубликатов
    hash_groups = {}

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        relative_path = image_path.relative_to(
            IMAGE_DIR
        )

        label_path = (
            LABEL_DIR
            / relative_path.with_suffix(".txt")
        )

        row = {
            "image": str(relative_path),
            "label": str(
                label_path.relative_to(LABEL_DIR)
            ),
            "image_exists": True,
            "label_exists": label_path.exists(),
            "valid_label": False,
            "classes": "",
            "objects": 0,
            "md5": "",
            "status": "",
            "reason": "",
        }

        # ----------------------------------------------------
        # MD5 фотографии
        # ----------------------------------------------------

        try:

            file_hash = calculate_md5(
                image_path
            )

            row["md5"] = file_hash

            if file_hash not in hash_groups:

                hash_groups[file_hash] = []

            hash_groups[file_hash].append(
                str(relative_path)
            )

        except Exception as error:

            row["status"] = "exclude"

            row["reason"] = (
                f"Ошибка чтения файла: {error}"
            )

            rows.append(row)

            continue

        # ----------------------------------------------------
        # Нет разметки
        # ----------------------------------------------------

        if not label_path.exists():

            row["status"] = "exclude"

            row["reason"] = (
                "отсутствует файл разметки"
            )

            rows.append(row)

            continue

        # ----------------------------------------------------
        # Проверяем разметку
        # ----------------------------------------------------

        valid, classes, errors = validate_label(
            label_path
        )

        row["valid_label"] = valid
        row["objects"] = len(classes)

        row["classes"] = ",".join(
            map(str, sorted(set(classes)))
        )

        if valid:

            row["status"] = "valid"

        else:

            row["status"] = "exclude"

            row["reason"] = "; ".join(errors)

        rows.append(row)

        if (
            index % 100 == 0
            or index == len(image_files)
        ):

            print(
                f"Проверено: "
                f"{index}/{len(image_files)}"
            )

    dataframe = pd.DataFrame(rows)

    return dataframe, hash_groups


# ============================================================
# 7. ПОИСК ДУБЛИКАТОВ
# ============================================================

def mark_duplicates(
    dataframe,
    hash_groups
):

    duplicate_hashes = {
        file_hash
        for file_hash, files
        in hash_groups.items()
        if len(files) > 1
    }

    dataframe["duplicate"] = (
        dataframe["md5"]
        .isin(duplicate_hashes)
    )

    # Отдельно отмечаем дубликаты
    # только среди пригодных данных.
    mask = (
        dataframe["duplicate"]
        & (dataframe["status"] == "valid")
    )

    dataframe.loc[
        mask,
        "status"
    ] = "exclude"

    dataframe.loc[
        mask,
        "reason"
    ] = "полный дубликат изображения"

    return dataframe


# ============================================================
# 8. ПОДГОТОВКА СТАТИСТИКИ ДЛЯ РАЗБИВКИ
# ============================================================

def make_split_key(row):

    """
    Для detection одна фотография может содержать
    несколько классов.

    Создаём условный "отпечаток" набора классов:

    0
    0,1
    1
    0,2

    Его используем как дополнительный ориентир
    для сохранения похожего распределения.
    """

    return row["classes"]


# ============================================================
# 9. РАЗБИЕНИЕ TRAIN / VALIDATION
# ============================================================

def split_dataset(dataframe):

    valid_data = dataframe[
        dataframe["status"] == "valid"
    ].copy()

    if len(valid_data) < 2:

        raise ValueError(
            "Недостаточно корректных данных "
            "для разбиения."
        )

    valid_data["split_key"] = (
        valid_data.apply(
            make_split_key,
            axis=1
        )
    )

    # Проверяем, можно ли использовать stratify.
    value_counts = (
        valid_data["split_key"]
        .value_counts()
    )

    can_stratify = (
        len(value_counts) > 1
        and value_counts.min() >= 2
    )

    if can_stratify:

        train_df, val_df = train_test_split(
            valid_data,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=valid_data["split_key"]
        )

        print(
            "\nИспользовано "
            "стратифицированное разбиение."
        )

    else:

        train_df, val_df = train_test_split(
            valid_data,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE
        )

        print(
            "\nПредупреждение:"
        )

        print(
            "Стратифицированное разбиение "
            "невозможно для текущего "
            "набора данных."
        )

    return (
        train_df,
        val_df
    )


# ============================================================
# 10. КОПИРОВАНИЕ ФАЙЛОВ
# ============================================================

def copy_split(
    dataframe,
    images_output_dir,
    labels_output_dir
):

    for _, row in dataframe.iterrows():

        image_path = (
            IMAGE_DIR
            / row["image"]
        )

        label_path = (
            LABEL_DIR
            / row["label"]
        )

        output_image = (
            images_output_dir
            / Path(row["image"]).name
        )

        output_label = (
            labels_output_dir
            / Path(row["label"]).name
        )

        shutil.copy2(
            image_path,
            output_image
        )

        shutil.copy2(
            label_path,
            output_label
        )


# ============================================================
# 11. СОЗДАНИЕ DATASET.YAML
# ============================================================

def create_dataset_yaml(dataframe):

    all_classes = set()

    for classes in dataframe["classes"]:

        if not classes:
            continue

        for class_id in classes.split(","):

            all_classes.add(
                int(class_id)
            )

    if not all_classes:
        return

    max_class_id = max(all_classes)

    # Пока названия неизвестны.
    # Их нужно будет заменить после получения
    # реального описания классов.
    names = [
        f"class_{class_id}"
        for class_id in range(
            max_class_id + 1
        )
    ]

    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names)}",
        "names:",
    ]

    for class_id, name in enumerate(names):

        yaml_lines.append(
            f"  {class_id}: {name}"
        )

    yaml_path = (
        OUTPUT_DIR / "dataset.yaml"
    )

    yaml_path.write_text(
        "\n".join(yaml_lines),
        encoding="utf-8"
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    # Проверяем основные папки
    if not IMAGE_DIR.exists():

        print(
            f"Ошибка: папка "
            f"{IMAGE_DIR} не найдена."
        )

        return

    if not LABEL_DIR.exists():

        print(
            "Папка labels не найдена."
        )

        print(
            "Скрипт не может подготовить "
            "датасет для supervised detection, "
            "пока отсутствует разметка."
        )

        return

    # --------------------------------------------------------
    # Анализ
    # --------------------------------------------------------

    result = analyze_dataset()

    if not result:
        return

    dataframe, hash_groups = result

    # --------------------------------------------------------
    # Дубликаты
    # --------------------------------------------------------

    dataframe = mark_duplicates(
        dataframe,
        hash_groups
    )

    # --------------------------------------------------------
    # Сохраняем полный отчёт
    # --------------------------------------------------------

    dataframe.to_csv(
        REPORT_DIR / "preparation_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Если ничего не осталось
    # --------------------------------------------------------

    valid_count = int(
        (
            dataframe["status"]
            == "valid"
        ).sum()
    )

    print()
    print(
        f"Корректных пар image+label: "
        f"{valid_count}"
    )

    if valid_count < 2:

        print(
            "Недостаточно корректных "
            "данных для обучения."
        )

        return

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    train_df, val_df = split_dataset(
        dataframe
    )

    print(
        f"Train:      {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    # --------------------------------------------------------
    # Копирование
    # --------------------------------------------------------

    copy_split(
        train_df,
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR
    )

    copy_split(
        val_df,
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR
    )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    create_dataset_yaml(
        dataframe[
            dataframe["status"] == "valid"
        ]
    )

    # --------------------------------------------------------
    # Отдельный отчёт по исключённым данным
    # --------------------------------------------------------

    excluded = dataframe[
        dataframe["status"]
        == "exclude"
    ]

    excluded.to_csv(
        REPORT_DIR / "excluded_data.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "Подготовка завершена."
    )

    print(
        f"Результат: "
        f"{OUTPUT_DIR.resolve()}"
    )


# ============================================================
# 13. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()

очистка данных
