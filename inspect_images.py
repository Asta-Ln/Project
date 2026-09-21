from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from PIL import Image, ImageStat


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

# Папка с фотографиями
IMAGE_DIR = Path("dataset/images")

# Папка для отчётов
REPORT_DIR = Path("dataset_report")

# Какие файлы считаем изображениями
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

# Предполагаемый стандартный размер из описания кейса
EXPECTED_SIZE = (1600, 1200)

# Проверять ли соответствие ожидаемому размеру
CHECK_EXPECTED_SIZE = True

# Нужно ли искать изображения во вложенных папках
SEARCH_SUBFOLDERS = True

# ------------------------------------------------------------
# Простые эвристики качества.
# Это НЕ критерии конкурса.
# Они только помогают найти подозрительные изображения.
# ------------------------------------------------------------

DARK_THRESHOLD = 40
BRIGHT_THRESHOLD = 215
LOW_CONTRAST_THRESHOLD = 10


# ============================================================
# 2. ПОДГОТОВКА
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. ПОИСК ФОТОГРАФИЙ
# ============================================================

def find_images():
    """
    Возвращает список найденных изображений.
    """

    if SEARCH_SUBFOLDERS:
        all_files = IMAGE_DIR.rglob("*")
    else:
        all_files = IMAGE_DIR.glob("*")

    images = []

    for path in all_files:

        if not path.is_file():
            continue

        extension = path.suffix.lower()

        if extension in SUPPORTED_EXTENSIONS:
            images.append(path)

    return sorted(images)


# ============================================================
# 4. MD5
# ============================================================

def calculate_md5(file_path):
    """
    Вычисляет хеш файла.
    Используется для поиска полностью одинаковых файлов.
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
# 5. ЯРКОСТЬ И КОНТРАСТ
# ============================================================

def analyze_brightness_and_contrast(image):
    """
    Вычисляет приблизительную среднюю яркость и контраст.

    Изображение уменьшается перед расчётом,
    чтобы ускорить обработку большого набора фотографий.
    """

    image_copy = image.copy()

    # Уменьшаем изображение для быстрого анализа.
    image_copy.thumbnail((400, 400))

    gray = image_copy.convert("L")

    statistics = ImageStat.Stat(gray)

    mean_brightness = statistics.mean[0]
    contrast = statistics.stddev[0]

    return mean_brightness, contrast


# ============================================================
# 6. АНАЛИЗ ОДНОЙ ФОТОГРАФИИ
# ============================================================

def analyze_image(image_path):

    result = {
        "filename": image_path.name,
        "relative_path": str(
            image_path.relative_to(IMAGE_DIR)
        ),

        "extension": image_path.suffix.lower(),

        "file_size_mb": round(
            image_path.stat().st_size / (1024 * 1024),
            3
        ),

        "valid": False,

        "width": None,
        "height": None,

        "format": None,
        "mode": None,

        "md5": None,

        "size_ok": None,

        "mean_brightness": None,
        "contrast": None,

        "warning": "",
        "error": "",
    }

    try:

        # ----------------------------------------------------
        # Проверка целостности
        # ----------------------------------------------------

        with Image.open(image_path) as image:
            image.verify()

        # После verify() открываем изображение заново.
        with Image.open(image_path) as image:

            width, height = image.size

            result["valid"] = True

            result["width"] = width
            result["height"] = height

            result["format"] = image.format
            result["mode"] = image.mode

            # ------------------------------------------------
            # Проверка размера
            # ------------------------------------------------

            result["size_ok"] = (
                (width, height) == EXPECTED_SIZE
            )

            # ------------------------------------------------
            # Хеш
            # ------------------------------------------------

            result["md5"] = calculate_md5(
                image_path
            )

            # ------------------------------------------------
            # Яркость и контраст
            # ------------------------------------------------

            brightness, contrast = (
                analyze_brightness_and_contrast(image)
            )

            result["mean_brightness"] = round(
                brightness,
                2
            )

            result["contrast"] = round(
                contrast,
                2
            )

            # ------------------------------------------------
            # Предупреждения
            # ------------------------------------------------

            warnings = []

            if (
                CHECK_EXPECTED_SIZE
                and not result["size_ok"]
            ):
                warnings.append(
                    "нестандартный размер"
                )

            if brightness < DARK_THRESHOLD:
                warnings.append(
                    "слишком тёмное"
                )

            if brightness > BRIGHT_THRESHOLD:
                warnings.append(
                    "слишком светлое"
                )

            if contrast < LOW_CONTRAST_THRESHOLD:
                warnings.append(
                    "низкий контраст"
                )

            # Например, для цветного кейса можно обратить внимание
            # на изображения, которые оказались не RGB.
            if image.mode not in {"RGB", "RGBA"}:
                warnings.append(
                    f"необычный цветовой режим: {image.mode}"
                )

            result["warning"] = "; ".join(
                warnings
            )

    except Exception as error:

        result["error"] = str(error)

    return result


# ============================================================
# 7. ПОИСК ДУБЛИКАТОВ
# ============================================================

def find_duplicates(dataframe):

    """
    Группирует файлы с одинаковым MD5.
    """

    hash_groups = defaultdict(list)

    for _, row in dataframe.iterrows():

        # Повреждённые файлы не учитываем.
        if not row["valid"]:
            continue

        # Без хеша сравнение невозможно.
        if not row["md5"]:
            continue

        hash_groups[row["md5"]].append(
            row["relative_path"]
        )

    duplicate_rows = []

    group_number = 1

    for file_hash, files in hash_groups.items():

        if len(files) < 2:
            continue

        for file_path in files:

            duplicate_rows.append(
                {
                    "duplicate_group": group_number,
                    "md5": file_hash,
                    "file": file_path,
                }
            )

        group_number += 1

    return pd.DataFrame(
        duplicate_rows
    )


# ============================================================
# 8. СВОДНАЯ ИНФОРМАЦИЯ
# ============================================================

def print_summary(
    dataframe,
    duplicate_dataframe
):

    total = len(dataframe)

    valid = int(
        dataframe["valid"].sum()
    )

    invalid = total - valid

    print()
    print("=" * 60)
    print("        ОТЧЁТ О ПРОВЕРКЕ ФОТОГРАФИЙ")
    print("=" * 60)

    print(
        f"Всего изображений:       {total}"
    )

    print(
        f"Корректных:              {valid}"
    )

    print(
        f"Повреждённых:            {invalid}"
    )

    if total == 0:
        return

    # --------------------------------------------------------
    # Размеры
    # --------------------------------------------------------

    print()
    print("Размеры изображений:")

    size_distribution = (
        dataframe[
            dataframe["valid"]
        ][
            ["width", "height"]
        ]
        .value_counts()
    )

    for (width, height), count in (
        size_distribution.items()
    ):

        print(
            f"  {width}x{height}: {count}"
        )

    # --------------------------------------------------------
    # Ожидаемый размер
    # --------------------------------------------------------

    if CHECK_EXPECTED_SIZE:

        correct_size = int(
            dataframe["size_ok"]
            .fillna(False)
            .sum()
        )

        incorrect_size = (
            valid - correct_size
        )

        print()
        print(
            f"Ожидаемый размер "
            f"{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}:"
        )

        print(
            f"  соответствует:         {correct_size}"
        )

        print(
            f"  не соответствует:      {incorrect_size}"
        )

    # --------------------------------------------------------
    # Форматы
    # --------------------------------------------------------

    print()
    print("Форматы:")

    format_distribution = (
        dataframe["format"]
        .value_counts(dropna=False)
    )

    for image_format, count in (
        format_distribution.items()
    ):

        print(
            f"  {image_format}: {count}"
        )

    # --------------------------------------------------------
    # Цветовые режимы
    # --------------------------------------------------------

    print()
    print("Цветовые режимы:")

    mode_distribution = (
        dataframe["mode"]
        .value_counts(dropna=False)
    )

    for mode, count in (
        mode_distribution.items()
    ):

        print(
            f"  {mode}: {count}"
        )

    # --------------------------------------------------------
    # Предупреждения
    # --------------------------------------------------------

    warning_count = int(
        dataframe["warning"]
        .fillna("")
        .ne("")
        .sum()
    )

    print()
    print(
        f"Изображений с предупреждениями: "
        f"{warning_count}"
    )

    # --------------------------------------------------------
    # Дубликаты
    # --------------------------------------------------------

    print(
        f"Файлов, участвующих в дубликатах: "
        f"{len(duplicate_dataframe)}"
    )

    print("=" * 60)


# ============================================================
# 9. ОСНОВНАЯ ПРОГРАММА
# ============================================================

def main():

    # --------------------------------------------------------
    # Проверяем папку
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        print(
            f"Ошибка: папка не найдена: "
            f"{IMAGE_DIR}"
        )

        print(
            "Измени IMAGE_DIR в настройках."
        )

        return

    # --------------------------------------------------------
    # Ищем изображения
    # --------------------------------------------------------

    image_files = find_images()

    print(
        f"Найдено изображений: "
        f"{len(image_files)}"
    )

    if not image_files:

        print(
            "JPEG/PNG изображения не найдены."
        )

        return

    # --------------------------------------------------------
    # Анализируем каждое изображение
    # --------------------------------------------------------

    results = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        result = analyze_image(
            image_path
        )

        results.append(result)

        if (
            index % 100 == 0
            or index == len(image_files)
        ):

            print(
                f"Обработано: "
                f"{index}/{len(image_files)}"
            )

    # --------------------------------------------------------
    # Создаём таблицу
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Ищем дубликаты
    # --------------------------------------------------------

    duplicate_dataframe = (
        find_duplicates(dataframe)
    )

    # --------------------------------------------------------
    # Сохраняем общий отчёт
    # --------------------------------------------------------

    dataframe.to_csv(
        REPORT_DIR / "images_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Сохраняем дубликаты
    # --------------------------------------------------------

    duplicate_dataframe.to_csv(
        REPORT_DIR / "duplicates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Сохраняем только проблемные изображения
    # --------------------------------------------------------

    problematic = dataframe[
        (
            dataframe["warning"]
            .fillna("")
            .ne("")
        )
        |
        (
            dataframe["error"]
            .fillna("")
            .ne("")
        )
    ]

    problematic.to_csv(
        REPORT_DIR / "problematic_images.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Показываем результат
    # --------------------------------------------------------

    print_summary(
        dataframe,
        duplicate_dataframe
    )

    print()
    print(
        "Отчёты сохранены в:"
    )

    print(
        REPORT_DIR.resolve()
    )


# ============================================================
# 10. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()

анализ данных

*****
