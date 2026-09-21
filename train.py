from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Путь к предобученной модели.
# Лучше хранить .pt локально в проекте.
MODEL_PATH = Path("models/yolo26n.pt")

# Путь к описанию датасета.
DATA_YAML = Path("prepared_dataset/dataset.yaml")

# Количество эпох.
# На конкурсе можно менять в зависимости от времени.
EPOCHS = 30

# Размер изображения, который используется моделью при обучении.
# Для baseline начнём с 640.
IMGSZ = 640

# Размер batch.
# 4 — достаточно осторожное значение для компьютера
# с ограниченной видеопамятью.
BATCH = 4

# Сколько эпох ждать улучшения перед остановкой.
PATIENCE = 10

# Фиксируем случайность для воспроизводимости.
SEED = 42

# Название эксперимента.
RUN_NAME = "baseline"

# Папка, куда сохраняются результаты.
PROJECT_DIR = "runs/detect"

# Для Windows безопаснее начать с 0 workers.
WORKERS = 0


# ============================================================
# ОПРЕДЕЛЯЕМ УСТРОЙСТВО
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    print("Используется NVIDIA GPU.")
else:
    DEVICE = "cpu"
    print("CUDA GPU не найден. Используется CPU.")


# ============================================================
# ПРОВЕРКИ
# ============================================================

def check_files():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}\n"
            f"Положи .pt файл в папку models/"
        )

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"Файл датасета не найден: {DATA_YAML}\n"
            f"Сначала подготовь датасет."
        )


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train():

    check_files()

    print()
    print("=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ")
    print("=" * 60)

    print(f"Модель:       {MODEL_PATH}")
    print(f"Датасет:      {DATA_YAML}")
    print(f"Epochs:       {EPOCHS}")
    print(f"Image size:   {IMGSZ}")
    print(f"Batch:        {BATCH}")
    print(f"Device:       {DEVICE}")
    print(f"Run name:     {RUN_NAME}")
    print("=" * 60)

    # --------------------------------------------------------
    # Загружаем предобученную модель
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Обучение
    # --------------------------------------------------------

    results = model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,

        patience=PATIENCE,

        device=DEVICE,

        seed=SEED,

        workers=WORKERS,

        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # Сохранять графики и результаты
        plots=True,
    )

    return results


# ============================================================
# ПРОВЕРКА ЛУЧШЕЙ МОДЕЛИ
# ============================================================

def validate():

    best_model_path = (
        Path(PROJECT_DIR)
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():

        print()
        print(
            "Файл best.pt не найден."
        )

        print(
            "Проверь папку с результатами обучения."
        )

        return

    print()
    print("=" * 60)
    print("ПРОВЕРКА ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 60)

    print(
        f"Загрузка: {best_model_path}"
    )

    best_model = YOLO(
        str(best_model_path)
    )

    metrics = best_model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        device=DEVICE,
    )

    print()
    print("Валидация завершена.")


# ============================================================
# MAIN
# ============================================================

def main():

    train()

    validate()

    print()
    print("=" * 60)
    print("ГОТОВО")
    print("=" * 60)

    print(
        "Лучшая модель:"
    )

    print(
        Path(PROJECT_DIR)
        / RUN_NAME
        / "weights"
        / "best.pt"
    )


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()
