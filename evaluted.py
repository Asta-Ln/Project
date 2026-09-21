from pathlib import Path
import json

import torch
from ultralytics import YOLO


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Лучшая модель после обучения
MODEL_PATH = Path(
    "runs/detect/baseline/weights/best.pt"
)

# Описание датасета
DATA_YAML = Path(
    "prepared_dataset/dataset.yaml"
)

# Размер изображения
IMGSZ = 640

# Batch для валидации
BATCH = 4

# Порог уверенности
CONF = 0.25

# Папка с результатами оценки
REPORT_DIR = Path("evaluation")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# УСТРОЙСТВО
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0

    print("Используется NVIDIA GPU.")

else:

    DEVICE = "cpu"

    print("Используется CPU.")


# ============================================================
# ПРОВЕРКА ФАЙЛОВ
# ============================================================

def check_files():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Модель не найдена:\n"
            f"{MODEL_PATH}\n\n"
            f"Сначала выполни обучение."
        )

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"Файл dataset.yaml не найден:\n"
            f"{DATA_YAML}"
        )


# ============================================================
# ВАЛИДАЦИЯ
# ============================================================

def evaluate_model():

    check_files()

    print()
    print("=" * 60)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 60)

    print(f"Модель:      {MODEL_PATH}")
    print(f"Датасет:     {DATA_YAML}")
    print(f"Image size:  {IMGSZ}")
    print(f"Batch:       {BATCH}")
    print(f"Confidence:  {CONF}")
    print(f"Device:      {DEVICE}")

    print("=" * 60)

    # --------------------------------------------------------
    # Загружаем обученную модель
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Запускаем validation
    # --------------------------------------------------------

    metrics = model.val(
        data=str(DATA_YAML),

        imgsz=IMGSZ,

        batch=BATCH,

        conf=CONF,

        device=DEVICE,

        plots=True,

        save_json=True,
    )

    return metrics


# ============================================================
# СОХРАНЕНИЕ ОСНОВНЫХ МЕТРИК
# ============================================================

def save_metrics(metrics):

    # Основные detection-метрики
    box_metrics = metrics.box

    result = {
        "mAP50-95": float(
            box_metrics.map
        ),

        "mAP50": float(
            box_metrics.map50
        ),

        "mAP75": float(
            box_metrics.map75
        ),
    }

    # Средние Precision / Recall,
    # если объект metrics их предоставляет.
    if hasattr(box_metrics, "mp"):

        result["precision"] = float(
            box_metrics.mp
        )

    if hasattr(box_metrics, "mr"):

        result["recall"] = float(
            box_metrics.mr
        )

    # --------------------------------------------------------
    # Сохраняем JSON
    # --------------------------------------------------------

    output_file = (
        REPORT_DIR / "metrics.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Выводим на экран
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    for metric_name, value in result.items():

        print(
            f"{metric_name:12}: "
            f"{value:.4f}"
        )

    print("=" * 60)

    return result


# ============================================================
# МЕТРИКИ ПО КЛАССАМ
# ============================================================

def save_class_metrics(metrics):

    box_metrics = metrics.box

    class_map = box_metrics.maps

    class_results = []

    for class_id, map_value in enumerate(
        class_map
    ):

        class_results.append(
            {
                "class_id": class_id,
                "mAP50-95": float(map_value),
            }
        )

    if not class_results:
        return

    output_file = (
        REPORT_DIR / "class_metrics.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("mAP50-95 по классам:")

    for item in class_results:

        print(
            f"  Класс {item['class_id']}: "
            f"{item['mAP50-95']:.4f}"
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(metrics):

    try:

        matrix = (
            metrics.confusion_matrix
        )

        dataframe = matrix.to_df()

        output_file = (
            REPORT_DIR
            / "confusion_matrix.csv"
        )

        dataframe.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            "Confusion matrix сохранена."
        )

    except Exception as error:

        print()
        print(
            "Не удалось сохранить "
            "confusion matrix:"
        )

        print(error)


# ============================================================
# MAIN
# ============================================================

def main():

    metrics = evaluate_model()

    save_metrics(
        metrics
    )

    save_class_metrics(
        metrics
    )

    save_confusion_matrix(
        metrics
    )

    print()
    print(
        "Оценка завершена."
    )

    print(
        f"Результаты находятся в: "
        f"{REPORT_DIR.resolve()}"
    )


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()
