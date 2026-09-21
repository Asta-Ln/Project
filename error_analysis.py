from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Обученная модель
MODEL_PATH = Path(
    "runs/detect/baseline/weights/best.pt"
)

# Validation-изображения
IMAGE_DIR = Path(
    "prepared_dataset/images/val"
)

# Validation-разметка
LABEL_DIR = Path(
    "prepared_dataset/labels/val"
)

# Куда сохранять результаты анализа
OUTPUT_DIR = Path(
    "error_analysis"
)

# Размер изображения для модели
IMGSZ = 640

# Порог уверенности
CONF_THRESHOLD = 0.25

# Порог IoU для сопоставления
# истинного объекта и предсказания
IOU_THRESHOLD = 0.5

# Максимальное количество примеров
# каждого типа ошибки
MAX_EXAMPLES_PER_TYPE = 10


# ============================================================
# СОЗДАЁМ ПАПКИ
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MISSED_DIR = OUTPUT_DIR / "missed"
FALSE_POSITIVE_DIR = OUTPUT_DIR / "false_positive"
WRONG_CLASS_DIR = OUTPUT_DIR / "wrong_class"
LOW_CONF_DIR = OUTPUT_DIR / "low_confidence"

for directory in [
    MISSED_DIR,
    FALSE_POSITIVE_DIR,
    WRONG_CLASS_DIR,
    LOW_CONF_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# УТИЛИТЫ
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Intersection over Union.

    box:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# ЧТЕНИЕ YOLO-РАЗМЕТКИ
# ============================================================

def load_ground_truth(
    label_path,
    image_width,
    image_height,
):
    """
    Читает YOLO-разметку:

    class x_center y_center width height

    и переводит координаты в пиксели:

    x1 y1 x2 y2
    """

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with label_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])

                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                continue

            # ------------------------------------------------
            # Нормализованные координаты → пиксели
            # ------------------------------------------------

            x_center_px = (
                x_center * image_width
            )

            y_center_px = (
                y_center * image_height
            )

            width_px = (
                width * image_width
            )

            height_px = (
                height * image_height
            )

            x1 = (
                x_center_px
                - width_px / 2
            )

            y1 = (
                y_center_px
                - height_px / 2
            )

            x2 = (
                x_center_px
                + width_px / 2
            )

            y2 = (
                y_center_px
                + height_px / 2
            )

            ground_truth.append(
                {
                    "class_id": class_id,
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

    return ground_truth


# ============================================================
# ПРЕДСКАЗАНИЯ МОДЕЛИ
# ============================================================

def get_predictions(
    model,
    image_path,
):
    """
    Запускает модель на одном изображении.
    """

    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    result = results[0]

    predictions = []

    if result.boxes is None:
        return predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )
    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    for box, class_id, confidence in zip(
        boxes,
        classes,
        confidences,
    ):

        predictions.append(
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": box.tolist(),
            }
        )

    return predictions


# ============================================================
# СОПОСТАВЛЕНИЕ PREDICTION ↔ GROUND TRUTH
# ============================================================

def match_objects(
    ground_truth,
    predictions,
):
    """
    Сопоставляет реальные объекты
    с предсказаниями.

    Результат:

    matches
    missed
    false_positive
    wrong_class
    """

    matches = []

    missed = []

    false_positive = []

    wrong_class = []

    used_predictions = set()

    # --------------------------------------------------------
    # Сначала пытаемся сопоставить каждый GT
    # с лучшим предсказанием
    # --------------------------------------------------------

    for gt_index, gt in enumerate(
        ground_truth
    ):

        best_prediction_index = None
        best_iou = 0.0

        for prediction_index, prediction in enumerate(
            predictions
        ):

            if prediction_index in used_predictions:
                continue

            iou = calculate_iou(
                gt["box"],
                prediction["box"]
            )

            if iou > best_iou:

                best_iou = iou
                best_prediction_index = (
                    prediction_index
                )

        # ----------------------------------------------------
        # Нет достаточно хорошего пересечения
        # ----------------------------------------------------

        if (
            best_prediction_index is None
            or best_iou < IOU_THRESHOLD
        ):

            missed.append(
                {
                    "gt_index": gt_index,
                    "gt": gt,
                }
            )

            continue

        # ----------------------------------------------------
        # Нашли подходящее предсказание
        # ----------------------------------------------------

        prediction = predictions[
            best_prediction_index
        ]

        used_predictions.add(
            best_prediction_index
        )

        # ----------------------------------------------------
        # Проверяем класс
        # ----------------------------------------------------

        if (
            prediction["class_id"]
            != gt["class_id"]
        ):

            wrong_class.append(
                {
                    "gt": gt,
                    "prediction": prediction,
                    "iou": best_iou,
                }
            )

        else:

            matches.append(
                {
                    "gt": gt,
                    "prediction": prediction,
                    "iou": best_iou,
                }
            )

    # --------------------------------------------------------
    # Оставшиеся prediction → false positive
    # --------------------------------------------------------

    for prediction_index, prediction in enumerate(
        predictions
    ):

        if prediction_index not in used_predictions:

            false_positive.append(
                {
                    "prediction": prediction
                }
            )

    return (
        matches,
        missed,
        false_positive,
        wrong_class,
    )


# ============================================================
# РИСОВАНИЕ
# ============================================================

def draw_box(
    image,
    box,
    label,
):
    """
    Рисует прямоугольник и подпись.
    """

    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def save_error_image(
    image_path,
    ground_truth,
    predictions,
    output_path,
):
    """
    Рисует реальные объекты и предсказания.

    GT  → зелёный
    Pred → красный
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return

    # --------------------------------------------------------
    # Реальная разметка
    # --------------------------------------------------------

    for gt in ground_truth:

        class_id = gt["class_id"]

        x1, y1, x2, y2 = [
            int(value)
            for value in gt["box"]
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"GT: class {class_id}",
            (
                x1,
                max(20, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------------
    # Предсказания
    # --------------------------------------------------------

    for prediction in predictions:

        class_id = prediction["class_id"]
        confidence = prediction["confidence"]

        x1, y1, x2, y2 = [
            int(value)
            for value in prediction["box"]
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            (
                f"Pred: class {class_id} "
                f"{confidence:.2f}"
            ),
            (
                x1,
                min(
                    image.shape[0] - 10,
                    y2 + 20
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(
        str(output_path),
        image
    )


# ============================================================
# ОСНОВНОЙ АНАЛИЗ
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Модель не найдена:\n"
            f"{MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Папка validation "
            f"не найдена:\n"
            f"{IMAGE_DIR}"
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    image_files = sorted(
        [
            path
            for path in IMAGE_DIR.iterdir()
            if path.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
            }
        ]
    )

    print(
        f"Validation изображений: "
        f"{len(image_files)}"
    )

    report_rows = []

    missed_count = 0
    false_positive_count = 0
    wrong_class_count = 0
    low_confidence_count = 0

    # --------------------------------------------------------
    # Проходим по изображениям
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        ground_truth = load_ground_truth(
            label_path,
            image_width,
            image_height,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = get_predictions(
            model,
            image_path,
        )

        # ----------------------------------------------------
        # Сопоставление
        # ----------------------------------------------------

        (
            matches,
            missed,
            false_positive,
            wrong_class,
        ) = match_objects(
            ground_truth,
            predictions,
        )

        # ----------------------------------------------------
        # Обрабатываем missed
        # ----------------------------------------------------

        if missed:

            missed_count += len(missed)

            if (
                missed_count
                <= MAX_EXAMPLES_PER_TYPE
            ):

                output_path = (
                    MISSED_DIR
                    / image_path.name
                )

                save_error_image(
                    image_path,
                    ground_truth,
                    predictions,
                    output_path,
                )

                report_rows.append(
                    {
                        "image": image_path.name,
                        "error_type": "missed",
                        "count": len(missed),
                        "description": (
                            "Реальный объект "
                            "не был обнаружен"
                        ),
                    }
                )

        # ----------------------------------------------------
        # False positive
        # ----------------------------------------------------

        if false_positive:

            false_positive_count += (
                len(false_positive)
            )

            if (
                false_positive_count
                <= MAX_EXAMPLES_PER_TYPE
            ):

                output_path = (
                    FALSE_POSITIVE_DIR
                    / image_path.name
                )

                save_error_image(
                    image_path,
                    ground_truth,
                    predictions,
                    output_path,
                )

                report_rows.append(
                    {
                        "image": image_path.name,
                        "error_type": "false_positive",
                        "count": len(false_positive),
                        "description": (
                            "Модель обнаружила "
                            "несуществующий объект"
                        ),
                    }
                )

        # ----------------------------------------------------
        # Wrong class
        # ----------------------------------------------------

        if wrong_class:

            wrong_class_count += (
                len(wrong_class)
            )

            if (
                wrong_class_count
                <= MAX_EXAMPLES_PER_TYPE
            ):

                output_path = (
                    WRONG_CLASS_DIR
                    / image_path.name
                )

                save_error_image(
                    image_path,
                    ground_truth,
                    predictions,
                    output_path,
                )

                report_rows.append(
                    {
                        "image": image_path.name,
                        "error_type": "wrong_class",
                        "count": len(wrong_class),
                        "description": (
                            "Объект найден, "
                            "но определён "
                            "неверный класс"
                        ),
                    }
                )

        # ----------------------------------------------------
        # Low confidence
        # ----------------------------------------------------

        for prediction in predictions:

            confidence = (
                prediction["confidence"]
            )

            if (
                confidence
                < 0.5
            ):

                low_confidence_count += 1

                if (
                    low_confidence_count
                    <= MAX_EXAMPLES_PER_TYPE
                ):

                    output_path = (
                        LOW_CONF_DIR
                        / (
                            f"{image_path.stem}"
                            "_low_conf.jpg"
                        )
                    )

                    save_error_image(
                        image_path,
                        ground_truth,
                        predictions,
                        output_path,
                    )

        if (
            index % 50 == 0
            or index == len(image_files)
        ):

            print(
                f"Обработано: "
                f"{index}/{len(image_files)}"
            )

    # ========================================================
    # СОХРАНЯЕМ ОТЧЁТ
    # ========================================================

    report_df = pd.DataFrame(
        report_rows
    )

    report_df.to_csv(
        OUTPUT_DIR
        / "error_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # СВОДКА
    # ========================================================

    print()
    print("=" * 60)
    print("АНАЛИЗ ОШИБОК")
    print("=" * 60)

    print(
        f"Пропущенные объекты: "
        f"{missed_count}"
    )

    print(
        f"Ложные обнаружения: "
        f"{false_positive_count}"
    )

    print(
        f"Неверный класс: "
        f"{wrong_class_count}"
    )

    print(
        f"Низкая уверенность: "
        f"{low_confidence_count}"
    )

    print()
    print(
        "Примеры ошибок сохранены в:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
