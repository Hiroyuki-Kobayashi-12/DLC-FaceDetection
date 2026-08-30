# tools/onnx_widerface_difficulty_evaluator.py
# ============================================================
#
# WIDER FACE ValidationへONNX推論を行い、
# Easy、Medium、Hardごとに顔検出性能を評価します。
#
# 判定:
#   TP     対象難易度GTとIoU 0.5以上で1対1対応
#   IGNORE 対象外難易度の有効GTとIoU 0.5以上
#   FP     どの有効GTとも対応しないPrediction
#   FN     Predictionと対応しない対象難易度GT
#
# 主指標:
#   AP@0.5
#
# 補助指標:
#   Precision、Recall、F1、TP、FP、FN、IGNORE
#
# Dataset構造:
#   data/widerface_pytorch_json/val/
#   ├── image/<event>/<image>.jpg
#   └── anno/<event>/<image>.json
#
# ============================================================

from pathlib import Path
import csv
import json

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


# ============================================================
# Settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "results" / "model_best.onnx"
VAL_ROOT = PROJECT_ROOT / "data" / "widerface_pytorch_json" / "val"
IMAGE_ROOT = VAL_ROOT / "image"
ANNO_ROOT = VAL_ROOT / "anno"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "widerface_difficulty_evaluation"

DIFFICULTIES = ("easy", "medium", "hard")
IMAGE_SIZE = 640
AP_CONFIDENCE_THRESHOLD = 0.001
METRIC_CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
MAX_IMAGES = None
PROVIDERS = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
    ]

# ============================================================
# Data
# ============================================================

def load_annotation(path):
    """画像1枚分のJSONを読み込みます。"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_sample(annotation_path):
    """JSONから元画像、相対パス、全有効GTを取得します。"""
    annotation = load_annotation(annotation_path)
    relative_path = annotation["image"]["relative_path"]
    image = Image.open(IMAGE_ROOT / relative_path).convert("RGB")

    boxes = []
    face_indices = []
    levels = []

    for face in annotation["faces"]:
        box = face["bbox"]["xyxy"]

        if not face["valid_bbox"] or box is None:
            continue

        boxes.append(box)
        face_indices.append(int(face["face_index"]))
        levels.append(tuple(face["levels"]))

    return (
        image,
        relative_path,
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(face_indices, dtype=np.int64),
        levels,
    )


# ============================================================
# ONNX Inference
# ============================================================

def preprocess(image):
    """RGB画像を640x640のNCHW float32へ変換します。"""
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def xywh_to_xyxy(boxes):
    """中心座標xywhをxyxyへ変換します。"""
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def iou_matrix(boxes_a, boxes_b):
    """2組のxyxy bbox間のIoU行列を返します。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = size[..., 0] * size[..., 1]

    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-7)


def nms(boxes, scores):
    """信頼度順に重複Predictionを除去します。"""
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0 and len(keep) < MAX_DETECTIONS:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        remaining = order[1:]
        overlaps = iou_matrix(boxes[current:current + 1], boxes[remaining])[0]
        order = remaining[overlaps < NMS_IOU_THRESHOLD]

    return np.asarray(keep, dtype=np.int64)


def postprocess(raw_output, original_width, original_height):
    """YOLOv5出力へ信頼度フィルタ、NMS、座標復元を適用します。"""
    predictions = np.asarray(raw_output)

    if predictions.ndim == 3:
        predictions = predictions[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4] * predictions[:, 5]
    mask = scores >= AP_CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    boxes = xywh_to_xyxy(boxes)
    keep = nms(boxes, scores)
    boxes = boxes[keep]
    scores = scores[keep]

    boxes[:, [0, 2]] *= original_width / float(IMAGE_SIZE)
    boxes[:, [1, 3]] *= original_height / float(IMAGE_SIZE)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    return boxes.astype(np.float32), scores.astype(np.float32)


# ============================================================
# Difficulty Matching
# ============================================================

def match_difficulty(
    prediction_boxes,
    prediction_scores,
    gt_boxes,
    gt_face_indices,
    gt_levels,
    difficulty,
    relative_path,
):
    """Predictionと対象GTを1対1対応し、対象外GTはIGNOREにします。"""
    target_mask = np.asarray(
        [difficulty in levels for levels in gt_levels],
        dtype=bool,
    )

    target_boxes = gt_boxes[target_mask]
    target_indices = gt_face_indices[target_mask]
    ignore_boxes = gt_boxes[~target_mask]
    ignore_indices = gt_face_indices[~target_mask]

    matched_targets = set()
    rows = []

    for prediction_index in np.argsort(prediction_scores)[::-1]:
        box = prediction_boxes[prediction_index]
        status = "FP"
        matched_face_index = ""
        matched_iou = 0.0

        if len(target_boxes) > 0:
            overlaps = iou_matrix(box[None], target_boxes)[0]
            target_index = int(np.argmax(overlaps))
            target_iou = float(overlaps[target_index])

            if target_iou >= MATCH_IOU_THRESHOLD:
                matched_face_index = int(target_indices[target_index])
                matched_iou = target_iou

                if target_index not in matched_targets:
                    status = "TP"
                    matched_targets.add(target_index)

        if status == "FP" and matched_iou < MATCH_IOU_THRESHOLD and len(ignore_boxes) > 0:
            overlaps = iou_matrix(box[None], ignore_boxes)[0]
            ignore_index = int(np.argmax(overlaps))
            ignore_iou = float(overlaps[ignore_index])

            if ignore_iou >= MATCH_IOU_THRESHOLD:
                status = "IGNORE"
                matched_face_index = int(ignore_indices[ignore_index])
                matched_iou = ignore_iou

        rows.append({
            "difficulty": difficulty,
            "image_relative_path": relative_path,
            "record_type": "PREDICTION",
            "prediction_index": int(prediction_index),
            "score": float(prediction_scores[prediction_index]),
            "status": status,
            "matched_face_index": matched_face_index,
            "matched_iou": matched_iou,
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
        })

    for target_index, box in enumerate(target_boxes):
        if target_index in matched_targets:
            continue

        rows.append({
            "difficulty": difficulty,
            "image_relative_path": relative_path,
            "record_type": "GROUND_TRUTH",
            "prediction_index": "",
            "score": "",
            "status": "FN",
            "matched_face_index": int(target_indices[target_index]),
            "matched_iou": 0.0,
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
        })

    return rows, len(target_boxes)


# ============================================================
# Metrics
# ============================================================

def calculate_ap(recalls, precisions):
    """Precision-Recall曲線の包絡線からAPを計算します。"""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    points = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[points + 1] - recalls[points]) * precisions[points + 1]))


def evaluate_rows(rows, num_gt):
    """IGNOREとFN行を除外してAPを計算し、固定閾値の指標も返します。"""
    predictions = [
        row for row in rows
        if row["record_type"] == "PREDICTION" and row["status"] != "IGNORE"
    ]
    predictions.sort(key=lambda row: row["score"], reverse=True)

    tp_flags = np.asarray([row["status"] == "TP" for row in predictions], dtype=float)
    fp_flags = 1.0 - tp_flags
    cumulative_tp = np.cumsum(tp_flags)
    cumulative_fp = np.cumsum(fp_flags)
    recalls = cumulative_tp / max(num_gt, 1)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    ap_50 = calculate_ap(recalls, precisions) if predictions else 0.0

    selected = [
        row for row in rows
        if row["record_type"] == "PREDICTION"
        and row["score"] >= METRIC_CONFIDENCE_THRESHOLD
    ]
    tp = sum(row["status"] == "TP" for row in selected)
    fp = sum(row["status"] == "FP" for row in selected)
    ignored = sum(row["status"] == "IGNORE" for row in selected)
    fn = num_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(num_gt, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    metrics = {
        "ap_50": ap_50,
        "num_ground_truths": num_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap_confidence_threshold": AP_CONFIDENCE_THRESHOLD,
        "metric_confidence_threshold": METRIC_CONFIDENCE_THRESHOLD,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
    }

    curve = [
        {
            "rank": index + 1,
            "score": predictions[index]["score"],
            "precision": float(precisions[index]),
            "recall": float(recalls[index]),
        }
        for index in range(len(predictions))
    ]
    return metrics, curve


# ============================================================
# Output
# ============================================================

def write_csv(path, rows, fieldnames):
    """CSVをUTF-8 BOM付きで保存します。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():
    """Validation全画像を推論し、難易度別の評価結果を保存します。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(ANNO_ROOT.rglob("*.json"))

    if MAX_IMAGES is not None:
        annotation_files = annotation_files[:MAX_IMAGES]

    session = ort.InferenceSession(str(MODEL_PATH), providers=PROVIDERS)
    input_name = session.get_inputs()[0].name

    rows = {difficulty: [] for difficulty in DIFFICULTIES}
    gt_counts = {difficulty: 0 for difficulty in DIFFICULTIES}

    for annotation_path in tqdm(annotation_files, desc="ONNX evaluation"):
        image, relative_path, gt_boxes, gt_face_indices, gt_levels = load_sample(
            annotation_path
        )
        width, height = image.size
        raw_output = session.run(None, {input_name: preprocess(image)})[0]
        prediction_boxes, prediction_scores = postprocess(raw_output, width, height)

        for difficulty in DIFFICULTIES:
            image_rows, num_gt = match_difficulty(
                prediction_boxes,
                prediction_scores,
                gt_boxes,
                gt_face_indices,
                gt_levels,
                difficulty,
                relative_path,
            )
            rows[difficulty].extend(image_rows)
            gt_counts[difficulty] += num_gt

    summary = []

    for difficulty in DIFFICULTIES:
        output_directory = OUTPUT_ROOT / difficulty
        output_directory.mkdir(parents=True, exist_ok=True)
        metrics, curve = evaluate_rows(rows[difficulty], gt_counts[difficulty])
        summary.append({"difficulty": difficulty, **metrics})

        with open(output_directory / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

        write_csv(
            output_directory / "prediction_details.csv",
            rows[difficulty],
            [
                "difficulty", "image_relative_path", "record_type",
                "prediction_index", "score", "status", "matched_face_index",
                "matched_iou", "x1", "y1", "x2", "y2",
            ],
        )
        write_csv(
            output_directory / "precision_recall_curve.csv",
            curve,
            ["rank", "score", "precision", "recall"],
        )

    write_csv(
        OUTPUT_ROOT / "metrics_summary.csv",
        summary,
        [
            "difficulty", "ap_50", "num_ground_truths", "tp", "fp", "fn",
            "ignored_predictions", "precision", "recall", "f1",
            "ap_confidence_threshold", "metric_confidence_threshold",
            "match_iou_threshold", "nms_iou_threshold",
        ],
    )

    with open(OUTPUT_ROOT / "metrics_all.json", "w", encoding="utf-8") as file:
        json.dump(
            {row["difficulty"]: {k: v for k, v in row.items() if k != "difficulty"} for row in summary},
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()