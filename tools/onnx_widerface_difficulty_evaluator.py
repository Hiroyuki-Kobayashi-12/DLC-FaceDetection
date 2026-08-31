# tools/onnx_widerface_difficulty_evaluator.py
# ============================================================
# WIDER FACE ValidationへONNX推論を行い、Easy、Medium、Hardごとに
# 顔検出モデルを数値評価し、評価指標を教材用グラフへ可視化します。
#
# 物体検出では、クラスだけでなく位置も正しい必要があります。
# PredictionとGround Truthの重なりをIoUで調べ、1対1で対応させます。
#
# IoU = 交差領域 / 和集合領域
#   0は重なりなし、1は完全一致です。
#   本評価ではIoU 0.5以上を位置が一致した検出とします。
#
# TP: 対象難易度GTを検出できた
# FP: 顔でない場所、位置ずれ、または同じGTを重複検出した
# FN: 対象難易度GTを検出できなかった
# IGNORE: 対象外難易度の実在する顔を検出したため評価から除外する
#
# Precision = TP / (TP + FP)
#   検出結果のうち正しかった割合です。高いほど誤検出が少なくなります。
#
# Recall = TP / (TP + FN)
#   評価対象の顔を検出できた割合です。高いほど見逃しが少なくなります。
#
# F1 = 2 * Precision * Recall / (Precision + Recall)
#   PrecisionとRecallのバランスを1つの値で示します。
#
# AP@0.5:
#   IoU 0.5を正解条件としたPrecision-Recall曲線の面積です。
#   Confidence閾値を1つに固定せず、検出順位全体を評価します。
#
# mAP:
#   複数クラスまたは複数IoU条件のAPを平均した値です。
#   今回は顔1クラス、IoU 0.5固定なので、AP@0.5は1クラスにおける
#   mAP@0.5と同値です。COCO形式のmAP@0.5:0.95ではありません。
#
# 難易度別評価:
#   各画像を一度だけ推論し、同じPredictionを3難易度で評価します。
#   対象外難易度の有効顔に一致したPredictionはIGNOREにします。
#
# 検出結果行列:
#   物体検出では背景候補の総数を定義できないためTNは計算しません。
#   TP、FP、FNを表示し、TN相当セルにはN/Aを表示します。
#
# Dataset構造:
#   data/widerface_pytorch_json/val/
#   |-- image/<event>/<image>.jpg
#   `-- anno/<event>/<image>.json
#
# 出力:
#   results/
#   |-- metrics_all.json
#   `-- visualization/
#       |-- ap50_by_difficulty.png
#       |-- precision_recall_by_difficulty.png
#       |-- metrics_by_difficulty.png
#       |-- detection_matrix_easy.png
#       |-- detection_matrix_medium.png
#       `-- detection_matrix_hard.png
# ============================================================

from pathlib import Path
import json

import matplotlib.pyplot as plt
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
RESULTS_ROOT = PROJECT_ROOT / "results"
VISUALIZATION_ROOT = RESULTS_ROOT / "visualization"
METRICS_PATH = RESULTS_ROOT / "metrics_all.json"

DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_LABELS = ("Easy", "Medium", "Hard")
IMAGE_SIZE = 640
AP_CONFIDENCE_THRESHOLD = 0.001
METRIC_CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
MAX_IMAGES = None
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ============================================================
# Data
# ============================================================

def load_annotation(path):
    """画像1枚分のJSONアノテーションを読み込みます。"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_sample(annotation_path):
    """元画像、相対パス、全有効GT、難易度を返します。"""
    annotation = load_annotation(annotation_path)
    relative_path = annotation["image"]["relative_path"]
    image = Image.open(IMAGE_ROOT / relative_path).convert("RGB")

    boxes = []
    levels = []

    # 全有効GTを保持し、評価時に対象GTとIGNORE対象GTへ分けます。
    for face in annotation["faces"]:
        box = face["bbox"]["xyxy"]

        if not face["valid_bbox"] or box is None:
            continue

        boxes.append(box)
        levels.append(tuple(face["levels"]))

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return image, relative_path, boxes, levels


# ============================================================
# ONNX Inference
# ============================================================

def preprocess(image):
    """RGB画像をモデル入力用のNCHW float32へ変換します。"""
    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = image_array.transpose(2, 0, 1)
    image_array = image_array[None]
    return np.ascontiguousarray(image_array)


def xywh_to_xyxy(boxes):
    """中心座標xywhを左上・右下座標xyxyへ変換します。"""
    converted_boxes = np.zeros_like(boxes, dtype=np.float32)

    for index, box in enumerate(boxes):
        center_x, center_y, width, height = box
        converted_boxes[index] = [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ]

    return converted_boxes


def calculate_iou(box_a, box_b):
    """2つのxyxy bbox間のIoUを、式の順番どおりに計算します。"""
    # 交差領域の左上座標を求めます。
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])

    # 交差領域の右下座標を求めます。
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])

    # 重なりがない場合、幅または高さは0になります。
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    # 2つのbboxそれぞれの面積を求めます。
    box_a_width = max(0.0, box_a[2] - box_a[0])
    box_a_height = max(0.0, box_a[3] - box_a[1])
    box_a_area = box_a_width * box_a_height

    box_b_width = max(0.0, box_b[2] - box_b[0])
    box_b_height = max(0.0, box_b[3] - box_b[1])
    box_b_area = box_b_width * box_b_height

    # 和集合は、2つの面積から重複分を1回引いて求めます。
    union_area = box_a_area + box_b_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def find_best_match(prediction_box, ground_truth_boxes):
    """Predictionと最もIoUが高いGTの番号とIoUを返します。"""
    best_index = -1
    best_iou = 0.0

    for gt_index, ground_truth_box in enumerate(ground_truth_boxes):
        iou = calculate_iou(prediction_box, ground_truth_box)

        if iou > best_iou:
            best_index = gt_index
            best_iou = iou

    return best_index, best_iou


def nms(boxes, scores):
    """高Confidenceのbboxから順番に、重複候補を除去します。"""
    # Confidenceが高い順にPrediction番号を並べます。
    sorted_indices = list(np.argsort(scores)[::-1])
    kept_indices = []

    while sorted_indices and len(kept_indices) < MAX_DETECTIONS:
        # 現在最もConfidenceが高いPredictionを採用します。
        current_index = sorted_indices.pop(0)
        kept_indices.append(current_index)

        remaining_indices = []

        # 残りのPredictionを1つずつ比較します。
        for other_index in sorted_indices:
            iou = calculate_iou(boxes[current_index], boxes[other_index])

            # 重なりが閾値未満なら、別の検出候補として残します。
            if iou < NMS_IOU_THRESHOLD:
                remaining_indices.append(other_index)

        sorted_indices = remaining_indices

    return np.asarray(kept_indices, dtype=np.int64)


def postprocess(raw_output, original_width, original_height):
    """Confidence抽出、NMS、元画像座標への復元を行います。"""
    predictions = np.asarray(raw_output)

    if predictions.ndim == 3:
        predictions = predictions[0]

    boxes = predictions[:, :4]

    # YOLOv5の最終ConfidenceはObjectnessと顔クラス確率の積です。
    objectness_scores = predictions[:, 4]
    face_class_scores = predictions[:, 5]
    scores = objectness_scores * face_class_scores

    confidence_mask = scores >= AP_CONFIDENCE_THRESHOLD
    boxes = boxes[confidence_mask]
    scores = scores[confidence_mask]

    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    boxes = xywh_to_xyxy(boxes)
    kept_indices = nms(boxes, scores)
    boxes = boxes[kept_indices]
    scores = scores[kept_indices]

    width_scale = original_width / float(IMAGE_SIZE)
    height_scale = original_height / float(IMAGE_SIZE)

    for box in boxes:
        box[0] = max(0.0, min(box[0] * width_scale, original_width))
        box[1] = max(0.0, min(box[1] * height_scale, original_height))
        box[2] = max(0.0, min(box[2] * width_scale, original_width))
        box[3] = max(0.0, min(box[3] * height_scale, original_height))

    return boxes.astype(np.float32), scores.astype(np.float32)


# ============================================================
# Difficulty Matching
# ============================================================

def match_difficulty(
    prediction_boxes,
    prediction_scores,
    gt_boxes,
    gt_levels,
    difficulty,
):
    """PredictionをTP、FP、IGNOREへ分類し、対象GT数を返します。"""
    target_boxes = []
    ignore_boxes = []

    # 現在の難易度に含まれるGTと、対象外のGTへ分けます。
    for gt_box, levels in zip(gt_boxes, gt_levels):
        if difficulty in levels:
            target_boxes.append(gt_box)
        else:
            ignore_boxes.append(gt_box)

    matched_target_indices = set()
    prediction_records = []
    sorted_prediction_indices = np.argsort(prediction_scores)[::-1]

    # Confidenceが高いPredictionから順番にGTへ対応させます。
    for prediction_index in sorted_prediction_indices:
        prediction_box = prediction_boxes[prediction_index]
        status = "FP"

        # 最初に現在の難易度のGTと照合します。
        target_index, target_iou = find_best_match(prediction_box, target_boxes)

        if (
            target_index >= 0
            and target_iou >= MATCH_IOU_THRESHOLD
            and target_index not in matched_target_indices
        ):
            status = "TP"
            matched_target_indices.add(target_index)

        # 対象GTへ一致しなかったPredictionだけを対象外GTと照合します。
        if status == "FP":
            _, ignore_iou = find_best_match(prediction_box, ignore_boxes)

            if ignore_iou >= MATCH_IOU_THRESHOLD:
                status = "IGNORE"

        prediction_records.append({
            "score": float(prediction_scores[prediction_index]),
            "status": status,
        })

    return prediction_records, len(target_boxes)


# ============================================================
# Metrics
# ============================================================

def calculate_ap(recalls, precisions):
    """Precision包絡線を作り、Recall増分ごとの面積を合計します。"""
    recall_values = [0.0] + list(recalls) + [1.0]
    precision_values = [0.0] + list(precisions) + [0.0]

    # 右側のPrecisionが高ければ、その値で左側を補間します。
    for index in range(len(precision_values) - 2, -1, -1):
        precision_values[index] = max(
            precision_values[index],
            precision_values[index + 1],
        )

    ap = 0.0

    # Recallが増えた区間の幅とPrecisionを掛けて面積を足します。
    for index in range(len(recall_values) - 1):
        recall_width = recall_values[index + 1] - recall_values[index]

        if recall_width > 0.0:
            ap += recall_width * precision_values[index + 1]

    return float(ap)


def evaluate_records(records, num_gt):
    """AP@0.5と固定Confidence閾値の検出指標を計算します。"""
    # IGNOREは正解にも誤りにも数えません。
    predictions = [
        record
        for record in records
        if record["status"] != "IGNORE"
    ]
    predictions.sort(key=lambda record: record["score"], reverse=True)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []

    # Confidence順にPredictionを追加しながらPR曲線を作ります。
    for prediction in predictions:
        if prediction["status"] == "TP":
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        recall = cumulative_tp / max(num_gt, 1)
        precision = cumulative_tp / max(cumulative_tp + cumulative_fp, 1)
        recalls.append(recall)
        precisions.append(precision)

    ap_50 = calculate_ap(recalls, precisions) if predictions else 0.0

    # 固定Confidence閾値以上のPredictionだけで運用点を評価します。
    selected_records = [
        record
        for record in records
        if record["score"] >= METRIC_CONFIDENCE_THRESHOLD
    ]

    tp = sum(record["status"] == "TP" for record in selected_records)
    fp = sum(record["status"] == "FP" for record in selected_records)
    ignored = sum(record["status"] == "IGNORE" for record in selected_records)
    fn = num_gt - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
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

    curve = {
        "recall": recalls,
        "precision": precisions,
    }

    return metrics, curve


# ============================================================
# Visualization
# ============================================================

def save_ap50_graph(metrics_all):
    """難易度別AP@0.5を棒グラフで比較します。"""
    values = [metrics_all[difficulty]["ap_50"] for difficulty in DIFFICULTIES]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(DIFFICULTY_LABELS, values)
    axis.set_title("AP@0.5 by Difficulty")
    axis.set_xlabel("Difficulty")
    axis.set_ylabel("AP@0.5")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y")

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(VISUALIZATION_ROOT / "ap50_by_difficulty.png", dpi=150)
    plt.close(figure)


def save_precision_recall_graph(curves):
    """PrecisionとRecallの関係を難易度別に描きます。"""
    figure, axis = plt.subplots(figsize=(8, 6))

    for difficulty, label in zip(DIFFICULTIES, DIFFICULTY_LABELS):
        axis.plot(
            curves[difficulty]["recall"],
            curves[difficulty]["precision"],
            label=label,
        )

    axis.set_title("Precision-Recall Curve by Difficulty")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(VISUALIZATION_ROOT / "precision_recall_by_difficulty.png", dpi=150)
    plt.close(figure)


def save_metric_graph(metrics_all):
    """固定Confidence閾値のPrecision、Recall、F1を比較します。"""
    x_positions = np.arange(len(DIFFICULTIES))
    bar_width = 0.24
    precision_values = [metrics_all[d]["precision"] for d in DIFFICULTIES]
    recall_values = [metrics_all[d]["recall"] for d in DIFFICULTIES]
    f1_values = [metrics_all[d]["f1"] for d in DIFFICULTIES]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x_positions - bar_width, precision_values, bar_width, label="Precision")
    axis.bar(x_positions, recall_values, bar_width, label="Recall")
    axis.bar(x_positions + bar_width, f1_values, bar_width, label="F1")
    axis.set_title(
        f"Metrics by Difficulty at Confidence {METRIC_CONFIDENCE_THRESHOLD}"
    )
    axis.set_xlabel("Difficulty")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions, DIFFICULTY_LABELS)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(VISUALIZATION_ROOT / "metrics_by_difficulty.png", dpi=150)
    plt.close(figure)


def save_detection_matrix(metrics, difficulty):
    """TP、FP、FNを白から青の濃淡で表示します。"""
    matrix = np.asarray([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], 0],
    ], dtype=float)

    figure, axis = plt.subplots(figsize=(6, 5))

    # Bluesは0に近い値を白、大きい値を濃い青で表示します。
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=max(float(np.max(matrix)), 1.0),
    )

    axis.set_title(f"Detection Outcome Matrix: {difficulty.title()}")
    axis.set_xlabel("Evaluation Outcome")
    axis.set_ylabel("Record Source")
    axis.set_xticks([0, 1], ["Correct", "Error"])
    axis.set_yticks([0, 1], ["Prediction", "Ground Truth"])

    labels = [
        [f"TP\n{metrics['tp']}", f"FP\n{metrics['fp']}"],
        [f"FN\n{metrics['fn']}", "TN\nN/A"],
    ]

    # 背景が濃いセルでは白文字、薄いセルでは黒文字にします。
    color_threshold = max(float(np.max(matrix)), 1.0) / 2.0

    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            text_color = "white" if value > color_threshold else "black"

            axis.text(
                column,
                row,
                labels[row][column],
                ha="center",
                va="center",
                color=text_color,
            )

    axis.text(
        0.5,
        -0.15,
        "TN is not defined for object detection.",
        transform=axis.transAxes,
        ha="center",
    )
    figure.colorbar(image, ax=axis, label="Count")
    figure.tight_layout()
    figure.savefig(
        VISUALIZATION_ROOT / f"detection_matrix_{difficulty}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_visualizations(metrics_all, curves):
    """数値評価から教材用グラフを保存します。"""
    VISUALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    save_ap50_graph(metrics_all)
    save_precision_recall_graph(curves)
    save_metric_graph(metrics_all)

    for difficulty in DIFFICULTIES:
        save_detection_matrix(metrics_all[difficulty], difficulty)


# ============================================================
# Main
# ============================================================

def main():
    """数値評価をJSONへ保存し、教材用グラフを作成します。"""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(ANNO_ROOT.rglob("*.json"))

    if MAX_IMAGES is not None:
        annotation_files = annotation_files[:MAX_IMAGES]

    session = ort.InferenceSession(str(MODEL_PATH), providers=PROVIDERS)
    input_name = session.get_inputs()[0].name
    records = {difficulty: [] for difficulty in DIFFICULTIES}
    gt_counts = {difficulty: 0 for difficulty in DIFFICULTIES}

    for annotation_path in tqdm(annotation_files, desc="ONNX evaluation"):
        image, _, gt_boxes, gt_levels = load_sample(annotation_path)
        original_width, original_height = image.size

        # 画像1枚につきONNX推論は1回だけ行います。
        model_input = preprocess(image)
        raw_output = session.run(None, {input_name: model_input})[0]
        prediction_boxes, prediction_scores = postprocess(
            raw_output,
            original_width,
            original_height,
        )

        # 同じPredictionを使い、対象GTだけを難易度ごとに切り替えます。
        for difficulty in DIFFICULTIES:
            image_records, num_gt = match_difficulty(
                prediction_boxes,
                prediction_scores,
                gt_boxes,
                gt_levels,
                difficulty,
            )
            records[difficulty].extend(image_records)
            gt_counts[difficulty] += num_gt

    metrics_all = {}
    curves = {}

    for difficulty in DIFFICULTIES:
        metrics, curve = evaluate_records(
            records[difficulty],
            gt_counts[difficulty],
        )
        metrics_all[difficulty] = metrics
        curves[difficulty] = curve

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_all, file, ensure_ascii=False, indent=2)

    save_visualizations(metrics_all, curves)


if __name__ == "__main__":
    main()
