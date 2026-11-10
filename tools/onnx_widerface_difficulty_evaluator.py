# tools/onnx_widerface_difficulty_evaluator.py
# ============================================================
# WIDER FACE ValidationをONNXモデルで推論し、Easy / Medium / Hardごとに
# 数値評価、全体ダッシュボード、苦手画像ランキングを出力します。
#
# 主な出力:
#   results/metrics_all.json
#   results/visualization/evaluation_dashboard.png
#   results/visualization/precision_recall_by_difficulty.png
#   results/visualization/detection_matrix_<difficulty>.png
#   results/visualization/difficult_images/summary.json
#   results/visualization/difficult_images/ranked/<difficulty>/...
#   results/visualization/difficult_images/focus/...
#
# 難易度別の判定規則:
#   対象難易度の未対応GTとIoU >= 0.50  -> TP
#   対象難易度の対応済みGTとIoU >= 0.50 -> FP（重複検出）
#   対象GTとのIoU < 0.50 かつ対象外GTとIoU >= 0.50 -> IGNORE
#   どのGTともIoU < 0.50 -> FP
#   最後まで未対応の対象GT -> FN
#
# valid_bbox=Falseでも利用可能なbboxがある顔はIGNORE領域として扱います。
# bboxが壊れている、または存在しない顔はマッチングに使用しません。
# ============================================================

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
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
DIFFICULT_ROOT = VISUALIZATION_ROOT / "difficult_images"
METRICS_PATH = RESULTS_ROOT / "metrics_all.json"
DIFFICULT_SUMMARY_PATH = DIFFICULT_ROOT / "summary.json"

DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_LABELS = ("Easy", "Medium", "Hard")
IMAGE_SIZE = 640
AP_CONFIDENCE_THRESHOLD = 0.001
METRIC_CONFIDENCE_THRESHOLD = 0.75
NMS_IOU_THRESHOLD = 0.45
MATCH_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
MAX_IMAGES = None
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 苦手画像ランキングは3観点を分け、各観点の代表画像だけを出力します。
RANKING_TOP_COUNT = 15
FN_RATE_MIN_GT_COUNT = 3
SMALL_SUPPORT_MAX_GT_COUNT = 2
FOCUS_MAX_IMAGES_PER_PATH = 50
LABEL_VISIBLE_MAX_FACE_COUNT = 30
CONTACT_SHEET_COLUMNS = 5
CONTACT_SHEET_THUMBNAIL_SIZE = (320, 240)

# eventフォルダ名または画像相対パスを指定します。空配列なら重点出力しません。
# 例: ["0--Parade", "2--Demonstration/ なっちゃってたんで、ああ、なんだこれ、音声が入ってきちゃってる、すいません、ええ、止まんないんだけどこれ。2_Demonstration_Demonstrators_2_1.jpg"]
FOCUS_PATHS = ["0--Parade"]

# 描画色はRGBです。
COLOR_TP_GT = (25, 170, 90)
COLOR_TP_PREDICTION = (40, 210, 125)
COLOR_FN_GT = (230, 55, 55)
COLOR_FP_PREDICTION = (35, 115, 220)
COLOR_IGNORE_PREDICTION = (240, 185, 20)
COLOR_IGNORED_GT = (150, 150, 150)


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class GroundTruth:
    """元画像座標のGT bboxと評価情報です。"""

    box: tuple[float, float, float, float]
    levels: tuple[str, ...]
    valid_bbox: bool
    face_index: int


@dataclass(frozen=True)
class PredictionMatch:
    """Prediction 1件の難易度別マッチング結果です。"""

    prediction_index: int
    box: tuple[float, float, float, float]
    score: float
    status: str
    matched_target_index: int | None = None
    matched_ignore_index: int | None = None
    matched_iou: float = 0.0


@dataclass
class ImageDifficultyResult:
    """画像1枚・難易度1種類の評価結果と描画情報です。"""

    relative_path: str
    difficulty: str
    image_width: int
    image_height: int
    target_ground_truths: list[GroundTruth]
    ignore_ground_truths: list[GroundTruth]
    prediction_matches: list[PredictionMatch]
    matched_target_indices: set[int]
    metric_confidence_threshold: float
    metrics: dict = field(default_factory=dict)


# ============================================================
# Data Loading
# ============================================================

def load_annotation(path):
    """画像1枚分のJSONアノテーションを読み込みます。"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_valid_box(box):
    """bboxが4座標を持ち、正の幅と高さを持つか確認します。"""
    if box is None or len(box) != 4:
        return False

    x1, y1, x2, y2 = [float(value) for value in box]
    return x2 > x1 and y2 > y1


def load_ground_truths(annotation):
    """評価対象GTとIGNORE候補の両方を保持します。"""
    ground_truths = []

    for fallback_index, face in enumerate(annotation["faces"]):
        box = face.get("bbox", {}).get("xyxy")

        # 座標自体が壊れている顔は、安全に照合できないため使用しません。
        if not is_valid_box(box):
            continue

        ground_truths.append(
            GroundTruth(
                box=tuple(float(value) for value in box),
                levels=tuple(face.get("levels", [])),
                valid_bbox=bool(face.get("valid_bbox", False)),
                face_index=int(face.get("face_index", fallback_index)),
            )
        )

    return ground_truths


def load_sample(annotation_path):
    """元画像、相対パス、全GT情報を返します。"""
    annotation = load_annotation(annotation_path)
    relative_path = annotation["image"]["relative_path"]
    image = Image.open(IMAGE_ROOT / relative_path).convert("RGB")
    ground_truths = load_ground_truths(annotation)
    return image, relative_path, ground_truths


# ============================================================
# ONNX Inference
# ============================================================

def preprocess(image):
    """RGB画像をモデル入力用のNCHW float32へ変換します。"""
    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    image_array = image_array.transpose(2, 0, 1)[None]
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
    """2つのxyxy bbox間のIoUを順番どおりに計算します。"""
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    box_a_area = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    box_b_area = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def find_best_match(prediction_box, ground_truths):
    """Predictionと最もIoUが高いGTの番号とIoUを返します。"""
    best_index = -1
    best_iou = 0.0

    for gt_index, ground_truth in enumerate(ground_truths):
        iou = calculate_iou(prediction_box, ground_truth.box)

        if iou > best_iou:
            best_index = gt_index
            best_iou = iou

    return best_index, best_iou


def nms(boxes, scores):
    """高Confidenceのbboxから順に重複候補を除去します。"""
    sorted_indices = list(np.argsort(scores)[::-1])
    kept_indices = []

    while sorted_indices and len(kept_indices) < MAX_DETECTIONS:
        current_index = int(sorted_indices.pop(0))
        kept_indices.append(current_index)
        remaining_indices = []

        for other_index in sorted_indices:
            iou = calculate_iou(boxes[current_index], boxes[other_index])

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
    scores = predictions[:, 4] * predictions[:, 5]
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

def split_ground_truths(ground_truths, difficulty):
    """現在の難易度の対象GTとIGNORE対象GTへ分けます。"""
    target_ground_truths = []
    ignore_ground_truths = []

    for ground_truth in ground_truths:
        if ground_truth.valid_bbox and difficulty in ground_truth.levels:
            target_ground_truths.append(ground_truth)
        else:
            # 別難易度またはvalid_bbox=Falseでも、利用可能なbboxはIGNORE領域です。
            ignore_ground_truths.append(ground_truth)

    return target_ground_truths, ignore_ground_truths


def match_prediction(
    prediction_index,
    prediction_box,
    prediction_score,
    target_ground_truths,
    ignore_ground_truths,
    matched_target_indices,
):
    """Prediction 1件をTP・FP・IGNOREのいずれかへ分類します。"""
    target_index, target_iou = find_best_match(
        prediction_box,
        target_ground_truths,
    )

    # 対象GTと一致した場合は、そのGTが対応済みでもIGNORE判定へ進めません。
    # 未対応ならTP、対応済みなら重複検出としてFPに確定します。
    if target_iou >= MATCH_IOU_THRESHOLD:
        if target_index not in matched_target_indices:
            matched_target_indices.add(target_index)
            return PredictionMatch(
                prediction_index=prediction_index,
                box=tuple(float(value) for value in prediction_box),
                score=float(prediction_score),
                status="TP",
                matched_target_index=target_index,
                matched_iou=float(target_iou),
            )

        return PredictionMatch(
            prediction_index=prediction_index,
            box=tuple(float(value) for value in prediction_box),
            score=float(prediction_score),
            status="FP",
            matched_target_index=target_index,
            matched_iou=float(target_iou),
        )

    ignore_index, ignore_iou = find_best_match(
        prediction_box,
        ignore_ground_truths,
    )

    if ignore_iou >= MATCH_IOU_THRESHOLD:
        return PredictionMatch(
            prediction_index=prediction_index,
            box=tuple(float(value) for value in prediction_box),
            score=float(prediction_score),
            status="IGNORE",
            matched_ignore_index=ignore_index,
            matched_iou=float(ignore_iou),
        )

    return PredictionMatch(
        prediction_index=prediction_index,
        box=tuple(float(value) for value in prediction_box),
        score=float(prediction_score),
        status="FP",
        matched_iou=float(max(target_iou, ignore_iou)),
    )


def calculate_image_metrics(result):
    """固定Confidence閾値で画像単位の指標を計算します。"""
    selected_matches = [
        match
        for match in result.prediction_matches
        if match.score >= result.metric_confidence_threshold
    ]
    tp_matches = [match for match in selected_matches if match.status == "TP"]
    fp_matches = [match for match in selected_matches if match.status == "FP"]
    ignored_matches = [match for match in selected_matches if match.status == "IGNORE"]

    # 低ConfidenceのTPは固定閾値では採用されないため、そのGTはFNです。
    selected_matched_targets = {
        match.matched_target_index
        for match in tp_matches
        if match.matched_target_index is not None
    }

    target_count = len(result.target_ground_truths)
    tp = len(selected_matched_targets)
    fp = len(fp_matches)
    fn = target_count - tp
    ignored = len(ignored_matches)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(target_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    fn_rate = fn / max(target_count, 1)
    mean_tp_iou = float(np.mean([match.matched_iou for match in tp_matches])) if tp_matches else 0.0
    mean_tp_confidence = float(np.mean([match.score for match in tp_matches])) if tp_matches else 0.0

    return {
        "num_ground_truths": target_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fn_rate": fn_rate,
        "mean_tp_iou": mean_tp_iou,
        "mean_tp_confidence": mean_tp_confidence,
        "selected_matched_target_indices": sorted(selected_matched_targets),
    }


def match_difficulty(
    relative_path,
    image_width,
    image_height,
    prediction_boxes,
    prediction_scores,
    ground_truths,
    difficulty,
):
    """画像1枚について難易度別のマッチング結果を作成します。"""
    target_ground_truths, ignore_ground_truths = split_ground_truths(
        ground_truths,
        difficulty,
    )
    matched_target_indices = set()
    prediction_matches = []
    sorted_prediction_indices = np.argsort(prediction_scores)[::-1]

    for prediction_index in sorted_prediction_indices:
        prediction_match = match_prediction(
            prediction_index=int(prediction_index),
            prediction_box=prediction_boxes[prediction_index],
            prediction_score=prediction_scores[prediction_index],
            target_ground_truths=target_ground_truths,
            ignore_ground_truths=ignore_ground_truths,
            matched_target_indices=matched_target_indices,
        )
        prediction_matches.append(prediction_match)

    result = ImageDifficultyResult(
        relative_path=relative_path,
        difficulty=difficulty,
        image_width=image_width,
        image_height=image_height,
        target_ground_truths=target_ground_truths,
        ignore_ground_truths=ignore_ground_truths,
        prediction_matches=prediction_matches,
        matched_target_indices=matched_target_indices,
        metric_confidence_threshold=METRIC_CONFIDENCE_THRESHOLD,
    )
    result.metrics = calculate_image_metrics(result)
    return result


# ============================================================
# Dataset Metrics
# ============================================================

def calculate_ap(recalls, precisions):
    """Precision包絡線からAPを計算します。"""
    recall_values = [0.0] + list(recalls) + [1.0]
    precision_values = [0.0] + list(precisions) + [0.0]

    for index in range(len(precision_values) - 2, -1, -1):
        precision_values[index] = max(
            precision_values[index],
            precision_values[index + 1],
        )

    ap = 0.0

    for index in range(len(recall_values) - 1):
        recall_width = recall_values[index + 1] - recall_values[index]

        if recall_width > 0.0:
            ap += recall_width * precision_values[index + 1]

    return float(ap)


def evaluate_difficulty_results(image_results):
    """画像単位結果を集約し、難易度全体の指標とPR曲線を返します。"""
    all_matches = []
    num_gt = 0

    for result in image_results:
        all_matches.extend(result.prediction_matches)
        num_gt += len(result.target_ground_truths)

    ap_predictions = [match for match in all_matches if match.status != "IGNORE"]
    ap_predictions.sort(key=lambda match: match.score, reverse=True)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []

    for prediction in ap_predictions:
        if prediction.status == "TP":
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        recalls.append(cumulative_tp / max(num_gt, 1))
        precisions.append(cumulative_tp / max(cumulative_tp + cumulative_fp, 1))

    selected = [
        match
        for match in all_matches
        if match.score >= METRIC_CONFIDENCE_THRESHOLD
    ]
    tp = sum(match.status == "TP" for match in selected)
    fp = sum(match.status == "FP" for match in selected)
    ignored = sum(match.status == "IGNORE" for match in selected)
    fn = num_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(num_gt, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    metrics = {
        "ap_50": calculate_ap(recalls, precisions) if ap_predictions else 0.0,
        "num_ground_truths": num_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_100_gt": fp / max(num_gt, 1) * 100.0,
        "fn_per_100_gt": fn / max(num_gt, 1) * 100.0,
        "ap_confidence_threshold": AP_CONFIDENCE_THRESHOLD,
        "metric_confidence_threshold": METRIC_CONFIDENCE_THRESHOLD,
        "match_iou_threshold": MATCH_IOU_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
    }
    curve = {"recall": recalls, "precision": precisions}
    return metrics, curve


# ============================================================
# Ranking
# ============================================================

def rank_results(image_results, ranking_name, top_count=RANKING_TOP_COUNT):
    """指定観点で苦手画像を順位付けします。"""
    if ranking_name == "fn_rate":
        candidates = [
            result
            for result in image_results
            if result.metrics["num_ground_truths"] >= FN_RATE_MIN_GT_COUNT
        ]
        key = lambda result: (
            result.metrics["fn_rate"],
            result.metrics["fn"],
            result.metrics["fp"],
            -result.metrics["mean_tp_iou"],
        )
    elif ranking_name == "fn_count":
        candidates = [result for result in image_results if result.metrics["num_ground_truths"] > 0]
        key = lambda result: (
            result.metrics["fn"],
            result.metrics["fn_rate"],
            result.metrics["fp"],
        )
    elif ranking_name == "fp_count":
        candidates = list(image_results)
        key = lambda result: (
            result.metrics["fp"],
            result.metrics["fn"],
            result.metrics["fn_rate"],
        )
    else:
        raise ValueError(f"Unknown ranking: {ranking_name}")

    return sorted(candidates, key=key, reverse=True)[:top_count]


def small_support_complete_misses(image_results):
    """GT 1〜2件で全件見逃した画像を別枠として返します。"""
    return [
        result
        for result in image_results
        if 1 <= result.metrics["num_ground_truths"] <= SMALL_SUPPORT_MAX_GT_COUNT
        and result.metrics["recall"] == 0.0
    ]


def is_focus_target(relative_path, focus_path):
    """eventフォルダ名または完全相対パスに一致するか判定します。"""
    normalized_relative = Path(relative_path).as_posix()
    normalized_focus = Path(focus_path).as_posix().rstrip("/")
    return normalized_relative == normalized_focus or normalized_relative.startswith(normalized_focus + "/")


def select_focus_results(image_results_by_difficulty):
    """Configで指定されたパスの画像を難易度別に抽出します。"""
    focus_results = {}

    for focus_path in FOCUS_PATHS:
        focus_results[focus_path] = {}

        for difficulty in DIFFICULTIES:
            matched_results = [
                result
                for result in image_results_by_difficulty[difficulty]
                if is_focus_target(result.relative_path, focus_path)
            ]
            focus_results[focus_path][difficulty] = matched_results[:FOCUS_MAX_IMAGES_PER_PATH]

    return focus_results


# ============================================================
# Drawing Helpers
# ============================================================

def safe_name(value):
    """ファイル名に使用しにくい文字を置換します。"""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "item"


def draw_dashed_rectangle(draw, box, color, width=3, dash=10):
    """Pillowで破線矩形を描きます。"""
    x1, y1, x2, y2 = [int(round(value)) for value in box]

    for start in range(x1, x2, dash * 2):
        draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=width)

    for start in range(y1, y2, dash * 2):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=width)


def draw_label(draw, position, text, color, font):
    """矩形付近へ読みやすいラベルを描きます。"""
    x, y = [int(round(value)) for value in position]
    text_box = draw.textbbox((x, y), text, font=font)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def selected_matches(result):
    """固定Confidence閾値以上のPredictionを返します。"""
    return [
        match
        for match in result.prediction_matches
        if match.score >= result.metric_confidence_threshold
    ]


def draw_result_boxes(image, result):
    """TP・FP・FN・IGNOREを元画像へ描画します。"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    show_labels = (
        len(result.target_ground_truths) + len(selected_matches(result))
        <= LABEL_VISIBLE_MAX_FACE_COUNT
    )
    matched_target_indices = set(result.metrics["selected_matched_target_indices"])

    for target_index, ground_truth in enumerate(result.target_ground_truths):
        if target_index in matched_target_indices:
            color = COLOR_TP_GT
            label = "TP GT"
        else:
            color = COLOR_FN_GT
            label = "FN GT"

        draw.rectangle(ground_truth.box, outline=color, width=3)

        if show_labels:
            draw_label(draw, (ground_truth.box[0], max(0, ground_truth.box[1] - 12)), label, color, font)

    # 評価対象外GTは薄い灰色で任意表示し、IGNOREの根拠を確認しやすくします。
    for ground_truth in result.ignore_ground_truths:
        draw_dashed_rectangle(draw, ground_truth.box, COLOR_IGNORED_GT, width=1, dash=6)

    for match in selected_matches(result):
        if match.status == "TP":
            color = COLOR_TP_PREDICTION
            label = f"TP {match.score:.2f} IoU {match.matched_iou:.2f}"
        elif match.status == "FP":
            color = COLOR_FP_PREDICTION
            label = f"FP {match.score:.2f}"
        else:
            color = COLOR_IGNORE_PREDICTION
            label = f"IGNORE {match.score:.2f}"

        draw_dashed_rectangle(draw, match.box, color, width=3, dash=8)

        if show_labels:
            draw_label(draw, (match.box[0], match.box[1]), label, color, font)

    return image


def add_information_panel(image, result, ranking_name=None, rank=None):
    """元画像を隠さないよう、画像上部へ評価情報パネルを追加します。"""
    panel_height = 92
    panel = Image.new("RGB", (image.width, panel_height), (12, 38, 62))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    ranking_text = ""

    if ranking_name is not None and rank is not None:
        ranking_text = f" | {ranking_name} Rank {rank:03d}"

    line_1 = f"{result.difficulty.title()}{ranking_text} | {result.relative_path}"
    line_2 = (
        f"GT {result.metrics['num_ground_truths']} | TP {result.metrics['tp']} | "
        f"FN {result.metrics['fn']} | FP {result.metrics['fp']} | "
        f"IGNORE {result.metrics['ignored_predictions']}"
    )
    line_3 = (
        f"Precision {result.metrics['precision']:.3f} | Recall {result.metrics['recall']:.3f} | "
        f"FN Rate {result.metrics['fn_rate']:.3f} | Mean TP IoU {result.metrics['mean_tp_iou']:.3f} | "
        f"Mean TP Confidence {result.metrics['mean_tp_confidence']:.3f}"
    )

    draw.text((12, 10), line_1, fill=(255, 255, 255), font=font)
    draw.text((12, 36), line_2, fill=(220, 235, 245), font=font)
    draw.text((12, 62), line_3, fill=(220, 235, 245), font=font)

    output = Image.new("RGB", (image.width, image.height + panel_height), (255, 255, 255))
    output.paste(panel, (0, 0))
    output.paste(image, (0, panel_height))
    return output


def render_result_image(result, save_path, ranking_name=None, rank=None):
    """元画像へ評価矩形と情報パネルを描き、保存します。"""
    image = Image.open(IMAGE_ROOT / result.relative_path).convert("RGB")
    image = draw_result_boxes(image, result)
    image = add_information_panel(image, result, ranking_name, rank)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path, quality=94)


# ============================================================
# Contact Sheets
# ============================================================

def thumbnail_with_caption(image_path, caption):
    """個別画像から説明付きサムネイルを作ります。"""
    width, height = CONTACT_SHEET_THUMBNAIL_SIZE
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((width, height - 35), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), (245, 248, 251))
    x = (width - image.width) // 2
    tile.paste(image, (x, 0))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((8, height - 27), caption, fill=(20, 45, 65), font=font)
    return tile


def create_contact_sheet(entries, save_path, title):
    """ランキング画像を一覧で比較できるコンタクトシートへまとめます。"""
    if not entries:
        return

    columns = CONTACT_SHEET_COLUMNS
    rows = math.ceil(len(entries) / columns)
    tile_width, tile_height = CONTACT_SHEET_THUMBNAIL_SIZE
    title_height = 60
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height + title_height),
        (230, 238, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((15, 20), title, fill=(10, 45, 75), font=font)

    for index, entry in enumerate(entries):
        tile = thumbnail_with_caption(
            entry["output_path"],
            f"Rank {entry['rank']:03d} | {entry['relative_path']}",
        )
        column = index % columns
        row = index // columns
        sheet.paste(tile, (column * tile_width, title_height + row * tile_height))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(save_path, quality=92)


# ============================================================
# Existing and Dashboard Visualizations
# ============================================================

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
    axis.grid(True, alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(VISUALIZATION_ROOT / "precision_recall_by_difficulty.png", dpi=160)
    plt.close(figure)


def save_detection_matrix(metrics, difficulty):
    """TP・FP・FNを白から青の濃淡で表示します。"""
    matrix = np.asarray([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], 0],
    ], dtype=float)
    figure, axis = plt.subplots(figsize=(6, 5))
    maximum = max(float(np.max(matrix)), 1.0)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
    axis.set_title(f"Detection Outcome Matrix: {difficulty.title()}")
    axis.set_xlabel("Evaluation Outcome")
    axis.set_ylabel("Record Source")
    axis.set_xticks([0, 1], ["Correct", "Error"])
    axis.set_yticks([0, 1], ["Prediction", "Ground Truth"])
    labels = [
        [f"TP\n{metrics['tp']:,}", f"FP\n{metrics['fp']:,}"],
        [f"FN\n{metrics['fn']:,}", "TN\nN/A"],
    ]

    for row in range(2):
        for column in range(2):
            text_color = "white" if matrix[row, column] > maximum / 2.0 else "black"
            axis.text(column, row, labels[row][column], ha="center", va="center", color=text_color)

    axis.text(0.5, -0.15, "TN is not defined for object detection.", transform=axis.transAxes, ha="center")
    figure.colorbar(image, ax=axis, label="Count")
    figure.tight_layout()
    figure.savefig(
        VISUALIZATION_ROOT / f"detection_matrix_{difficulty}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_evaluation_dashboard(metrics_all, curves):
    """難易度比較と正規化指標を1枚へまとめます。"""
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

    # 難易度別スコア
    axis_scores = figure.add_subplot(grid[0, 0])
    x = np.arange(len(DIFFICULTIES))
    width = 0.2
    for offset, metric, label in [
        (-1.5, "ap_50", "AP@0.5"),
        (-0.5, "precision", "Precision"),
        (0.5, "recall", "Recall"),
        (1.5, "f1", "F1"),
    ]:
        axis_scores.bar(
            x + offset * width,
            [metrics_all[difficulty][metric] for difficulty in DIFFICULTIES],
            width,
            label=label,
        )
    axis_scores.set_xticks(x, DIFFICULTY_LABELS)
    axis_scores.set_ylim(0.0, 1.0)
    axis_scores.set_title("Performance by Difficulty")
    axis_scores.grid(axis="y", alpha=0.3)
    axis_scores.legend(fontsize=8)

    # 検出件数
    axis_counts = figure.add_subplot(grid[0, 1])
    bottom = np.zeros(len(DIFFICULTIES))
    for metric, label, color in [
        ("tp", "TP", "#1C9B62"),
        ("fp", "FP", "#1F6EAA"),
        ("fn", "FN", "#D94B4B"),
        ("ignored_predictions", "IGNORE", "#E3A008"),
    ]:
        values = np.asarray([metrics_all[difficulty][metric] for difficulty in DIFFICULTIES])
        axis_counts.bar(DIFFICULTY_LABELS, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis_counts.set_title("Detection Counts")
    axis_counts.set_ylabel("Count")
    axis_counts.legend(fontsize=8)
    axis_counts.grid(axis="y", alpha=0.3)

    # 正規化指標
    axis_normalized = figure.add_subplot(grid[0, 2])
    axis_normalized.bar(x - 0.18, [metrics_all[d]["fp_per_100_gt"] for d in DIFFICULTIES], 0.36, label="FP / 100 GT")
    axis_normalized.bar(x + 0.18, [metrics_all[d]["fn_per_100_gt"] for d in DIFFICULTIES], 0.36, label="FN / 100 GT")
    axis_normalized.set_xticks(x, DIFFICULTY_LABELS)
    axis_normalized.set_title("Normalized Errors")
    axis_normalized.set_ylabel("Errors per 100 GT")
    axis_normalized.legend(fontsize=8)
    axis_normalized.grid(axis="y", alpha=0.3)

    # PR曲線
    axis_pr = figure.add_subplot(grid[1, :2])
    for difficulty, label in zip(DIFFICULTIES, DIFFICULTY_LABELS):
        axis_pr.plot(curves[difficulty]["recall"], curves[difficulty]["precision"], label=label)
    axis_pr.set_xlim(0, 1)
    axis_pr.set_ylim(0, 1)
    axis_pr.set_xlabel("Recall")
    axis_pr.set_ylabel("Precision")
    axis_pr.set_title("Precision-Recall Curves")
    axis_pr.grid(alpha=0.3)
    axis_pr.legend()

    # 数値表
    axis_table = figure.add_subplot(grid[1, 2])
    axis_table.axis("off")
    rows = []
    for difficulty in DIFFICULTIES:
        metrics = metrics_all[difficulty]
        rows.append([
            difficulty.title(),
            f"{metrics['ap_50']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
        ])
    table = axis_table.table(
        cellText=rows,
        colLabels=["Difficulty", "AP", "P", "R", "F1"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    axis_table.set_title("Metric Summary")

    figure.suptitle(
        f"WIDER FACE Evaluation Dashboard | Conf {METRIC_CONFIDENCE_THRESHOLD} | IoU {MATCH_IOU_THRESHOLD}",
        fontsize=16,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(VISUALIZATION_ROOT / "evaluation_dashboard.png", dpi=170)
    plt.close(figure)


def save_visualizations(metrics_all, curves):
    """全体評価グラフを保存します。"""
    VISUALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    save_evaluation_dashboard(metrics_all, curves)
    save_precision_recall_graph(curves)

    for difficulty in DIFFICULTIES:
        save_detection_matrix(metrics_all[difficulty], difficulty)


# ============================================================
# Ranked and Focus Outputs
# ============================================================

def result_summary(result, rank=None, output_path=None):
    """JSONへ保存可能な画像単位サマリーを返します。"""
    summary = {
        "relative_path": result.relative_path,
        "difficulty": result.difficulty,
        **{
            key: value
            for key, value in result.metrics.items()
            if key != "selected_matched_target_indices"
        },
    }

    if rank is not None:
        summary["rank"] = rank

    if output_path is not None:
        summary["output_image"] = output_path.as_posix()

    return summary


def save_ranked_images(image_results_by_difficulty):
    """難易度・観点別の個別画像とコンタクトシートを保存します。"""
    summary = {difficulty: {} for difficulty in DIFFICULTIES}

    for difficulty in DIFFICULTIES:
        for ranking_name in ("fn_rate", "fn_count", "fp_count"):
            ranked_results = rank_results(
                image_results_by_difficulty[difficulty],
                ranking_name,
            )
            ranking_directory = DIFFICULT_ROOT / "ranked" / difficulty / ranking_name
            contact_entries = []
            summary[difficulty][ranking_name] = []

            for rank, result in enumerate(ranked_results, start=1):
                file_name = (
                    f"rank_{rank:03d}_"
                    f"{safe_name(Path(result.relative_path).stem)}.jpg"
                )
                save_path = ranking_directory / file_name
                render_result_image(result, save_path, ranking_name, rank)
                contact_entries.append({
                    "rank": rank,
                    "relative_path": result.relative_path,
                    "output_path": save_path,
                })
                summary[difficulty][ranking_name].append(
                    result_summary(result, rank, save_path)
                )

            create_contact_sheet(
                contact_entries,
                ranking_directory / f"{difficulty}_{ranking_name}_contact_sheet.jpg",
                f"{difficulty.title()} | {ranking_name} Top {len(contact_entries)}",
            )

        small_results = small_support_complete_misses(
            image_results_by_difficulty[difficulty]
        )
        summary[difficulty]["small_support_complete_miss"] = [
            result_summary(result)
            for result in small_results
        ]

    return summary


def save_focus_images(image_results_by_difficulty):
    """Config指定パスの画像を難易度別に保存します。"""
    focus_selection = select_focus_results(image_results_by_difficulty)
    summary = {}

    for focus_path, difficulty_results in focus_selection.items():
        focus_name = safe_name(focus_path)
        summary[focus_path] = {}

        for difficulty, results in difficulty_results.items():
            output_directory = DIFFICULT_ROOT / "focus" / focus_name / difficulty
            summary[focus_path][difficulty] = []

            for index, result in enumerate(results, start=1):
                save_path = output_directory / (
                    f"{index:03d}_{safe_name(Path(result.relative_path).stem)}.jpg"
                )
                render_result_image(result, save_path, "focus", index)
                summary[focus_path][difficulty].append(
                    result_summary(result, index, save_path)
                )

    return summary


def save_difficult_image_outputs(image_results_by_difficulty):
    """ランキング・重点確認画像・JSONサマリーを保存します。"""
    DIFFICULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "settings": {
            "ranking_top_count": RANKING_TOP_COUNT,
            "fn_rate_min_gt_count": FN_RATE_MIN_GT_COUNT,
            "small_support_max_gt_count": SMALL_SUPPORT_MAX_GT_COUNT,
            "focus_paths": FOCUS_PATHS,
            "focus_max_images_per_path": FOCUS_MAX_IMAGES_PER_PATH,
            "metric_confidence_threshold": METRIC_CONFIDENCE_THRESHOLD,
            "match_iou_threshold": MATCH_IOU_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
        },
        "ranked": save_ranked_images(image_results_by_difficulty),
        "focus": save_focus_images(image_results_by_difficulty),
    }

    with open(DIFFICULT_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    """ONNX評価、全体可視化、苦手画像出力を実行します。"""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(ANNO_ROOT.rglob("*.json"))

    if MAX_IMAGES is not None:
        annotation_files = annotation_files[:MAX_IMAGES]

    session = ort.InferenceSession(str(MODEL_PATH), providers=PROVIDERS)
    input_name = session.get_inputs()[0].name
    image_results_by_difficulty = {
        difficulty: []
        for difficulty in DIFFICULTIES
    }

    for annotation_path in tqdm(annotation_files, desc="ONNX evaluation"):
        image, relative_path, ground_truths = load_sample(annotation_path)
        original_width, original_height = image.size
        model_input = preprocess(image)
        raw_output = session.run(None, {input_name: model_input})[0]
        prediction_boxes, prediction_scores = postprocess(
            raw_output,
            original_width,
            original_height,
        )

        for difficulty in DIFFICULTIES:
            result = match_difficulty(
                relative_path=relative_path,
                image_width=original_width,
                image_height=original_height,
                prediction_boxes=prediction_boxes,
                prediction_scores=prediction_scores,
                ground_truths=ground_truths,
                difficulty=difficulty,
            )
            image_results_by_difficulty[difficulty].append(result)

    metrics_all = {}
    curves = {}

    for difficulty in DIFFICULTIES:
        metrics, curve = evaluate_difficulty_results(
            image_results_by_difficulty[difficulty]
        )
        metrics_all[difficulty] = metrics
        curves[difficulty] = curve

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_all, file, ensure_ascii=False, indent=2)

    save_visualizations(metrics_all, curves)
    save_difficult_image_outputs(image_results_by_difficulty)


if __name__ == "__main__":
    main()
