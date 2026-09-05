# Cell 1 Config
# ============================================================
#
# このセルの役割:
#
#   後続セルで使用する設定値を一括管理します。
#   通常の学習条件変更は、このセルだけを編集します。
#
# 設計ルール:
#
#   - 定数指定だけを置く
#   - モデル実装の固定値は各モデルクラスへ置く
#   - 関数、クラス、処理、例外処理は置かない
#   - 設定を縦に詰め、差分を確認しやすくする
#
# ============================================================

# Model
MODEL_NAME = "yolov5"
NUM_CLASSES = 1
CLASS_NAMES = {0: "face"}
DEVICE_NAME = "auto"
USE_PRETRAINED_WEIGHTS = True

# Dataset
IMAGE_SIZE = 640
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

# YOLOv5 Loss
YOLOV5_BOX_LOSS_WEIGHT = 0.05
YOLOV5_OBJECTNESS_LOSS_WEIGHT = 1.0
YOLOV5_CLASSIFICATION_LOSS_WEIGHT = 0.5
YOLOV5_OBJECTNESS_POSITIVE_WEIGHT = 1.0
YOLOV5_CLASSIFICATION_POSITIVE_WEIGHT = 1.0
YOLOV5_ANCHOR_MATCH_THRESHOLD = 4.0
YOLOV5_LABEL_SMOOTHING = 0.0
YOLOV5_FOCAL_GAMMA = 0.0
YOLOV5_FOCAL_ALPHA = 0.25
YOLOV5_OBJECTNESS_IOU_RATIO = 1.0
YOLOV5_OBJECTNESS_BALANCE = [4.0, 1.0, 0.4]

# Augmentation
# 配列の上から順番に適用します。
# probabilityは各処理を適用する確率で、0.0から1.0の範囲です。
# 使用しない処理はコメントアウトし、使用する場合だけコメントを外します。
TRAIN_AUGMENTATIONS = [
    # 画像とbboxを左右反転します。
    {
        "name": "horizontal_flip",
        "probability": 0.5,
    },
    # 明るさを変更します。1.0が元画像、1.0未満が暗く、1.0超が明るくなります。
    {
        "name": "brightness",
        "probability": 0.5,
        "min_factor": 0.8,
        "max_factor": 1.5,
    },
    # 明暗差を変更します。1.0未満で弱く、1.0超で強くなります。
    {
        "name": "contrast",
        "probability": 0.5,
        "min_factor": 0.8,
        "max_factor": 2.3,
    },

    # 色の鮮やかさを変更します。0.0で白黒、1.0で元画像です。
    {
        "name": "saturation",
        "probability": 0.4,
        "min_factor": 0.8,
        "max_factor": 1.2,
    },

    # 画像の鮮明さを変更します。0.0でぼかし、1.0で元画像です。
    # {
    #     "name": "sharpness",
    #     "probability": 0.2,
    #     "min_factor": 0.8,
    #     "max_factor": 1.5,
    # },

    # Gaussian Blurを適用します。radiusが大きいほど強くぼかします。
    {
        "name": "gaussian_blur",
        "probability": 0.5,
        "min_radius": 0.1,
        "max_radius": 1.5,
    },

    # 白黒画像へ変換した後、モデル入力用の3チャンネルRGBへ戻します。
    # {
    #     "name": "grayscale",
    #     "probability": 0.05,
    # },

    # Gaussian Noiseを加えます。stddevが大きいほどノイズが強くなります。
    {
        "name": "gaussian_noise",
        "probability": 0.2,
        "min_stddev": 3.0,
        "max_stddev": 12.0,
    },
]

# Augmentation Preview
# 毎回同じ16枚と同じ乱数系列を使い、4x4で変換結果を確認します。
AUGMENTATION_PREVIEW_FILE_NAME = "augmentation_preview.png"
AUGMENTATION_PREVIEW_SAMPLE_COUNT = 16
AUGMENTATION_PREVIEW_COLUMN_COUNT = 4
AUGMENTATION_PREVIEW_SEED = 12345
SHOW_AUGMENTATION_PREVIEW = False

# Optimizer
OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
SGD_MOMENTUM = 0.9

# Scheduler
SCHEDULER_NAME = "CosineAnnealingLR"
NUM_EPOCHS = 50
STEP_LR_STEP_SIZE = 10
STEP_LR_GAMMA = 0.1
COSINE_ANNEALING_MIN_LR = 1e-6

# Output
OUTPUT_DIRECTORY = "/kaggle/working/dlc26_outputs"
CHECKPOINT_DIRECTORY_NAME = "checkpoints"
CHECKPOINT_FILE_PREFIX = "epoch"
HISTORY_JSON_FILE_NAME = "training_history.json"
HISTORY_IMAGE_FILE_NAME = "training_history.png"
BEST_ONNX_FILE_NAME = "model_best.onnx"
FINAL_ONNX_FILE_NAME = "model_final.onnx"
SHOW_HISTORY = True

# ONNX
ONNX_OPSET_VERSION = 12
ONNX_DYNAMIC_BATCH = True
