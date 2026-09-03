# Cell 3 Augmentation Create
# ============================================================
#
# このセルの役割:
#
#   ConfigのTRAIN_AUGMENTATIONSに記載された順番で、
#   学習画像へ適用するオーギュメンテーションを作成します。
#
#   画像の位置を変更する処理では、bboxも同時に変換します。
#   色や画質だけを変更する処理では、bboxを変更しません。
#
# Configから受け取る定数:
#
#   TRAIN_AUGMENTATIONS
#   SEED                         任意
#
# Configの記載例:
#
#   TRAIN_AUGMENTATIONS = [
#       {
#           "name": "horizontal_flip",
#           "probability": 0.5,
#       },
#       {
#           "name": "brightness",
#           "probability": 0.3,
#           "min_factor": 0.8,
#           "max_factor": 1.2,
#       },
#       {
#           "name": "contrast",
#           "probability": 0.3,
#           "min_factor": 0.8,
#           "max_factor": 1.2,
#       },
#   ]
#
# 上記は、次の順番で実行します。
#
#   1. horizontal_flip
#   2. brightness
#   3. contrast
#
# 後続セルへ渡すもの:
#
#   train_augmentation
#   visualize_train_augmentation()
#
# Dataset側の適用位置:
#
#   画像とbboxのリサイズ
#   -> オーギュメンテーション
#   -> Tensor変換
#
# ============================================================

import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance, ImageFilter, ImageOps


# ============================================================
# Data Augmentation
# ============================================================

class DataAugmentation:
    """Config配列に記載された順番で画像とtargetを変換します。"""

    def __init__(self, augmentation_configs, seed=None):
        """設定一覧、乱数生成器、処理名と関数の対応を準備します。"""
        self.augmentation_configs = list(augmentation_configs)
        self.random = random.Random(seed)

        self.augmentation_methods = {
            "horizontal_flip": self.horizontal_flip,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "sharpness": self.sharpness,
            "gaussian_blur": self.gaussian_blur,
            "grayscale": self.grayscale,
        }

        self.validate_configs()

    def validate_probability(self, probability, augmentation_name):
        """適用確率が0以上1以下であることを確認します。"""
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{augmentation_name}: probability must be between 0 and 1. "
                f"value={probability}"
            )

    def validate_factor_range(self, config, augmentation_name):
        """画像調整倍率の最小値と最大値を確認します。"""
        if "min_factor" not in config or "max_factor" not in config:
            raise ValueError(
                f"{augmentation_name}: min_factor and max_factor are required."
            )

        min_factor = float(config["min_factor"])
        max_factor = float(config["max_factor"])

        if min_factor < 0.0:
            raise ValueError(
                f"{augmentation_name}: min_factor must be 0 or greater."
            )

        if min_factor > max_factor:
            raise ValueError(
                f"{augmentation_name}: min_factor must not exceed max_factor."
            )

    def validate_radius_range(self, config):
        """Gaussian Blurの半径の最小値と最大値を確認します。"""
        if "min_radius" not in config or "max_radius" not in config:
            raise ValueError(
                "gaussian_blur: min_radius and max_radius are required."
            )

        min_radius = float(config["min_radius"])
        max_radius = float(config["max_radius"])

        if min_radius < 0.0:
            raise ValueError(
                "gaussian_blur: min_radius must be 0 or greater."
            )

        if min_radius > max_radius:
            raise ValueError(
                "gaussian_blur: min_radius must not exceed max_radius."
            )

    def validate_config(self, config):
        """オーギュメンテーション設定1件を検証します。"""
        if not isinstance(config, dict):
            raise TypeError("Each augmentation config must be dict.")

        if "name" not in config:
            raise ValueError("Augmentation config requires name.")

        augmentation_name = str(config["name"])

        if augmentation_name not in self.augmentation_methods:
            raise ValueError(
                f"Unknown augmentation: {augmentation_name}"
            )

        probability = float(config.get("probability", 1.0))
        self.validate_probability(
            probability=probability,
            augmentation_name=augmentation_name,
        )

        factor_augmentations = {
            "brightness",
            "contrast",
            "saturation",
            "sharpness",
        }

        if augmentation_name in factor_augmentations:
            self.validate_factor_range(
                config=config,
                augmentation_name=augmentation_name,
            )

        if augmentation_name == "gaussian_blur":
            self.validate_radius_range(config)

    def validate_configs(self):
        """Config配列を記載順に検証します。"""
        for config in self.augmentation_configs:
            self.validate_config(config)

    def should_apply(self, config):
        """設定された確率に従って処理を適用するか決定します。"""
        probability = float(config.get("probability", 1.0))
        random_value = self.random.random()
        return random_value < probability

    def random_factor(self, config):
        """設定範囲から画像調整倍率を1つ選びます。"""
        min_factor = float(config["min_factor"])
        max_factor = float(config["max_factor"])
        return self.random.uniform(min_factor, max_factor)

    def horizontal_flip(self, image, target, config):
        """画像とbboxを左右反転します。"""
        image_width = image.width
        transformed_image = ImageOps.mirror(image)
        transformed_boxes = target["boxes"].clone()

        if transformed_boxes.shape[0] > 0:
            original_x1 = transformed_boxes[:, 0].clone()
            original_x2 = transformed_boxes[:, 2].clone()

            transformed_boxes[:, 0] = image_width - original_x2
            transformed_boxes[:, 2] = image_width - original_x1

        transformed_target = dict(target)
        transformed_target["boxes"] = transformed_boxes

        return transformed_image, transformed_target

    def brightness(self, image, target, config):
        """画像の明るさをランダムな倍率で変更します。"""
        factor = self.random_factor(config)
        transformed_image = ImageEnhance.Brightness(image).enhance(factor)
        return transformed_image, target

    def contrast(self, image, target, config):
        """画像のコントラストをランダムな倍率で変更します。"""
        factor = self.random_factor(config)
        transformed_image = ImageEnhance.Contrast(image).enhance(factor)
        return transformed_image, target

    def saturation(self, image, target, config):
        """画像の色の鮮やかさをランダムな倍率で変更します。"""
        factor = self.random_factor(config)
        transformed_image = ImageEnhance.Color(image).enhance(factor)
        return transformed_image, target

    def sharpness(self, image, target, config):
        """画像の鮮明さをランダムな倍率で変更します。"""
        factor = self.random_factor(config)
        transformed_image = ImageEnhance.Sharpness(image).enhance(factor)
        return transformed_image, target

    def gaussian_blur(self, image, target, config):
        """画像へランダムな半径のGaussian Blurを適用します。"""
        min_radius = float(config["min_radius"])
        max_radius = float(config["max_radius"])
        radius = self.random.uniform(min_radius, max_radius)
        transformed_image = image.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )
        return transformed_image, target

    def grayscale(self, image, target, config):
        """画像をグレースケール化し、3チャンネルRGBへ戻します。"""
        transformed_image = ImageOps.grayscale(image).convert("RGB")
        return transformed_image, target

    def apply_one(self, image, target, config):
        """設定1件に対応する処理を、適用確率に従って実行します。"""
        if not self.should_apply(config):
            return image, target

        augmentation_name = config["name"]
        augmentation_method = self.augmentation_methods[augmentation_name]

        return augmentation_method(
            image=image,
            target=target,
            config=config,
        )

    def apply(self, image, target):
        """Config配列を上から順番に適用します。"""
        transformed_image = image
        transformed_target = target

        for config in self.augmentation_configs:
            transformed_image, transformed_target = self.apply_one(
                image=transformed_image,
                target=transformed_target,
                config=config,
            )

        return transformed_image, transformed_target


# ============================================================
# Augmentation Visualization
# ============================================================

def augmentation_image_array(image_tensor):
    """CHW TensorをMatplotlibで表示できるHWC配列へ変換します。"""
    image_array = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_array, 0.0, 1.0)


def augmentation_draw_boxes(axis, boxes):
    """オーギュメンテーション後のbboxを画像上へ描画します。"""
    for box in boxes:
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        box_width = x2 - x1
        box_height = y2 - y1

        rectangle = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )

        axis.add_patch(rectangle)


def augmentation_sample_indices(dataset_size, sample_count):
    """毎回同じ先頭サンプルを可視化対象として返します。"""
    actual_sample_count = min(sample_count, dataset_size)
    return list(range(actual_sample_count))


def augmentation_axes(row_count, column_count):
    """指定行列数のFigureと1次元axes配列を作成します。"""
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(column_count * 4, row_count * 4),
    )

    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    return figure, axes_array


def augmentation_show_or_close(figure, show):
    """指定に応じてFigureを表示し、最後に閉じます。"""
    if show:
        plt.show()

    plt.close(figure)


def visualize_train_augmentation(
    train_dataset,
    save_path,
    sample_count=16,
    column_count=4,
    visualization_seed=12345,
    show=False,
):
    """固定した16枚を変換し、bbox付き4x4画像として保存します。"""
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0.")

    if column_count <= 0:
        raise ValueError("column_count must be greater than 0.")

    sample_indices = augmentation_sample_indices(
        dataset_size=len(train_dataset),
        sample_count=sample_count,
    )

    if len(sample_indices) == 0:
        raise ValueError("train_dataset has no samples.")

    row_count = (
        len(sample_indices) + column_count - 1
    ) // column_count

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    augmentation = train_dataset.augmentation
    original_random_state = None

    # 可視化専用Seedを使い、毎回同じ変換結果を作ります。
    # 可視化後に元の乱数状態へ戻すため、学習時の変換系列へ影響しません。
    if augmentation is not None:
        original_random_state = augmentation.random.getstate()
        augmentation.random.seed(visualization_seed)

    figure = None

    try:
        figure, axes = augmentation_axes(
            row_count=row_count,
            column_count=column_count,
        )

        for plot_index, sample_index in enumerate(sample_indices):
            image_tensor, target = train_dataset[sample_index]
            image_array = augmentation_image_array(image_tensor)
            axis = axes[plot_index]

            axis.imshow(image_array)
            augmentation_draw_boxes(
                axis=axis,
                boxes=target["boxes"],
            )
            axis.set_title(
                f"Sample {sample_index:03d} | "
                f"Faces {len(target['boxes'])}"
            )
            axis.axis("off")

        for plot_index in range(len(sample_indices), len(axes)):
            axes[plot_index].axis("off")

        figure.suptitle(
            "Training Augmentation Preview",
            fontsize=16,
        )
        figure.tight_layout()
        figure.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

        augmentation_show_or_close(
            figure=figure,
            show=show,
        )
        figure = None

    finally:
        if augmentation is not None and original_random_state is not None:
            augmentation.random.setstate(original_random_state)

        if figure is not None:
            plt.close(figure)


# ============================================================
# Train Augmentation Create
# ============================================================

# ConfigにSEEDがあれば、オーギュメンテーションの初期Seedにも使用します。
augmentation_seed = globals().get("SEED")

train_augmentation = DataAugmentation(
    augmentation_configs=TRAIN_AUGMENTATIONS,
    seed=augmentation_seed,
)
