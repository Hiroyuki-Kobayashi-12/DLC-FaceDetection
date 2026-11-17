# Cell 3 Dataset Create
# ============================================================
#
# このセルの役割:
#
#   WIDER FACEのJSONと画像を読み込み、
#   学習用とValidation用のDataset / DataLoaderを作成します。
#
# Datasetは難易度別に分割しません。
#
#   train_loader:
#     valid_bbox=True かつ valid_for_training=True の顔を使用します。
#
#   val_loader:
#     valid_bbox=True の顔をすべて使用します。
#
# Validationでは、各bboxに対応する難易度情報を
# target["levels"]へ残します。
#
# 例:
#
#   target["levels"] = [
#       ["easy", "medium", "hard"],
#       ["medium", "hard"],
#       ["hard"],
#   ]
#
# boxes、labels、face_indices、levelsは同じ順番で対応します。
# 難易度別のLoss計算や評価は後続処理で行います。
#
# Configから受け取る定数:
#
#   IMAGE_SIZE
#   BATCH_SIZE
#   NUM_WORKERS
#   PIN_MEMORY
#
# 後続セルへ渡す変数:
#
#   train_dataset
#   val_dataset
#   train_loader
#   val_loader
#
# ============================================================


from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

import torch
from torch.utils.data import Dataset, DataLoader

import random

# ============================================================
# WIDER FACE Dataset
# ============================================================

class WiderFaceDataset(Dataset):
    """WIDER FACEの画像と顔bboxを読み込むDatasetです。"""

    def __init__(self, split):
        """splitに対応する画像、JSON、サンプル一覧を準備します。"""
        self.split = split
        self.image_size = IMAGE_SIZE

        self.input_root = Path("/kaggle/input")
        self.dataset_root = self.widerface_dataset_root()

        self.image_directory = (
            self.dataset_root
            / self.split
            / "image"
        )

        self.annotation_directory = (
            self.dataset_root
            / self.split
            / "anno"
        )

        if not 0.0 <= BRIGHTNESS_PROBABILITY <= 1.0:
            raise ValueError(
                "BRIGHTNESS_PROBABILITY must be between 0 and 1."
            )

        if BRIGHTNESS_MIN_FACTOR <= 0.0:
            raise ValueError(
                "BRIGHTNESS_MIN_FACTOR must be greater than 0."
            )

        if BRIGHTNESS_MAX_FACTOR < BRIGHTNESS_MIN_FACTOR:
            raise ValueError(
                "BRIGHTNESS_MAX_FACTOR must be greater than "
                "or equal to BRIGHTNESS_MIN_FACTOR."
            )

        self.annotation_files = self.widerface_annotation_files()

    def widerface_dataset_root(self):
        """trainとvalを持つWIDER FACE Datasetのルートを取得します。"""
        for path in self.input_root.rglob("*"):
            if not path.is_dir():
                continue

            required_directories = [
                path / "train" / "image",
                path / "train" / "anno",
                path / "val" / "image",
                path / "val" / "anno",
            ]

            if all(
                directory.exists()
                for directory in required_directories
            ):
                return path

    def widerface_annotation_files(self):
        """splitに対応するJSONファイルを一定の順番で取得します。"""
        return sorted(
            self.annotation_directory.rglob("*.json")
        )

    def widerface_load_annotation(self, annotation_path):
        """画像1枚分のJSONを読み込みます。"""
        with open(
            annotation_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def widerface_face_is_target(self, face):
        """splitに応じて学習またはValidationで使用する顔を選びます。"""
        if not face["valid_bbox"]:
            return False

        if self.split == "train":
            return face["valid_for_training"]

        return True

    def widerface_target(self, annotation):
        """faces配列からbbox、ラベル、顔番号、難易度を作成します。"""
        boxes = []
        labels = []
        face_indices = []
        levels = []

        for face in annotation["faces"]:
            if not self.widerface_face_is_target(face):
                continue

            box = face["bbox"]["xyxy"]

            if box is None:
                continue

            boxes.append(box)
            labels.append(0)
            face_indices.append(face["face_index"])
            levels.append(list(face["levels"]))

        if len(boxes) == 0:
            boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.long,
            )

            face_indices_tensor = torch.zeros(
                (0,),
                dtype=torch.long,
            )

        else:
            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels_tensor = torch.tensor(
                labels,
                dtype=torch.long,
            )

            face_indices_tensor = torch.tensor(
                face_indices,
                dtype=torch.long,
            )

        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "face_indices": face_indices_tensor,
            "levels": levels,
        }

    def widerface_image(self, annotation):
        """JSONの相対パスからPIL RGB画像を読み込みます。"""
        relative_path = annotation["image"]["relative_path"]
        image_path = self.image_directory / relative_path

        return Image.open(image_path).convert("RGB")

    def widerface_resize(self, image, boxes):
        """画像を正方形へリサイズし、bboxも同じ比率で変換します。"""
        original_width, original_height = image.size

        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        boxes = boxes.clone()

        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return image, boxes

    def widerface_brightness_augmentation(self, image):
        # Validationにはデータ拡張を適用しない。
        if self.split != "train":
            return image

        # Configで無効化されている場合は元画像を返却
        if not USE_BRIGHTNESS_AUGMENTATION:
            return image

        # 指定確率に該当しない場合は元画像を返却
        if random.random() >= BRIGHTNESS_PROBABILITY:
            return image

        brightness_factor = random.uniform(
            BRIGHTNESS_MIN_FACTOR,
            BRIGHTNESS_MAX_FACTOR,
        )

        return ImageEnhance.Brightness(
            image
        ).enhance(
            brightness_factor
        )
    
    def widerface_horizontal_flip(self, image, boxes):

        if self.split != "train":
            return image, boxes

        if not USE_HORIZONTAL_FLIP:
            return image, boxes

        if random.random() >= HORIZONTAL_FLIP_PROBABILITY:
            return image, boxes

        image = ImageOps.mirror(image)
        boxes = boxes.clone()

        if boxes.shape[0] > 0:
            image_width = image.size[0]

            original_x1 = boxes[:, 0].clone()
            original_x2 = boxes[:, 2].clone()

            boxes[:, 0] = image_width - original_x2
            boxes[:, 2] = image_width - original_x1

        return image, boxes
    
    def widerface_blur_augmentation(self, image):
        """学習画像へ弱いGaussian Blurを適用します。"""

        # Validationには適用しません。
        if self.split != "train":
            return image

        # Configで無効なら何もしません。
        if not USE_BLUR_AUGMENTATION:
            return image

        # 指定確率に該当しなければ何もしません。
        if random.random() >= BLUR_AUGMENTATION_PROBABILITY:
            return image

        blur_radius = random.uniform(
            BLUR_MIN_RADIUS,
            BLUR_MAX_RADIUS,
        )

        return image.filter(
            ImageFilter.GaussianBlur(
                radius=blur_radius,
            )
        )

    def widerface_random_crop(self, image, target):
        """学習画像と対応するtargetへ弱いランダムクロップを適用します。"""

        if self.split != "train":
            return image, target

        if not USE_RANDOM_CROP:
            return image, target

        if random.random() >= RANDOM_CROP_PROBABILITY:
            return image, target

        image_width, image_height = image.size

        crop_scale = random.uniform(
            RANDOM_CROP_MIN_SCALE,
            1.0,
        )

        crop_width = max(
            1,
            int(image_width * crop_scale),
        )

        crop_height = max(
            1,
            int(image_height * crop_scale),
        )

        max_left = image_width - crop_width
        max_top = image_height - crop_height

        crop_left = random.randint(
            0,
            max_left,
        )

        crop_top = random.randint(
            0,
            max_top,
        )

        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height

        cropped_image = image.crop(
            (
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            )
        )

        boxes = target["boxes"]

        if boxes.shape[0] == 0:
            return cropped_image, target

        original_boxes = boxes.clone()

        # クロップ位置を原点にした座標へ移動します。
        cropped_boxes = boxes.clone()

        cropped_boxes[:, [0, 2]] -= float(crop_left)
        cropped_boxes[:, [1, 3]] -= float(crop_top)

        # クロップ画像内へ座標を制限します。
        cropped_boxes[:, 0].clamp_(
            min=0.0,
            max=float(crop_width),
        )

        cropped_boxes[:, 2].clamp_(
            min=0.0,
            max=float(crop_width),
        )

        cropped_boxes[:, 1].clamp_(
            min=0.0,
            max=float(crop_height),
        )

        cropped_boxes[:, 3].clamp_(
            min=0.0,
            max=float(crop_height),
        )

        original_widths = (
            original_boxes[:, 2]
            - original_boxes[:, 0]
        ).clamp(min=0.0)

        original_heights = (
            original_boxes[:, 3]
            - original_boxes[:, 1]
        ).clamp(min=0.0)

        original_areas = (
            original_widths
            * original_heights
        )

        cropped_widths = (
            cropped_boxes[:, 2]
            - cropped_boxes[:, 0]
        ).clamp(min=0.0)

        cropped_heights = (
            cropped_boxes[:, 3]
            - cropped_boxes[:, 1]
        ).clamp(min=0.0)

        cropped_areas = (
            cropped_widths
            * cropped_heights
        )

        visibility = (
            cropped_areas
            / original_areas.clamp(min=1e-12)
        )

        # クロップ後も十分に残っている顔だけ採用します。
        keep_mask = (
            (cropped_widths >= RANDOM_CROP_MIN_BOX_SIZE)
            & (cropped_heights >= RANDOM_CROP_MIN_BOX_SIZE)
            & (
                visibility
                >= RANDOM_CROP_MIN_VISIBILITY
            )
        )

        # 全bboxが消えるクロップは無効にします。
        if not bool(keep_mask.any()):
            return image, target

        keep_indices = torch.where(
            keep_mask
        )[0]

        target = {
            **target,
            "boxes": cropped_boxes[keep_indices],
            "labels": target["labels"][keep_indices],
            "face_indices": target["face_indices"][keep_indices],
            "levels": [
                target["levels"][index]
                for index in keep_indices.tolist()
            ],
        }

        return cropped_image, target

    def widerface_image_tensor(self, image):
        """PIL RGB画像を0から1のCHW FloatTensorへ変換します。"""
        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        return torch.from_numpy(
            image_array
        ).permute(2, 0, 1).contiguous()

    def widerface_scale_augmentation(self, image, target,):
        # Validationには適用しません。
        if self.split != "train":
            return image, target

        if not USE_SCALE_AUGMENTATION:
            return image, target

        if random.random() >= SCALE_AUGMENTATION_PROBABILITY:
            return image, target

        image_width, image_height = image.size

        scale_factor = random.uniform(
            SCALE_AUGMENTATION_MIN_FACTOR,
            SCALE_AUGMENTATION_MAX_FACTOR,
        )

        scaled_width = max(
            1,
            int(round(image_width * scale_factor)),
        )

        scaled_height = max(
            1,
            int(round(image_height * scale_factor)),
        )

        scaled_image = image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )

        original_boxes = target["boxes"].clone()
        scaled_boxes = original_boxes.clone()

        if scaled_boxes.shape[0] > 0:
            scaled_boxes[:, [0, 2]] *= scale_factor
            scaled_boxes[:, [1, 3]] *= scale_factor

        # --------------------------------------------------------
        # 縮小時
        #
        # 小さくした画像を元サイズのキャンバスへランダム配置します。
        # --------------------------------------------------------

        if scale_factor <= 1.0:
            max_offset_x = image_width - scaled_width
            max_offset_y = image_height - scaled_height

            offset_x = random.randint(
                0,
                max_offset_x,
            )

            offset_y = random.randint(
                0,
                max_offset_y,
            )

            output_image = Image.new(
                mode="RGB",
                size=(image_width, image_height),
                color=SCALE_AUGMENTATION_FILL_COLOR,
            )

            output_image.paste(
                scaled_image,
                (offset_x, offset_y),
            )

            if scaled_boxes.shape[0] > 0:
                scaled_boxes[:, [0, 2]] += float(offset_x)
                scaled_boxes[:, [1, 3]] += float(offset_y)

        # --------------------------------------------------------
        # 拡大時
        #
        # 拡大画像から元サイズ領域をランダムに切り出します。
        # --------------------------------------------------------

        else:
            max_crop_left = scaled_width - image_width
            max_crop_top = scaled_height - image_height

            crop_left = random.randint(
                0,
                max_crop_left,
            )

            crop_top = random.randint(
                0,
                max_crop_top,
            )

            crop_right = crop_left + image_width
            crop_bottom = crop_top + image_height

            output_image = scaled_image.crop(
                (
                    crop_left,
                    crop_top,
                    crop_right,
                    crop_bottom,
                )
            )

            if scaled_boxes.shape[0] > 0:
                scaled_boxes[:, [0, 2]] -= float(crop_left)
                scaled_boxes[:, [1, 3]] -= float(crop_top)

        # bboxが0個なら、そのまま返します。
        if scaled_boxes.shape[0] == 0:
            return output_image, target

        # クリップ前の面積を保存します。
        before_clip_widths = (
            scaled_boxes[:, 2]
            - scaled_boxes[:, 0]
        ).clamp(min=0.0)

        before_clip_heights = (
            scaled_boxes[:, 3]
            - scaled_boxes[:, 1]
        ).clamp(min=0.0)

        before_clip_areas = (
            before_clip_widths
            * before_clip_heights
        )

        # 画像内へbboxを収めます。
        scaled_boxes[:, 0].clamp_(
            min=0.0,
            max=float(image_width),
        )

        scaled_boxes[:, 2].clamp_(
            min=0.0,
            max=float(image_width),
        )

        scaled_boxes[:, 1].clamp_(
            min=0.0,
            max=float(image_height),
        )

        scaled_boxes[:, 3].clamp_(
            min=0.0,
            max=float(image_height),
        )

        after_clip_widths = (
            scaled_boxes[:, 2]
            - scaled_boxes[:, 0]
        ).clamp(min=0.0)

        after_clip_heights = (
            scaled_boxes[:, 3]
            - scaled_boxes[:, 1]
        ).clamp(min=0.0)

        after_clip_areas = (
            after_clip_widths
            * after_clip_heights
        )

        visibility = (
            after_clip_areas
            / before_clip_areas.clamp(min=1e-12)
        )

        keep_mask = (
            (
                after_clip_widths
                >= SCALE_AUGMENTATION_MIN_BOX_SIZE
            )
            & (
                after_clip_heights
                >= SCALE_AUGMENTATION_MIN_BOX_SIZE
            )
            & (
                visibility
                >= SCALE_AUGMENTATION_MIN_VISIBILITY
            )
        )

        # すべての顔が消える変換は不採用にします。
        if not bool(keep_mask.any()):
            return image, target

        keep_indices = torch.where(
            keep_mask
        )[0]

        updated_target = {
            **target,
            "boxes": scaled_boxes[keep_indices],
            "labels": target["labels"][keep_indices],
            "face_indices": target["face_indices"][keep_indices],
            "levels": [
                target["levels"][index]
                for index in keep_indices.tolist()
            ],
        }

        return output_image, updated_target

    def widerface_sample(self, index):
        """
        1サンプルを作成し、
        学習時のみデータ拡張を適用します。
        """

        # --------------------------------------------------------
        # 1. アノテーション読み込み
        # --------------------------------------------------------

        annotation_path = self.annotation_files[index]

        annotation = self.widerface_load_annotation(
            annotation_path
        )

        # --------------------------------------------------------
        # 2. 元画像とtargetを作成
        # --------------------------------------------------------

        image = self.widerface_image(
            annotation
        )

        target = self.widerface_target(
            annotation
        )

        # --------------------------------------------------------
        # 3. 弱いランダムクロップ
        # --------------------------------------------------------

        image, target = self.widerface_random_crop(
            image,
            target,
        )

        # --------------------------------------------------------
        # 4. 640×640へリサイズ
        # --------------------------------------------------------

        image, target["boxes"] = self.widerface_resize(
            image,
            target["boxes"],
        )

        # --------------------------------------------------------
        # 5. 弱いランダムスケール
        #
        # 出力サイズは640×640を維持します。
        # --------------------------------------------------------

        image, target = self.widerface_scale_augmentation(
            image,
            target,
        )

        # --------------------------------------------------------
        # 6. 左右反転
        # --------------------------------------------------------

        image, target["boxes"] = self.widerface_horizontal_flip(
            image,
            target["boxes"],
        )

        # --------------------------------------------------------
        # 7. Tensor変換
        # --------------------------------------------------------

        # Tensor変換の前に弱いぼかしを入れる
        image = self.widerface_blur_augmentation(image)
        image = self.widerface_image_tensor(
            image
        )

        # --------------------------------------------------------
        # 8. 補足情報
        # --------------------------------------------------------

        target["split"] = self.split

        target["image_relative_path"] = (
            annotation["image"]["relative_path"]
        )

        target["image_width"] = self.image_size
        target["image_height"] = self.image_size

        target["original_image_width"] = int(
            annotation["image"]["width"]
        )

        target["original_image_height"] = int(
            annotation["image"]["height"]
        )

        return image, target

    def __len__(self):
        """Datasetに含まれる画像数を返します。"""
        return len(self.annotation_files)

    def __getitem__(self, index):
        """指定indexの画像Tensorとtargetを返します。"""
        return self.widerface_sample(index)


# ============================================================
# DataLoader
# ============================================================

class WiderFaceDataLoader:
    """WIDER FACEのDatasetとDataLoaderを作成する専用クラスです。"""

    def widerface_collate(self, batch):
        """bbox数が異なるtargetをlistのまままとめます。"""
        images, targets = zip(*batch)

        return list(images), list(targets)

    def widerface_dataset(self, split):
        """splitに対応するWiderFaceDatasetを作成します。"""
        return WiderFaceDataset(
            split=split
        )

    def widerface_loader(self, dataset, shuffle):
        """指定DatasetからDataLoaderを作成します。"""
        return DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            collate_fn=self.widerface_collate,
            pin_memory=PIN_MEMORY,
        )

    def create(self):
        """学習用とValidation用のDataset / DataLoaderを作成します。"""
        train_dataset = self.widerface_dataset(
            split="train"
        )

        val_dataset = self.widerface_dataset(
            split="val"
        )

        train_loader = self.widerface_loader(
            dataset=train_dataset,
            shuffle=True,
        )

        val_loader = self.widerface_loader(
            dataset=val_dataset,
            shuffle=False,
        )

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "train_loader": train_loader,
            "val_loader": val_loader,
        }


# ============================================================
# Dataset / DataLoader Create
# ============================================================

widerface_data = WiderFaceDataLoader()
dataset_result = widerface_data.create()


# ============================================================
# Output
# ============================================================

train_dataset = dataset_result["train_dataset"]
val_dataset = dataset_result["val_dataset"]
train_loader = dataset_result["train_loader"]
val_loader = dataset_result["val_loader"]
