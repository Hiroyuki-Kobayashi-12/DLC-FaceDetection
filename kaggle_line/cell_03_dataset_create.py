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
# Data Augmentation:
#
#   USE_AUGMENTATION=True の場合、trainだけに適用します。
#   valは評価条件を揃えるため、リサイズのみ行います。
#
#   1. Scale Crop:
#        元画像から縦横比を保ったまま一部を切り出し、
#        リサイズ後の顔を相対的に大きくします。
#        bbox中心が切り出し範囲外の顔と、
#        切り出し後に極端に小さくなった顔は除外します。
#
#   2. Horizontal Flip:
#        画像とbboxを左右反転します。
#
# Configから受け取る定数:
#
#   IMAGE_SIZE
#   BATCH_SIZE
#   NUM_WORKERS
#   PIN_MEMORY
#   USE_AUGMENTATION
#   AUGMENT_SCALE_CROP_PROB
#   AUGMENT_SCALE_CROP_MIN
#   AUGMENT_MIN_BOX_SIZE
#   AUGMENT_HFLIP_PROB
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
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# WIDER FACE Dataset
# ============================================================

class WiderFaceDataset(Dataset):
    """WIDER FACEの画像と顔bboxを読み込むDatasetです。"""

    def __init__(self, split):
        """splitに対応する画像、JSON、サンプル一覧を準備します。"""
        self.split = split
        self.image_size = IMAGE_SIZE
        self.use_augmentation = (
            USE_AUGMENTATION
            and self.split == "train"
        )

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

    def widerface_scale_crop(self, image, target):
        """元画像の一部を切り出し、顔を相対的に拡大します。"""
        original_width, original_height = image.size

        # 切り出し倍率をAUGMENT_SCALE_CROP_MINから1.0の間で選びます。
        scale = (
            AUGMENT_SCALE_CROP_MIN
            + (1.0 - AUGMENT_SCALE_CROP_MIN) * torch.rand(1).item()
        )

        crop_width = max(1, int(original_width * scale))
        crop_height = max(1, int(original_height * scale))

        crop_x1 = int(torch.randint(
            0,
            original_width - crop_width + 1,
            (1,),
        ).item())

        crop_y1 = int(torch.randint(
            0,
            original_height - crop_height + 1,
            (1,),
        ).item())

        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        boxes = target["boxes"]

        # bbox中心が切り出し範囲に残る顔だけを対象にします。
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

        center_inside = (
            (center_x >= crop_x1)
            & (center_x < crop_x2)
            & (center_y >= crop_y1)
            & (center_y < crop_y2)
        )

        # 切り出し範囲の左上を原点とする座標へ変換し、範囲内へ収めます。
        cropped_boxes = boxes.clone()
        cropped_boxes[:, [0, 2]] = (
            cropped_boxes[:, [0, 2]] - crop_x1
        ).clamp(0.0, crop_width)
        cropped_boxes[:, [1, 3]] = (
            cropped_boxes[:, [1, 3]] - crop_y1
        ).clamp(0.0, crop_height)

        # 範囲外へはみ出して極端に小さくなった顔を除外します。
        box_width = cropped_boxes[:, 2] - cropped_boxes[:, 0]
        box_height = cropped_boxes[:, 3] - cropped_boxes[:, 1]

        keep = (
            center_inside
            & (box_width >= AUGMENT_MIN_BOX_SIZE)
            & (box_height >= AUGMENT_MIN_BOX_SIZE)
        )

        # boxes、labels、face_indices、levelsの対応を保ったまま絞り込みます。
        target["boxes"] = cropped_boxes[keep]
        target["labels"] = target["labels"][keep]
        target["face_indices"] = target["face_indices"][keep]
        target["levels"] = [
            levels
            for levels, kept in zip(target["levels"], keep.tolist())
            if kept
        ]

        image = image.crop(
            (crop_x1, crop_y1, crop_x2, crop_y2)
        )

        return image, target

    def widerface_horizontal_flip(self, image, boxes):
        """画像とbboxを左右反転します。"""
        image_width = image.size[0]

        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        # 反転後は、元の右端x2が新しい左端x1になります。
        flipped_boxes = boxes.clone()
        flipped_boxes[:, 0] = image_width - boxes[:, 2]
        flipped_boxes[:, 2] = image_width - boxes[:, 0]

        return image, flipped_boxes

    def widerface_augment(self, image, target):
        """trainだけに、Scale CropとHorizontal Flipを確率的に適用します。"""
        if torch.rand(1).item() < AUGMENT_SCALE_CROP_PROB:
            image, target = self.widerface_scale_crop(
                image,
                target,
            )

        if torch.rand(1).item() < AUGMENT_HFLIP_PROB:
            image, target["boxes"] = self.widerface_horizontal_flip(
                image,
                target["boxes"],
            )

        return image, target

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

    def widerface_image_tensor(self, image):
        """PIL RGB画像を0から1のCHW FloatTensorへ変換します。"""
        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        return torch.from_numpy(
            image_array
        ).permute(2, 0, 1).contiguous()

    def widerface_sample(self, index):
        """1サンプルを構成する各部品を順番に作成します。"""
        annotation_path = self.annotation_files[index]
        annotation = self.widerface_load_annotation(
            annotation_path
        )

        image = self.widerface_image(annotation)
        target = self.widerface_target(annotation)

        # データ拡張は元画像サイズで行い、その後に共通のリサイズを行います。
        if self.use_augmentation:
            image, target = self.widerface_augment(
                image,
                target,
            )

        image, target["boxes"] = self.widerface_resize(
            image,
            target["boxes"],
        )

        image = self.widerface_image_tensor(image)

        target["split"] = self.split
        target["image_relative_path"] = annotation["image"]["relative_path"]
        target["image_width"] = int(annotation["image"]["width"])
        target["image_height"] = int(annotation["image"]["height"])

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
