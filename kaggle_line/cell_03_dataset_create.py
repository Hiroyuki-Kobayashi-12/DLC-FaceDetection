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
#   OUTPUT_DIRECTORY
#   SAVE_AUGMENTATION_SAMPLES
#   AUGMENTATION_SAMPLE_DIRECTORY_NAME
#   AUGMENTATION_SAMPLE_NUM
#   AUGMENTATION_SAMPLE_ALL_EPOCHS
#
# Augmentation Sample:
#
#   SAVE_AUGMENTATION_SAMPLES=True の場合、学習中にtrainの
#   index < AUGMENTATION_SAMPLE_NUM の画像について拡張前後を保存する。
#   AUGMENTATION_SAMPLE_ALL_EPOCHS=False の場合は1エポック目だけ保存する。
#
#   保存先:
#     OUTPUT_DIRECTORY / AUGMENTATION_SAMPLE_DIRECTORY_NAME / epoch_XXX /
#       {index}_before.jpg  元画像 + 元bbox
#                           (緑=残った顔、赤=クロップで除外、黄=切り出し範囲)
#       {index}_after.jpg   拡張 + 640x640リサイズ後(モデル入力) + bbox
#       {index}_info.json   適用した拡張の内容
#
#   エポック番号はCell 11が train_dataset.epoch へ設定する。
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
from PIL import Image, ImageDraw

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

        # 拡張サンプルの保存先。エポック番号はCell 11から毎エポック設定される。
        self.epoch = 1
        self.sample_directory = (
            Path(OUTPUT_DIRECTORY)
            / AUGMENTATION_SAMPLE_DIRECTORY_NAME
        )

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

    # ランダムクロップ
    def widerface_scale_crop(self, image, target):
        """元画像の一部を切り出し、顔を相対的に拡大します。"""
        # 元画像の幅と高さを取得する。
        # bboxもこの時点では元画像のピクセル座標。
        original_width, original_height = image.size

        # 切り出し倍率をAUGMENT_SCALE_CROP_MINから1.0の間で選ぶ。
        # 例: AUGMENT_SCALE_CROP_MIN=0.5 なら [0.5, 1.0) の一様乱数。
        scale = (
            AUGMENT_SCALE_CROP_MIN
            + (1.0 - AUGMENT_SCALE_CROP_MIN) * torch.rand(1).item()
        )

        # 幅と高さに同じ倍率を掛け、縦横比を保った切り出しサイズを求める。
        # 例: 1024x768 で scale=0.7 なら 716x537 を切り出す。
        crop_width = max(1, int(original_width * scale))
        crop_height = max(1, int(original_height * scale))

        # 切り出し範囲の左上座標をランダムに選ぶ。
        # randintの上限は含まないため +1 し、
        # 左上を 0 から (元サイズ - 切り出しサイズ) の範囲に収める。
        # これにより切り出し範囲が必ず画像内に収まる。
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

        # 切り出し範囲の右下座標を求める。
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        # 元画像座標のbbox [N, 4] (x1, y1, x2, y2) を取り出す。
        boxes = target["boxes"]

        # bbox中心が切り出し範囲に残る顔だけを対象にする。
        # 判定は平行移動前の元画像座標で行う。
        # 中心が範囲外の顔は大部分が切れるため、学習対象から外す。
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

        center_inside = (
            (center_x >= crop_x1)
            & (center_x < crop_x2)
            & (center_y >= crop_y1)
            & (center_y < crop_y2)
        )

        # 切り出し範囲の左上を原点とする座標へ変換し、範囲内へ収める。
        # x座標(0, 2列目)からcrop_x1、y座標(1, 3列目)からcrop_y1を引いて平行移動し、
        # はみ出した部分はclampで切り出し範囲の端に切り詰める。
        # 例: x=[90, 150], crop_x1=100 → [-10, 50] → [0, 50]
        # 元のboxesを書き換えないようcloneしてから変換する。
        cropped_boxes = boxes.clone()
        cropped_boxes[:, [0, 2]] = (
            cropped_boxes[:, [0, 2]] - crop_x1
        ).clamp(0.0, crop_width)
        cropped_boxes[:, [1, 3]] = (
            cropped_boxes[:, [1, 3]] - crop_y1
        ).clamp(0.0, crop_height)

        # 範囲外へはみ出して極端に小さくなった顔を除外する。
        # clamp後のbboxの幅と高さを計算する。
        box_width = cropped_boxes[:, 2] - cropped_boxes[:, 0]
        box_height = cropped_boxes[:, 3] - cropped_boxes[:, 1]

        # 「中心が範囲内」かつ「幅・高さがAUGMENT_MIN_BOX_SIZE以上」の顔だけ残す。
        # AUGMENT_MIN_BOX_SIZEはリサイズ前の元画像ピクセル単位。
        keep = (
            center_inside
            & (box_width >= AUGMENT_MIN_BOX_SIZE)
            & (box_height >= AUGMENT_MIN_BOX_SIZE)
        )

        # boxes、labels、face_indices、levelsの対応を保ったまま絞り込む。
        # Tensorはkeepマスクで、listのlevelsはzipで同じ顔だけを残す。
        target["boxes"] = cropped_boxes[keep]
        target["labels"] = target["labels"][keep]
        target["face_indices"] = target["face_indices"][keep]
        target["levels"] = [
            levels
            for levels, kept in zip(target["levels"], keep.tolist())
            if kept
        ]

        # 画像を切り出し範囲で切り出す。
        # bboxは上で切り出し後の座標へ変換済み。
        image = image.crop(
            (crop_x1, crop_y1, crop_x2, crop_y2)
        )

        # 拡張サンプル保存用に、切り出し範囲と倍率を返す。
        crop_info = {
            "scale": scale,
            "box": [crop_x1, crop_y1, crop_x2, crop_y2],
        }

        return image, target, crop_info

    def widerface_horizontal_flip(self, image, boxes):
        """画像とbboxを左右反転します。"""
        # 反転の基準となる画像幅を取得する。
        # Scale Crop後に呼ばれた場合は、切り出し後の幅になる。
        image_width = image.size[0]

        # 画像を左右反転する。
        image = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        # 反転後は、元の右端x2が新しい左端x1になる。
        # x座標を x → image_width - x で変換し、x1 < x2 の関係を保つ。
        # 例: image_width=1000, x=[100, 200] → [800, 900]
        # y座標は反転の影響を受けないため変更しない。
        # bboxの並び順は変わらないため、labelsなど他のtargetとの対応も保たれる。
        flipped_boxes = boxes.clone()
        flipped_boxes[:, 0] = image_width - boxes[:, 2]
        flipped_boxes[:, 2] = image_width - boxes[:, 0]

        return image, flipped_boxes

    def widerface_augment(self, image, target):
        """trainだけに、Scale CropとHorizontal Flipを確率的に適用します。"""
        # 各拡張は別々の乱数で判定するため、互いに独立して適用される。
        # 適用順は Scale Crop → Horizontal Flip で固定。
        # torch.randを使うため、DataLoaderの各workerで異なる乱数になる。

        # 拡張サンプル保存用に、適用した拡張の内容を記録する。
        # scale_crop: 未適用ならNone、適用時は切り出し範囲と倍率
        augment_info = {
            "scale_crop": None,
            "horizontal_flip": False,
        }

        # 確率AUGMENT_SCALE_CROP_PROBでScale Cropを適用する。
        # bbox以外のtarget(labels、face_indices、levels)も同時に絞り込まれる。
        if torch.rand(1).item() < AUGMENT_SCALE_CROP_PROB:
            image, target, augment_info["scale_crop"] = self.widerface_scale_crop(
                image,
                target,
            )

        # 確率AUGMENT_HFLIP_PROBでHorizontal Flipを適用する。
        # 顔の数や順番は変わらないため、boxesだけを更新する。
        if torch.rand(1).item() < AUGMENT_HFLIP_PROB:
            image, target["boxes"] = self.widerface_horizontal_flip(
                image,
                target["boxes"],
            )
            augment_info["horizontal_flip"] = True

        return image, target, augment_info

    def widerface_should_save_sample(self, index):
        """このサンプルの拡張前後を保存するかを判定します。"""
        # 拡張しないval、または保存OFFの場合は保存しない。
        if not (self.use_augmentation and SAVE_AUGMENTATION_SAMPLES):
            return False

        # AUGMENTATION_SAMPLE_ALL_EPOCHS=False の場合は1エポック目だけ保存する。
        if not AUGMENTATION_SAMPLE_ALL_EPOCHS and self.epoch != 1:
            return False

        # workerが並列に動いても同じ画像を選べるよう、indexで判定する。
        return index < AUGMENTATION_SAMPLE_NUM

    def widerface_draw_boxes(self, draw, boxes, color, width):
        """xyxy形式のbboxを描画します。"""
        for x1, y1, x2, y2 in boxes.tolist():
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=width,
            )

    def widerface_save_sample(
        self,
        index,
        annotation,
        before_image,
        before_target,
        after_image,
        after_target,
        augment_info,
    ):
        """拡張前後の画像と適用内容をエポック別ディレクトリへ保存します。"""
        epoch_directory = self.sample_directory / f"epoch_{self.epoch:03d}"
        epoch_directory.mkdir(parents=True, exist_ok=True)

        # クロップで除外された顔を、face_indicesの差分から求める。
        kept_face_indices = set(after_target["face_indices"].tolist())
        kept_mask = torch.tensor(
            [
                face_index in kept_face_indices
                for face_index in before_target["face_indices"].tolist()
            ],
            dtype=torch.bool,
        )

        # before: 元画像に、残った顔(緑)、除外された顔(赤)、切り出し範囲(黄)を描画する。
        # 線幅は元画像の大きさに合わせて調整する。
        before_image = before_image.copy()
        before_draw = ImageDraw.Draw(before_image)
        line_width = max(2, round(max(before_image.size) / 400))

        before_boxes = before_target["boxes"]
        self.widerface_draw_boxes(
            before_draw, before_boxes[kept_mask], "lime", line_width
        )
        self.widerface_draw_boxes(
            before_draw, before_boxes[~kept_mask], "red", line_width
        )

        if augment_info["scale_crop"] is not None:
            before_draw.rectangle(
                augment_info["scale_crop"]["box"],
                outline="yellow",
                width=line_width * 2,
            )

        # after: 拡張とリサイズ後のモデル入力画像に、変換後のbbox(緑)を描画する。
        after_image = after_image.copy()
        after_draw = ImageDraw.Draw(after_image)
        self.widerface_draw_boxes(
            after_draw, after_target["boxes"], "lime", 2
        )

        before_image.save(epoch_directory / f"{index:05d}_before.jpg", quality=90)
        after_image.save(epoch_directory / f"{index:05d}_after.jpg", quality=90)

        # 適用した拡張の内容をJSONで保存する。
        removed_face_indices = [
            face_index
            for face_index, kept in zip(
                before_target["face_indices"].tolist(),
                kept_mask.tolist(),
            )
            if not kept
        ]

        info = {
            "epoch": self.epoch,
            "index": index,
            "image_relative_path": annotation["image"]["relative_path"],
            "original_size": list(before_image.size),
            "after_size": list(after_image.size),
            "scale_crop": augment_info["scale_crop"],
            "horizontal_flip": augment_info["horizontal_flip"],
            "num_faces_before": len(before_boxes),
            "num_faces_after": len(after_target["boxes"]),
            "removed_face_indices": removed_face_indices,
        }

        with open(
            epoch_directory / f"{index:05d}_info.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(info, file, ensure_ascii=False, indent=2)

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

        # 拡張サンプルを保存する場合、拡張前の画像とtargetを控えておく。
        # 拡張処理はtargetのdictを書き換えるため、浅いコピーを残す。
        save_sample = self.widerface_should_save_sample(index)

        if save_sample:
            before_image = image
            before_target = dict(target)

        # データ拡張は元画像サイズで行い、その後に共通のリサイズを行います。
        augment_info = None

        if self.use_augmentation:
            image, target, augment_info = self.widerface_augment(
                image,
                target,
            )

        image, target["boxes"] = self.widerface_resize(
            image,
            target["boxes"],
        )

        # リサイズ後(モデル入力)の画像と拡張前の画像を保存する。
        if save_sample:
            self.widerface_save_sample(
                index=index,
                annotation=annotation,
                before_image=before_image,
                before_target=before_target,
                after_image=image,
                after_target=target,
                augment_info=augment_info,
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
