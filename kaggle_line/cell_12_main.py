# Cell 12 Main
# ============================================================
#
# このセルの役割:
#
#   Cell 2からCell 11で作成した部品を接続し、
#   学習、Validation、履歴出力、モデル保存を実行します。
#
# 保存方針:
#
#   - 各エポックを個別の.pthへ保存する
#   - Validation Lossが最小のエポックをbestとして記録する
#   - ONNXはbestと最終エポックの2つだけ出力する
#
# Configから受け取る定数:
#
#   NUM_EPOCHS
#   OUTPUT_DIRECTORY
#   CHECKPOINT_DIRECTORY_NAME
#   CHECKPOINT_FILE_PREFIX
#   HISTORY_JSON_FILE_NAME
#   HISTORY_IMAGE_FILE_NAME
#   BEST_ONNX_FILE_NAME
#   FINAL_ONNX_FILE_NAME
#   SHOW_HISTORY
#   AUGMENTATION_PREVIEW_FILE_NAME
#   AUGMENTATION_PREVIEW_SAMPLE_COUNT
#   AUGMENTATION_PREVIEW_COLUMN_COUNT
#   AUGMENTATION_PREVIEW_SEED
#   SHOW_AUGMENTATION_PREVIEW
#
# 前のセルから受け取るもの:
#
#   model
#   DEVICE
#   MODEL_METADATA
#   train_loader
#   val_loader
#   criterion
#   optimizer
#   scheduler
#   train_one_epoch()
#   validate_one_epoch()
#   create_history()
#   add_history()
#   save_history()
#   visualize_history()
#   save_checkpoint()
#   load_checkpoint_model()
#   export_onnx()
#   visualize_train_augmentation()
#
# 最終的に残すもの:
#
#   history
#   best_validation_loss
#   best_checkpoint_path
#   final_checkpoint_path
#
# ============================================================


from pathlib import Path
import copy


# ============================================================
# Output Paths
# ============================================================

output_directory = Path(OUTPUT_DIRECTORY)
checkpoint_directory = output_directory / CHECKPOINT_DIRECTORY_NAME
history_json_path = output_directory / HISTORY_JSON_FILE_NAME
history_image_path = output_directory / HISTORY_IMAGE_FILE_NAME
best_onnx_path = output_directory / BEST_ONNX_FILE_NAME
final_onnx_path = output_directory / FINAL_ONNX_FILE_NAME
augmentation_preview_path = output_directory / AUGMENTATION_PREVIEW_FILE_NAME

output_directory.mkdir(parents=True, exist_ok=True)
checkpoint_directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Augmentation Preview
# ============================================================

# 毎回同じ16枚と同じ可視化Seedを使用します。
# 可視化関数は実行前の乱数状態を復元するため、学習時の変換へ影響しません。
visualize_train_augmentation(
    train_dataset=train_dataset,
    save_path=augmentation_preview_path,
    sample_count=AUGMENTATION_PREVIEW_SAMPLE_COUNT,
    column_count=AUGMENTATION_PREVIEW_COLUMN_COUNT,
    visualization_seed=AUGMENTATION_PREVIEW_SEED,
    show=SHOW_AUGMENTATION_PREVIEW,
)


# ============================================================
# Training State
# ============================================================

history = create_history()
best_validation_loss = float("inf")
best_checkpoint_path = None
final_checkpoint_path = None


# ============================================================
# Training Loop
# ============================================================

for epoch in range(1, NUM_EPOCHS + 1):
    train_result = train_one_epoch(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    validation_result = validate_one_epoch(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
    )

    add_history(
        history=history,
        epoch=epoch,
        train_result=train_result,
        validation_result=validation_result,
        optimizer=optimizer,
    )

    if scheduler is not None:
        scheduler.step()

    epoch_checkpoint_path = (
        checkpoint_directory
        / f"{CHECKPOINT_FILE_PREFIX}_{epoch:03d}.pth"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        model_metadata=MODEL_METADATA,
        epoch=epoch,
        validation_loss=validation_result["loss"],
        save_path=epoch_checkpoint_path,
    )

    final_checkpoint_path = epoch_checkpoint_path

    if validation_result["loss"] < best_validation_loss:
        best_validation_loss = validation_result["loss"]
        best_checkpoint_path = epoch_checkpoint_path

    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
        f"Train {train_result['loss']:.6f} | "
        f"Validation {validation_result['loss']:.6f}"
    )


# ============================================================
# History Output
# ============================================================

save_history(
    history=history,
    save_path=history_json_path,
)

visualize_history(
    history=history,
    save_path=history_image_path,
    show=SHOW_HISTORY,
)


# ============================================================
# ONNX Output
# ============================================================

# 最終エポックのモデルを、そのままONNXへ出力します。
export_onnx(
    model=model,
    model_metadata=MODEL_METADATA,
    save_path=final_onnx_path,
)

# 現在の最終モデルを変更しないように複製し、
# bestの.pthを読み込んでONNXへ出力します。
best_model = load_checkpoint_model(
    model=copy.deepcopy(model),
    checkpoint_path=best_checkpoint_path,
)

export_onnx(
    model=best_model,
    model_metadata=MODEL_METADATA,
    save_path=best_onnx_path,
)
