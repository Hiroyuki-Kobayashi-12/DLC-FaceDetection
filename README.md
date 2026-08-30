# DLC Face Detection

WIDER FACEを使用した顔検出モデルの学習・評価を、GitHub、Kaggle、ローカル環境で管理するためのリポジトリです。

参加者は、自分の実験Branchでコードを作成・変更し、GitHubへCommit・Pushした後、Kaggleから対象BranchをCloneして学習します。学習後は、ローカルの`tools/`で数値評価を行い、各自の`results/`に可視化と考察を残します。

```text
VS Codeで実験コードを準備
↓
自分の実験BranchへCommit・Push
↓
Kaggleで自分の実験BranchをClone
↓
kaggle_lineをファイル名順に実行
↓
学習成果物をローカルへ取得
↓
toolsで数値評価
↓
resultsへ可視化・考察を記録
↓
実験完了CommitへTagを付与
```

---

## 1. 運用の基本原則

1. 日常の実験は、自分の実験Branch内で行う
2. Kaggleで使用するコードは、事前にCommit・Pushする
3. Kaggleでは、GitHub上の自分の実験BranchをCloneする
4. Cloneした`kaggle_line`をファイル名順に実行する
5. Kaggle上だけにコード変更を残さない
6. 学習後は、共通評価スクリプトで数値評価する
7. 各自で評価結果を可視化し、結果を説明できる状態にする
8. `main`は初期環境と年度ごとの整理に使用し、日常の実験では操作しない
9. 大容量データ、モデル、秘密情報をGitHubへ登録しない

> Kaggleで使用したコード、ローカルで評価したコード、`results/`へ記録した結果が、同じ実験として追跡できる状態を維持してください。

---

## 2. リポジトリ構成

```text
DLC-FaceDetection/
├── data/
├── kaggle_line/
├── tools/
├── results/
├── kaggle.ipynb
├── .gitignore
└── README.md
```

### `data/`

ローカル評価で使用する画像、JSONアノテーション、評価用情報を配置します。

WIDER FACE画像本体などの大容量データは、原則としてGitHubへ登録しません。

### `kaggle_line/`

Kaggleで実行する学習コードを配置します。

初期版では、処理ごとにPythonファイルを分割しています。

```text
kaggle_line/
├── cell_01_config.py
├── cell_02_model_create.py
├── cell_03_dataset_create.py
├── cell_04_loss_create.py
├── cell_05_optimizer_create.py
├── cell_06_scheduler_create.py
├── cell_07_train_one_epoch.py
├── cell_08_validate_one_epoch.py
├── cell_09_history_visualize.py
├── cell_10_model_export.py
└── cell_11_main.py
```

この構成は初期例です。参加者は、自分の実験Branch内で初期版を参考にしながら、`kaggle_line`を自由に作成・変更できます。

変更例：

- モデルの追加・変更
- Lossの追加・変更
- OptimizerやSchedulerの変更
- Dataset処理の変更
- TransformやData Augmentationの追加
- 学習ループの変更
- Pythonファイルの追加・削除
- 処理単位やファイル構成の再設計

ファイル名順に実行する場合は、実行順が分かる名前を使用してください。

```text
cell_01_*.py
cell_02_*.py
cell_03_*.py
```

`kaggle_line`内へ、実行対象と区別できないバックアップファイルを置かないでください。

```text
cell_04_loss_create_old.py
cell_04_loss_create_copy.py
```

過去のコードはGitの履歴から確認します。

### `kaggle.ipynb`

GitHubから取得した実験コードを、Kaggle上で起動するためのNotebookです。

Notebookは、Cloneしたリポジトリの`kaggle_line`をファイル名順に実行します。学習コードの正本は`kaggle_line`で管理し、Notebook内へ重複して保持しません。

### `tools/`

ローカルでモデルを評価するスクリプトを配置します。

初期版として、全参加者が共通して使用できる数値評価スクリプトを用意します。参加者は共通評価スクリプトを基準として、同じ条件でモデル性能を確認します。

### `results/`

各参加者が、数値評価を可視化し、実験結果を説明するための成果物を配置します。

具体的な構成と記録形式は、今後の運用で整備します。

---

## 3. Branchの役割

### `main`

`main`は、次の目的に限定して使用します。

- プロジェクト開始時の初期環境を保持する
- 年度ごとに共通資産と成果を整理する

`main`は、日常的な実験を行う場所ではありません。参加者は基本的に`main`を直接操作せず、自分の実験Branchで作業してください。

`main`へ反映する必要がある変更は、年度の整理または共通環境の改善として内容を確認し、Pull Requestを使用します。

### 自分の実験Branch

各参加者は、自分専用の実験Branchを作成します。

```text
sandbox/<GitHubユーザー名>
```

例：

```text
sandbox/kobayashi
sandbox/sato
```

モデル作成、学習、評価、可視化、実験記録は、自分の実験Branch内で行います。

自分の実験Branchでは、途中の変更、失敗した試行、一時的に動かない状態も許容します。成功した結果だけでなく、試行錯誤をCommitとして残してください。

### 他の参加者のBranch

他の参加者のBranchは、コードや実験内容を確認するために参照できます。

本人の同意なく、他の参加者のBranchへCommit・Pushしないでください。

---

## 4. 初回準備

### 必要な環境

- GitHubアカウント
- Git
- Visual Studio Code
- VS CodeのPython拡張機能
- VS CodeのJupyter拡張機能
- Git GraphまたはGitLens
- Kaggleアカウント
- ローカル評価用のPython環境

### リポジトリをCloneする

1. GitHubで`DLC-FaceDetection`を開く
2. `Code`からHTTPS URLをコピーする
3. VS Codeでコマンドパレットを開く
4. `Git: Clone`を選択する
5. URLを貼り付ける
6. 保存先を選択する
7. Cloneしたフォルダを開く

### 自分の実験Branchを作る

1. `main`へ切り替える
2. PullまたはFetchで最新状態を取得する
3. `main`から自分の実験Branchを作成する
4. `Publish Branch`でGitHubへ公開する
5. VS Code左下が自分の実験Branchになっていることを確認する

```text
main
└── sandbox/<GitHubユーザー名>
```

---

## 5. 実験準備とGitHubへの反映

### 1. 自分の実験Branchを確認する

作業開始前に、現在のBranchを確認します。

```text
sandbox/<GitHubユーザー名>
```

`main`や他の参加者のBranchで作業しないでください。

### 2. `kaggle_line`を準備する

初期版を参考に、実験内容に合わせて自由に変更します。

初期構成では、通常の学習条件を次のファイルで管理します。

```text
kaggle_line/cell_01_config.py
```

初期構成を維持する必要はありません。自分の実験に適した構成へ変更できます。

### 3. 差分を確認する

VS Codeのソース管理画面で、変更内容を確認します。

- 実験と関係のない変更が含まれていないか
- 設定値が意図した内容か
- Datasetやモデルが含まれていないか
- 秘密情報が含まれていないか

### 4. Commitする

Kaggleで学習する前に、使用するコードをCommitします。

```text
experiment: 入力画像サイズを640へ変更
experiment: Focal Lossを追加
experiment: Schedulerを変更
fix: Validation時の出力処理を修正
```

1つのCommitには、可能な限り1つの目的を持たせます。

### 5. Pushする

Commit後は、自分の実験BranchをGitHubへPushします。

```text
変更
↓
Stage
↓
Commit
↓
Push
↓
origin/sandbox/<GitHubユーザー名>
```

CommitしただけではKaggleから取得できません。必ずPushまで実行してください。

---

## 6. Kaggleでモデルを学習する

### 1. 自分の実験Branchを指定する

Kaggleの起動用Notebookで、自分の実験Branchを指定します。

```python
GIT_REFERENCE = "sandbox/<GitHubユーザー名>"
```

例：

```python
GIT_REFERENCE = "sandbox/kobayashi"
```

Kaggleは、指定したBranchの最新CommitをCloneします。

実行するコードを固定する場合は、Commit SHAまたはTagを指定します。

```python
GIT_REFERENCE = "完全なCommit SHA"
```

```python
GIT_REFERENCE = "exp/kobayashi/001-073%"
```

### 2. `kaggle_line`を実行する

起動用Notebookは、Cloneしたリポジトリの`kaggle_line`をファイル名順に実行します。

```text
自分の実験BranchをClone
↓
kaggle_line/cell_*.pyを取得
↓
ファイル名順に実行
↓
学習と成果物出力
```

Kaggle上でPythonファイルを貼り替える必要はありません。

### 3. 学習成果物を保存する

初期版では、次の成果物をKaggle Outputへ保存します。

```text
/kaggle/working/dlc26_outputs/
├── checkpoints/
│   ├── epoch_001.pth
│   ├── epoch_002.pth
│   └── ...
├── training_history.json
├── training_history.png
├── model_best.onnx
└── model_final.onnx
```

実験コードを変更した場合は、各自の実装に従って必要な成果物を保存します。

### 4. Kaggle上だけに変更を残さない

Kaggle上で一時的にコードを変更した場合は、正式な変更をVS Code側へ反映します。

```text
Kaggleで原因を確認
↓
VS Code側の自分の実験Branchへ反映
↓
Commit・Push
↓
Kaggleで自分の実験Branchを再取得
↓
正式に再実行
```

正式な実験は、GitHubへPush済みのコードで行います。

---

## 7. `tools/`で数値評価する

Kaggleで学習が完了したら、ONNXモデルと必要な成果物をローカル環境へ取得します。

`tools/`には、モデルを共通条件で数値評価するスクリプトを配置します。初期版として、WIDER FACE Validationを使用し、Easy、Medium、Hardごとに評価できる共通スクリプトを用意します。

初期版の主な評価内容は次のとおりです。

- WIDER FACE ValidationへのONNX推論
- PredictionとGround TruthのIoUマッチング
- 対象外難易度に対応したPredictionの除外
- Easy、Medium、HardごとのAP@0.5
- Precision
- Recall
- F1
- TP、FP、FN、IGNORE
- Prediction単位の評価結果

共通評価スクリプトは、参加者間で数値を比較するための基準です。共通指標の意味や判定方法を変更する場合は、他の実験との比較可能性を確認してください。

評価時は、少なくとも次を記録します。

- 評価したモデル
- 実験Branch
- Commit SHAまたはTag
- 使用したDataset
- Confidence閾値
- NMSのIoU閾値
- GTマッチングのIoU閾値
- Easy、Medium、Hardの評価値

数値評価が完了しただけでは、実験完了とはしません。次の`results/`で結果を可視化し、結果を説明できる状態まで整理します。

---

## 8. `results/`で可視化・考察する

`results/`には、`tools/`で取得した数値評価を基に、各参加者が作成した可視化と実験の説明を配置します。

共通評価スクリプトは数値評価の初期基盤です。参加者は数値を確認するだけで終わらず、モデルの特徴、改善点、失敗傾向を説明するための可視化へ取り組んでください。

可視化の例：

- Easy、Medium、Hardの評価値比較
- Precision-Recall Curve
- Confidence閾値ごとのPrecision、Recall、F1
- TP、FP、FN、IGNOREの件数比較
- PredictionとGround Truthの重ね合わせ
- 検出に成功した代表画像
- 誤検出した代表画像
- 見逃した代表画像
- 小さい顔、遮蔽された顔、密集した顔の分析
- 実験間の評価値比較

可視化は装飾を目的とするものではありません。数値から読み取れる傾向を示し、モデルの結果を他の参加者へ説明するために使用します。

各実験では、少なくとも次を説明できる状態にします。

- 何を変更したか
- 変更した理由
- どの評価値が改善・悪化したか
- Easy、Medium、Hardで傾向が異なるか
- どのような顔を検出できたか
- どのような顔を見逃したか
- どのような誤検出が発生したか
- 結果から何が分かったか
- 次に何を試すか

`results/`の具体的なディレクトリ構成と記録形式は、今後の運用で整備します。それまでは、各自の実験Branch内で、数値評価、可視化、説明の対応関係が分かる形で保存してください。

```text
数値評価
↓
可視化
↓
モデルの傾向を分析
↓
結果を文章で説明
↓
次の実験へつなげる
```

学習、数値評価、可視化、考察まで完了した時点を、1つの実験完了として扱います。

---

## 9. Tagによる実験記録

Tagは、モデルの学習、数値評価、可視化、考察まで完了した実験を固定するために使用します。

途中経過や学習コードだけの状態にはTagを付けません。細かな変更はCommitとして記録し、次の作業が完了した時点でAnnotated Tagを作成します。

1. 自分の実験Branchでコードを準備する
2. 実験コードをCommit・Pushする
3. Kaggleでモデルの学習を完了する
4. 学習成果物をローカルへ取得する
5. `tools/`で数値評価する
6. `results/`へ可視化と考察を記録する
7. 評価結果と実験記録をCommit・Pushする
8. 実験完了CommitへAnnotated Tagを付ける
9. Tagを`origin`へPushする

```text
学習コードをCommit・Push
↓
Kaggleで学習
↓
toolsで数値評価
↓
resultsで可視化・考察
↓
実験記録をCommit・Push
↓
Annotated Tagを作成・Push
```

推奨形式：

```text
exp/<GitHubユーザー名>/<3桁連番>
```

例：

```text
exp/kobayashi/001
exp/kobayashi/002
exp/sato/001
```

Tag名には精度を含めず、Tagメッセージと実験記録へ記載します。

Tagメッセージには、少なくとも次を記載します。

- 実験概要
- 使用モデル
- Validation Loss
- Easy AP
- Medium AP
- Hard AP
- Kaggle Outputの保存先
- `results/`内の実験記録
- 関連Issue

Kaggleで同じ実験コードを再現する場合は、Tagを指定します。

```python
GIT_REFERENCE = "exp/kobayashi/001-073%"
```

PTHやONNXなどの大容量成果物はTagに含まれません。実験記録に記載された保存先から取得します。

---

## 10. IssueとPull Request

### Issue

Issueは、課題、疑問、仮説、相談事項を共有するために使用します。

例：

- Validation Lossが安定しない
- Hardの顔を検出できない
- ONNX出力に失敗する
- Kaggleとローカルで結果が異なる
- 評価スクリプトを改善したい
- 可視化方法を相談したい

関連するCommitにはIssue番号を記載できます。

```text
fix: ONNX出力処理を修正 #5
```

### Pull Request

自分の実験Branchの全変更を`main`へ反映する必要はありません。

`main`へ提案するのは、年度のまとめや、複数の参加者が利用する価値のある変更です。

- 初期環境の不具合修正
- 共通Dataset処理の改善
- 共通数値評価スクリプトの改善
- 安定したモデル実装
- READMEの改善
- 年度成果の整理

`main`へ直接Pushせず、Pull Requestを使用します。

---

## 11. `main`への反映

`main`は、次のタイミングで整理します。

### 初期環境の整備

参加者が実験を開始できる基準環境を配置します。

### 年度ごとのまとめ

年度内の実験結果を確認し、共通利用するコード、評価スクリプト、文書、成果を整理して反映します。

日常の細かな実験を、都度`main`へMergeする運用にはしません。

---

## 12. GitHubへ登録しないもの

原則として、次をGitHubへ登録しません。

- WIDER FACE画像本体
- Kaggle Dataset本体
- 各エポックのPTH
- ONNXモデル
- 大量の予測結果
- 大量の可視化画像
- キャッシュ
- 一時ファイル
- 仮想環境

次の秘密情報も登録しないでください。

- Kaggle API Token
- `kaggle.json`
- GitHub Token
- パスワード
- アクセストークン
- `.env`
- 個人情報
- 社内限定情報

Commit前に、Stageしたファイルを必ず確認してください。

---

## 13. 実験前チェックリスト

- [ ] 自分の実験Branchにいる
- [ ] 実験内容に合わせて`kaggle_line`を準備した
- [ ] Configの設定値を確認した
- [ ] VS Codeで差分を確認した
- [ ] Datasetや秘密情報がStageされていない
- [ ] 実験コードをCommitした
- [ ] 自分の実験BranchをPushした
- [ ] Kaggleの`GIT_REFERENCE`を確認した
- [ ] Kaggleへ必要なDatasetが追加されている
- [ ] KaggleのAcceleratorとInternet設定を確認した

---

## 14. 実験完了チェックリスト

- [ ] Kaggleで使用したBranch、Commit SHA、Tagを確認した
- [ ] モデル学習が完了した
- [ ] 学習履歴を確認した
- [ ] 必要なPTHとONNXを保存した
- [ ] Kaggle Outputの保存先を記録した
- [ ] `tools/`で数値評価した
- [ ] Easy、Medium、Hardの評価値を記録した
- [ ] `results/`へ可視化を保存した
- [ ] 評価結果を文章で説明した
- [ ] 分かったことを記録した
- [ ] 失敗したことを記録した
- [ ] 次に試すことを記録した
- [ ] 評価結果と実験記録をCommit・Pushした
- [ ] 実験完了CommitへAnnotated Tagを付けた
- [ ] Tagを`origin`へPushした

---

## 15. 困ったとき

### Commitした内容がKaggleへ反映されない

Commit後にPushしたか確認してください。KaggleはGitHub上のコードを取得します。

### Kaggleで古いコードが実行される

- `GIT_REFERENCE`を確認する
- 対象Branchへ最新CommitをPushしたか確認する
- 実行時に表示されたCommit SHAを確認する

### `kaggle_line`のファイルが重複実行される

実行対象に一致するバックアップファイルがないか確認してください。

### 他の参加者の変更が見えない

Fetchを実行してください。

### CommitしたのにGitHubへ反映されない

Commitはローカル保存です。Pushまたは変更の同期を実行してください。

### Branchを切り替えられない

未Commitの変更が残っていないか、VS Codeのソース管理画面で確認してください。

---

## 16. 最小運用ルール

```text
mainから自分の実験Branchを作成
↓
自分の実験Branchでkaggle_lineを準備
↓
Commit・Push
↓
Kaggleで自分の実験BranchをClone
↓
kaggle_lineをファイル名順に実行
↓
学習成果物をローカルへ取得
↓
toolsで共通数値評価
↓
resultsで各自が可視化・考察
↓
実験記録をCommit・Push
↓
実験完了CommitへTagを作成・Push
```

**日常の実験は自分のBranchで行い、GitHubへPush済みのコードをKaggleで学習し、共通評価と各自の可視化・考察まで完了させる。**

これを本プロジェクトの基本運用とします。
