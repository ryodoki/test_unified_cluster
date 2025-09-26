# Unified Cluster Ja — README

日本語テキスト向けクラスタリング統合ツールです。次の2系統の処理を1つのCLIにまとめています。

* **ngram**: 文字 n-gram TF‑IDF と KMeans によるクラスタリング（単一テキスト列）
* **qa**: Sentence Transformers による意味ベクトル化とクラスタリング（「質問箇所」「質問内容」の2列）
* **both**: 上記2つを同じ入力から実行し、1つのExcelファイルにシート分けで出力

本ツールは `unified_cluster_ja.py`（キャンバスのコード）として提供されています。

---

## 目次

* [動作環境](#動作環境)
* [インストール](#インストール)
* [基本の使い方](#基本の使い方)
* [各モードの詳細](#各モードの詳細)

  * [ngram モード](#ngram-モード)
  * [qa モード](#qa-モード)
  * [both モード](#both-モード)
* [対話ウィザードと非対話環境](#対話ウィザードと非対話環境)

  * [UNIFIED\_WIZARD\_PRESET（無人実行のためのプリセット）](#unified_wizard_preset無人実行のためのプリセット)
* [出力仕様](#出力仕様)
* [自己テスト（--self-test）](#自己テスト--self-test)
* [列の自動推定ロジック（qa）](#列の自動推定ロジックqa)
* [トラブルシュート](#トラブルシュート)
* [よくある質問](#よくある質問)

---

## 動作環境

* Python 3.9 以上を推奨
* OS は問いません（Windows/macOS/Linux）

## インストール

依存パッケージ（最低限）:

```bash
pip install pandas numpy scikit-learn openpyxl
```

qa モードを Sentence Transformers で使う場合:

```bash
pip install sentence-transformers
```

オプション（`--method hdbscan` を使う場合）:

```bash
pip install hdbscan
```

> すぐ試したいだけなら、`--model debug-tfidf` を使えば追加のモデルダウンロードなしで動作します。

---

## 基本の使い方

### コマンド構文

```bash
# n-gram モード（単一テキスト列）
python unified_cluster_ja.py ngram <入力.xlsx> <出力.xlsx> \
  [--text-col 列名] [--min-df 2] [--ngram-min 2] [--ngram-max 10]

# QA モード（質問箇所 + 質問内容）
python unified_cluster_ja.py qa <入力.xlsx> <出力.xlsx> \
  [--sheet SHEET] [--col_place 質問箇所列名] [--col_body 質問内容列名] \
  [--method auto_kmeans|hdbscan] \
  [--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2]

# 両方まとめて実行
python unified_cluster_ja.py both <入力.xlsx> <出力.xlsx> [共通/各モードのオプション]
```

### 例

```bash
# 例1: n-gram
python unified_cluster_ja.py ngram 相談一覧.xlsx 出力_ngram.xlsx --text-col 本文

# 例2: QA（Sentence Transformers）
python unified_cluster_ja.py qa 質疑集約.xlsx 出力_qa.xlsx \
  --col_place 質問箇所 --col_body 質問内容

# 例3: QA（軽量デバッグ埋め込み: モデルDL不要）
python unified_cluster_ja.py qa 質疑集約.xlsx 出力_qa.xlsx \
  --col_place 質問箇所 --col_body 質問内容 --model debug-tfidf

# 例4: 2系統まとめて
python unified_cluster_ja.py both 入力.xlsx 出力_両方.xlsx \
  --text-col 本文 --col_place 質問箇所 --col_body 質問内容
```

Windows PowerShell の例:

```powershell
python .\unified_cluster_ja.py ngram .\input.xlsx .\out.xlsx --text-col 本文
```

---

## 各モードの詳細

### ngram モード

* **対象**: 単一のテキスト列
* **手順**: 前処理 → 文字 n-gram TF‑IDF → KMeans（クラスタ数はシルエット係数で自動探索）
* **主要オプション**

  * `--text-col`: テキスト列名。未指定時は `'text'` または最初の文字列列を自動使用
  * `--min-df`: TF‑IDF の `min_df`（既定: 2）
  * `--ngram-min`, `--ngram-max`: 文字 n-gram 範囲（既定: 2〜10）

### qa モード

* **対象**: 「質問箇所」と「質問内容」の2列（自動推定あり）
* **手順**: ヘッダ行推定 → 列名推定 → 文埋め込み → KMeans または HDBSCAN → 要約テーブル
* **主要オプション**

  * `--sheet`: シート名（未指定は先頭）
  * `--col_place`, `--col_body`: 列名を明示指定（自動推定が失敗した場合に使用）
  * `--method`: `auto_kmeans`（既定）または `hdbscan`
  * `--model`: 既定は `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`。
    ダウンロード不要の軽量確認には `debug-tfidf` を使います。

### both モード

* **対象**: ngram と qa を連続実行し、1つのExcelにシート分けで保存
* **補足**: 一時ファイルにそれぞれ書き出した後、`ngram_*/qa_*` のシート名でマージします。

---

## 対話ウィザードと非対話環境

* **引数なし + 対話可能**: 対話ウィザードが起動します。質問に答えるだけで実行できます。
* **引数なし + 対話不可（サーバやノートブック等）**: `input()` は呼ばず、ヘルプを表示して終了します。
* **`--interactive`**: 明示的にウィザードを起動します。

### UNIFIED\_WIZARD\_PRESET（無人実行のためのプリセット）

非対話環境で**引数なし**でも実行したい場合、環境変数またはJSONファイルでプリセットを渡せます。

環境変数の例:

```bash
export UNIFIED_WIZARD_PRESET='{
  "mode":"ngram",
  "input":"in.xlsx",
  "output":"out.xlsx",
  "text_col":"本文",
  "min_df":1,
  "ngram_min":2,
  "ngram_max":5
}'
python unified_cluster_ja.py
```

ファイルの例（カレントに `unified_wizard_preset.json` を置く）:

```json
{
  "mode": "qa",
  "input": "in.xlsx",
  "output": "out.xlsx",
  "sheet": "Sheet1",
  "col_place": "質問箇所",
  "col_body": "質問内容",
  "method": "auto_kmeans",
  "model": "debug-tfidf"
}
```

> プリセットで指定可能なキーは、各モードのオプション名と同じです。

---

## 出力仕様

* **ngram**

  * `rows`: 入力行 + `cleaned_text`, `cluster`, `center_distance`
  * `clusters`: クラスタIDごとの上位 n-gram（最大15件）
  * `metrics`: `silhouette_cosine` を1行で格納
* **qa**

  * `rows`: 入力行 + `cluster_id`, `cluster_name`, `cluster_rep`, `__concat_text__`
  * `summary`: `cluster_id`, `cluster_name`, `count`, `representative`
* **both**

  * `ngram_rows`, `ngram_clusters`, `ngram_metrics`, `qa_rows`, `qa_summary` といった**接頭辞付き**シート名になります。

---

## 自己テスト（--self-test）

ツール内部に軽量テストを内蔵しています（外部モデル不要）。

```bash
python unified_cluster_ja.py --self-test
```

検証内容（抜粋）:

* n-gram: 行数整合、`cleaned_text` の存在
* QA: `rows` と `summary` の存在、`summary.count` 合計 = `rows` 件数
* both: `ngram_*` と `qa_*` の両系統シートが作成されること
* プリセット経由ノー引数実行の成立（`UNIFIED_WIZARD_PRESET` で無人走行）

---

## 列の自動推定ロジック（qa）

* **ヘッダ行推定**: 上から最大5行（`header=0..4`）を試し、候補列が同時に見つかった最初の行を採用
* **候補名**

  * 質問箇所: `質問箇所/質疑箇所/設問箇所/質問場所/設問場所/該当箇所/対象箇所/質問部位/質問位置/記載箇所` 等
  * 質問内容: `質問内容/質疑内容/設問内容/照会内容/問合せ内容/問い合わせ内容/質問事項/質疑/問合せ` 等
* 上記で見つからない場合でも、文字列に「質問」「箇所」「場所」「内容」等が含まれる列をヒューリスティックに選びます。

---

## トラブルシュート

* **`OSError: [Errno 29] I/O error` が `input()` で発生**

  * 非対話環境での `input()` 呼び出しが原因です。本ツールは**対話不可時は自動で `input()` を避ける**よう改善済みです。
  * ノー引数で無人実行したい場合は、[UNIFIED\_WIZARD\_PRESET](#unified_wizard_preset無人実行のためのプリセット) を使ってください。
* **`KeyError: 必要な列が見つかりません`（qa）**

  * 自動推定に失敗。`--col_place` と `--col_body` を明示してください。
* **`sentence-transformers` が無い**

  * `pip install sentence-transformers` を行うか、軽量確認用に `--model debug-tfidf` を使ってください。
* **クラスタ数の自動決定が極端**（ngram/qa）

  * 入力が少ない、または文が類似しすぎるとシルエット係数の計算が不安定です。入力を増やす、前処理を調整するなどで改善します。

---

## よくある質問

**Q. Excelファイルのシートを指定したい**
A. `--sheet SHEET` を使ってください（qa/both）。未指定時は先頭シートを使います。

**Q. 列名が微妙に違う**
A. 自動推定が効かない場合は `--col_place`, `--col_body` を明示してください。

**Q. モデルのダウンロードを避けたい**
A. `--model debug-tfidf` を使えば追加ダウンロード無しで動きます（精度は簡易版）。

**Q. 非対話のバッチ実行にしたい**
A. サブコマンドに必要な引数をすべて渡してください。ノー引数で無人実行したい場合はプリセット（環境変数/JSON）をご利用ください。

---

以上です。運用ポリシーとして、**対話不可環境で引数なしの場合は実行せずヘルプ表示**を既定としています。必要に応じてプリセットをご活用ください。
