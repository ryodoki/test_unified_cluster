#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_cluster_ja_v3.py
v3 fix: sheet_name=None のときに pandas が全シート dict を返して落ちる問題を修正。
未指定なら「先頭シート(0)」を使う。
他は v2 と同様。

このファイルは、Excel 上の「質問箇所」と「質問内容」などの列から
文章を生成し、多言語対応の埋め込みモデルでベクトル化してクラスタリングします。
結果は行ごとの付与情報とクラスタのサマリーの 2 シートで出力します。

使い方:
  python qa_cluster_ja_v3.py input.xlsx output.xlsx
  python qa_cluster_ja_v3.py input.xlsx output.xlsx --sheet Sheet1
  python qa_cluster_ja_v3.py input.xlsx output.xlsx --col_place 質問箇所 --col_body 質問内容

主な処理の流れ:
  1) 見出し行の自動検出（最大 5 行を走査）
  2) 列名の自動推定（候補語とヒューリスティック）
  3) 文のクリーニングと連結（「箇所。 内容」）
  4) SentenceTransformer による埋め込み生成
  5) KMeans（自動 k 推定）または HDBSCAN によるクラスタリング
  6) 各クラスタの代表文とクラスタ名（TF-IDF 上位語の連結）の作成
  7) Excel へ出力（rows / summary）
"""
import argparse
import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# HDBSCAN は任意依存。未インストールでも動くように try-import。
try:
    import hdbscan as _hdbscan
except Exception:
    _hdbscan = None


def norm(s: str) -> str:
    """
    列名などの比較用に、空白・改行・全角空白を除去し、小文字化した正規化文字列を返す。
    例: " 質問 箇所 " -> "質問箇所"

    引数:
        s: 入力文字列（非文字列は str() して扱う）

    戻り値:
        正規化された文字列
    """
    s = str(s)
    # 改行や全角空白を単純化
    s = s.replace("\n", "").replace("\r", "").replace("\u3000", " ")
    # 連続空白を除去（完全に詰める）
    s = re.sub(r"\s+", "", s)
    return s.lower()


def clean_text(s: str) -> str:
    """
    文面をモデル入力用に整形する。
    改行をスペースにし、連続空白を 1 個にし、前後の空白を削る。

    引数:
        s: 入力文字列

    戻り値:
        クリーンアップ後の文字列
    """
    s = str(s).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 列名自動推定に使う候補語。実運用でよくある日本語のカラム名を網羅。
CAND_PLACE = ["質問箇所","質疑箇所","設問箇所","質問場所","設問場所","該当箇所","対象箇所","質問部位","質問位置","記載箇所"]
CAND_BODY  = ["質問内容","質疑内容","設問内容","照会内容","問合せ内容","問い合わせ内容","質問事項","質疑","問合せ"]


def guess_columns(df: pd.DataFrame) -> (Optional[str], Optional[str]):
    """
    データフレームから「箇所（place）」と「本文（body）」の列名を推定する。
    1) よくある列名の候補語から厳密一致（正規化比較）
    2) 候補がなければヒューリスティック（「質問」かつ「箇所/場所/部位/位置」など）

    引数:
        df: 入力 DataFrame（ヘッダは読み込み時点のまま）

    戻り値:
        (col_place, col_body)。見つからない場合は None を返す。
    """
    cols = list(df.columns)
    # 正規化名 -> 実列名 の辞書を作る
    mapping = {norm(c): c for c in cols}

    # まず候補語から厳密一致
    place = None
    for p in CAND_PLACE:
        if norm(p) in mapping:
            place = mapping[norm(p)]
            break

    body = None
    for b in CAND_BODY:
        if norm(b) in mapping:
            body = mapping[norm(b)]
            break

    # 候補が無ければヒューリスティックに走査
    if place is None:
        for c in cols:
            n = norm(c)
            # 「質問+箇所/場所/部位/位置」含み、または「設計」で番号/日/名を含まないもの
            if "質問" in n and ("箇所" in n or "場所" in n or "部位" in n or "位置" in n):
                place = c
                break
            if "設計" in n and ("番号" not in n and "日" not in n and "名" not in n):
                place = c
                break

    if body is None:
        for c in cols:
            n = norm(c)
            # 「質問/質疑/照会/問合」を含み、かつ「内容/事項」を含むか、十分な長さの列名
            if ("質問" in n or "質疑" in n or "照会" in n or "問合" in n) and ("内容" in n or "事項" in n or len(n) >= 4):
                body = c
                break

    return place, body


def autodetect_header(path: str, sheet_name=None, max_rows=5) -> int:
    """
    Excel の見出し行を自動検出する。
    先頭から max_rows 行までをヘッダ候補として読み、guess_columns が両列を
    検出できた時点の行番号をヘッダとして採用する。見つからなければ 0。

    v3 fix: sheet_name が None の場合でも、pandas が全シート dict を返さないよう
    既定で 0（先頭シート）を使う。

    引数:
        path: Excel ファイルパス
        sheet_name: シート名またはインデックス（None なら 0 を使う）
        max_rows: 試行するヘッダ候補の最大行数

    戻り値:
        見出し行のインデックス（0 始まり）
    """
    # v3 fix: デフォルトで先頭シート(0)を使う
    sheet = sheet_name if sheet_name is not None else 0

    for h in range(0, max_rows):
        # header=h で読み、その状態で列推定が成立するか試す
        df = pd.read_excel(path, sheet_name=sheet, header=h, nrows=30)
        p, b = guess_columns(df)
        if p and b:
            return h
    return 0


def representative_texts(emb: np.ndarray, labels: np.ndarray, texts: List[str]) -> Dict[int, str]:
    """
    各クラスタの中心（平均ベクトル）に最も近い文を「代表文」として抽出する。

    引数:
        emb: 文埋め込み（shape: [N, D]）
        labels: クラスタラベル（shape: [N]）
        texts: 元の文リスト（長さ N）

    戻り値:
        {cluster_id: representative_text}
    """
    reps = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        # クラスタ内の中心ベクトルを算出
        center = emb[idx].mean(axis=0, keepdims=True)
        # 中心と各点のコサイン類似度を計算
        sims = cosine_similarity(center, emb[idx])[0]
        # 最も近い文を代表として採用
        reps[int(c)] = texts[idx[sims.argmax()]]
    return reps


def build_cluster_names(texts: List[str], labels: np.ndarray, topn: int = 6) -> Dict[int, str]:
    """
    各クラスタの代表「名前」を簡易 TF-IDF によって自動作成する。
    手順:
      1) ひらがな/カタカナ/漢字/英数の連続をトークンとみなす簡易分割
      2) 文内頻度を正規化（L1）した TF を作成
      3) 文章出現頻度から IDF を計算
      4) クラスタごとに TF-IDF の平均ベクトルを取り、上位語を連結して名前にする

    引数:
        texts: 文リスト
        labels: クラスタラベル
        topn: 名前に採用する上位語の最大数

    戻り値:
        {cluster_id: "term1 / term2 / ..."} の辞書
    """
    # 日本語と英数の連続をトークンとして抽出
    tokens_list = [re.findall(r"[ぁ-んァ-ン一-龥a-zA-Z0-9]+", t) for t in texts]

    # 語彙を index 化
    vocab = {}
    for toks in tokens_list:
        for w in toks:
            vocab.setdefault(w, len(vocab))

    # 文によってはトークンが全く取れないことがある
    if not vocab:
        return {int(c): f"cluster_{c}" for c in np.unique(labels)}

    # TF 行列を作成（文ごとに合計 1 になるよう正規化）
    tf = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, toks in enumerate(tokens_list):
        for w in toks:
            tf[i, vocab[w]] += 1
        s = tf[i].sum()
        if s > 0:
            tf[i] /= s

    # DF（語が出現した文数）から IDF を計算し、TF-IDF を作成
    dfreq = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + dfreq)) + 1.0
    tfidf = tf * idf

    inv = {v: k for k, v in vocab.items()}

    names = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        # クラスタの TF-IDF 中心を取り、寄与の大きい順に語を並べる
        centroid = tfidf[idx].mean(axis=0)
        order = np.argsort(-centroid)[:topn]
        terms = [inv[j] for j in order if centroid[j] > 0]
        names[int(c)] = " / ".join(terms) if terms else f"cluster_{int(c)}"
    return names


def auto_kmeans(emb: np.ndarray, k_min: int = 2, k_max: int = 20, random_state: int = 42):
    """
    KMeans のクラスタ数 k を自動選択する。
    k_min..k_max を総当たりで学習し、シルエットスコアが最も高い k を採用する。
    データが極端に少ない場合は単一クラスタにフォールバック。

    引数:
        emb: 埋め込み（[N, D]）
        k_min, k_max: 探索する k の範囲
        random_state: 乱数シード

    戻り値:
        (labels, best_k, best_score)
        labels: shape [N] のラベル配列
        best_k: 最良と判断されたクラスタ数
        best_score: そのときのシルエットスコア
    """
    n = emb.shape[0]
    if n < 3:
        # データが少ない場合は 1 クラスタ扱い
        return np.zeros(n, dtype=int), 1, 0.0

    # k_max はデータ数を超えない範囲に丸める
    k_max = max(k_min, min(k_max, n))

    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        # n_init="auto" は scikit-learn 1.4+ の推奨設定
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(emb)

        # 単一クラスタになった場合はスコア計算しても意味が薄いのでスキップ
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(emb, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    # もし全候補でうまく分かれなければ単一クラスタにフォールバック
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1
        best_score = 0.0

    return best_labels, best_k, best_score


def main():
    """
    コマンドライン引数を受け取り、クラスタリング処理を一括実行するエントリポイント。
    出力 Excel には
      - rows シート: 元データ + cluster_id/cluster_name/cluster_rep/__concat_text__
      - summary シート: クラスタごとの件数と代表文
    を書き出す。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sheet", default=None, help="シート名（未指定は先頭シート）")
    ap.add_argument("--col_place", default=None, help="列名を明示指定（例: 質問箇所）")
    ap.add_argument("--col_body", default=None, help="列名を明示指定（例: 質問内容）")
    ap.add_argument("--method", choices=["auto_kmeans","hdbscan"], default="auto_kmeans")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = ap.parse_args()

    # v3 fix: sheet=None の場合に pandas が dict を返すのを避けるため、必ず 0 を使う
    sheet = args.sheet if args.sheet is not None else 0  # v3 fix

    # 見出し行を自動検出してから本読み込み
    header_row = autodetect_header(args.input, sheet_name=sheet, max_rows=5)
    df = pd.read_excel(args.input, sheet_name=sheet, header=header_row)

    # 列の特定。明示指定があればそれを優先、無ければ自動推定。
    if args.col_place and args.col_body:
        col_place, col_body = args.col_place, args.col_body
    else:
        col_place, col_body = guess_columns(df)

    # どちらかでも見つからなければエラーを出し、列名候補の手がかりを見せる
    if col_place is None or col_body is None:
        cols_show = [repr(c) for c in df.columns]
        raise KeyError(
            "必要な列が見つかりません。--col_place と --col_body で列名を指定してください。\n"
            f"候補になりそうな列を自動検出できませんでした。現列名: {cols_show}"
        )

    # 前の行の箇所を引き継ぐ（結合セル相当のデータに対応）。本文は欠損なら空文字に。
    df[col_place] = df[col_place].ffill()
    df[col_body] = df[col_body].fillna("")

    # 箇所と本文を連結して 1 文に。後続のモデル入力を想定して簡易クリーニング。
    texts = (df[col_place].astype(str) + "。 " + df[col_body].astype(str)).map(clean_text).tolist()

    # 多言語 SentenceTransformer でベクトル化。正規化しておくと類似度の扱いが楽。
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # クラスタリング手法の選択
    if args.method == "hdbscan":
        if _hdbscan is None:
            raise RuntimeError("hdbscan がインストールされていません。pip install hdbscan")
        n = emb.shape[0]
        # データ規模に応じた min_cluster_size を自動設定（50 件に 1 つ or 5 の大きい方）
        mcs = max(5, n // 50)
        clusterer = _hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        labels = clusterer.fit_predict(emb)

        # HDBSCAN は外れ値を -1 にするため、出力の利便性のため連番に張り替える。
        uniq = [c for c in sorted(set(labels)) if c != -1]
        remap = {c: i for i, c in enumerate(uniq)}
        next_id = len(uniq)
        labels2 = np.empty_like(labels)
        for i, c in enumerate(labels):
            labels2[i] = remap.get(c, next_id)
            if c == -1:
                next_id += 1
        labels = labels2
        best_k = len(set(labels))            # 実質的なクラスタ数
        best_score = np.nan                  # HDBSCAN はシルエットを評価指標に使っていないため NaN
    else:
        # KMeans で自動 k 推定
        labels, best_k, best_score = auto_kmeans(emb)

    # クラスタ代表文と、TF-IDF によるクラスタ名を作る
    reps = representative_texts(emb, labels, texts)
    names = build_cluster_names(texts, labels, topn=6)

    # 出力データフレームを作成
    out = df.copy()
    out["cluster_id"] = labels
    out["cluster_name"] = [names.get(int(c), f"cluster_{int(c)}") for c in labels]
    out["cluster_rep"] = [reps.get(int(c), "") for c in labels]
    out["__concat_text__"] = texts

    # 要約テーブル（クラスタごとの件数と代表文）
    summary = (
        out.groupby(["cluster_id", "cluster_name"], as_index=False)
           .agg(count=("cluster_id","size"),
                representative=("cluster_rep","first"))
           .sort_values(["count","cluster_id"], ascending=[False, True])
    )

    # Excel へ 2 シート出力。openpyxl を使うので .xlsx を想定。
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="rows", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)

    # 実行ログを標準出力へ
    print(f"header_row={header_row}, place='{col_place}', body='{col_body}'")
    print(f"clusters: {best_k}, silhouette: {best_score}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
