#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excelの「質問箇所」「質問内容」を結合して、意味類似でクラスタリングします。
使い方:
    python qa_cluster_ja.py input.xlsx output.xlsx --method auto_kmeans
    python qa_cluster_ja.py input.xlsx output.xlsx --method hdbscan

前提:
    pip install -U pandas openpyxl scikit-learn sentence-transformers hdbscan janome numpy

出力:
  - output.xlsx
    - Sheet "rows": 元の行＋ cluster_id, cluster_name, cluster_rep を付与
    - Sheet "summary": クラスタごとの件数、代表文、代表キーワード

メモ:
  - 列名は「質問箇所」「質問内容」を期待します。違う場合は --col_place --col_body で指定してください。
  - マージセル起源の欠損は自動で前方補完(ffill)します。
  - 文章は「質問箇所 + '。' + 質問内容」で結合します。
"""
import argparse
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# オプション: HDBSCAN
try:
    import hdbscan as _hdbscan
except Exception:
    _hdbscan = None

# 日本語トークナイズでクラスタ名をそれっぽく
try:
    from janome.tokenizer import Tokenizer
    _tknz = Tokenizer()
    def tokenize_ja(text: str) -> List[str]:
        tokens = []
        for t in _tknz.tokenize(text):
            pos = t.part_of_speech.split(",")[0]
            base = t.base_form if t.base_form != "*" else t.surface
            if pos in ("名詞", "動詞", "形容詞"):
                # 記号・数字っぽいのを弾く
                if not re.fullmatch(r"[\d０-９一二三四五六七八九十百千万億兆％/.-]+", base):
                    tokens.append(base)
        return tokens
except Exception:
    # janomeが無い場合はスペース分割の簡易版
    def tokenize_ja(text: str) -> List[str]:
        return re.findall(r"\w+", text, flags=re.UNICODE)

def clean_text(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_cluster_names(texts: List[str], labels: np.ndarray, topn: int = 6) -> Dict[int, str]:
    # 簡易TF-IDF（自前実装）
    docs_tokens = [tokenize_ja(x) for x in texts]
    # 語彙
    vocab = {}
    for toks in docs_tokens:
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)
    if not vocab:
        return {int(c): f"cluster_{c}" for c in np.unique(labels)}

    # TF
    tf = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, toks in enumerate(docs_tokens):
        if not toks:
            continue
        for w in toks:
            tf[i, vocab[w]] += 1.0
        if tf[i].sum() > 0:
            tf[i] /= tf[i].sum()

    # IDF
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + df)) + 1.0

    # TF-IDF
    tfidf = tf * idf

    inv_vocab = {v: k for k, v in vocab.items()}
    names = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centroid = tfidf[idx].mean(axis=0)
        order = np.argsort(-centroid)[:topn]
        terms = [inv_vocab[j] for j in order if centroid[j] > 0]
        names[int(c)] = " / ".join(terms) if terms else f"cluster_{int(c)}"
    return names

def auto_kmeans(emb: np.ndarray, k_min: int = 2, k_max: int = 20, random_state: int = 42):
    n = emb.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int), 1, 0.0
    k_max = max(k_min, min(k_max, n))
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(emb)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1
        best_score = 0.0
    return best_labels, best_k, best_score

def cluster_hdbscan(emb: np.ndarray):
    if _hdbscan is None:
        raise RuntimeError("hdbscan がインストールされていません。pip install hdbscan")
    # データ量に応じて最小クラスターサイズを調整
    n = emb.shape[0]
    mcs = max(5, n // 50)  # おおよそ2%目安
    clusterer = _hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
    labels = clusterer.fit_predict(emb)
    return labels

def representative_texts(emb: np.ndarray, labels: np.ndarray, texts: List[str]):
    reps = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        center = emb[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(center, emb[idx])[0]
        reps[int(c)] = texts[idx[sims.argmax()]]
    return reps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="入力Excelファイル (.xlsx)")
    parser.add_argument("output", help="出力Excelファイル (.xlsx)")
    parser.add_argument("--col_place", default="質問箇所")
    parser.add_argument("--col_body", default="質問内容")
    parser.add_argument("--method", choices=["auto_kmeans", "hdbscan"], default="auto_kmeans")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = parser.parse_args()

    df = pd.read_excel(args.input)

    # 列が無いときはエラー
    if args.col_place not in df.columns or args.col_body not in df.columns:
        raise KeyError(f"必要な列が見つかりません: {args.col_place}, {args.col_body}")

    # マージセル起源の欠損を補完してから結合
    df[args.col_place] = df[args.col_place].ffill()
    df[args.col_body] = df[args.col_body].fillna("")
    texts = (df[args.col_place].astype(str) + "。 " + df[args.col_body].astype(str)).map(clean_text).tolist()

    # 埋め込み
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # クラスタリング
    if args.method == "hdbscan":
        labels = cluster_hdbscan(emb)
        # HDBSCANのノイズ(-1)は最大クラスタID+1から連番に付け替え
        unique = [c for c in sorted(set(labels)) if c != -1]
        remap = {c: i for i, c in enumerate(unique)}
        next_id = len(unique)
        labels2 = np.empty_like(labels)
        for i, c in enumerate(labels):
            if c == -1:
                labels2[i] = next_id
                next_id += 1
            else:
                labels2[i] = remap[c]
        labels = labels2
        best_k = len(set(labels))
        best_score = np.nan
    else:
        labels, best_k, best_score = auto_kmeans(emb)

    # 代表文とクラスタ名
    reps = representative_texts(emb, labels, texts)
    names = build_cluster_names(texts, labels, topn=6)

    # 出力
    out = df.copy()
    out["cluster_id"] = labels
    out["cluster_name"] = [names.get(int(c), f"cluster_{int(c)}") for c in labels]
    out["cluster_rep"] = [reps.get(int(c), "") for c in labels]
    out["__concat_text__"] = texts  # デバッグ用

    # サマリ
    summary = (
        out.groupby(["cluster_id", "cluster_name"], as_index=False)
          .agg(count=("cluster_id", "size"),
               representative=("cluster_rep", "first"))
          .sort_values(["count", "cluster_id"], ascending=[False, True])
    )
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="rows", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)

    print(f"clusters: {best_k}, silhouette: {best_score}")
    print(f"saved: {args.output}")

if __name__ == "__main__":
    main()
