#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_cluster_ja_v3.py（拡張: target 切替）
v3 fix: sheet_name=None のときに pandas が全シート dict を返して落ちる問題を修正。
さらにリクエスト対応で「質問箇所のみ」「質問内容のみ」「両方（別々）」で
クラスタリングできる `--target` を追加。既定は従来通りの連結（concat）。

使い方例:
  # 従来通り: 箇所 + 内容 を連結して 1 本の文でクラスタリング
  python qa_cluster_ja_v3.py in.xlsx out.xlsx --target concat

  # 質問箇所だけでクラスタリング（「どの場所の話が多いか」を見る）
  python qa_cluster_ja_v3.py in.xlsx out.xlsx --target place

  # 質問内容だけでクラスタリング（テキスト内容のまとまりを見る）
  python qa_cluster_ja_v3.py in.xlsx out.xlsx --target body

  # 箇所と内容を“別々に”両方クラスタリングし、4シート出力
  # rows_place / summary_place / rows_body / summary_body
  python qa_cluster_ja_v3.py in.xlsx out.xlsx --target both

出力:
  target=concat/place/body のときは rows と summary の 2 シート。
  target=both のときは rows_place, summary_place, rows_body, summary_body。
"""
import argparse
import re
from typing import List, Dict, Optional, Tuple

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
    s = str(s)
    s = s.replace("\n", "").replace("\r", "").replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def clean_text(s: str) -> str:
    s = str(s).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


CAND_PLACE = ["質問箇所","質疑箇所","設問箇所","質問場所","設問場所","該当箇所","対象箇所","質問部位","質問位置","記載箇所"]
CAND_BODY  = ["質問内容","質疑内容","設問内容","照会内容","問合せ内容","問い合わせ内容","質問事項","質疑","問合せ"]


def guess_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    mapping = {norm(c): c for c in cols}

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

    if place is None:
        for c in cols:
            n = norm(c)
            if "質問" in n and ("箇所" in n or "場所" in n or "部位" in n or "位置" in n):
                place = c
                break
            if "設計" in n and ("番号" not in n and "日" not in n and "名" not in n):
                place = c
                break

    if body is None:
        for c in cols:
            n = norm(c)
            if ("質問" in n or "質疑" in n or "照会" in n or "問合" in n) and ("内容" in n or "事項" in n or len(n) >= 4):
                body = c
                break

    return place, body


def autodetect_header(path: str, sheet_name=None, max_rows=5) -> int:
    sheet = sheet_name if sheet_name is not None else 0
    for h in range(0, max_rows):
        df = pd.read_excel(path, sheet_name=sheet, header=h, nrows=30)
        p, b = guess_columns(df)
        if p and b:
            return h
    return 0


def representative_texts(emb: np.ndarray, labels: np.ndarray, texts: List[str]) -> Dict[int, str]:
    reps = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        center = emb[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(center, emb[idx])[0]
        reps[int(c)] = texts[idx[sims.argmax()]]
    return reps


def build_cluster_names(texts: List[str], labels: np.ndarray, topn: int = 6) -> Dict[int, str]:
    tokens_list = [re.findall(r"[ぁ-んァ-ン一-龥a-zA-Z0-9]+", t) for t in texts]
    vocab = {}
    for toks in tokens_list:
        for w in toks:
            vocab.setdefault(w, len(vocab))
    if not vocab:
        return {int(c): f"cluster_{c}" for c in np.unique(labels)}

    tf = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, toks in enumerate(tokens_list):
        for w in toks:
            tf[i, vocab[w]] += 1
        s = tf[i].sum()
        if s > 0:
            tf[i] /= s

    dfreq = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(texts)) / (1 + dfreq)) + 1.0
    tfidf = tf * idf

    inv = {v: k for k, v in vocab.items()}
    names = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = tfidf[idx].mean(axis=0)
        order = np.argsort(-centroid)[:topn]
        terms = [inv[j] for j in order if centroid[j] > 0]
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


def run_clustering(texts: List[str], method: str, model: SentenceTransformer):
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    if method == "hdbscan":
        if _hdbscan is None:
            raise RuntimeError("hdbscan がインストールされていません。pip install hdbscan")
        n = emb.shape[0]
        mcs = max(5, n // 50)
        clusterer = _hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        labels = clusterer.fit_predict(emb)
        uniq = [c for c in sorted(set(labels)) if c != -1]
        remap = {c: i for i, c in enumerate(uniq)}
        next_id = len(uniq)
        labels2 = np.empty_like(labels)
        for i, c in enumerate(labels):
            labels2[i] = remap.get(c, next_id)
            if c == -1:
                next_id += 1
        labels = labels2
        best_k = len(set(labels))
        best_score = np.nan
    else:
        labels, best_k, best_score = auto_kmeans(emb)

    reps = representative_texts(emb, labels, texts)
    names = build_cluster_names(texts, labels, topn=6)

    out_cols = pd.DataFrame({
        "cluster_id": labels,
        "cluster_name": [names.get(int(c), f"cluster_{int(c)}") for c in labels],
        "cluster_rep": [reps.get(int(c), "") for c in labels],
        "__text__": texts,
    })

    summary = (
        out_cols.groupby(["cluster_id", "cluster_name"], as_index=False)
                .agg(count=("cluster_id", "size"), representative=("cluster_rep", "first"))
                .sort_values(["count", "cluster_id"], ascending=[False, True])
    )

    meta = {"k": best_k, "silhouette": best_score}
    return out_cols, summary, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--sheet", default=None, help="シート名（未指定は先頭シート）")
    ap.add_argument("--col_place", default=None, help="列名を明示指定（例: 質問箇所）")
    ap.add_argument("--col_body", default=None, help="列名を明示指定（例: 質問内容）")
    ap.add_argument("--method", choices=["auto_kmeans", "hdbscan"], default="auto_kmeans")
    ap.add_argument("--target", choices=["concat", "place", "body", "both"], default="concat")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = ap.parse_args()

    sheet = args.sheet if args.sheet is not None else 0

    header_row = autodetect_header(args.input, sheet_name=sheet, max_rows=5)
    df = pd.read_excel(args.input, sheet_name=sheet, header=header_row)

    if args.col_place and args.col_body:
        col_place, col_body = args.col_place, args.col_body
    else:
        col_place, col_body = guess_columns(df)

    if col_place is None or col_body is None:
        cols_show = [repr(c) for c in df.columns]
        raise KeyError(
            "必要な列が見つかりません。--col_place と --col_body で列名を指定してください。\n"
            f"現列名: {cols_show}"
        )

    df[col_place] = df[col_place].ffill()
    df[col_body] = df[col_body].fillna("")

    model = SentenceTransformer(args.model)

    def make_texts(target: str) -> List[str]:
        if target == "place":
            return df[col_place].astype(str).map(clean_text).tolist()
        if target == "body":
            return df[col_body].astype(str).map(clean_text).tolist()
        return (df[col_place].astype(str) + "。 " + df[col_body].astype(str)).map(clean_text).tolist()

    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        if args.target in ("concat", "place", "body"):
            texts = make_texts(args.target)
            out_cols, summary, meta = run_clustering(texts, args.method, model)
            out = pd.concat([df.copy(), out_cols], axis=1)
            out.to_excel(w, sheet_name="rows", index=False)
            summary.to_excel(w, sheet_name="summary", index=False)
            print(f"header_row={header_row}, place='{col_place}', body='{col_body}'")
            print(f"target={args.target}, clusters={meta['k']}, silhouette={meta['silhouette']}")
            print(f"saved: {args.output}")
        else:
            texts_p = make_texts("place")
            out_p, sum_p, meta_p = run_clustering(texts_p, args.method, model)
            pd.concat([df.copy(), out_p], axis=1).to_excel(w, sheet_name="rows_place", index=False)
            sum_p.to_excel(w, sheet_name="summary_place", index=False)

            texts_b = make_texts("body")
            out_b, sum_b, meta_b = run_clustering(texts_b, args.method, model)
            pd.concat([df.copy(), out_b], axis=1).to_excel(w, sheet_name="rows_body", index=False)
            sum_b.to_excel(w, sheet_name="summary_body", index=False)

            print(f"header_row={header_row}, place='{col_place}', body='{col_body}'")
            print(
                f"target=both, place_clusters={meta_p['k']} (sil={meta_p['silhouette']}), "
                f"body_clusters={meta_b['k']} (sil={meta_b['silhouette']})"
            )
            print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
