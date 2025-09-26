#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Japanese text clustering tool

This script merges two workflows into one CLI:
  1) ngram: character n-gram TF-IDF + KMeans clustering for a single text column
  2) qa: semantic clustering of QA-style sheets using sentence-transformers
     with auto header/column detection and KMeans/HDBSCAN

USAGE:
  # 1) n-gram mode (single text column)
  python unified_cluster_ja.py ngram input.xlsx output.xlsx [--text-col TEXT] \
      [--min-df 2] [--ngram-min 2] [--ngram-max 10]

  # 2) QA mode (place/body columns + sentence embeddings)
  python unified_cluster_ja.py qa input.xlsx output.xlsx [--sheet SHEET] \
      [--col_place 列名] [--col_body 列名] [--method auto_kmeans|hdbscan] \
      [--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2]

  # 3) BOTH mode (run both and save into one Excel)
  python unified_cluster_ja.py both input.xlsx output.xlsx [common/options...]

Extras:
  --self-test    内蔵の簡易テストを実行（外部モデル不要）
  --interactive  引数なしや不足時に対話ウィザードを起動（標準入力が対話不可なら自動フォールバック）

Wizard preset (非対話環境の救済):
  - 環境変数 UNIFIED_WIZARD_PRESET に JSON を入れると、引数なしでもウィザード相当を無人実行
    例:
      export UNIFIED_WIZARD_PRESET='{"mode":"ngram","input":"in.xlsx","output":"out.xlsx","text_col":"本文","min_df":1,"ngram_min":2,"ngram_max":5}'
  - またはカレントに unified_wizard_preset.json を置く（同じJSON）

Outputs:
  - ngram mode: Excel with sheets [rows, clusters, metrics]
  - qa mode:   Excel with sheets [rows, summary]
  - both mode: Excel with sheets [ngram_* , qa_*]

Dependencies:
  pandas, numpy, scikit-learn, openpyxl
  (qa mode only) sentence-transformers, optionally hdbscan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# =============== Common helpers ===============

def _safe_str(s) -> str:
    if not isinstance(s, str):
        return "" if s is None else str(s)
    return s

# =============== Mode: ngram ===============
NGRAM_DEFAULT_TEXT_COL = "text"
NGRAM_DEFAULT_MIN_DF = 2
NGRAM_DEFAULT_RANGE = (2, 10)

_PUNCS = "「」『』【】［］（）()〔〕〈〉《》“”\"'、。・，．,.\u3000:：;；!?！？…━—-‐−〜~*＊/／\\｜|＋+＝=＜><>"
TRANS_TABLE = str.maketrans({c: " " for c in _PUNCS})

PHRASES = [
    "でしょうか", "ありませんでしょうか",
    "考えてよろしいでしょうか", "お考えでしょうか", "よろしいでしょうか", "宜しいでしょうか",
    "ご教示願います", "ご教示ください", "ご教示下さい", "ご教授ください", "ご教授下さい",
    "御教示願います", "御教示ください", "御教示下さい", "御教授ください", "御教授下さい",
    "ご教示お願いします", "ご教示お願い致します", "該当しますでしょうか",
    "ご教示のほど", "お願い申し上げます", "よろしくお願いいたします", "よろしくお願いします",
    "考慮していますか", "という理解で",
]


def ngram_normalize_text(s: str) -> str:
    s = _safe_str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\b", " ", s)
    for phrase in PHRASES:
        s = re.sub(re.escape(phrase), " ", s)
    s = s.translate(TRANS_TABLE)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ngram_auto_kmeans(X, k_min=2, k_max=12, random_state=42):
    n = X.shape[0]
    if n < 2:
        km = KMeans(n_clusters=1, n_init=10, random_state=random_state).fit(X)
        return km, np.zeros(n, dtype=int), np.nan
    ks = [k for k in range(k_min, min(k_max, n - 1) + 1)]
    best_km, best_score, best_labels = None, -1.0, None
    for k in ks:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            if score > best_score:
                best_km, best_score, best_labels = km, score, labels
        except Exception:
            continue
    if best_km is None:
        km = KMeans(n_clusters=1, n_init=10, random_state=random_state).fit(X)
        return km, np.zeros(n, dtype=int), np.nan
    return best_km, best_labels, best_score


def ngram_cluster(in_path: str, out_path: str, text_col: Optional[str], min_df: int, ngram_min: int, ngram_max: int):
    df = pd.read_excel(in_path)

    # Choose text column
    if text_col and text_col in df.columns:
        col = text_col
    elif NGRAM_DEFAULT_TEXT_COL in df.columns:
        col = NGRAM_DEFAULT_TEXT_COL
    else:
        col = None
        for c in df.columns:
            if df[c].dtype == object:
                col = c
                print(f"[info] auto-selected TEXT_COL='{col}'")
                break
        if col is None:
            raise ValueError(f"テキスト列が見つかりません: {list(df.columns)}")

    cleaned = df[col].apply(ngram_normalize_text).fillna("")
    if cleaned.str.len().sum() == 0:
        raise ValueError("前処理後にテキストが空です。除外フレーズや列選択を見直してください。")

    vec = TfidfVectorizer(analyzer="char", ngram_range=(ngram_min, ngram_max), min_df=min_df)
    X = vec.fit_transform(cleaned)

    km, labels, sil = ngram_auto_kmeans(X)

    df_out = df.copy()
    df_out["cleaned_text"] = cleaned
    df_out["cluster"] = labels

    try:
        dists = km.transform(X)
        min_dist = dists[np.arange(dists.shape[0]), labels]
        df_out["center_distance"] = min_dist
    except Exception:
        df_out["center_distance"] = np.nan

    feats = np.array(vec.get_feature_names_out())
    cluster_rows = []
    if hasattr(km, "cluster_centers_"):
        for i, center in enumerate(km.cluster_centers_):
            idx = np.argsort(center)[::-1][:15]
            cluster_rows.append({"cluster": i, "top_char_ngrams": ", ".join(feats[idx])})
    df_kw = pd.DataFrame(cluster_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_out.to_excel(w, index=False, sheet_name="rows")
        if len(df_kw) > 0:
            meta = pd.DataFrame({"metric": ["silhouette_cosine"], "value": [sil if sil is not None else np.nan]})
            df_kw.to_excel(w, index=False, sheet_name="clusters")
            meta.to_excel(w, index=False, sheet_name="metrics")

    print(f"[done] ngram clusters={len(set(labels))} silhouette={sil}")


# =============== Mode: qa ===============
CAND_PLACE = [
    "質問箇所", "質疑箇所", "設問箇所", "質問場所", "設問場所", "該当箇所", "対象箇所", "質問部位", "質問位置", "記載箇所",
]
CAND_BODY = [
    "質問内容", "質疑内容", "設問内容", "照会内容", "問合せ内容", "問い合わせ内容", "質問事項", "質疑", "問合せ",
]


def _norm_col(s: str) -> str:
    s = _safe_str(s)
    s = s.replace("\n", "").replace("\r", "").replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def qa_guess_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    mapping = {_norm_col(c): c for c in cols}
    place = None
    for p in CAND_PLACE:
        if _norm_col(p) in mapping:
            place = mapping[_norm_col(p)]
            break
    body = None
    for b in CAND_BODY:
        if _norm_col(b) in mapping:
            body = mapping[_norm_col(b)]
            break
    if place is None:
        for c in cols:
            n = _norm_col(c)
            if "質問" in n and ("箇所" in n or "場所" in n or "部位" in n or "位置" in n):
                place = c
                break
            if "設計" in n and ("番号" not in n and "日" not in n and "名" not in n):
                place = c
                break
    if body is None:
        for c in cols:
            n = _norm_col(c)
            if ("質問" in n or "質疑" in n or "照会" in n or "問合" in n) and ("内容" in n or "事項" in n or len(n) >= 4):
                body = c
                break
    return place, body


def qa_autodetect_header(path: str, sheet_name=None, max_rows=5) -> int:
    sheet = sheet_name if sheet_name is not None else 0
    for h in range(0, max_rows):
        df = pd.read_excel(path, sheet_name=sheet, header=h, nrows=30)
        p, b = qa_guess_columns(df)
        if p and b:
            return h
    return 0


def qa_clean_text(s: str) -> str:
    s = _safe_str(s).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def qa_representative_texts(emb: np.ndarray, labels: np.ndarray, texts: List[str]) -> Dict[int, str]:
    reps = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        center = emb[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(center, emb[idx])[0]
        reps[int(c)] = texts[idx[sims.argmax()]]
    return reps


def qa_build_cluster_names(texts: List[str], labels: np.ndarray, topn: int = 6) -> Dict[int, str]:
    tokens_list = [re.findall(r"[ぁ-んァ-ン一-龥a-zA-Z0-9]+", t) for t in texts]
    vocab: Dict[str, int] = {}
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
    names: Dict[int, str] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = tfidf[idx].mean(axis=0)
        order = np.argsort(-centroid)[:topn]
        terms = [inv[j] for j in order if centroid[j] > 0]
        names[int(c)] = " / ".join(terms) if terms else f"cluster_{int(c)}"
    return names


def qa_auto_kmeans(emb: np.ndarray, k_min: int = 2, k_max: int = 20, random_state: int = 42):
    n = emb.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int), 1, 0.0
    k_max = max(k_min, min(k_max, n))
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(emb)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_labels is None:
        return np.zeros(n, dtype=int), 1, 0.0
    return best_labels, best_k, best_score


def _qa_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    if model_name.lower() in {"debug-tfidf", "debug_tfidf"}:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
        X = vec.fit_transform(texts).astype(np.float32)
        X = X.toarray()
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers が必要です。pip install sentence-transformers") from e
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def qa_cluster(
    in_path: str,
    out_path: str,
    sheet: Optional[str],
    col_place: Optional[str],
    col_body: Optional[str],
    method: str,
    model_name: str,
):
    sheet_to_read = sheet if sheet is not None else 0
    header_row = qa_autodetect_header(in_path, sheet_name=sheet_to_read, max_rows=5)
    df = pd.read_excel(in_path, sheet_name=sheet_to_read, header=header_row)

    if col_place and col_body:
        place_col, body_col = col_place, col_body
    else:
        place_col, body_col = qa_guess_columns(df)

    if place_col is None or body_col is None:
        cols_show = [repr(c) for c in df.columns]
        raise KeyError(
            "必要な列が見つかりません。--col_place と --col_body で列名を指定してください。\n"
            f"候補を自動検出できませんでした。現列名: {cols_show}"
        )

    df[place_col] = df[place_col].ffill()
    df[body_col] = df[body_col].fillna("")
    texts = (df[place_col].astype(str) + "。 " + df[body_col].astype(str)).map(qa_clean_text).tolist()

    emb = _qa_embeddings(texts, model_name)

    if method == "hdbscan":
        try:
            import hdbscan as _hdbscan
        except Exception as e:
            raise RuntimeError("hdbscan がインストールされていません。pip install hdbscan") from e
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
        labels, best_k, best_score = qa_auto_kmeans(emb)

    reps = qa_representative_texts(emb, labels, texts)
    names = qa_build_cluster_names(texts, labels, topn=6)

    out = df.copy()
    out["cluster_id"] = labels
    out["cluster_name"] = [names.get(int(c), f"cluster_{int(c)}") for c in labels]
    out["cluster_rep"] = [reps.get(int(c), "") for c in labels]
    out["__concat_text__"] = texts

    summary = (
        out.groupby(["cluster_id", "cluster_name"], as_index=False)
        .agg(count=("cluster_id", "size"), representative=("cluster_rep", "first"))
        .sort_values(["count", "cluster_id"], ascending=[False, True])
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="rows", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)

    print(f"[done] qa header_row={header_row}, place='{place_col}', body='{body_col}'")
    print(f"[done] qa clusters: {best_k}, silhouette: {best_score}")


# =============== CLI / Interactive / Tests ===============

def _print_examples(parser: argparse.ArgumentParser) -> None:
    print("\nExamples:")
    print("  python unified_cluster_ja.py ngram 相談一覧.xlsx 出力_ngram.xlsx --text-col 本文")
    print("  python unified_cluster_ja.py qa    質疑集約.xlsx 出力_qa.xlsx --col_place 質問箇所 --col_body 質問内容")
    print("  python unified_cluster_ja.py both  入力.xlsx 出力_両方.xlsx --text-col 本文 --col_place 質問箇所 --col_body 質問内容")
    print("  # モデルDL不要のデバッグ: --model debug-tfidf")


def _load_wizard_preset() -> Dict[str, object]:
    """Load wizard preset from env var or local json file."""
    data = os.environ.get("UNIFIED_WIZARD_PRESET")
    if data:
        try:
            return json.loads(data)
        except Exception as e:
            print(f"[warn] UNIFIED_WIZARD_PRESET のJSONが不正です: {e}")
            return {}
    path = os.path.join(os.getcwd(), "unified_wizard_preset.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[warn] unified_wizard_preset.json の読込に失敗: {e}")
    return {}


def _wizard() -> None:
    print("\n[interactive] 引数が無いので対話ウィザードを起動します。Ctrl-C で中断できます。\n")
    preset = _load_wizard_preset()

    # 非対話環境でプリセットも無いなら安全に抜ける
    if not sys.stdin.isatty() and not preset:
        print("[info] 標準入力が対話不可のため、ウィザードを起動できません。CLI引数または UNIFIED_WIZARD_PRESET を使用してください。")
        _print_examples(argparse.ArgumentParser())
        return

    try:
        # mode
        if preset:
            mode = str(preset.get("mode", "ngram")).lower()
        else:
            mode = input("mode [ngram/qa/both] (default: ngram): ").strip().lower() or "ngram"
        if mode not in {"ngram", "qa", "both"}:
            print(f"[warn] 未知のmode '{mode}' を 'ngram' に置換")
            mode = "ngram"

        # input/output
        if preset:
            inp = str(preset.get("input", "")).strip()
            outp = str(preset.get("output", "")).strip()
            if not inp or not outp:
                print("[warn] プリセットに 'input' と 'output' が必要です。")
                return
        else:
            inp = input("入力Excelパス: ").strip()
            while not inp:
                inp = input("入力Excelパス（必須）: ").strip()
            default_out = os.path.splitext(inp)[0] + ("_both.xlsx" if mode == "both" else f"_{mode}.xlsx")
            outp = input(f"出力Excelパス (default: {default_out}): ").strip() or default_out

        # common options
        text_col = preset.get("text_col") if preset else None
        try:
            min_df = int(preset.get("min_df", NGRAM_DEFAULT_MIN_DF)) if preset else NGRAM_DEFAULT_MIN_DF
            nmin = int(preset.get("ngram_min", NGRAM_DEFAULT_RANGE[0])) if preset else NGRAM_DEFAULT_RANGE[0]
            nmax = int(preset.get("ngram_max", NGRAM_DEFAULT_RANGE[1])) if preset else NGRAM_DEFAULT_RANGE[1]
        except Exception:
            min_df, nmin, nmax = NGRAM_DEFAULT_MIN_DF, NGRAM_DEFAULT_RANGE[0], NGRAM_DEFAULT_RANGE[1]

        sheet = preset.get("sheet") if preset else None
        col_place = preset.get("col_place") if preset else None
        col_body = preset.get("col_body") if preset else None
        method = str(preset.get("method", "auto_kmeans")).lower() if preset else "auto_kmeans"
        if method not in {"auto_kmeans", "hdbscan"}:
            method = "auto_kmeans"
        model = str(preset.get("model", "debug-tfidf")) if preset else "debug-tfidf"

        if not preset and mode in {"ngram", "both"}:
            text_col = input(f"[ngram] テキスト列名 (default: {NGRAM_DEFAULT_TEXT_COL} or 自動): ").strip() or text_col
            try:
                _min_df = input(f"[ngram] min_df (default: {min_df}): ").strip()
                min_df = int(_min_df) if _min_df else min_df
                _nmin = input(f"[ngram] ngram-min (default: {nmin}): ").strip()
                nmin = int(_nmin) if _nmin else nmin
                _nmax = input(f"[ngram] ngram-max (default: {nmax}): ").strip()
                nmax = int(_nmax) if _nmax else nmax
            except ValueError:
                print("[warn] 数値でない入力がありました。既定値を使います。")

        if not preset and mode in {"qa", "both"}:
            sheet = input("[qa] シート名 (default: 先頭): ").strip() or sheet
            _cp = input("[qa] 質問箇所の列名 (空なら自動推定): ").strip()
            col_place = _cp or col_place
            _cb = input("[qa] 質問内容の列名 (空なら自動推定): ").strip()
            col_body = _cb or col_body
            m = input("[qa] method [auto_kmeans/hdbscan] (default: auto_kmeans): ").strip() or method
            method = m if m in {"auto_kmeans", "hdbscan"} else "auto_kmeans"
            _model = input("[qa] model (default: debug-tfidf): ").strip()
            model = _model or model

        # Dispatch
        if mode == "ngram":
            ngram_cluster(inp, outp, text_col=text_col, min_df=min_df, ngram_min=nmin, ngram_max=nmax)
        elif mode == "qa":
            qa_cluster(inp, outp, sheet=sheet, col_place=col_place, col_body=col_body, method=method, model_name=model)
        else:
            tmp_ngram = tmp_qa = None
            try:
                fd1, tmp_ngram = tempfile.mkstemp(prefix="ngram_", suffix=".xlsx"); os.close(fd1)
                fd2, tmp_qa = tempfile.mkstemp(prefix="qa_", suffix=".xlsx"); os.close(fd2)
                ngram_cluster(inp, tmp_ngram, text_col=text_col, min_df=min_df, ngram_min=nmin, ngram_max=nmax)
                qa_cluster(inp, tmp_qa, sheet=sheet, col_place=col_place, col_body=col_body, method=method, model_name=model)
                with pd.ExcelWriter(outp, engine="openpyxl") as w:
                    for name, df in pd.read_excel(tmp_ngram, sheet_name=None).items():
                        df.to_excel(w, index=False, sheet_name=f"ngram_{name}")
                    for name, df in pd.read_excel(tmp_qa, sheet_name=None).items():
                        df.to_excel(w, index=False, sheet_name=f"qa_{name}")
                print(f"[done] both: combined results saved to {outp}")
            finally:
                for p in [tmp_ngram, tmp_qa]:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
    except (KeyboardInterrupt, EOFError, OSError):
        print("\n[interactive] 対話入力ができない/中断されました。CLI引数または UNIFIED_WIZARD_PRESET をご利用ください。")


def _self_test() -> None:
    print("[self-test] start")
    df_both = pd.DataFrame(
        {
            "text": [
                "杭基礎の配筋確認しました。", "配筋写真の解像度が不足", "図面の寸法が不一致", "監理者への連絡方法",
                "仮設通路の安全対策", "雨天時の打設条件",
            ],
            "質問箇所": ["構造", "構造", "意匠", "施工", "安全", "施工"],
            "質問内容": [
                "柱主筋の継手長さの基準を確認したい。",
                "提出した配筋写真の解像度要件はありますか。",
                "A-201とA-501で寸法が違うがどちらが正？",
                "監理者への連絡はメールかポータルか。",
                "仮設通路に手摺は必要ですか。",
                "雨天時のコンクリート打設基準は。",
            ],
        }
    )

    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "toy.xlsx")
        out_ng = os.path.join(td, "out_ng.xlsx")
        out_qa = os.path.join(td, "out_qa.xlsx")
        out_both = os.path.join(td, "out_both.xlsx")
        df_both.to_excel(inp, index=False)

        # ngram test
        ngram_cluster(inp, out_ng, text_col="text", min_df=1, ngram_min=2, ngram_max=4)
        sheets = pd.read_excel(out_ng, sheet_name=None)
        assert "rows" in sheets and len(sheets["rows"]) == len(df_both)
        assert "cleaned_text" in sheets["rows"].columns

        # qa test (debug-tfidf)
        qa_cluster(inp, out_qa, sheet=None, col_place="質問箇所", col_body="質問内容", method="auto_kmeans", model_name="debug-tfidf")
        sheets2 = pd.read_excel(out_qa, sheet_name=None)
        assert "rows" in sheets2 and "summary" in sheets2
        assert sheets2["summary"]["count"].sum() == len(sheets2["rows"])  # sanity

        # both mode assembly test
        from pandas import ExcelWriter
        tmp1 = os.path.join(td, "tmp1.xlsx")
        tmp2 = os.path.join(td, "tmp2.xlsx")
        ngram_cluster(inp, tmp1, text_col="text", min_df=1, ngram_min=2, ngram_max=4)
        qa_cluster(inp, tmp2, sheet=None, col_place="質問箇所", col_body="質問内容", method="auto_kmeans", model_name="debug-tfidf")
        with ExcelWriter(out_both, engine="openpyxl") as w:
            for name, df in pd.read_excel(tmp1, sheet_name=None).items():
                df.to_excel(w, index=False, sheet_name=f"ngram_{name}")
            for name, df in pd.read_excel(tmp2, sheet_name=None).items():
                df.to_excel(w, index=False, sheet_name=f"qa_{name}")
        sheets3 = pd.read_excel(out_both, sheet_name=None)
        assert any(s.startswith("ngram_") for s in sheets3)
        assert any(s.startswith("qa_") for s in sheets3)

        # wizard preset test (non-interactive safe path)
        os.environ["UNIFIED_WIZARD_PRESET"] = json.dumps({
            "mode": "ngram", "input": inp, "output": os.path.join(td, "wiz.xlsx"),
            "text_col": "text", "min_df": 1, "ngram_min": 2, "ngram_max": 4
        })
        _saved_argv = sys.argv[:]
        try:
            sys.argv = ["unified_cluster_ja.py"]  # no-args -> wizard path
            main()
        finally:
            sys.argv = _saved_argv
            os.environ.pop("UNIFIED_WIZARD_PRESET", None)
        wiz_sheets = pd.read_excel(os.path.join(td, "wiz.xlsx"), sheet_name=None)
        assert "rows" in wiz_sheets

    print("[self-test] ok")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified Japanese text clustering tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--self-test", action="store_true", help="内蔵の簡易テストを実行")
    ap.add_argument("--interactive", action="store_true", help="対話ウィザードを起動")

    sub = ap.add_subparsers(dest="mode")  # not required; we handle no-arg ourselves

    # ngram subcommand
    ng = sub.add_parser("ngram", help="Char n-gram TF-IDF + KMeans clustering")
    ng.add_argument("input", help="入力Excelファイル")
    ng.add_argument("output", help="出力Excelファイル")
    ng.add_argument("--text-col", default=None, help=f"テキスト列名（既定: '{NGRAM_DEFAULT_TEXT_COL}' か最初の文字列列）")
    ng.add_argument("--min-df", type=int, default=NGRAM_DEFAULT_MIN_DF, help="min_df（頻度閾値）")
    ng.add_argument("--ngram-min", type=int, default=NGRAM_DEFAULT_RANGE[0], help="文字n-gram下限")
    ng.add_argument("--ngram-max", type=int, default=NGRAM_DEFAULT_RANGE[1], help="文字n-gram上限")

    # qa subcommand
    qa = sub.add_parser("qa", help="SentenceTransformer によるQA様式の意味クラスタリング")
    qa.add_argument("input", help="入力Excelファイル")
    qa.add_argument("output", help="出力Excelファイル")
    qa.add_argument("--sheet", default=None, help="シート名（未指定は先頭シート）")
    qa.add_argument("--col_place", default=None, help="列名を明示指定（例: 質問箇所）")
    qa.add_argument("--col_body", default=None, help="列名を明示指定（例: 質問内容）")
    qa.add_argument("--method", choices=["auto_kmeans", "hdbscan"], default="auto_kmeans", help="クラスタリング手法")
    qa.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="SentenceTransformerモデル名（テスト・軽量用途は 'debug-tfidf'）",
    )

    # both subcommand
    both = sub.add_parser("both", help="n-gram と QA をまとめて実行して1つのExcelへ出力")
    both.add_argument("input", help="入力Excelファイル（両モードで共用）")
    both.add_argument("output", help="出力Excelファイル（両モードの結果をシート分けで保存）")
    both.add_argument("--sheet", default=None, help="QA用: シート名（未指定は先頭シート）")
    both.add_argument("--text-col", default=None, help=f"n-gram用: テキスト列名（既定: '{NGRAM_DEFAULT_TEXT_COL}' か最初の文字列列）")
    both.add_argument("--min-df", type=int, default=NGRAM_DEFAULT_MIN_DF, help="n-gram用: min_df（頻度閾値）")
    both.add_argument("--ngram-min", type=int, default=NGRAM_DEFAULT_RANGE[0], help="n-gram用: 文字n-gram下限")
    both.add_argument("--ngram-max", type=int, default=NGRAM_DEFAULT_RANGE[1], help="n-gram用: 文字n-gram上限")
    both.add_argument("--col_place", default=None, help="QA用: 列名を明示指定（例: 質問箇所）")
    both.add_argument("--col_body", default=None, help="QA用: 列名を明示指定（例: 質問内容）")
    both.add_argument("--method", choices=["auto_kmeans", "hdbscan"], default="auto_kmeans", help="QA用: クラスタリング手法")
    both.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="QA用: SentenceTransformerモデル名（テスト・軽量用途は 'debug-tfidf'）",
    )

    return ap


def main() -> None:
    parser = build_parser()

    # 引数ゼロなら人間ファーストでウィザード（非対話ならプリセット/ヘルプにフォールバック）
    if len(sys.argv) == 1:
        _wizard()
        return

    # Parse
    try:
        args = parser.parse_args()
    except SystemExit:
        # Argparse would raise SystemExit on error; show help nicely and return
        parser.print_help()
        _print_examples(parser)
        return

    # Self-test
    if getattr(args, "self_test", False):
        _self_test()
        return

    if getattr(args, "interactive", False):
        _wizard()
        return

    try:
        if args.mode == "ngram":
            ngram_cluster(
                in_path=args.input,
                out_path=args.output,
                text_col=args.text_col,
                min_df=args.min_df,
                ngram_min=args.ngram_min,
                ngram_max=args.ngram_max,
            )
        elif args.mode == "qa":
            qa_cluster(
                in_path=args.input,
                out_path=args.output,
                sheet=args.sheet,
                col_place=args.col_place,
                col_body=args.col_body,
                method=args.method,
                model_name=args.model,
            )
        elif args.mode == "both":
            tmp_ngram = None
            tmp_qa = None
            try:
                fd1, tmp_ngram = tempfile.mkstemp(prefix="ngram_", suffix=".xlsx"); os.close(fd1)
                fd2, tmp_qa = tempfile.mkstemp(prefix="qa_", suffix=".xlsx"); os.close(fd2)

                ngram_cluster(
                    in_path=args.input,
                    out_path=tmp_ngram,
                    text_col=args.text_col,
                    min_df=args.min_df,
                    ngram_min=args.ngram_min,
                    ngram_max=args.ngram_max,
                )
                qa_cluster(
                    in_path=args.input,
                    out_path=tmp_qa,
                    sheet=args.sheet,
                    col_place=args.col_place,
                    col_body=args.col_body,
                    method=args.method,
                    model_name=args.model,
                )

                with pd.ExcelWriter(args.output, engine="openpyxl") as w:
                    try:
                        ng_sheets = pd.read_excel(tmp_ngram, sheet_name=None)
                        for name, df in ng_sheets.items():
                            df.to_excel(w, index=False, sheet_name=f"ngram_{name}")
                    except Exception as e:
                        print(f"[warn] n-gram結果の結合でエラー: {e}")
                    try:
                        qa_sheets = pd.read_excel(tmp_qa, sheet_name=None)
                        for name, df in qa_sheets.items():
                            df.to_excel(w, index=False, sheet_name=f"qa_{name}")
                    except Exception as e:
                        print(f"[warn] QA結果の結合でエラー: {e}")
                print(f"[done] both: combined results saved to {args.output}")
            finally:
                for p in [tmp_ngram, tmp_qa]:
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
        else:
            parser.print_help()
            _print_examples(parser)
            print("\nmode が未指定です。ngram / qa / both のいずれかを指定してください。")
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
