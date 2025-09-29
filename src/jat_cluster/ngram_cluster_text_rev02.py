# cluster_texts.py
# usage: python cluster_texts.py input.xlsx output.xlsx
import sys
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import re
from typing import List, Dict, Optional

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

# 列名自動推定に使う候補語。実運用でよくある日本語のカラム名を網羅。
CAND_PLACE = ["質問箇所","質疑箇所","設問箇所","質問場所","設問場所","該当箇所","対象箇所","質問部位","質問位置","記載箇所"]
CAND_BODY  = ["質問内容","質疑内容","設問内容","照会内容","問合せ内容","問い合わせ内容","質問事項","質疑","問合せ"]

# ===== 設定 =====
TEXT_COL = "text"  # ← Excelのテキスト列名に合わせて変更
MIN_DF = 2         # 何行以上に出る n-gram を採用するか
NGRAM = (2, 10)     # 日本語向けに文字2〜10グラム

# 除外したい定型フレーズ（必要に応じて増やしてOK）
PHRASES = [
    "でしょうか","ありませんでしょうか",
    "考えてよろしいでしょうか", "お考えでしょうか","よろしいでしょうか", "宜しいでしょうか",
    "ご教示願います", "ご教示ください","ご教示下さい", "ご教授ください", "ご教授下さい",
    "御教示願います", "御教示ください","御教示下さい", "御教授ください", "御教授下さい",
    "ご教示お願いします","ご教示お願い致します","該当しますでしょうか"
    "ご教示のほど","お願い申し上げます", "よろしくお願いいたします", "よろしくお願いします", 
    "考慮していますか","という理解で",
]

# 記号類をスペースに寄せる（日本語・英数字の混在に配慮）
_PUNCS = "「」『』【】［］（）()〔〕〈〉《》“”\"'、。・，．,.\u3000:：;；!?！？…━—-‐−〜~*＊/／\\｜|＋+＝=＜><>"
TRANS_TABLE = str.maketrans({c: " " for c in _PUNCS})

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    # 全角半角など正規化
    s = unicodedata.normalize("NFKC", s)
    # URL, メール、数字だけのノイズを先に除去
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\b", " ", s)
    # 定型フレーズ除去（句読点や空白に挟まれていても消えるように）
    for phrase in PHRASES:
        pat = re.compile(re.escape(phrase))
        s = pat.sub(" ", s)
    # 記号をスペースに
    s = s.translate(TRANS_TABLE)
    # 数字だけの断片は潰す（桁を要素にしたくない場合）
    s = re.sub(r"\d+", " ", s)
    # 余分なスペース
    s = re.sub(r"\s+", " ", s).strip()
    return s

def auto_kmeans(X, k_min=2, k_max=12, random_state=42):
    # データ数によって上限を調整
    n = X.shape[0]
    if n < 2:
        return None, np.array([0]), None
    ks = [k for k in range(k_min, min(k_max, n - 1) + 1)]
    best_k, best_score, best_model = None, -1, None
    for k in ks:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)
            # 全部同じクラスタ回避
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            if score > best_score:
                best_k, best_score, best_model = k, score, km
        except Exception:
            continue
    # 見つからない場合は1クラスタ固定
    if best_model is None:
        km = KMeans(n_clusters=1, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        return km, labels, None
    return best_model, best_model.labels_, best_score

def cluster_keywords(km: KMeans, vectorizer: TfidfVectorizer, topn=12):
    if km is None or getattr(km, "cluster_centers_", None) is None:
        return {}
    feats = np.array(vectorizer.get_feature_names_out())
    kw = {}
    for i, center in enumerate(km.cluster_centers_):
        idx = np.argsort(center)[::-1][:topn]
        kw[i] = feats[idx].tolist()
    return kw

def main(in_path, out_path):
    df = pd.read_excel(in_path)
    if TEXT_COL not in df.columns:
        # それっぽい最初の文字列カラムを自動選択
        candidate = None
        for c in df.columns:
            if df[c].dtype == object:
                candidate = c
                break
        if candidate is None:
            raise ValueError(f"テキスト列が見つからん: {df.columns.tolist()}")
        print(f"[info] TEXT_COL を '{candidate}' に自動設定")
        text_col = candidate
    else:
        text_col = TEXT_COL

    # 正規化と除外
    cleaned = df[text_col].apply(normalize_text).fillna("")

    # すべて空になったらどうにもならん
    if cleaned.str.len().sum() == 0:
        raise ValueError("前処理後にテキストが空になってる。除外フレーズが強すぎるかも。")

    # ベクトル化（形態素いらん版）
    vec = TfidfVectorizer(analyzer="char", ngram_range=NGRAM, min_df=MIN_DF)
    X = vec.fit_transform(cleaned)

    # クラスタ数自動決定
    km, labels, sil = auto_kmeans(X)
    df_out = df.copy()
    df_out["cleaned_text"] = cleaned
    df_out["cluster"] = labels

    # 各行のクラスタ中心への距離（小さいほど中心に近い）
    try:
        dists = km.transform(X)
        min_dist = dists[np.arange(dists.shape[0]), labels]
        df_out["center_distance"] = min_dist
    except Exception:
        df_out["center_distance"] = np.nan

    # クラスタ代表 n-gram
    kw = cluster_keywords(km, vec, topn=15)
    cluster_rows = []
    for cid, words in kw.items():
        cluster_rows.append({"cluster": cid, "top_char_ngrams": ", ".join(words)})
    df_kw = pd.DataFrame(cluster_rows)

    # 書き出し
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_out.to_excel(w, index=False, sheet_name="rows")
        if len(df_kw) > 0:
            meta = pd.DataFrame({
                "metric": ["silhouette_cosine"],
                "value": [sil if sil is not None else np.nan]
            })
            df_kw.to_excel(w, index=False, sheet_name="clusters")
            meta.to_excel(w, index=False, sheet_name="metrics")

    print(f"[done] clusters={len(set(labels))}  silhouette={sil}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cluster_texts.py input.xlsx output.xlsx")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
