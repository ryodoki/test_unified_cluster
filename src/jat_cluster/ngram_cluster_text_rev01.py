# cluster_texts.py
# usage: python cluster_texts.py input.xlsx output.xlsx
import sys
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
