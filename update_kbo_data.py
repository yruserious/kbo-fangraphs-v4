import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_raw"
DATA_DIR.mkdir(exist_ok=True, parents=True)

CURRENT_SEASON = 2026
REQUEST_TIMEOUT = 20

KBO_URLS = {
    "HitterBasic1": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx",
    "HitterBasic2": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic2.aspx",
    "PitcherBasic": "https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx",
    "DefenseBasic": "https://www.koreabaseball.com/Record/Player/Defense/Basic.aspx",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.koreabaseball.com/",
}


def season_dir(season: int) -> Path:
    d = DATA_DIR / f"{season}_kbo_official"
    d.mkdir(exist_ok=True, parents=True)
    return d


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[-1]).strip() for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def read_all_tables(html: str):
    tables = pd.read_html(StringIO(html))
    cleaned = []
    for t in tables:
        df = t.copy()
        df = clean_cols(df)
        cleaned.append(df)
    return cleaned


def score_table(df: pd.DataFrame, required_cols):
    cols = set(df.columns.tolist())
    score = sum(1 for c in required_cols if c in cols)
    return score


def filter_valid_rows(df: pd.DataFrame, name_candidates=None):
    if name_candidates is None:
        name_candidates = ["선수명", "선수", "player", "Player", "이름"]

    name_col = None
    for c in name_candidates:
        if c in df.columns:
            name_col = c
            break

    if name_col is None:
        return df

    out = df.copy()
    out[name_col] = out[name_col].astype(str).str.strip()

    bad_words = [
        "합계", "평균", "순위", "기록", "선수명", "player", "등록", "검색"
    ]
    out = out[out[name_col] != ""]
    out = out[~out[name_col].isin(["nan", "None"])]

    for w in bad_words:
        out = out[~out[name_col].str.contains(w, case=False, na=False)]

    return out


def extract_best_table(html: str, required_cols, label: str):
    tables = read_all_tables(html)

    best = None
    best_score = -1
    best_len = -1

    print(f"\n[{label}] tables found: {len(tables)}")

    for i, df in enumerate(tables):
        temp = filter_valid_rows(df)
        score = score_table(temp, required_cols)
        row_count = len(temp)

        print(f"[{label}] table {i}: rows={row_count}, score={score}, cols={temp.columns.tolist()[:15]}")

        if score > best_score or (score == best_score and row_count > best_len):
            best = temp
            best_score = score
            best_len = row_count

    if best is None or len(best) == 0:
        raise ValueError(f"{label}: 적절한 테이블을 찾지 못했습니다.")

    print(f"[{label}] selected rows={len(best)}, score={best_score}")
    return clean_cols(best)


def save_meta(season: int, status: str, msg: str = ""):
    df = pd.DataFrame([{
        "season": season,
        "status": status,
        "msg": msg,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }])
    safe_write_csv(df, season_dir(season) / "_meta.csv")


def main():
    d = season_dir(CURRENT_SEASON)

    try:
        h1 = extract_best_table(
            fetch_html(KBO_URLS["HitterBasic1"]),
            ["선수명", "팀명", "AVG", "G", "PA", "AB", "R", "H", "2B", "3B", "HR"],
            "HitterBasic1"
        )

        h2 = extract_best_table(
            fetch_html(KBO_URLS["HitterBasic2"]),
            ["선수명", "팀명", "BB", "HBP", "SLG", "OBP", "OPS"],
            "HitterBasic2"
        )

        p = extract_best_table(
            fetch_html(KBO_URLS["PitcherBasic"]),
            ["선수명", "팀명", "ERA", "G", "W", "L", "SV", "HLD", "IP", "H", "HR", "BB", "HBP", "SO", "R", "ER"],
            "PitcherBasic"
        )

        df_def = extract_best_table(
            fetch_html(KBO_URLS["DefenseBasic"]),
            ["선수명", "팀명", "POS", "G", "E", "PO", "A", "FPCT"],
            "DefenseBasic"
        )

        print("\n=== FINAL ROW COUNTS ===")
        print("hitter_basic1 rows:", len(h1))
        print("hitter_basic2 rows:", len(h2))
        print("pitcher_basic rows:", len(p))
        print("defense_basic rows:", len(df_def))

        safe_write_csv(h1, d / "hitter_basic1.csv")
        safe_write_csv(h2, d / "hitter_basic2.csv")
        safe_write_csv(p, d / "pitcher_basic.csv")
        safe_write_csv(df_def, d / "defense_basic.csv")

        save_meta(CURRENT_SEASON, "success", "update completed")
        print("\nsuccess")

    except Exception as e:
        save_meta(CURRENT_SEASON, "fail", str(e))
        print("\nfail:", e)
        raise


if __name__ == "__main__":
    main()