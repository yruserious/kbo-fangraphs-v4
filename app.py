import re
import time
from io import StringIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


# =====================================
# 기본 설정
# =====================================
st.set_page_config(page_title="KBO FanGraphs-style Leaderboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_raw"
DATA_DIR.mkdir(exist_ok=True, parents=True)

CURRENT_SEASON = 2026
LIVE_CACHE_TTL_SECONDS = 60 * 30  # 30분
REQUEST_TIMEOUT = 20

KBO_URLS = {
    "HitterBasic1": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx",
    "HitterBasic2": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic2.aspx",
    "PitcherBasic": "https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx",
    "DefenseBasic": "https://www.koreabaseball.com/Record/Player/Defense/Basic.aspx",
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.koreabaseball.com/",
}


# =====================================
# 공통 유틸
# =====================================
def season_dir(season: int) -> Path:
    d = DATA_DIR / f"{season}_kbo_official"
    d.mkdir(exist_ok=True, parents=True)
    return d


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            s = s.str.replace(",", "", regex=False)
            s = s.replace({"-": np.nan, "nan": np.nan, "None": np.nan})
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", "", s)
    return s.strip()


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    if "순위" in df.columns:
        df = df.drop(columns=["순위"])
    df.insert(0, "순위", range(1, len(df) + 1))
    return df


def color_minus(val):
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return ""
    if v < 90:
        return "color: green; font-weight: 700;"
    if v > 110:
        return "color: red; font-weight: 700;"
    return ""


def parse_ip_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if " " in s and "/" in s:
        a, frac = s.split(" ")
        num, den = frac.split("/")
        return float(a) + float(num) / float(den)

    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)

    try:
        return float(s)
    except Exception:
        return np.nan


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def safe_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def apply_search(df: pd.DataFrame, search: str) -> pd.DataFrame:
    if not search:
        return df
    mask = df.astype(str).apply(lambda r: r.str.contains(search, case=False, na=False)).any(axis=1)
    return df[mask]


def safe_sort(df: pd.DataFrame, default_candidates, default_ascending=False):
    cols = df.columns.tolist()

    default_index = 0
    for c in default_candidates:
        if c in cols:
            default_index = cols.index(c)
            break

    sort_col = st.selectbox("Sort column", cols, index=default_index)
    ascending = st.checkbox("Ascending", value=default_ascending)

    try:
        df = df.sort_values(by=sort_col, ascending=ascending, na_position="last")
    except Exception:
        pass

    df = add_rank(df)
    return df


def get_available_seasons():
    seasons = []
    if DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if p.is_dir():
                m = re.match(r"(\d{4})_kbo_official$", p.name)
                if m:
                    seasons.append(int(m.group(1)))
    return sorted(set(seasons + [CURRENT_SEASON]))


def list_files(d: Path):
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if p.is_file()])


# =====================================
# 네트워크 / HTML 처리
# =====================================
def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def extract_best_table_from_html(html: str, required_cols=None) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    best = None
    best_score = -1

    for t in tables:
        df = clean_cols(t.copy())

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[-1]).strip() for c in df.columns]

        score = 0
        if required_cols:
            score = sum(1 for c in required_cols if c in df.columns)

        if len(df) > 0 and score > best_score:
            best_score = score
            best = df

    if best is None or len(best) == 0:
        raise ValueError("기록 테이블을 찾지 못했습니다.")

    return clean_cols(best)


# =====================================
# 라이브 데이터 fetch
# 캐시 TTL 적용
# =====================================
@st.cache_data(ttl=LIVE_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_kbo_hitter_live_cached():
    html1 = fetch_html(KBO_URLS["HitterBasic1"])
    html2 = fetch_html(KBO_URLS["HitterBasic2"])

    df1 = extract_best_table_from_html(
        html1,
        required_cols=["선수명", "팀명", "AVG", "G", "PA", "AB", "R", "H", "2B", "3B", "HR"]
    )
    df2 = extract_best_table_from_html(
        html2,
        required_cols=["선수명", "팀명", "BB", "HBP", "SLG", "OBP", "OPS"]
    )

    return df1, df2


@st.cache_data(ttl=LIVE_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_kbo_pitcher_live_cached():
    html = fetch_html(KBO_URLS["PitcherBasic"])
    df = extract_best_table_from_html(
        html,
        required_cols=["선수명", "팀명", "ERA", "G", "W", "L", "SV", "HLD", "IP", "H", "HR", "BB", "HBP", "SO", "R", "ER"]
    )
    return df


@st.cache_data(ttl=LIVE_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_kbo_defense_live_cached():
    html = fetch_html(KBO_URLS["DefenseBasic"])
    df = extract_best_table_from_html(
        html,
        required_cols=["선수명", "팀명", "POS", "G", "E", "PO", "A", "FPCT"]
    )
    return df


def clear_live_cache():
    fetch_kbo_hitter_live_cached.clear()
    fetch_kbo_pitcher_live_cached.clear()
    fetch_kbo_defense_live_cached.clear()


# =====================================
# CSV 경로
# =====================================
def hitter_csv_paths(season: int):
    d = season_dir(season)
    return d / "hitter_basic1.csv", d / "hitter_basic2.csv"


def pitcher_csv_path(season: int):
    return season_dir(season) / "pitcher_basic.csv"


def defense_csv_path(season: int):
    return season_dir(season) / "defense_basic.csv"


def meta_csv_path(season: int):
    return season_dir(season) / "_meta.csv"


def save_meta(season: int, source: str, status: str, message: str = ""):
    meta = pd.DataFrame(
        [{
            "season": season,
            "source": source,
            "status": status,
            "message": message,
            "updated_at_unix": int(time.time()),
            "updated_at_text": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }]
    )
    safe_write_csv(meta, meta_csv_path(season))


def load_meta(season: int):
    path = meta_csv_path(season)
    if path.exists():
        try:
            return safe_read_csv(path)
        except Exception:
            return None
    return None


# =====================================
# 라이브 fetch + CSV 저장 + fallback
# =====================================
def refresh_current_season_data(season: int):
    if season != CURRENT_SEASON:
        return

    d = season_dir(season)
    h1_path, h2_path = hitter_csv_paths(season)
    p_path = pitcher_csv_path(season)
    d_path = defense_csv_path(season)

    hitter1, hitter2 = fetch_kbo_hitter_live_cached()
    pitcher = fetch_kbo_pitcher_live_cached()
    defense = fetch_kbo_defense_live_cached()

    safe_write_csv(hitter1, h1_path)
    safe_write_csv(hitter2, h2_path)
    safe_write_csv(pitcher, p_path)
    safe_write_csv(defense, d_path)

    save_meta(season, source="live_kbo", status="success", message="live fetch and save completed")


def try_refresh_current_season_data(season: int):
    try:
        refresh_current_season_data(season)
        return True, "라이브 데이터 갱신 성공"
    except Exception as e:
        save_meta(season, source="live_kbo", status="failed", message=str(e))
        return False, str(e)


# =====================================
# 데이터 로드
# =====================================
def load_hitter(season: int, use_live_for_current=True):
    h1_path, h2_path = hitter_csv_paths(season)

    if season == CURRENT_SEASON and use_live_for_current:
        ok, msg = try_refresh_current_season_data(season)
        if not ok and (not h1_path.exists() or not h2_path.exists()):
            raise RuntimeError(f"라이브 타자 데이터도 실패했고 저장 CSV도 없습니다: {msg}")

    if not h1_path.exists() or not h2_path.exists():
        raise FileNotFoundError(f"타자 CSV 없음: {h1_path.name}, {h2_path.name}")

    df1 = clean_cols(safe_read_csv(h1_path))
    df2 = clean_cols(safe_read_csv(h2_path))

    key_candidates = ["선수명", "선수", "player", "Player", "이름"]
    team_candidates = ["팀명", "팀", "team", "Team"]

    name1 = find_col(df1, key_candidates)
    name2 = find_col(df2, key_candidates)
    team1 = find_col(df1, team_candidates)
    team2 = find_col(df2, team_candidates)

    if not name1 or not name2:
        raise ValueError("타자 파일에서 선수명 컬럼을 찾지 못했습니다.")

    df1["__name__"] = df1[name1].apply(normalize_name)
    df2["__name__"] = df2[name2].apply(normalize_name)

    merge_keys = ["__name__"]

    if team1 and team2:
        df1["__team__"] = df1[team1].astype(str).str.strip()
        df2["__team__"] = df2[team2].astype(str).str.strip()
        merge_keys.append("__team__")

    overlap_cols = set(df1.columns) & set(df2.columns)
    drop_from_df2 = [c for c in overlap_cols if c not in merge_keys]
    df2 = df2.drop(columns=drop_from_df2, errors="ignore")

    df = pd.merge(df1, df2, on=merge_keys, how="inner")

    df["선수명"] = df[name1]
    df["팀명"] = df[team1] if team1 else ""

    rename_map = {"희타": "SAC", "희비": "SF"}
    df = df.rename(columns=rename_map)

    num_cols = [
        "H", "2B", "3B", "HR", "AB", "SF", "SAC", "BB", "HBP", "PA",
        "R", "RBI", "TB", "OBP", "SLG", "OPS", "AVG", "G", "SO"
    ]
    df = to_num(df, num_cols)

    if all(c in df.columns for c in ["H", "2B", "3B", "HR"]):
        df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]

    if "PA" in df.columns and df["PA"].notna().any():
        denom = df["PA"]
    else:
        denom = (
            df.get("AB", 0)
            + df.get("BB", 0)
            + df.get("HBP", 0)
            + df.get("SF", 0)
        )

    needed = ["BB", "HBP", "1B", "2B", "3B", "HR"]
    if all(c in df.columns for c in needed):
        df["wOBA"] = (
            0.69 * df["BB"]
            + 0.72 * df["HBP"]
            + 0.89 * df["1B"]
            + 1.27 * df["2B"]
            + 1.62 * df["3B"]
            + 2.10 * df["HR"]
        ) / denom.replace(0, np.nan)
        df["wOBA"] = df["wOBA"].round(3)
    else:
        df["wOBA"] = np.nan

    if "wOBA" in df.columns and df["wOBA"].notna().any():
        if "PA" in df.columns and df["PA"].notna().any() and df["PA"].sum() > 0:
            woba_lg = (df["wOBA"] * df["PA"]).sum() / df["PA"].sum()
        else:
            valid = denom.replace(0, np.nan)
            woba_lg = (df["wOBA"] * valid).sum() / valid.sum()

        if pd.notna(woba_lg) and woba_lg != 0:
            df["wRC+"] = ((df["wOBA"] / woba_lg) * 100).round(0).astype("Int64")
        else:
            df["wRC+"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        woba_lg = np.nan
        df["wRC+"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "OBP" in df.columns and "SLG" in df.columns:
        df["OPS"] = df["OBP"] + df["SLG"]

    if "OPS" in df.columns and df["OPS"].notna().any():
        ops_lg = df["OPS"].mean()
        if pd.notna(ops_lg) and ops_lg != 0:
            df["OPS+"] = ((df["OPS"] / ops_lg) * 100).round(0).astype("Int64")
        else:
            df["OPS+"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["OPS+"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    preferred_cols = [
        "선수명", "팀명", "AVG", "G", "PA", "AB", "R", "H", "2B", "3B",
        "HR", "TB", "RBI", "BB", "HBP", "SO", "SAC", "SF", "OBP", "SLG", "OPS",
        "1B", "wOBA", "wRC+", "OPS+"
    ]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols and not c.startswith("__")]
    df = df[existing_cols + other_cols]

    return df, float(woba_lg) if pd.notna(woba_lg) else np.nan


def load_pitcher(season: int, use_live_for_current=True):
    path = pitcher_csv_path(season)

    if season == CURRENT_SEASON and use_live_for_current:
        ok, msg = try_refresh_current_season_data(season)
        if not ok and not path.exists():
            raise RuntimeError(f"라이브 투수 데이터도 실패했고 저장 CSV도 없습니다: {msg}")

    if not path.exists():
        raise FileNotFoundError(f"투수 CSV 없음: {path.name}")

    df = clean_cols(safe_read_csv(path))

    name_col = find_col(df, ["선수명", "선수", "player", "Player", "이름"])
    team_col = find_col(df, ["팀명", "팀", "team", "Team"])

    df["선수명"] = df[name_col] if name_col else "Unknown"
    df["팀명"] = df[team_col] if team_col else ""

    df = to_num(df, ["ERA", "H", "HR", "BB", "HBP", "SO", "K", "R", "ER", "G", "W", "L", "SV", "HLD", "WHIP"])

    if "SO" not in df.columns and "K" in df.columns:
        df["SO"] = df["K"]

    if "IP" in df.columns:
        df["IP"] = df["IP"].apply(parse_ip_value)
        df["IP"] = pd.to_numeric(df["IP"], errors="coerce")

    era_lg = df["ERA"].mean() if "ERA" in df.columns else np.nan

    if all(c in df.columns for c in ["HR", "BB", "HBP", "SO", "IP"]):
        raw = (
            13 * df["HR"]
            + 3 * (df["BB"] + df["HBP"])
            - 2 * df["SO"]
        ) / df["IP"].replace(0, np.nan)

        c = era_lg - raw.mean() if pd.notna(era_lg) else 0
        df["FIP"] = (raw + c).round(2)
    else:
        df["FIP"] = np.nan

    if "ERA" in df.columns and pd.notna(era_lg) and era_lg != 0:
        df["ERA-"] = ((df["ERA"] / era_lg) * 100).round(0).astype("Int64")
    else:
        df["ERA-"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "FIP" in df.columns and df["FIP"].notna().any():
        fip_lg = df["FIP"].mean()
        if pd.notna(fip_lg) and fip_lg != 0:
            df["FIP-"] = ((df["FIP"] / fip_lg) * 100).round(0).astype("Int64")
        else:
            fip_lg = np.nan
            df["FIP-"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        fip_lg = np.nan
        df["FIP-"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    preferred_cols = [
        "선수명", "팀명", "ERA", "G", "W", "L", "SV", "HLD",
        "IP", "H", "HR", "BB", "HBP", "SO", "R", "ER", "WHIP", "FIP", "ERA-", "FIP-"
    ]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    return df, float(era_lg) if pd.notna(era_lg) else np.nan, float(fip_lg) if pd.notna(fip_lg) else np.nan


def load_defense(season: int, use_live_for_current=True):
    path = defense_csv_path(season)

    if season == CURRENT_SEASON and use_live_for_current:
        ok, msg = try_refresh_current_season_data(season)
        if not ok and not path.exists():
            raise RuntimeError(f"라이브 수비 데이터도 실패했고 저장 CSV도 없습니다: {msg}")

    if not path.exists():
        raise FileNotFoundError(f"수비 CSV 없음: {path.name}")

    df = clean_cols(safe_read_csv(path))

    name_col = find_col(df, ["선수명", "선수", "player", "Player", "이름"])
    team_col = find_col(df, ["팀명", "팀", "team", "Team"])
    pos_col = find_col(df, ["포지션", "POS", "Position"])

    df["선수명"] = df[name_col] if name_col else "Unknown"
    df["팀명"] = df[team_col] if team_col else ""
    df["포지션"] = df[pos_col] if pos_col else "Unknown"

    df = to_num(df, ["G", "PO", "A", "E", "FPCT", "SB", "CS"])

    if "CS%" in df.columns:
        df["CS%"] = pd.to_numeric(
            df["CS%"].astype(str).str.replace("%", "", regex=False).replace("-", np.nan),
            errors="coerce"
        ) / 100.0

    if "FPCT" not in df.columns and all(c in df.columns for c in ["PO", "A", "E"]):
        denom = (df["PO"] + df["A"] + df["E"]).replace(0, np.nan)
        df["FPCT"] = ((df["PO"] + df["A"]) / denom).round(3)

    if "포지션" in df.columns and "FPCT" in df.columns:
        pos_avg = df.groupby("포지션")["FPCT"].transform("mean")
        df["포지션평균_FPCT"] = pos_avg.round(3)
        df["FPCT+"] = ((df["FPCT"] / pos_avg) * 100).round(0).astype("Int64")
    else:
        df["포지션평균_FPCT"] = np.nan
        df["FPCT+"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "CS%" not in df.columns and all(c in df.columns for c in ["SB", "CS"]):
        denom = (df["SB"] + df["CS"]).replace(0, np.nan)
        df["CS%"] = (df["CS"] / denom).round(3)

    if all(c in df.columns for c in ["E", "G", "포지션"]):
        df["E_per_G"] = df["E"] / df["G"].replace(0, np.nan)
        pos_avg_e = df.groupby("포지션")["E_per_G"].transform("mean")
        df["DEF_impact"] = ((pos_avg_e - df["E_per_G"]) * df["G"]).round(2)
    else:
        df["DEF_impact"] = np.nan

    preferred_cols = [
        "선수명", "팀명", "포지션", "G", "PO", "A", "E", "FPCT",
        "포지션평균_FPCT", "FPCT+", "DEF_impact", "SB", "CS", "CS%"
    ]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    return df


# =====================================
# 사이드바
# =====================================
st.sidebar.title("KBO FanGraphs-style Leaderboard")

seasons = get_available_seasons()
default_index = seasons.index(CURRENT_SEASON) if CURRENT_SEASON in seasons else len(seasons) - 1

season = st.sidebar.selectbox("Season", seasons, index=default_index)
mode = st.sidebar.selectbox("Select", ["Hitter", "Pitcher", "Defense"])

use_live_for_current = st.sidebar.checkbox("Use live KBO data for current season", value=True)
auto_refresh = st.sidebar.checkbox("Auto refresh every 30 min (via cache TTL)", value=True)
show_debug = st.sidebar.checkbox("Show debug", value=False)

if st.sidebar.button("Force refresh now"):
    clear_live_cache()
    ok, msg = try_refresh_current_season_data(CURRENT_SEASON)
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

if mode == "Hitter":
    st.title("KBO Hitter Leaderboard")
elif mode == "Pitcher":
    st.title("KBO Pitcher Leaderboard")
else:
    st.title("KBO Defense Leaderboard")

search = st.text_input("Search", "")

if season == CURRENT_SEASON and use_live_for_current:
    st.caption("현재 시즌은 앱 접속 시 라이브 데이터를 재확인하고, 성공하면 CSV도 자동 저장합니다.")

if not auto_refresh:
    # 자동 갱신 끄면 사실상 캐시를 계속 쓰는 느낌이 아니라
    # 현재 코드에서는 live 사용 여부만 유지하되 강제 clear를 안 하는 구조
    pass


# =====================================
# 메인
# =====================================
try:
    with st.spinner("KBO 데이터를 불러오는 중..."):
        if mode == "Hitter":
            df, woba_lg = load_hitter(season, use_live_for_current=use_live_for_current)

            if "PA" in df.columns and df["PA"].notna().any():
                max_pa = int(df["PA"].max())
                default_pa = min(30, max_pa)
                min_pa = st.slider("Minimum PA", 0, max_pa, default_pa)
                df = df[df["PA"] >= min_pa]

            df = apply_search(df, search)

            st.caption(f"Rows: {len(df)}")
            c1, c2 = st.columns(2)
            c1.metric("League wOBA", f"{woba_lg:.3f}" if pd.notna(woba_lg) else "NA")
            c2.metric("League wRC+", "100")

            df = safe_sort(df, ["wRC+", "wOBA", "OPS+"], default_ascending=False)

            st.subheader("Top 10")
            top = df.head(10)

            if "wRC+" in top.columns and len(top) > 0:
                tooltip_cols = [c for c in ["선수명", "팀명", "wRC+", "wOBA", "PA"] if c in top.columns]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("wRC+:Q"),
                    y=alt.Y("선수명:N", sort="-x"),
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart, use_container_width=True)

            st.subheader("All")
            height = min(1200, 35 * (len(df) + 1))
            st.dataframe(df, use_container_width=True, hide_index=True, height=height)

            st.subheader("Player page")
            players = sorted(df["선수명"].dropna().astype(str).unique().tolist())
            if players:
                player = st.selectbox("Select a hitter", players)
                p = df[df["선수명"].astype(str) == player].head(1)

                if len(p) > 0:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Player", player)
                    c2.metric("Team", str(p["팀명"].iloc[0]) if "팀명" in p.columns else "-")
                    c3.metric("wRC+", str(p["wRC+"].iloc[0]) if "wRC+" in p.columns else "-")
                    c4.metric("OPS+", str(p["OPS+"].iloc[0]) if "OPS+" in p.columns else "-")
                    st.dataframe(p.T, use_container_width=True, hide_index=False)

        elif mode == "Pitcher":
            df, era_lg, fip_lg = load_pitcher(season, use_live_for_current=use_live_for_current)

            if "IP" in df.columns and df["IP"].notna().any():
                max_ip = int(df["IP"].max())
                min_ip = st.slider("Minimum IP", 0, max_ip, 0)
                df = df[df["IP"] >= min_ip]

            df = apply_search(df, search)

            st.caption(f"Rows: {len(df)}")
            c1, c2 = st.columns(2)
            c1.metric("League ERA", f"{era_lg:.2f}" if pd.notna(era_lg) else "NA")
            c2.metric("League FIP", f"{fip_lg:.2f}" if pd.notna(fip_lg) else "NA")

            df = safe_sort(df, ["FIP-", "ERA-", "FIP", "ERA"], default_ascending=True)

            st.subheader("Top 10")
            top = df.head(10)

            if "FIP-" in top.columns and top["FIP-"].notna().any():
                tooltip_cols = [c for c in ["선수명", "팀명", "FIP-", "ERA-", "IP"] if c in top.columns]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("FIP-:Q"),
                    y=alt.Y("선수명:N", sort="x"),
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart, use_container_width=True)

            elif "ERA-" in top.columns and top["ERA-"].notna().any():
                tooltip_cols = [c for c in ["선수명", "팀명", "ERA-", "ERA", "IP"] if c in top.columns]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("ERA-:Q"),
                    y=alt.Y("선수명:N", sort="x"),
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart, use_container_width=True)

            st.subheader("All")
            all_df = df.copy()
            style_cols = [c for c in ["ERA-", "FIP-"] if c in all_df.columns]
            height = min(1200, 35 * (len(all_df) + 1))

            if style_cols:
                styled_df = all_df.style.applymap(color_minus, subset=style_cols)
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height)
            else:
                st.dataframe(all_df, use_container_width=True, hide_index=True, height=height)

            st.subheader("Player page")
            players = sorted(df["선수명"].dropna().astype(str).unique().tolist())
            if players:
                player = st.selectbox("Select a pitcher", players)
                p = df[df["선수명"].astype(str) == player].head(1)

                if len(p) > 0:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Player", player)
                    c2.metric("Team", str(p["팀명"].iloc[0]) if "팀명" in p.columns else "-")
                    c3.metric("ERA-", str(p["ERA-"].iloc[0]) if "ERA-" in p.columns else "-")
                    c4.metric("FIP-", str(p["FIP-"].iloc[0]) if "FIP-" in p.columns else "-")
                    st.dataframe(p.T, use_container_width=True, hide_index=False)

        else:
            df = load_defense(season, use_live_for_current=use_live_for_current)

            positions = sorted(df["포지션"].dropna().astype(str).unique().tolist()) if "포지션" in df.columns else []
            selected_pos = st.selectbox("Position", ["All"] + positions)

            if selected_pos != "All":
                df = df[df["포지션"].astype(str) == selected_pos]

            catcher_only = st.checkbox("Catchers only (for CS%)", value=False)
            if catcher_only and "포지션" in df.columns:
                df = df[df["포지션"].astype(str).str.contains("포수|C", case=False, na=False)]

            df = apply_search(df, search)

            st.caption(f"Rows: {len(df)}")

            c1, c2, c3 = st.columns(3)
            if "FPCT" in df.columns and len(df["FPCT"].dropna()) > 0:
                c1.metric("Avg FPCT", f"{df['FPCT'].dropna().mean():.3f}")
            if "FPCT+" in df.columns and len(df["FPCT+"].dropna()) > 0:
                c2.metric("Avg FPCT+", f"{df['FPCT+'].dropna().mean():.0f}")
            if "CS%" in df.columns and len(df["CS%"].dropna()) > 0:
                c3.metric("Avg CS%", f"{df['CS%'].dropna().mean():.3f}")

            default_candidates = ["CS%"] if catcher_only else ["FPCT+", "FPCT"]
            df = safe_sort(df, default_candidates, default_ascending=False)

            st.subheader("Top 10")
            top = df.head(10)

            if catcher_only and "CS%" in top.columns and len(top) > 0:
                tooltip_cols = [c for c in ["선수명", "팀명", "포지션", "CS%", "CS", "SB"] if c in top.columns]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("CS%:Q"),
                    y=alt.Y("선수명:N", sort="-x"),
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart, use_container_width=True)

            elif "FPCT+" in top.columns and len(top) > 0:
                tooltip_cols = [c for c in ["선수명", "팀명", "포지션", "FPCT", "포지션평균_FPCT", "FPCT+", "E"] if c in top.columns]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("FPCT+:Q"),
                    y=alt.Y("선수명:N", sort="-x"),
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart, use_container_width=True)

            elif "FPCT" in top.columns and len(top) > 0:
                tooltip_cols = [c for c in ["선수명", "팀명", "포지션", "FPCT", "E"] if c in top.columns]
                chart = alt.Chart(top).mark_bar().encode(
                    x=alt.X("FPCT:Q"),
                    y=alt.Y("선수명:N", sort="-x"),
                    tooltip=tooltip_cols
                )
                st.altair_chart(chart, use_container_width=True)

            st.subheader("All")
            height = min(1200, 35 * (len(df) + 1))
            st.dataframe(df, use_container_width=True, hide_index=True, height=height)

            st.subheader("Player page")
            players = sorted(df["선수명"].dropna().astype(str).unique().tolist())
            if players:
                player = st.selectbox("Select a player", players)
                p = df[df["선수명"].astype(str) == player].head(1)

                if len(p) > 0:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Player", player)
                    m2.metric("Team", str(p["팀명"].iloc[0]) if "팀명" in p.columns else "-")
                    m3.metric("Position", str(p["포지션"].iloc[0]) if "포지션" in p.columns else "-")

                    if catcher_only and "CS%" in p.columns and pd.notna(p["CS%"].iloc[0]):
                        m4.metric("CS%", str(round(float(p["CS%"].iloc[0]), 3)))
                    elif "FPCT+" in p.columns and pd.notna(p["FPCT+"].iloc[0]):
                        m4.metric("FPCT+", str(p["FPCT+"].iloc[0]))
                    elif "FPCT" in p.columns and pd.notna(p["FPCT"].iloc[0]):
                        m4.metric("FPCT", str(p["FPCT"].iloc[0]))
                    else:
                        m4.metric("Metric", "-")

                    st.dataframe(p.T, use_container_width=True, hide_index=False)

    # 상태 표시
    meta = load_meta(CURRENT_SEASON)
    if meta is not None and len(meta) > 0:
        row = meta.iloc[0]
        st.caption(
            f"Last update: {row.get('updated_at_text', '-')}"
            f" | source: {row.get('source', '-')}"
            f" | status: {row.get('status', '-')}"
        )

except Exception as e:
    st.error("데이터 처리 중 오류가 발생했습니다.")
    st.code(str(e))

    if show_debug:
        d = season_dir(season)
        st.subheader("Debug")
        st.write("Season:", season)
        st.write("Mode:", mode)
        st.write("Current season:", CURRENT_SEASON)
        st.write("Use live current season:", use_live_for_current)
        st.write("Season dir:", str(d))
        st.write("Files:", list_files(d))
        meta = load_meta(season)
        st.write("Meta exists:", meta is not None)
        if meta is not None:
            st.dataframe(meta, use_container_width=True)