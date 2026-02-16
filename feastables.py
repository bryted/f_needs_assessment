# ---------------------------------------------
# Feastables Needs Assessment — Recommendations Cleaning
# FULL UPDATED CODE (Other replacement + IGA replacement + 0/1 flags)
# ---------------------------------------------
# What this script does:
# 1) Loads the raw Excel export
# 2) Replaces 'other' tokens in:
#    - recommendations-d36a_child_actions  -> uses recommendations-d36b_child_actions_other
#    - recommendations-d37a_guardian_actions -> uses recommendations-d37b_guardian_actions_other
#    - recommendations-d38a_community_actions -> uses recommendations-d38b_community_actions_other
#    (replaces 'other' with other_<slugified_text>)
# 3) Household IGA rule:
#    - If 'iga' is selected in recommendations-d37a_guardian_actions, remove 'iga' and
#      replace it with IGA options from recommendations-d37b_guardian_actions_iga as iga_<slugified_phrase>
#    (e.g., "Additional livelihood training" -> iga_additional_livelihood_training)
# 4) Builds:
#    - child_visits_clean_resolved_other_v3.csv
#    - child_visits_actions_01_resolved_other_v3.csv  (wide 0/1 flags)
#    - action_dictionary_resolved_other_v3.csv
#
# NOTE:
# - This script assumes "coded" multi-select columns are space-separated tokens.
# - For IGA, values are treated as full phrases (NOT split on spaces). We split only on ; , /.
# ---------------------------------------------

import re
from pathlib import Path

import pandas as pd

# Sheet and column anchors
WORKER_REQUIRED_COLS = [
    "visit-known-d10_child_name",
    "recommendations-d36a_child_actions",
    "recommendations-d37a_guardian_actions",
    "recommendations-d38a_community_actions",
    "KEY",
]

HH_REQUIRED_COLS = [
    "member_name-b28a_member_name",
    "member_name-b30_member_sex",
    "PARENT_KEY",
]

# Recommendation columns (as aligned)
CHILD_MAIN   = "recommendations-d36a_child_actions"
CHILD_VOC    = "recommendations-d36b_child_actions_vocational"
CHILD_OTHER  = "recommendations-d36b_child_actions_other"

HH_MAIN      = "recommendations-d37a_guardian_actions"
HH_IGA       = "recommendations-d37b_guardian_actions_iga"
HH_OTHER     = "recommendations-d37b_guardian_actions_other"

COM_MAIN     = "recommendations-d38a_community_actions"
COM_OTHER    = "recommendations-d38b_community_actions_other"

# Base fields for visuals (child name is visit-known-d10_child_name in your file)
BASE_COLS = {
    "meta-instanceID": "visit_id",
    "SubmissionDate": "submission_date",
    "visit-d11_visit_date": "visit_date",
    "household": "household_id",
    "visit-d09_child_code": "child_code",
    "visit-known-d10_child_name": "child_name",
    "age": "child_age",
    "visit-known-d01_cooperative": "cooperative",
    "visit-known-d03_district": "district",
    "visit-known-d04_community_name": "community",
    "visit-known-d07_farmer_clmrs_code": "farmer_clmrs_code",
    "visit-known-d08_farmer_name": "farmer_name",
}

# =========================
# HELPERS
# =========================
def clean_multiselect(series: pd.Series) -> pd.Series:
    """Normalize multi-select strings to single-spaced tokens."""
    return (series.fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True))

def split_free_text_list(text: str):
    """
    Split free text into possible multiple items.
    Conservative splitting: ; , / only (keeps phrases with spaces intact).
    """
    if not text or not str(text).strip():
        return []
    t = str(text).strip()
    parts = re.split(r"[;,/]+", t)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [t]

def slugify(s: str) -> str:
    """Make a stable code from a phrase."""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def resolve_other_token(codes_series: pd.Series,
                        other_series: pd.Series,
                        other_token: str = "other",
                        other_prefix: str = "other_") -> pd.Series:
    """
    If 'other' appears in codes_series (space-tokenized), remove it and replace with other_<slugified_text>
    from other_series (split into multiple items by ; , /).
    """
    codes_series = clean_multiselect(codes_series)
    other_series = other_series.fillna("").astype(str)

    resolved = []
    for codes, oth in zip(codes_series.tolist(), other_series.tolist()):
        tokens = [t for t in codes.split(" ") if t] if codes else []
        if other_token in tokens:
            tokens = [t for t in tokens if t != other_token]
            items = split_free_text_list(oth)
            for item in items:
                sl = slugify(item)
                if sl:
                    tokens.append(other_prefix + sl)
        tokens = dedupe_preserve_order(tokens)
        resolved.append(" ".join(tokens))

    return pd.Series(resolved, index=codes_series.index)

def resolve_iga_token(hh_codes: pd.Series,
                      iga_series: pd.Series,
                      iga_token: str = "iga",
                      iga_prefix: str = "iga_") -> pd.Series:
    """
    If 'iga' appears in HH codes, remove it and replace it with IGA-specific items from iga_series.
    IGA values are phrases; we split only on ; , / (NOT spaces).
    """
    hh_codes = clean_multiselect(hh_codes)
    iga_series = iga_series.fillna("").astype(str)

    resolved = []
    for codes, iga in zip(hh_codes.tolist(), iga_series.tolist()):
        tokens = [t for t in codes.split(" ") if t] if codes else []
        if iga_token in tokens:
            tokens = [t for t in tokens if t != iga_token]
            iga_items = split_free_text_list(iga)
            for item in iga_items:
                sl = slugify(item)
                if sl:
                    tokens.append(iga_prefix + sl)
        tokens = dedupe_preserve_order(tokens)
        resolved.append(" ".join(tokens))

    return pd.Series(resolved, index=hh_codes.index)

def unique_tokens(series: pd.Series):
    """Unique space-separated tokens across a series."""
    toks = set()
    for v in series:
        if not v:
            continue
        toks.update([t for t in v.split(" ") if t])
    return sorted(toks)


def prompt_for_input_file() -> Path:
    """
    Ask the user to choose an Excel file.
    Tries a GUI file picker first, then falls back to terminal input.
    """
    selected = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select Feastables Needs Assessment Excel File",
            filetypes=[
                ("Excel files", "*.xlsx;*.xlsm;*.xls"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception:
        selected = ""

    if not selected:
        selected = input("Enter full path to your Excel file: ").strip().strip('"')

    if not selected:
        raise ValueError("No file selected.")

    path = Path(selected).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def get_output_paths() -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_clean = out_dir / "child_visits_clean_resolved_other_v3.csv"
    out_flags = out_dir / "child_visits_actions_01_resolved_other_v3.csv"
    out_dict = out_dir / "action_dictionary_resolved_other_v3.csv"
    return out_clean, out_flags, out_dict


def normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_booleanish(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "1.0", "true", "yes", "y"})


def resolve_workers_sheet(xl: pd.ExcelFile) -> str:
    for sheet in xl.sheet_names:
        cols = pd.read_excel(xl, sheet_name=sheet, nrows=0).columns.tolist()
        if all(col in cols for col in WORKER_REQUIRED_COLS):
            return sheet
    raise ValueError(
        "Could not find workers sheet with required columns: "
        + ", ".join(WORKER_REQUIRED_COLS)
    )


def resolve_hh_members_sheet(xl: pd.ExcelFile) -> str | None:
    preferred = "HH_members"
    if preferred in xl.sheet_names:
        cols = pd.read_excel(xl, sheet_name=preferred, nrows=0).columns.tolist()
        if all(col in cols for col in HH_REQUIRED_COLS):
            return preferred

    for sheet in xl.sheet_names:
        cols = pd.read_excel(xl, sheet_name=sheet, nrows=0).columns.tolist()
        if all(col in cols for col in HH_REQUIRED_COLS):
            return sheet
    return None


def extract_child_gender_lookup(hh_members: pd.DataFrame) -> pd.DataFrame:
    if hh_members.empty:
        return pd.DataFrame(columns=["__child_name_norm", "child_gender"])

    required = ["member_name-b28a_member_name", "member_name-b30_member_sex", "PARENT_KEY"]
    missing = [c for c in required if c not in hh_members.columns]
    if missing:
        return pd.DataFrame(columns=["__child_name_norm", "child_gender"])

    work = hh_members.copy()
    work["__child_name_norm"] = normalize_name(work["member_name-b28a_member_name"])
    work["child_gender"] = work["member_name-b30_member_sex"].fillna("").astype(str).str.strip().str.lower()

    if "member_name-is_child" in work.columns:
        work = work[parse_booleanish(work["member_name-is_child"])].copy()

    work = work[(work["__child_name_norm"] != "") & (work["child_gender"] != "")].copy()
    if work.empty:
        return pd.DataFrame(columns=["__child_name_norm", "child_gender"])

    # Keep only names with a single observed gender to avoid ambiguous assignments.
    gender_counts = (
        work.groupby(["__child_name_norm", "child_gender"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["__child_name_norm", "n"], ascending=[True, False])
    )
    gender_options = (
        gender_counts.groupby("__child_name_norm")["child_gender"]
        .nunique()
        .reset_index(name="gender_n")
    )
    chosen = gender_counts.drop_duplicates(subset=["__child_name_norm"], keep="first").merge(
        gender_options, on="__child_name_norm", how="left"
    )
    chosen = chosen[chosen["gender_n"] == 1].copy()
    return chosen[["__child_name_norm", "child_gender"]].reset_index(drop=True)

# =========================
# LOAD
# =========================
INPUT_PATH = prompt_for_input_file()
OUT_CLEAN, OUT_FLAGS, OUT_DICT = get_output_paths()

xl = pd.ExcelFile(INPUT_PATH)
workers_sheet = resolve_workers_sheet(xl)
hh_members_sheet = resolve_hh_members_sheet(xl)

df = pd.read_excel(xl, sheet_name=workers_sheet)

if hh_members_sheet:
    hh_members = pd.read_excel(xl, sheet_name=hh_members_sheet)
    gender_lookup = extract_child_gender_lookup(hh_members)
else:
    hh_members = pd.DataFrame()
    gender_lookup = pd.DataFrame(columns=["__child_name_norm", "child_gender"])

if not gender_lookup.empty:
    df["__child_name_norm"] = normalize_name(df["visit-known-d10_child_name"])
    df = df.merge(gender_lookup, on="__child_name_norm", how="left")
else:
    df["child_gender"] = pd.NA

# =========================
# APPLY RESOLUTION RULES
# =========================
df_res = df.copy()

# Replace 'other' in main multi-selects with slugified free text
df_res[CHILD_MAIN] = resolve_other_token(df[CHILD_MAIN], df[CHILD_OTHER], other_token="other", other_prefix="other_")
df_res[HH_MAIN]    = resolve_other_token(df[HH_MAIN],    df[HH_OTHER],    other_token="other", other_prefix="other_")
df_res[COM_MAIN]   = resolve_other_token(df[COM_MAIN],   df[COM_OTHER],   other_token="other", other_prefix="other_")

# Replace 'iga' in HH main with IGA options (as iga_<slug>)
df_res[HH_MAIN] = resolve_iga_token(df_res[HH_MAIN], df[HH_IGA], iga_token="iga", iga_prefix="iga_")

# =========================
# BUILD CLEAN VISIT TABLE
# =========================
missing_base = [c for c in BASE_COLS if c not in df_res.columns]
if missing_base:
    raise ValueError(f"Missing required base columns: {missing_base}")

child_visits_clean = df_res[list(BASE_COLS.keys())].rename(columns=BASE_COLS)

# Canonical child identifier for downstream reporting
child_visits_clean["child_id"] = child_visits_clean["visit_id"].astype(str)
child_visits_clean["child_gender"] = df_res["child_gender"].fillna("").astype(str).str.strip().str.lower()

# Keep resolved strings for traceability (and original IGA text / other text)
child_visits_clean["child_raw_resolved"] = df_res[CHILD_MAIN]
child_visits_clean["child_voc_raw"]      = df_res[CHILD_VOC].fillna("")
child_visits_clean["household_raw_resolved"] = df_res[HH_MAIN]
child_visits_clean["community_raw_resolved"] = df_res[COM_MAIN]

child_visits_clean["child_other_text"]      = df[CHILD_OTHER]
child_visits_clean["household_other_text"]  = df[HH_OTHER]
child_visits_clean["community_other_text"]  = df[COM_OTHER]
child_visits_clean["household_iga_text"]    = df[HH_IGA]   # traceability

# Flags
child_visits_clean["child__other_flag"]     = df[CHILD_OTHER].fillna("").astype(str).str.strip().ne("").astype(int)
child_visits_clean["household__other_flag"] = df[HH_OTHER].fillna("").astype(str).str.strip().ne("").astype(int)
child_visits_clean["community__other_flag"] = df[COM_OTHER].fillna("").astype(str).str.strip().ne("").astype(int)

# Whether IGA was selected in the *original* HH main list
child_visits_clean["household__iga_flag"] = clean_multiselect(df[HH_MAIN]).str.contains(
    r"(?:^|\s)iga(?:\s|$)", regex=True
).astype(int)

# Date cleaning
child_visits_clean["submission_date"] = pd.to_datetime(child_visits_clean["submission_date"], errors="coerce")
child_visits_clean["visit_date"]      = pd.to_datetime(child_visits_clean["visit_date"], errors="coerce")
child_visits_clean["visit_date"]      = child_visits_clean["visit_date"].fillna(child_visits_clean["submission_date"])
child_visits_clean["visit_date_date"] = child_visits_clean["visit_date"].dt.date

# =========================
# BUILD 0/1 FLAGS TABLE
# =========================
# We build 0/1 flags from resolved columns:
REC_GROUPS = {
    "child": CHILD_MAIN,
    "child_voc": CHILD_VOC,
    "household": HH_MAIN,       # includes iga_* now (replaced)
    "community": COM_MAIN,
}

actions = child_visits_clean.copy()

for group, col in REC_GROUPS.items():
    codes = clean_multiselect(df_res[col])
    toks = unique_tokens(codes)

    for code in toks:
        colname = f"{group}__{code}"
        actions[colname] = codes.str.contains(
            fr"(?:^|\s){re.escape(code)}(?:\s|$)", regex=True
        ).astype(int)

# Keep only rows with at least one recommendation across child/household/community,
# excluding explicit "*__none" options.
indicator_cols: list[str] = []
for group in REC_GROUPS:
    prefix = f"{group}__"
    group_cols = [
        c
        for c in actions.columns
        if c.startswith(prefix) and not c.endswith("_flag") and c != f"{group}__none"
    ]
    indicator_cols.extend(group_cols)
indicator_cols = sorted(set(indicator_cols))

if indicator_cols:
    has_any_recommendation = (
        actions[indicator_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        > 0
    )
else:
    has_any_recommendation = pd.Series(False, index=actions.index)

child_visits_clean = child_visits_clean.loc[has_any_recommendation].reset_index(drop=True)
actions = actions.loc[has_any_recommendation].reset_index(drop=True)

# Build dictionary from filtered actions only.
action_dict_rows: list[dict[str, object]] = []
for group in REC_GROUPS:
    prefix = f"{group}__"
    group_cols = [c for c in actions.columns if c.startswith(prefix) and not c.endswith("_flag")]
    for colname in group_cols:
        selected_count = int(pd.to_numeric(actions[colname], errors="coerce").fillna(0).sum())
        if selected_count <= 0:
            continue
        action_dict_rows.append(
            {
                "action_group": group,
                "action_code": colname[len(prefix):],
                "selected_count": selected_count,
            }
        )

if action_dict_rows:
    action_dictionary = (
        pd.DataFrame(action_dict_rows)
        .drop_duplicates(subset=["action_group", "action_code"])
        .sort_values(["action_group", "selected_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
else:
    action_dictionary = pd.DataFrame(columns=["action_group", "action_code", "selected_count"])

# =========================
# EXPORT
# =========================
child_visits_clean.to_csv(OUT_CLEAN, index=False)
actions.to_csv(OUT_FLAGS, index=False)
action_dictionary.to_csv(OUT_DICT, index=False)

print("Done.")
print(f"Workers sheet used: {workers_sheet}")
print(f"HH members sheet used: {hh_members_sheet if hh_members_sheet else 'Not found'}")
if "child_gender" in child_visits_clean.columns:
    gender_non_blank = int(child_visits_clean["child_gender"].fillna("").astype(str).str.strip().ne("").sum())
    print(f"Rows with mapped child gender: {gender_non_blank}")
print(f"Rows with at least one recommendation: {len(actions)}")
print(f"Rows excluded with no recommendations: {len(df) - len(actions)}")
print("Clean visits:", OUT_CLEAN)
print("0/1 flags:", OUT_FLAGS)
print("Action dictionary:", OUT_DICT)

# Optional sanity checks:
# - How many IGA-coded tokens were produced?
iga_codes = action_dictionary[
    (action_dictionary["action_group"] == "household") &
    (action_dictionary["action_code"].str.startswith("iga_"))
]
print("\nHousehold IGA derived codes:")
print(iga_codes)
