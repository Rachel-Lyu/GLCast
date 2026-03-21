import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


@dataclass
class ExperimentConfig:
    """Container for one prepared dataset and its walk-forward hyperparameters."""

    dataset_name: str
    target_name: str
    target_wide: pd.DataFrame
    feature_wides: Dict[str, pd.DataFrame]
    valid_states: Set[str]
    feature_lags: Sequence[int]
    ar_lags: Sequence[int]
    baseline_gammas: Sequence[float]
    k_candidates: int
    s_screen: int
    q_tau: float
    unit: str
    retrain_every: int
    test_window: int
    val_window: int
    train_window: int


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(os.path.expanduser(path)))[0]


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------
def _align_features_to_target(
    target_wide: pd.DataFrame,
    feature_wides_raw: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Align auxiliary wide tables to the target index and clean obvious issues."""
    target_wide = target_wide.sort_index().copy()
    target_idx = pd.to_datetime(target_wide.index)

    feature_wides: Dict[str, pd.DataFrame] = {}
    for sig_name, wide in feature_wides_raw.items():
        if wide is None or len(wide) == 0:
            continue
        wide = wide.sort_index().copy()
        wide.index = pd.to_datetime(wide.index)

        wide_aligned = wide.reindex(target_idx)
        wide_aligned = wide_aligned.mask(wide_aligned < 0, 0.0)
        wide_aligned = wide_aligned.fillna(0.0)

        nonconstant_cols = (wide_aligned.abs().sum(axis=0) > 0)
        wide_aligned = wide_aligned.loc[:, nonconstant_cols]
        if wide_aligned.shape[1] == 0:
            continue
        feature_wides[sig_name] = wide_aligned
    return feature_wides


def _infer_states(target_wide: pd.DataFrame, valid_states: Optional[Iterable[str]] = None) -> Set[str]:
    """Infer two-letter state columns from the target table."""
    states = [c for c in target_wide.columns if isinstance(c, str) and c.isalpha() and len(c) == 2]
    if len(states) == 0:
        states = list(target_wide.columns)

    if valid_states is not None:
        valid_states = set(valid_states)
        states = [st for st in states if st in valid_states]
    return set(states)


# -----------------------------------------------------------------------------
# COVID dataset
# -----------------------------------------------------------------------------
COVID_TARGET_FILE = "~/PheOpt/map_explore/target/covidcast-hhs-confirmed_admissions_covid_1d-2019-12-31-to-2024-04-26.csv"
COVID_AUX_FILES = [
    "~/PheOpt/map_explore/auxiliary/covidcast-fb-survey-raw_wcli-2020-04-06-to-2022-06-25.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-fb-survey-raw_whh_cmnty_cli-2020-04-06-to-2022-06-25.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-fb-survey-smoothed_wtested_14d-2020-04-06-to-2022-06-25.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-fb-survey-smoothed_wtested_positive_14d-2020-04-06-to-2022-06-25.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s01_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s02_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s03_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s04_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s05_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s06_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/auxiliary/covidcast-google-symptoms-s07_raw_search-2019-12-31-to-2024-04-26.csv",
    "~/PheOpt/map_explore/target/covidcast-jhu-csse-confirmed_incidence_num-2020-01-22-to-2023-03-09.csv",
    "~/PheOpt/map_explore/target/covidcast-jhu-csse-deaths_incidence_num-2020-01-22-to-2023-03-09.csv",
]
COVID_VALID_STATES = {
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL",
    "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VA", "VT", "WA", "WI", "WV", "WY",
}


def _read_covid_signal_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.expanduser(path))
    df["time_value"] = pd.to_datetime(df["time_value"])
    df["geo_value"] = df["geo_value"].astype(str).str.upper()
    wide = df.pivot(index="time_value", columns="geo_value", values="value").sort_index()
    wide = wide.fillna(0.0)
    return wide



def load_covid_config() -> ExperimentConfig:
    signal_wides: Dict[str, pd.DataFrame] = {}
    for fp in [COVID_TARGET_FILE] + COVID_AUX_FILES:
        name = _basename_no_ext(fp)
        if name == "fetch":
            continue
        signal_wides[name] = _read_covid_signal_csv(fp)

    target_name = _basename_no_ext(COVID_TARGET_FILE)
    target_wide = signal_wides[target_name].iloc[92:].sort_index()

    all_signal_wides = {_basename_no_ext(fp): signal_wides[_basename_no_ext(fp)] for fp in COVID_AUX_FILES}
    feature_wides = _align_features_to_target(target_wide, all_signal_wides)
    valid_states = _infer_states(target_wide, COVID_VALID_STATES)

    return ExperimentConfig(
        dataset_name="covid",
        target_name=target_name,
        target_wide=target_wide,
        feature_wides=feature_wides,
        valid_states=valid_states,
        # covid_
        # feature_lags=(1, 2, 3, 4),
        # covid
        feature_lags=(0, 1, 2, 3),
        ar_lags=(1, 2, 3, 4),
        baseline_gammas=[0.0, 0.01, 0.05, 0.1],
        k_candidates=3,
        s_screen=50,
        q_tau=1.0,
        unit="D",
        retrain_every=30,
        test_window=30,
        val_window=30,
        train_window=180,
    )


# -----------------------------------------------------------------------------
# ILI dataset
# -----------------------------------------------------------------------------
ILI_STATE_ABBREVIATIONS = {
    "Alabama": "AL",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Total Us Mulo": "US",
    "Total Us": "US",
    "Total US": "US",
}

ILI_VALID_CODES = [
    "Vitamin C", "Vit C_IMM", "Kids Vit C", "ADCC", "PDCC", "ADSA", "PDSA", "UR", "Rfg OJ", "Hand San",
    "Disinfectant", "Cough Drops", "Lip", "Therm", "SS BOTTLED APPLE JUICE _ Uni_0",
    "SS BOTTLED GRAPE JUICE _ Uni_1", "SS BOTTLED ORANGE JUICE _ Un_2", "SS BOTTLED TOMATO_VEGETABLE _3",
    "COLD_ALLERGY_SINUS LIQUID_PO_4", "COLD_ALLERGY_SINUS TABLETS_P_5", "FACIAL TISSUE _ Unit Sales",
    "COUGH SYRUP _ Unit Sales", "SORE THROAT REMEDY LIQUIDS _11", "ANTACID LIQUID_POWDER _ Uni_12",
    "INTERNAL ANALGESIC LIQUIDS _13", "INTERNAL ANALGESIC TABLETS _14", "NASAL ASPIRATORS _ Unit Sales",
    "NASAL SPRAY_DROPS_INHALER __16", "NASAL STRIPS _ Unit Sales", "SLEEPING AID LIQUIDS _ Unit_18",
    "SLEEPING AID TABLETS _ Unit_19",
]

ILI_RENAME_DICT = {
    "Vitamin C": "Vitamin C Type",
    "Vit C_IMM": "Vitamin C Immunity Category",
    "Kids Vit C": "Kids Vitamin C",
    "ADCC": "Adult Cough Cold Category",
    "PDCC": "Pediatric Cough Cold Category",
    "ADSA": "Adult Sinus Allergy Category",
    "PDSA": "Pediatric Allergy Sinus Category",
    "UR": "Total Upper Respiratory",
    "Rfg OJ": "RFG Orange Juice",
    "Hand San": "Hand Sanitizers",
    "Disinfectant": "All Purpose Cleaner / Disinfectant",
    "Cough Drops": "Cough Drops",
    "Lip": "Lip Treatment",
    "Therm": "Personal Thermometers",
    "SS BOTTLED APPLE JUICE _ Uni_0": "Bottled Apple Juice",
    "SS BOTTLED GRAPE JUICE _ Uni_1": "Bottled Grape Juice",
    "SS BOTTLED ORANGE JUICE _ Un_2": "Bottled Orange Juice",
    "SS BOTTLED TOMATO_VEGETABLE _3": "Bottled Tomato Vegetable Juice",
    "COLD_ALLERGY_SINUS LIQUID_PO_4": "Cold Allergy Sinus Liquid",
    "COLD_ALLERGY_SINUS TABLETS_P_5": "Cold Allergy Sinus Tablets",
    "FACIAL TISSUE _ Unit Sales": "Facial Tissue",
    "COUGH SYRUP _ Unit Sales": "Cough Syrup",
    "SORE THROAT REMEDY LIQUIDS _11": "Sore Throat Remedy Liquids",
    "ANTACID LIQUID_POWDER _ Uni_12": "Antacid Liquid / Powder",
    "INTERNAL ANALGESIC LIQUIDS _13": "Internal Analgesic Liquids",
    "INTERNAL ANALGESIC TABLETS _14": "Internal Analgesic Tablets",
    "NASAL ASPIRATORS _ Unit Sales": "Nasal Aspirators",
    "NASAL SPRAY_DROPS_INHALER __16": "Nasal Spray / Drops / Inhalers",
    "NASAL STRIPS _ Unit Sales": "Nasal Strips",
    "SLEEPING AID LIQUIDS _ Unit_18": "Sleeping Aid Liquids",
    "SLEEPING AID TABLETS _ Unit_19": "Sleeping Aid Tablets",
}

ILI_VALID_STATES = {
    "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN", "KS", "KY", "LA",
    "MA", "MD", "ME", "MI", "MN", "MO", "MS", "NC", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
}

ILI_FAN_FILE = "/net/dali/home/mscbio/rul98/PheOpt/iRi/FAN FLU states Units all cats.xlsx"
ILI_IF_FILE = "/net/dali/home/mscbio/rul98/PheOpt/iRi/illness forecast.xlsx"
ILI_FLU_FILE = "/net/dali/home/mscbio/rul98/PheOpt/iRi/flu_data_all.csv"
ILI_SIGNAL_NAMES = [
    "Vitamin C Type", "Vitamin C Immunity Category", "Kids Vitamin C", "Adult Cough Cold Category",
    "Pediatric Cough Cold Category", "Adult Sinus Allergy Category", "Pediatric Allergy Sinus Category",
    "Total Upper Respiratory", "RFG Orange Juice", "Hand Sanitizers",
    "All Purpose Cleaner / Disinfectant", "Cough Drops", "Lip Treatment", "Personal Thermometers",
    "Bottled Apple Juice", "Bottled Grape Juice", "Bottled Orange Juice", "Bottled Tomato Vegetable Juice",
    "Cold Allergy Sinus Liquid", "Cold Allergy Sinus Tablets", "Facial Tissue", "Cough Syrup",
    "Sore Throat Remedy Liquids", "Antacid Liquid / Powder", "Internal Analgesic Liquids",
    "Internal Analgesic Tablets", "Nasal Aspirators", "Nasal Spray / Drops / Inhalers", "Nasal Strips",
    "Sleeping Aid Liquids", "Sleeping Aid Tablets",
]


def _get_week_start_date(time_str: str) -> str:
    date_part = str(time_str).replace("1 week ending ", "").replace("Week Ending ", "")
    date = pd.to_datetime(date_part)
    week_start_date = date - pd.Timedelta(days=7)
    return week_start_date.strftime("%Y%m%d")



def _read_weekly_excel_sheets(
    file_path: str,
    skiprows: Sequence[int],
    clean_state_cols: bool,
) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    out: Dict[str, pd.DataFrame] = {}
    for s_idx, sheet in enumerate(sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet, skiprows=skiprows[s_idx], index_col=None)
        df.columns = df.columns.str.replace(" - Multi Outlet", "", regex=False)
        if clean_state_cols:
            df.columns = (
                df.columns.str.replace("State - ", "", regex=False)
                .str.replace(" - MULO", "", regex=False)
                .str.title()
            )
        df = df.dropna(axis=0, how="all")
        df["Time"] = df["Time"].apply(_get_week_start_date).astype(int)
        df = df.set_index("Time")
        df.columns = [ILI_STATE_ABBREVIATIONS[c] for c in df.columns]
        df = df[(df.index >= 20120617) & (df.index <= 20170312)]
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        out[sheet] = df
    return out



def load_ili_config() -> ExperimentConfig:
    fan_skip = np.append(np.repeat(11, 8), np.repeat(3, 6))
    if_skip = np.repeat(3, len(pd.ExcelFile(ILI_IF_FILE).sheet_names))

    fan_dfs = _read_weekly_excel_sheets(ILI_FAN_FILE, skiprows=fan_skip, clean_state_cols=True)
    if_dfs = _read_weekly_excel_sheets(ILI_IF_FILE, skiprows=if_skip, clean_state_cols=False)

    dfs = {**fan_dfs, **if_dfs}
    valid_dfs = {code: dfs[code] for code in ILI_VALID_CODES}
    valid_dfs = {ILI_RENAME_DICT.get(key, key): valid_dfs[key] for key in ILI_VALID_CODES}

    flu_df = pd.read_csv(ILI_FLU_FILE)[
        ["region", "epiweek", "num_ili", "num_patients", "wili", "ili", "WEEK", "YEAR", "State"]
    ].drop_duplicates()
    flu_df = flu_df.loc[flu_df.YEAR < 2018]
    flu_df = flu_df.loc[flu_df.YEAR > 2010]
    flu_df = flu_df.groupby(["region", "epiweek"], as_index=False).last()
    flu_df = flu_df.sort_values(by=["region", "YEAR", "WEEK"])
    flu_df = flu_df.dropna(subset=["State"])
    flu_df["Time"] = pd.to_datetime(flu_df["epiweek"], format="%m/%d/%y")
    flu_pivot = flu_df.pivot_table(values="ili", index="Time", columns="region")
    valid_dfs["ili"] = flu_pivot

    target_name = "ili"
    target_wide = valid_dfs[target_name].sort_index()
    all_signal_wides = {name: valid_dfs[name] for name in ILI_SIGNAL_NAMES}
    feature_wides = _align_features_to_target(target_wide, all_signal_wides)
    valid_states = _infer_states(target_wide, ILI_VALID_STATES)

    return ExperimentConfig(
        dataset_name="ili",
        target_name=target_name,
        target_wide=target_wide,
        feature_wides=feature_wides,
        valid_states=valid_states,
        # ili_
        # feature_lags=(1, 2, 3, 4),
        # ar_lags=(3, 4, 5, 6),
        # ili
        feature_lags=(0, 1, 2, 3),
        ar_lags=(2, 3, 4, 5),
        baseline_gammas=[0.0, 0.01, 0.05, 0.1],
        k_candidates=3,
        s_screen=50,
        q_tau=1.0,
        unit="W",
        retrain_every=4,
        test_window=4,
        val_window=4,
        train_window=55,
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def get_experiment_config(dataset_name: str) -> ExperimentConfig:
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "covid":
        return load_covid_config()
    if dataset_name == "ili":
        return load_ili_config()
    raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from ['covid', 'ili'].")
