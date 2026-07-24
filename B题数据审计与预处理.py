from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 2026
DATA_FILE = "B题数据集.csv"
OUTPUT_DIR = Path("结果数据表") / "数据审计与预处理"
CLEAN_DIR = Path("预处理数据")

EXPECTED_COLUMNS = [
    "age", "annual_income", "education_level", "city_type",
    "daily_commute_km", "weekly_travel_distance_km", "current_vehicle_type",
    "vehicle_age_years", "fuel_expense_per_month",
    "charging_station_accessibility", "nearest_charging_station_km",
    "home_charging_available", "electricity_cost_per_kwh",
    "environmental_awareness_score", "government_incentive_awareness",
    "technology_affinity_score", "range_anxiety_score",
    "battery_replacement_concern", "ev_knowledge_score",
    "previous_ev_experience", "ev_adoption_likelihood",
    "monthly_energy_consumption_kwh", "monthly_charging_cost",
]

CATEGORICAL_COLUMNS = [
    "education_level", "city_type", "current_vehicle_type", "ev_adoption_likelihood"
]
BINARY_COLUMNS = ["home_charging_available", "previous_ev_experience"]
TARGET_COLUMNS = ["ev_adoption_likelihood", "home_charging_available", "range_anxiety_score"]
DOWNSTREAM_COLUMNS = ["monthly_energy_consumption_kwh", "monthly_charging_cost"]

SCORE_COLUMNS = [
    "charging_station_accessibility", "environmental_awareness_score",
    "government_incentive_awareness", "technology_affinity_score",
    "range_anxiety_score", "battery_replacement_concern", "ev_knowledge_score",
]

BUSINESS_RANGES = {
    "age": (18, 80),
    "annual_income": (0, np.inf),
    "daily_commute_km": (0, np.inf),
    "weekly_travel_distance_km": (0, np.inf),
    "vehicle_age_years": (0, np.inf),
    "fuel_expense_per_month": (0, np.inf),
    "charging_station_accessibility": (1, 10),
    "nearest_charging_station_km": (0, np.inf),
    "home_charging_available": (0, 1),
    "electricity_cost_per_kwh": (0, np.inf),
    "environmental_awareness_score": (1, 10),
    "government_incentive_awareness": (1, 10),
    "technology_affinity_score": (1, 10),
    "range_anxiety_score": (1, 10),
    "battery_replacement_concern": (1, 10),
    "ev_knowledge_score": (1, 10),
    "previous_ev_experience": (0, 1),
    "monthly_energy_consumption_kwh": (0, np.inf),
    "monthly_charging_cost": (0, np.inf),
}

TASKS = {
    "任务一": {
        "target": "ev_adoption_likelihood",
        "strict_exclude": [
            "ev_adoption_likelihood", "home_charging_available",
            "monthly_energy_consumption_kwh", "monthly_charging_cost",
        ],
        "relaxed_add": ["home_charging_available"],
    },
    "任务二": {
        "target": "home_charging_available",
        "strict_exclude": [
            "home_charging_available", "ev_adoption_likelihood",
            "range_anxiety_score", "monthly_energy_consumption_kwh",
            "monthly_charging_cost",
        ],
        "relaxed_add": ["range_anxiety_score", "battery_replacement_concern"],
    },
    "任务三": {
        "target": "range_anxiety_high",
        "strict_exclude": [
            "range_anxiety_score", "ev_adoption_likelihood",
            "home_charging_available", "monthly_energy_consumption_kwh",
            "monthly_charging_cost",
        ],
        "relaxed_add": ["home_charging_available"],
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"未找到数据文件：{path}")
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    if extra:
        print(f"Info: 存在额外字段：{extra}")
    if df.empty:
        raise ValueError("数据集为空")
    return df


def normalize_types(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    df = raw.copy()
    changes: list[dict] = []
    df.insert(0, "record_id", np.arange(1, len(df) + 1, dtype=int))

    for col in CATEGORICAL_COLUMNS:
        before = df[col].copy()
        df[col] = df[col].astype("string").str.strip()
        changed = before.astype("string").fillna("<NA>") != df[col].fillna("<NA>")
        for idx in df.index[changed]:
            changes.append({
                "记录键": int(df.at[idx, "record_id"]), "字段": col,
                "问题类型": "类别文本规范化", "原始值": before.at[idx],
                "处理后值": df.at[idx, col], "处理方式": "去除首尾空白",
            })

    numeric_columns = [c for c in EXPECTED_COLUMNS if c not in CATEGORICAL_COLUMNS]
    for col in numeric_columns:
        before = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        failed = before.notna() & df[col].isna()
        for idx in df.index[failed]:
            changes.append({
                "记录键": int(df.at[idx, "record_id"]), "字段": col,
                "问题类型": "数值解析失败", "原始值": before.at[idx],
                "处理后值": np.nan, "处理方式": "转为缺失值",
            })
    return df, changes


def apply_business_rules(df: pd.DataFrame, changes: list[dict]) -> pd.DataFrame:
    cleaned = df.copy()
    for col, (lower, upper) in BUSINESS_RANGES.items():
        values = cleaned[col]
        invalid = values.notna() & ((values < lower) | (values > upper))
        for idx in cleaned.index[invalid]:
            changes.append({
                "记录键": int(cleaned.at[idx, "record_id"]), "字段": col,
                "问题类型": "业务范围异常", "原始值": cleaned.at[idx, col],
                "处理后值": np.nan, "处理方式": f"超出[{lower}, {upper}]，转为缺失值",
            })
        cleaned.loc[invalid, col] = np.nan

    valid_categories = {
        "education_level": {"High School", "Bachelor", "Master", "PhD"},
        "city_type": {"Urban", "Suburban", "Rural"},
        "current_vehicle_type": {"Hatchback", "Sedan", "SUV", "Truck"},
        "ev_adoption_likelihood": {"Low", "Medium", "High"},
    }
    for col, allowed in valid_categories.items():
        invalid = cleaned[col].notna() & ~cleaned[col].isin(allowed)
        for idx in cleaned.index[invalid]:
            changes.append({
                "记录键": int(cleaned.at[idx, "record_id"]), "字段": col,
                "问题类型": "未知类别", "原始值": cleaned.at[idx, col],
                "处理后值": pd.NA, "处理方式": "转为缺失类别",
            })
        cleaned.loc[invalid, col] = pd.NA
    return cleaned


def remove_exact_duplicates(df: pd.DataFrame, changes: list[dict]) -> tuple[pd.DataFrame, int]:
    feature_cols = [c for c in df.columns if c != "record_id"]
    duplicate_mask = df.duplicated(subset=feature_cols, keep="first")
    for idx in df.index[duplicate_mask]:
        changes.append({
            "记录键": int(df.at[idx, "record_id"]), "字段": "整行",
            "问题类型": "完全重复记录", "原始值": "与前序记录完全相同",
            "处理后值": "删除", "处理方式": "保留首次出现记录",
        })
    return df.loc[~duplicate_mask].reset_index(drop=True), int(duplicate_mask.sum())


def add_quality_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range_anxiety_high"] = np.where(
        out["range_anxiety_score"].notna(),
        (out["range_anxiety_score"] > 5).astype(int), np.nan,
    )
    expected_cost = out["monthly_energy_consumption_kwh"] * out["electricity_cost_per_kwh"]
    out["charging_cost_absolute_error"] = out["monthly_charging_cost"] - expected_cost
    denominator = expected_cost.abs().clip(lower=1e-6)
    out["charging_cost_relative_error"] = out["charging_cost_absolute_error"].abs() / denominator
    out["weekly_daily_ratio"] = out["weekly_travel_distance_km"] / (
        5 * out["daily_commute_km"].clip(lower=1.0)
    )
    out["annual_income_log"] = np.log1p(out["annual_income"].clip(lower=0))
    out["fuel_burden_ratio"] = 12 * out["fuel_expense_per_month"] / out["annual_income"].clip(lower=1)
    out["station_commute_ratio"] = out["nearest_charging_station_km"] / out["daily_commute_km"].clip(lower=5)
    out["row_missing_count"] = out[EXPECTED_COLUMNS].isna().sum(axis=1)
    for col in EXPECTED_COLUMNS:
        if out[col].isna().any():
            out[f"{col}_missing"] = out[col].isna().astype(int)
    return out


def fit_reference_imputer(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """仅用于审计图表的固定参考预处理；正式建模须在每个训练折内重新估计。"""
    fit_idx, _ = train_test_split(
        np.arange(len(cleaned)), test_size=0.20, random_state=RANDOM_SEED,
        stratify=cleaned["ev_adoption_likelihood"],
    )
    train = cleaned.iloc[fit_idx]
    imputed = cleaned.copy()
    stats: dict[str, object] = {"fit_rows": int(len(fit_idx)), "seed": RANDOM_SEED}

    numeric_columns = [c for c in EXPECTED_COLUMNS if c not in CATEGORICAL_COLUMNS]
    for col in numeric_columns:
        median = float(train[col].median())
        imputed[col] = imputed[col].fillna(median)
        stats[col] = median

    for col in CATEGORICAL_COLUMNS:
        mode = train[col].dropna().mode()
        fill_value = str(mode.iloc[0]) if not mode.empty else "Missing"
        imputed[col] = imputed[col].fillna(fill_value)
        stats[col] = fill_value

    imputed = add_quality_fields(imputed)
    return imputed, stats


def field_audit(raw: pd.DataFrame, cleaned: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in EXPECTED_COLUMNS:
        raw_series = raw[col]
        clean_series = cleaned[col]
        rows.append({
            "字段": col,
            "原始类型": str(raw_series.dtype),
            "清洗后类型": str(clean_series.dtype),
            "原始缺失数": int(raw_series.isna().sum()),
            "规则清洗后缺失数": int(clean_series.isna().sum()),
            "参考插补后缺失数": int(imputed[col].isna().sum()),
            "唯一值数": int(clean_series.nunique(dropna=True)),
            "是否目标或派生结果": "是" if col in TARGET_COLUMNS + DOWNSTREAM_COLUMNS else "否",
        })
    return pd.DataFrame(rows)


def missing_comparison(raw: pd.DataFrame, cleaned: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(raw)
    for col in EXPECTED_COLUMNS:
        for stage, frame in [("原始数据", raw), ("规则清洗后", cleaned), ("参考插补后", imputed)]:
            count = int(frame[col].isna().sum())
            rows.append({"字段": col, "阶段": stage, "缺失数": count, "缺失率": count / n})
    return pd.DataFrame(rows)


def distribution_summary(raw: pd.DataFrame, cleaned: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "annual_income", "daily_commute_km", "weekly_travel_distance_km",
        "fuel_expense_per_month", "charging_station_accessibility",
        "nearest_charging_station_km", "ev_knowledge_score",
        "monthly_energy_consumption_kwh", "monthly_charging_cost",
    ]
    rows = []
    for col in selected:
        for stage, frame in [("原始数据", raw), ("规则清洗后", cleaned), ("参考插补后", imputed)]:
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            rows.append({
                "字段": col, "阶段": stage, "样本数": int(len(s)),
                "均值": float(s.mean()), "标准差": float(s.std()),
                "最小值": float(s.min()), "下四分位数": float(s.quantile(0.25)),
                "中位数": float(s.median()), "上四分位数": float(s.quantile(0.75)),
                "最大值": float(s.max()), "偏度": float(s.skew()),
            })
    return pd.DataFrame(rows)


def quantile_comparison(raw: pd.DataFrame, cleaned: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    selected = ["fuel_expense_per_month", "annual_income", "nearest_charging_station_km"]
    quantiles = np.linspace(0.01, 0.99, 99)
    rows = []
    for col in selected:
        for stage, frame in [("原始数据", raw), ("规则清洗后", cleaned), ("参考插补后", imputed)]:
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            for q, value in s.quantile(quantiles).items():
                rows.append({"字段": col, "阶段": stage, "分位点": float(q), "数值": float(value)})
    return pd.DataFrame(rows)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mappings = {
        "任务一": df["ev_adoption_likelihood"],
        "任务二": df["home_charging_available"].map({0: "无", 1: "有"}),
        "任务三": df["range_anxiety_high"].map({0: "低焦虑", 1: "高焦虑"}),
    }
    for task, series in mappings.items():
        counts = series.value_counts(dropna=False)
        for label, count in counts.items():
            rows.append({
                "任务": task, "类别": "缺失" if pd.isna(label) else str(label),
                "样本数": int(count), "比例": float(count / len(series)),
            })
    return pd.DataFrame(rows)


def categorical_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_COLUMNS + BINARY_COLUMNS:
        counts = df[col].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({
                "字段": col, "类别": "缺失" if pd.isna(value) else str(value),
                "样本数": int(count), "比例": float(count / len(df)),
            })
    return pd.DataFrame(rows)


def consistency_checks(cleaned: pd.DataFrame) -> pd.DataFrame:
    valid = cleaned[["monthly_energy_consumption_kwh", "electricity_cost_per_kwh", "monthly_charging_cost"]].dropna()
    expected = valid["monthly_energy_consumption_kwh"] * valid["electricity_cost_per_kwh"]
    abs_err = (valid["monthly_charging_cost"] - expected).abs()
    rel_err = abs_err / expected.abs().clip(lower=1e-6)
    return pd.DataFrame([
        {"检查项": "月充电成本乘积一致性", "统计量": "样本数", "数值": len(valid), "判定阈值": "-", "结论": "用于识别派生字段"},
        {"检查项": "月充电成本乘积一致性", "统计量": "绝对误差中位数", "数值": abs_err.median(), "判定阈值": "-", "结论": "越接近0越一致"},
        {"检查项": "月充电成本乘积一致性", "统计量": "相对误差中位数", "数值": rel_err.median(), "判定阈值": "-", "结论": "越接近0越一致"},
        {"检查项": "月充电成本乘积一致性", "统计量": "相对误差≤5%比例", "数值": (rel_err <= 0.05).mean(), "判定阈值": "0.95", "结论": "高比例说明字段为计算派生结果"},
    ])


def correlation_long(df: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "annual_income", "daily_commute_km", "weekly_travel_distance_km",
        "vehicle_age_years", "fuel_expense_per_month",
        "charging_station_accessibility", "nearest_charging_station_km",
        "electricity_cost_per_kwh", "environmental_awareness_score",
        "government_incentive_awareness", "technology_affinity_score",
        "range_anxiety_score", "battery_replacement_concern", "ev_knowledge_score",
    ]
    corr = df[selected].corr(method="spearman")
    rows = []
    for a in selected:
        for b in selected:
            rows.append({"变量一": a, "变量二": b, "Spearman相关系数": float(corr.loc[a, b])})
    return pd.DataFrame(rows)


def vif_table(imputed: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "age", "annual_income_log", "daily_commute_km", "weekly_travel_distance_km",
        "vehicle_age_years", "fuel_expense_per_month",
        "charging_station_accessibility", "nearest_charging_station_km",
        "electricity_cost_per_kwh", "environmental_awareness_score",
        "government_incentive_awareness", "technology_affinity_score",
        "battery_replacement_concern", "ev_knowledge_score",
    ]
    x = imputed[cols].astype(float)
    x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    rows = []
    for col in cols:
        y = x[col].to_numpy()
        others = x.drop(columns=col).to_numpy()
        r2 = LinearRegression().fit(others, y).score(others, y)
        vif = np.inf if r2 >= 1 else 1 / (1 - r2)
        rows.append({"字段": col, "R方": float(r2), "VIF": float(vif), "诊断": "高共线" if vif > 10 else "可接受"})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def missingness_diagnostics(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = {
        "任务一": cleaned["ev_adoption_likelihood"],
        "任务二": cleaned["home_charging_available"],
        "任务三": cleaned["range_anxiety_high"],
    }
    for col in EXPECTED_COLUMNS:
        indicator = cleaned[col].isna().astype(int)
        if indicator.sum() == 0:
            continue
        for task, target in targets.items():
            valid = target.notna()
            table = pd.crosstab(indicator[valid], target[valid])
            if table.shape[0] < 2 or table.shape[1] < 2:
                p_value = np.nan
            else:
                _, p_value, _, _ = chi2_contingency(table)
            rows.append({
                "缺失字段": col, "任务": task, "缺失数": int(indicator.sum()),
                "卡方检验P值": p_value,
                "判断": "缺失与标签可能相关" if pd.notna(p_value) and p_value < 0.05 else "未发现显著关联",
            })
    if not rows:
        rows.append({"缺失字段": "无", "任务": "全部", "缺失数": 0, "卡方检验P值": np.nan, "判断": "无缺失变量"})
    return pd.DataFrame(rows)


def single_feature_screen(imputed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    task_targets = {
        "任务一": imputed["ev_adoption_likelihood"].astype(str),
        "任务二": imputed["home_charging_available"].astype(int),
        "任务三": imputed["range_anxiety_high"].astype(int),
    }
    candidate_features = [c for c in EXPECTED_COLUMNS if c not in TARGET_COLUMNS]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for task, y in task_targets.items():
        majority = float(y.value_counts(normalize=True).max())
        for feature in candidate_features:
            x = imputed[[feature]]
            if feature in CATEGORICAL_COLUMNS:
                prep = ColumnTransformer([
                    ("cat", OneHotEncoder(handle_unknown="ignore"), [feature])
                ])
            else:
                prep = ColumnTransformer([
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), [feature])
                ])
            model = Pipeline([
                ("prep", prep),
                ("clf", LogisticRegression(max_iter=600, class_weight=None)),
            ])
            scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
            rows.append({
                "任务": task, "字段": feature, "五折准确率均值": float(scores.mean()),
                "五折准确率标准差": float(scores.std()), "多数类基线": majority,
                "超基线幅度": float(scores.mean() - majority),
                "泄漏风险": "高" if scores.mean() >= 0.95 else "中" if scores.mean() >= 0.85 else "低",
            })
    return pd.DataFrame(rows).sort_values(["任务", "五折准确率均值"], ascending=[True, False])


def feature_whitelist() -> pd.DataFrame:
    rows = []
    all_features = EXPECTED_COLUMNS.copy()
    for task, spec in TASKS.items():
        excluded = set(spec["strict_exclude"])
        relaxed_add = set(spec["relaxed_add"])
        for feature in all_features:
            if feature in excluded:
                status = "严格模型排除"
                reason = "目标、同步变量或下游派生结果"
            elif feature in relaxed_add:
                status = "宽松模型可加入"
                reason = "可能存在双向关系，需消融验证"
            else:
                status = "严格模型保留"
                reason = "可作为任务前置解释变量"
            rows.append({"任务": task, "字段": feature, "使用状态": status, "理由": reason})
    return pd.DataFrame(rows)


def preprocessing_rules() -> pd.DataFrame:
    return pd.DataFrame([
        {"步骤": 1, "处理对象": "字段与数据类型", "规则": "核对23个标准字段；类别去空白；数值解析失败转缺失", "建模阶段位置": "全数据确定性处理"},
        {"步骤": 2, "处理对象": "业务异常", "规则": "燃油费用等违反现实范围的值转缺失，不用孤立森林替代业务规则", "建模阶段位置": "全数据确定性处理"},
        {"步骤": 3, "处理对象": "完全重复记录", "规则": "保留首次出现记录并记录原记录键", "建模阶段位置": "全数据确定性处理"},
        {"步骤": 4, "处理对象": "派生字段", "规则": "月能耗和月充电成本仅做一致性审计，不进入三个主模型", "建模阶段位置": "特征白名单"},
        {"步骤": 5, "处理对象": "缺失值", "规则": "所有中位数、众数或模型插补参数仅在训练折估计", "建模阶段位置": "每个交叉验证训练折"},
        {"步骤": 6, "处理对象": "类别编码", "规则": "CatBoost保留原类别；线性模型在训练折独热编码", "建模阶段位置": "每个交叉验证训练折"},
        {"步骤": 7, "处理对象": "标准化", "规则": "仅Logistic/GAM等模型使用训练折参数标准化；树模型不强制", "建模阶段位置": "每个交叉验证训练折"},
        {"步骤": 8, "处理对象": "参考插补数据", "规则": "仅用于审计图表的前后分布展示，不作为最终测试输入", "建模阶段位置": "论文展示"},
    ])


def preprocessing_robustness(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fuel = pd.to_numeric(raw["fuel_expense_per_month"], errors="coerce")
    invalid = fuel < 0
    base_valid = fuel.mask(invalid)
    scenarios = {
        "负值转缺失后中位数插补": base_valid.fillna(base_valid.median()),
        "负值截断为0": fuel.clip(lower=0).fillna(fuel.clip(lower=0).median()),
        "删除负值记录": fuel.loc[~invalid].dropna(),
    }
    for name, series in scenarios.items():
        rows.append({
            "处理变量": "fuel_expense_per_month", "方案": name,
            "样本数": int(series.size), "均值": float(series.mean()),
            "标准差": float(series.std()), "中位数": float(series.median()),
            "下四分位数": float(series.quantile(0.25)), "上四分位数": float(series.quantile(0.75)),
        })

    for col in ["annual_income", "weekly_travel_distance_km", "monthly_charging_cost"]:
        s = pd.to_numeric(cleaned[col], errors="coerce").dropna()
        lower, upper = s.quantile([0.01, 0.99])
        clipped = s.clip(lower, upper)
        rows.extend([
            {"处理变量": col, "方案": "不缩尾", "样本数": int(len(s)), "均值": float(s.mean()), "标准差": float(s.std()), "中位数": float(s.median()), "下四分位数": float(s.quantile(0.25)), "上四分位数": float(s.quantile(0.75))},
            {"处理变量": col, "方案": "1%-99%缩尾对照", "样本数": int(len(clipped)), "均值": float(clipped.mean()), "标准差": float(clipped.std()), "中位数": float(clipped.median()), "下四分位数": float(clipped.quantile(0.25)), "上四分位数": float(clipped.quantile(0.75))},
        ])
    return pd.DataFrame(rows)


def build_audit_table(raw: pd.DataFrame, cleaned: pd.DataFrame, duplicate_count: int, changes: pd.DataFrame) -> pd.DataFrame:
    invalid_counts = changes[changes["问题类型"] == "业务范围异常"].groupby("字段").size().to_dict()
    rows = [
        {"等级": "Info", "检查项": "样本规模", "信息": f"原始{len(raw)}行、{len(raw.columns)}列", "处理方式": "记录并核验"},
        {"等级": "Info", "检查项": "字段完整性", "信息": f"标准字段{len(EXPECTED_COLUMNS)}个", "处理方式": "全部存在"},
        {"等级": "Warning" if raw.isna().sum().sum() else "Info", "检查项": "原始缺失值", "信息": f"共{int(raw.isna().sum().sum())}个", "处理方式": "确定性清洗后保留缺失；训练折内插补"},
        {"等级": "Warning" if duplicate_count else "Info", "检查项": "完全重复记录", "信息": f"{duplicate_count}行", "处理方式": "保留首次出现记录"},
        {"等级": "Warning" if invalid_counts else "Info", "检查项": "业务范围异常", "信息": json.dumps(invalid_counts, ensure_ascii=False), "处理方式": "异常值转缺失并保留清洗明细"},
        {"等级": "Warning", "检查项": "派生结果变量", "信息": "monthly_energy_consumption_kwh、monthly_charging_cost", "处理方式": "仅审计，不进入三个主模型"},
        {"等级": "Info", "检查项": "规则清洗后样本数", "信息": f"{len(cleaned)}行", "处理方式": "输出基础清洗数据"},
    ]
    return pd.DataFrame(rows)


def build_core_metrics(raw: pd.DataFrame, cleaned: pd.DataFrame, changes: pd.DataFrame, duplicates: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"指标": "原始样本数", "数值": len(raw), "单位": "行", "统计口径": "CSV数据行"},
        {"指标": "字段数", "数值": len(raw.columns), "单位": "列", "统计口径": "原始字段"},
        {"指标": "原始缺失单元格数", "数值": int(raw.isna().sum().sum()), "单位": "个", "统计口径": "全部23字段"},
        {"指标": "业务异常单元格数", "数值": int((changes["问题类型"] == "业务范围异常").sum()), "单位": "个", "统计口径": "业务范围规则"},
        {"指标": "完全重复记录数", "数值": duplicates, "单位": "行", "统计口径": "23字段完全一致"},
        {"指标": "规则清洗后样本数", "数值": len(cleaned), "单位": "行", "统计口径": "删除完全重复记录后"},
        {"指标": "任务一类别数", "数值": cleaned["ev_adoption_likelihood"].nunique(), "单位": "类", "统计口径": "Low/Medium/High"},
        {"指标": "任务二正类比例", "数值": cleaned["home_charging_available"].mean(), "单位": "比例", "统计口径": "home_charging_available=1"},
        {"指标": "任务三高焦虑比例", "数值": cleaned["range_anxiety_high"].mean(), "单位": "比例", "统计口径": "range_anxiety_score>5"},
    ])


def write_outputs(root: Path, tables: dict[str, pd.DataFrame], robustness: dict[str, pd.DataFrame], cleaned: pd.DataFrame, imputed: pd.DataFrame, stats: dict) -> None:
    output = root / OUTPUT_DIR
    clean_dir = root / CLEAN_DIR
    output.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    for name, frame in robustness.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")

    cleaned.to_csv(clean_dir / "B题基础清洗数据.csv", index=False, encoding="utf-8-sig")
    imputed.to_csv(clean_dir / "B题参考预处理展示数据.csv", index=False, encoding="utf-8-sig")
    (output / "参考预处理参数.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(root: Path, tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["核心指标"].set_index("指标")["数值"]
    labels = tables["标签分布"]
    anomalies = tables["异常处理对比"]
    top_single = tables["单特征预测能力"].groupby("任务", group_keys=False).head(5)
    report = f"""# B题数据审计与预处理报告\n\n## 1. 数据规模\n\n- 原始样本数：{int(metrics['原始样本数'])}\n- 原始字段数：{int(metrics['字段数'])}\n- 原始缺失单元格：{int(metrics['原始缺失单元格数'])}\n- 业务异常单元格：{int(metrics['业务异常单元格数'])}\n- 完全重复记录：{int(metrics['完全重复记录数'])}\n- 规则清洗后样本数：{int(metrics['规则清洗后样本数'])}\n\n## 2. 处理口径\n\n1. 字段核验、文本规范化和业务规则属于确定性清洗，可在全数据执行。\n2. 违反现实约束的值转为缺失，不用孤立森林替代业务规则。\n3. 统计插补、标准化、编码和特征选择必须在交叉验证训练折内重新估计。\n4. `monthly_energy_consumption_kwh` 与 `monthly_charging_cost` 作为下游派生结果，仅用于一致性审计。\n5. `B题参考预处理展示数据.csv` 仅用于前后分布图，不作为最终模型测试输入。\n\n## 3. 标签分布\n\n{labels.to_markdown(index=False)}\n\n## 4. 异常处理汇总\n\n{anomalies.to_markdown(index=False)}\n\n## 5. 单特征泄漏筛查前五项\n\n{top_single.to_markdown(index=False)}\n\n## 6. 后续建模接口\n\n- 正式模型读取 `预处理数据/B题基础清洗数据.csv`。\n- 每个任务按 `特征白名单.csv` 建立严格与宽松模型。\n- 缺失插补和标准化必须封装进交叉验证 Pipeline。\n- MATLAB 只读取后续汇总工作簿绘制正式论文图。\n"""
    (root / OUTPUT_DIR / "数据审计报告.md").write_text(report, encoding="utf-8")


def main() -> None:
    root = project_root()
    raw = ensure_input(root / DATA_FILE)
    typed, change_rows = normalize_types(raw)
    rule_cleaned = apply_business_rules(typed, change_rows)
    deduplicated, duplicate_count = remove_exact_duplicates(rule_cleaned, change_rows)
    cleaned = add_quality_fields(deduplicated)
    imputed, imputation_stats = fit_reference_imputer(cleaned)
    changes = pd.DataFrame(change_rows)
    if changes.empty:
        changes = pd.DataFrame([{
            "记录键": 0, "字段": "无", "问题类型": "无异常变更",
            "原始值": "-", "处理后值": "-", "处理方式": "无需处理",
        }])

    anomaly_summary = (
        changes.groupby(["字段", "问题类型", "处理方式"], dropna=False)
        .size().reset_index(name="原始问题数")
    )
    anomaly_summary["处理后问题数"] = 0

    tables = {
        "核心指标": build_core_metrics(raw, cleaned, changes, duplicate_count),
        "数据审计": build_audit_table(raw, cleaned, duplicate_count, changes),
        "字段审计": field_audit(raw, cleaned, imputed),
        "预处理规则": preprocessing_rules(),
        "特征白名单": feature_whitelist(),
        "缺失率对比": missing_comparison(raw, cleaned, imputed),
        "异常处理对比": anomaly_summary,
        "清洗明细": changes,
        "标签分布": label_distribution(cleaned),
        "类别变量分布": categorical_distribution(cleaned),
        "分布统计对比": distribution_summary(raw, cleaned, imputed),
        "分位点对比": quantile_comparison(raw, cleaned, imputed),
        "一致性检查": consistency_checks(cleaned),
        "相关性长表": correlation_long(imputed),
        "共线性诊断": vif_table(imputed),
        "缺失机制诊断": missingness_diagnostics(cleaned),
        "单特征预测能力": single_feature_screen(imputed),
    }
    robustness = {
        "预处理方案敏感性": preprocessing_robustness(raw, cleaned),
        "参考插补参数": pd.DataFrame([
            {"参数": key, "取值": value} for key, value in imputation_stats.items()
        ]),
    }

    write_outputs(root, tables, robustness, cleaned, imputed, imputation_stats)
    write_report(root, tables)
    print(f"完成：{root / OUTPUT_DIR}")


if __name__ == "__main__":
    main()
