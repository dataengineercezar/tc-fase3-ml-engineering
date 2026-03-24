"""Pipeline de modelagem supervisionada: Classificação e Regressão."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


# ---------- Preparação ----------

FEATURE_COLS = [
    "MONTH", "DAY_OF_WEEK", "DEP_HOUR", "DISTANCE",
    "SCHEDULED_TIME", "IS_WEEKEND", "IS_HOLIDAY", "NEAR_HOLIDAY",
    "FLIGHTS_SAME_HOUR_ORIGIN", "ROUTE_POPULARITY",
    "AIRLINE_ENC", "ORIGIN_ENC", "DEST_ENC",
    "PERIODO_DIA_ENC", "ESTACAO_ENC",
]


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode colunas categóricas."""
    out = df.copy()
    encoders = {}
    cat_cols = {
        "AIRLINE": "AIRLINE_ENC",
        "ORIGIN_AIRPORT": "ORIGIN_ENC",
        "DESTINATION_AIRPORT": "DEST_ENC",
        "PERIODO_DIA": "PERIODO_DIA_ENC",
        "ESTACAO": "ESTACAO_ENC",
    }
    for col, new_col in cat_cols.items():
        le = LabelEncoder()
        out[new_col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le
    return out, encoders


def split_data(df: pd.DataFrame, target_col: str, feature_cols: list,
               test_size: float = 0.2, random_state: int = 42):
    """Divide em treino/teste."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y if y.nunique() < 20 else None)


# ---------- Classificação ----------

def run_classification(df: pd.DataFrame) -> dict:
    """Executa pipeline de classificação: prever se voo atrasa (>= 15 min)."""
    df_enc, encoders = encode_categoricals(df)

    X_train, X_test, y_train, y_test = split_data(
        df_enc, "IS_DELAYED", FEATURE_COLS
    )

    # Amostragem para treino rápido (se dataset muito grande)
    if len(X_train) > 500_000:
        sample_idx = X_train.sample(n=500_000, random_state=42).index
        X_train_s = X_train.loc[sample_idx]
        y_train_s = y_train.loc[sample_idx]
    else:
        X_train_s, y_train_s = X_train, y_train

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Treinando: {name}")
        print(f"{'='*50}")

        if name == "Logistic Regression":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_s)
            X_te = scaler.transform(X_test)
        else:
            X_tr = X_train_s
            X_te = X_test

        model.fit(X_tr, y_train_s)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation (3-fold na amostra de treino para viabilidade)
        cv_sample_size = min(100_000, len(X_train_s))
        cv_idx = X_train_s.sample(n=cv_sample_size, random_state=42).index
        X_cv = X_train_s.loc[cv_idx]
        y_cv = y_train_s.loc[cv_idx]
        if name == "Logistic Regression":
            scaler_cv = StandardScaler()
            X_cv_scaled = scaler_cv.fit_transform(X_cv)
            cv_scores = cross_val_score(model.__class__(**model.get_params()),
                                        X_cv_scaled, y_cv, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                        scoring="roc_auc", n_jobs=-1)
        else:
            cv_scores = cross_val_score(model.__class__(**model.get_params()),
                                        X_cv, y_cv, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                        scoring="roc_auc", n_jobs=-1)

        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {auc:.4f}")
        print(f"CV AUC-ROC (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "model": model,
            "report": report,
            "auc": auc,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    # Salvar melhor modelo
    best_name = max(results, key=lambda k: results[k]["auc"])
    joblib.dump(results[best_name]["model"], OUTPUTS_DIR / "models" / "best_classifier.pkl")
    print(f"\n>> Melhor modelo (classificação): {best_name} (AUC={results[best_name]['auc']:.4f})")

    # Plots
    _plot_classification_results(results)
    _plot_feature_importance(results[best_name]["model"], FEATURE_COLS, best_name)

    return results


def _plot_classification_results(results: dict):
    """Gera gráficos de comparação dos classificadores."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1) Comparação de métricas
    metrics_df = pd.DataFrame({
        name: {
            "Accuracy": r["report"]["accuracy"],
            "Precision": r["report"]["weighted avg"]["precision"],
            "Recall": r["report"]["weighted avg"]["recall"],
            "F1-Score": r["report"]["weighted avg"]["f1-score"],
            "AUC-ROC": r["auc"],
            "CV AUC": r["cv_auc_mean"],
        }
        for name, r in results.items()
    }).T
    metrics_df.plot(kind="bar", ax=axes[0], rot=15)
    axes[0].set_title("Comparação de Métricas - Classificação")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower right", fontsize=8)

    # 2) Curvas ROC
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        axes[1].plot(fpr, tpr, label=f'{name} (AUC={r["auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Curvas ROC")
    axes[1].legend(fontsize=8)

    # 3) Matriz de confusão do melhor modelo
    best_name = max(results, key=lambda k: results[k]["auc"])
    cm = confusion_matrix(results[best_name]["y_test"], results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["No Delay", "Delayed"],
                yticklabels=["No Delay", "Delayed"])
    axes[2].set_title(f"Matriz de Confusão - {best_name}")
    axes[2].set_ylabel("Real")
    axes[2].set_xlabel("Previsto")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "figures" / "classification_results.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/classification_results.png")


def _plot_feature_importance(model, feature_names: list, model_name: str):
    """Feature importance do melhor modelo."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:15]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(idx)), importances[idx], color="steelblue")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.invert_yaxis()
        ax.set_title(f"Feature Importance - {model_name}")
        ax.set_xlabel("Importância")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "figures" / "feature_importance_clf.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        print("  >> Salvo: outputs/figures/feature_importance_clf.png")


# ---------- Regressão ----------

def run_regression(df: pd.DataFrame) -> dict:
    """Executa pipeline de regressão: prever minutos de atraso na chegada."""
    # Filtrar apenas voos que tiveram atraso > 0
    df_delayed = df[df["ARRIVAL_DELAY"] > 0].copy()
    df_enc, encoders = encode_categoricals(df_delayed)

    X = df_enc[FEATURE_COLS]
    y = df_enc["ARRIVAL_DELAY"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(X_train) > 300_000:
        sample_idx = X_train.sample(n=300_000, random_state=42).index
        X_train_s = X_train.loc[sample_idx]
        y_train_s = y_train.loc[sample_idx]
    else:
        X_train_s, y_train_s = X_train, y_train

    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Treinando: {name}")
        print(f"{'='*50}")

        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Cross-validation (3-fold na amostra de treino)
        cv_sample_size = min(80_000, len(X_train_s))
        cv_idx = X_train_s.sample(n=cv_sample_size, random_state=42).index
        X_cv = X_train_s.loc[cv_idx]
        y_cv = y_train_s.loc[cv_idx]
        cv_scores = cross_val_score(model.__class__(**model.get_params()),
                                    X_cv, y_cv, cv=3,
                                    scoring="neg_mean_absolute_error", n_jobs=-1)
        cv_mae_mean = -cv_scores.mean()
        cv_mae_std = cv_scores.std()

        print(f"MAE:  {mae:.2f} minutos")
        print(f"RMSE: {rmse:.2f} minutos")
        print(f"R²:   {r2:.4f}")
        print(f"CV MAE (3-fold): {cv_mae_mean:.2f} ± {cv_mae_std:.2f}")

        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    # Salvar melhor modelo
    best_name = min(results, key=lambda k: results[k]["mae"])
    joblib.dump(results[best_name]["model"], OUTPUTS_DIR / "models" / "best_regressor.pkl")
    print(f"\n>> Melhor modelo (regressão): {best_name} (MAE={results[best_name]['mae']:.2f})")

    _plot_regression_results(results)
    return results


def _plot_regression_results(results: dict):
    """Gráficos de avaliação dos regressores."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1) Comparação de métricas
    metrics_df = pd.DataFrame({
        name: {"MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"]}
        for name, r in results.items()
    }).T
    metrics_df[["MAE", "RMSE"]].plot(kind="bar", ax=axes[0], rot=15, color=["steelblue", "coral"])
    axes[0].set_title("MAE e RMSE (minutos)")
    axes[0].set_ylabel("Minutos")

    # 2) Real vs Previsto (melhor modelo)
    best_name = min(results, key=lambda k: results[k]["mae"])
    r = results[best_name]
    sample = np.random.RandomState(42).choice(len(r["y_test"]), size=min(5000, len(r["y_test"])), replace=False)
    axes[1].scatter(r["y_test"].values[sample], r["y_pred"][sample], alpha=0.2, s=5, color="steelblue")
    max_val = min(r["y_test"].values[sample].max(), 300)
    axes[1].plot([0, max_val], [0, max_val], "r--", alpha=0.7)
    axes[1].set_xlabel("Atraso Real (min)")
    axes[1].set_ylabel("Atraso Previsto (min)")
    axes[1].set_title(f"Real vs Previsto - {best_name}")
    axes[1].set_xlim(0, max_val)
    axes[1].set_ylim(0, max_val)

    # 3) Distribuição dos resíduos
    residuals = r["y_test"].values[sample] - r["y_pred"][sample]
    axes[2].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_title(f"Distribuição dos Resíduos - {best_name}")
    axes[2].set_xlabel("Resíduo (min)")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "figures" / "regression_results.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/regression_results.png")