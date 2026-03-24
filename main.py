"""
Tech Challenge Fase 3 - Machine Learning Engineering
Pipeline completo: EDA + Modelagem Supervisionada + Modelagem Não Supervisionada
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader import load_clean_data
from src.feature_engineering import prepare_features
from src.supervised import run_classification, run_regression
from src.unsupervised import run_clustering, run_pca_analysis, run_anomaly_detection

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

OUTPUTS_DIR = Path("outputs")
FIG_DIR = OUTPUTS_DIR / "figures"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def run_eda(df: pd.DataFrame, airlines: pd.DataFrame, airports: pd.DataFrame):
    """Análise Exploratória de Dados completa."""
    print("\n" + "=" * 70)
    print("1. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("=" * 70)

    # --- 1.1 Visão geral ---
    print("\n--- 1.1 Visão Geral ---")
    print(f"Shape: {df.shape}")
    print(f"Período: {df['FL_DATE'].min()} a {df['FL_DATE'].max()}")
    print(f"\nDistribuição da variável alvo (IS_DELAYED):")
    print(df["IS_DELAYED"].value_counts(normalize=True).round(4))
    print(f"\nEstatísticas de ARRIVAL_DELAY:")
    print(df["ARRIVAL_DELAY"].describe().round(2))
    print(f"\nValores nulos restantes (top 10):")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0].sort_values(ascending=False).head(10))

    # --- 1.2 Distribuição de atrasos ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histograma de atrasos
    delay_clipped = df["ARRIVAL_DELAY"].clip(-60, 180)
    axes[0, 0].hist(delay_clipped, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(0, color="red", linestyle="--", label="Sem atraso")
    axes[0, 0].axvline(15, color="orange", linestyle="--", label="Threshold (15min)")
    axes[0, 0].set_xlabel("Atraso na Chegada (min)")
    axes[0, 0].set_title("Distribuição de Atraso na Chegada")
    axes[0, 0].legend()

    # Top 15 aeroportos com mais atraso médio
    top_airports = (
        df.groupby("ORIGIN_AIRPORT")["ARRIVAL_DELAY"]
        .agg(["mean", "count"])
        .query("count >= 5000")
        .nlargest(15, "mean")
    )
    axes[0, 1].barh(top_airports.index, top_airports["mean"], color="coral")
    axes[0, 1].set_xlabel("Atraso Médio (min)")
    axes[0, 1].set_title("Top 15 Aeroportos - Maior Atraso Médio")
    axes[0, 1].invert_yaxis()

    # Atrasos por companhia aérea
    airline_delay = (
        df.groupby("AIRLINE_NAME")["ARRIVAL_DELAY"]
        .mean()
        .sort_values(ascending=True)
    )
    airline_delay.plot(kind="barh", ax=axes[1, 0], color="steelblue")
    axes[1, 0].set_xlabel("Atraso Médio (min)")
    axes[1, 0].set_title("Atraso Médio por Companhia Aérea")

    # Atrasos por dia da semana
    day_map = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sáb", 7: "Dom"}
    day_delay = df.groupby("DAY_OF_WEEK")["ARRIVAL_DELAY"].mean()
    day_delay.index = day_delay.index.map(day_map)
    day_delay.plot(kind="bar", ax=axes[1, 1], color="steelblue", rot=0)
    axes[1, 1].set_ylabel("Atraso Médio (min)")
    axes[1, 1].set_title("Atraso Médio por Dia da Semana")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/eda_overview.png")

    # --- 1.3 Padrões temporais ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Atraso por hora do dia
    hour_delay = df.groupby("DEP_HOUR")["ARRIVAL_DELAY"].mean()
    axes[0].plot(hour_delay.index, hour_delay.values, "o-", color="steelblue")
    axes[0].set_xlabel("Hora de Partida")
    axes[0].set_ylabel("Atraso Médio (min)")
    axes[0].set_title("Atraso Médio por Hora do Dia")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Atraso por mês
    month_delay = df.groupby("MONTH").agg(
        media=("ARRIVAL_DELAY", "mean"),
        pct_atrasados=("IS_DELAYED", "mean"),
    )
    axes[1].bar(month_delay.index, month_delay["media"], color="steelblue", alpha=0.7, label="Atraso médio")
    ax2 = axes[1].twinx()
    ax2.plot(month_delay.index, month_delay["pct_atrasados"] * 100, "ro-", label="% Atrasados")
    axes[1].set_xlabel("Mês")
    axes[1].set_ylabel("Atraso Médio (min)")
    ax2.set_ylabel("% Voos Atrasados")
    axes[1].set_title("Atrasos por Mês")
    axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Heatmap: dia da semana x hora
    heatmap_data = df.pivot_table(
        values="ARRIVAL_DELAY", index="DAY_OF_WEEK", columns="DEP_HOUR", aggfunc="mean"
    )
    heatmap_data.index = [day_map.get(i, i) for i in heatmap_data.index]
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=axes[2], cbar_kws={"label": "Atraso (min)"})
    axes[2].set_title("Atraso Médio: Dia da Semana × Hora")
    axes[2].set_xlabel("Hora de Partida")
    axes[2].set_ylabel("Dia da Semana")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_temporal.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/eda_temporal.png")

    # --- 1.4 Correlação ---
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_cols = [
        "DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE", "SCHEDULED_TIME",
        "ELAPSED_TIME", "AIR_TIME", "TAXI_OUT", "TAXI_IN",
        "AIR_SYSTEM_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"
    ]
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/eda_correlation.png")

    # --- 1.5 Composição dos atrasos ---
    fig, ax = plt.subplots(figsize=(10, 6))
    delay_types = ["AIR_SYSTEM_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
                   "WEATHER_DELAY", "SECURITY_DELAY"]
    delay_means = df[delay_types].mean().sort_values(ascending=True)
    delay_means.plot(kind="barh", ax=ax, color=["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"])
    ax.set_xlabel("Atraso Médio (min)")
    ax.set_title("Composição Média dos Tipos de Atraso")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_delay_types.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/eda_delay_types.png")

    # --- 1.6 Análise por Estado ---
    print("\n--- 1.6 Análise por Estado ---")
    state_stats = (
        df.groupby("ORIGIN_STATE")
        .agg(
            media_atraso=("ARRIVAL_DELAY", "mean"),
            pct_atrasados=("IS_DELAYED", "mean"),
            total_voos=("FLIGHT_NUMBER", "count"),
        )
        .query("total_voos >= 10000")
        .sort_values("media_atraso", ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    top_states = state_stats.head(20)
    axes[0].barh(top_states.index[::-1], top_states["media_atraso"].values[::-1], color="coral")
    axes[0].set_xlabel("Atraso Médio (min)")
    axes[0].set_title("Top 20 Estados — Maior Atraso Médio")
    (state_stats["pct_atrasados"].head(20) * 100).sort_values().plot(
        kind="barh", ax=axes[1], color="steelblue"
    )
    axes[1].set_xlabel("% Voos Atrasados")
    axes[1].set_title("Top 20 Estados — % de Atrasos")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_states.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/eda_states.png")

    # --- 1.7 Mapa geográfico de atrasos por aeroporto ---
    if HAS_PLOTLY:
        print("\n--- 1.7 Mapa Geográfico ---")
        airport_geo = (
            df.groupby("ORIGIN_AIRPORT")
            .agg(
                media_atraso=("ARRIVAL_DELAY", "mean"),
                pct_atrasados=("IS_DELAYED", "mean"),
                total_voos=("FLIGHT_NUMBER", "count"),
                lat=("ORIGIN_LAT", "first"),
                lon=("ORIGIN_LON", "first"),
            )
            .query("total_voos >= 1000")
            .reset_index()
        )
        fig_map = px.scatter_geo(
            airport_geo,
            lat="lat", lon="lon",
            size="total_voos",
            color="media_atraso",
            hover_name="ORIGIN_AIRPORT",
            hover_data={"total_voos": ":,", "pct_atrasados": ":.1%", "media_atraso": ":.1f"},
            color_continuous_scale="RdYlGn_r",
            size_max=25,
            title="Mapa de Atrasos Médios por Aeroporto (EUA 2015)",
            scope="usa",
        )
        fig_map.update_layout(geo=dict(showland=True, landcolor="lightgray"))
        fig_map.write_html(str(FIG_DIR / "mapa_atrasos.html"))
        print("  >> Salvo: outputs/figures/mapa_atrasos.html")
        try:
            fig_map.write_image(str(FIG_DIR / "mapa_atrasos.png"), width=1200, height=700)
            print("  >> Salvo: outputs/figures/mapa_atrasos.png")
        except Exception:
            print("  [AVISO] Exportação PNG do mapa falhou (Chrome não encontrado). "
                  "O mapa HTML foi salvo normalmente.")
    else:
        print("\n[INFO] Plotly não disponível — mapa geográfico não gerado.")


def main():
    # =========================
    # Carregar e preparar dados
    # =========================
    print("Carregando dados...")
    flights, airlines, airports = load_clean_data()
    print(f"Dados carregados: {flights.shape[0]:,} voos")

    print("Engenharia de features...")
    flights = prepare_features(flights)
    print(f"Features criadas. Colunas: {flights.shape[1]}")

    # Salvar dados processados
    processed_path = Path("data/processed/flights_processed.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    flights.to_csv(processed_path, index=False)
    print(f"Dados processados salvos em: {processed_path} ({flights.shape[0]:,} linhas)")

    # =========================
    # EDA
    # =========================
    run_eda(flights, airlines, airports)

    # =========================
    # Modelagem Supervisionada
    # =========================
    print("\n" + "=" * 70)
    print("2. MODELAGEM SUPERVISIONADA")
    print("=" * 70)

    print("\n--- 2.1 Classificação ---")
    clf_results = run_classification(flights)

    print("\n--- 2.2 Regressão ---")
    reg_results = run_regression(flights)

    # =========================
    # Modelagem Não Supervisionada
    # =========================
    print("\n" + "=" * 70)
    print("3. MODELAGEM NÃO SUPERVISIONADA")
    print("=" * 70)

    print("\n--- 3.1 Clusterização de Aeroportos ---")
    airport_profiles, cluster_info = run_clustering(flights)

    print("\n--- 3.2 Análise PCA ---")
    pca_results = run_pca_analysis(flights)

    print("\n--- 3.3 Detecção de Anomalias ---")
    anomaly_results = run_anomaly_detection(flights)

    # =========================
    # Conclusões
    # =========================
    print("\n" + "=" * 70)
    print("4. CONCLUSÕES E RESULTADOS")
    print("=" * 70)

    best_clf = max(clf_results, key=lambda k: clf_results[k]["auc"])
    best_reg = min(reg_results, key=lambda k: reg_results[k]["mae"])

    print(f"""
RESUMO DOS RESULTADOS:

CLASSIFICAÇÃO (Voo atrasa >= 15 min?):
  Melhor modelo: {best_clf}
  AUC-ROC: {clf_results[best_clf]['auc']:.4f}
  CV AUC-ROC (3-fold): {clf_results[best_clf]['cv_auc_mean']:.4f} ± {clf_results[best_clf]['cv_auc_std']:.4f}
  Accuracy: {clf_results[best_clf]['report']['accuracy']:.4f}

REGRESSÃO (Quantos minutos de atraso?):
  Melhor modelo: {best_reg}
  MAE: {reg_results[best_reg]['mae']:.2f} min
  CV MAE (3-fold): {reg_results[best_reg]['cv_mae_mean']:.2f} ± {reg_results[best_reg]['cv_mae_std']:.2f}
  RMSE: {reg_results[best_reg]['rmse']:.2f} min
  R²: {reg_results[best_reg]['r2']:.4f}

CLUSTERIZAÇÃO:
  Número ótimo de clusters (KMeans): {cluster_info['best_k']}
  Clusters DBSCAN: {cluster_info['n_clusters_dbscan']}
  Aeroportos analisados: {len(airport_profiles)}

PCA:
  Componentes para 90%% variância: {pca_results['n_components_90']}

DETECÇÃO DE ANOMALIAS:
  Anomalias detectadas: {anomaly_results['n_anomalies']:,} ({anomaly_results['pct_anomalies']:.1f}%)

LIMITAÇÕES:
  - Dados referentes apenas ao ano de 2015 (EUA)
  - Features externas (clima em tempo real, eventos) não disponíveis
  - Variáveis de atraso detalhadas (AIRLINE_DELAY etc.) só existem para voos atrasados
  - Amostragem foi necessária para modelos em dados de alto volume
  - KMeans assume clusters esféricos; DBSCAN foi adicionado como comparação

PRÓXIMOS PASSOS:
  - Incorporar dados meteorológicos (temperatura, precipitação)
  - Testar modelos de deep learning (LSTM para séries temporais)
  - Criar dashboard interativo com Streamlit ou Plotly Dash
  - Aplicar SMOTE ou balanceamento de classes
  - Feature engineering mais avançado (atraso médio histórico da rota)
  - Explorar semi-supervisionado com pseudo-labels
""")


if __name__ == "__main__":
    main()