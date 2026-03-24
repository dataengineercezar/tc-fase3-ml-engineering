"""Pipeline de modelagem não supervisionada: Clusterização e PCA."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def create_airport_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Cria perfil agregado por aeroporto de origem para clusterização."""
    profile = df.groupby("ORIGIN_AIRPORT").agg(
        total_voos=("FLIGHT_NUMBER", "count"),
        media_atraso_partida=("DEPARTURE_DELAY", "mean"),
        media_atraso_chegada=("ARRIVAL_DELAY", "mean"),
        mediana_atraso_chegada=("ARRIVAL_DELAY", "median"),
        std_atraso_chegada=("ARRIVAL_DELAY", "std"),
        pct_atrasados=("IS_DELAYED", "mean"),
        media_distancia=("DISTANCE", "mean"),
        media_taxi_out=("TAXI_OUT", "mean"),
        media_taxi_in=("TAXI_IN", "mean"),
        num_companhias=("AIRLINE", "nunique"),
        num_destinos=("DESTINATION_AIRPORT", "nunique"),
    ).reset_index()

    # Filtrar aeroportos com volume mínimo (para robustez)
    profile = profile[profile["total_voos"] >= 1000].copy()
    profile.reset_index(drop=True, inplace=True)
    return profile


def run_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clusterização de aeroportos por perfil operacional."""
    profile = create_airport_profile(df)

    feature_cols = [
        "total_voos", "media_atraso_partida", "media_atraso_chegada",
        "pct_atrasados", "media_distancia", "media_taxi_out",
        "num_companhias", "num_destinos"
    ]

    X = profile[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Método do cotovelo para definir k
    inertias = []
    sil_scores = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Plot cotovelo e silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(K_range, inertias, "bo-")
    axes[0].set_xlabel("Número de Clusters (k)")
    axes[0].set_ylabel("Inércia")
    axes[0].set_title("Método do Cotovelo")

    axes[1].plot(K_range, sil_scores, "ro-")
    axes[1].set_xlabel("Número de Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score por k")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "figures" / "elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/elbow_silhouette.png")

    # Escolher k ótimo (maior silhouette)
    best_k = K_range[np.argmax(sil_scores)]
    print(f"\nMelhor k (silhouette): {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    profile["CLUSTER"] = km_final.fit_predict(X_scaled)

    # Estatísticas por cluster
    cluster_stats = profile.groupby("CLUSTER")[feature_cols].mean()
    print("\n=== Perfil Médio por Cluster ===")
    print(cluster_stats.round(2).to_string())

    # Plot dos clusters
    _plot_clusters(profile, X_scaled, feature_cols, best_k)

    # DBSCAN comparativo
    print("\n--- DBSCAN (comparação) ---")
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    db_labels = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    print(f"Clusters DBSCAN: {n_clusters_db}")
    print(f"Outliers (ruído): {n_noise} aeroportos")
    if n_clusters_db >= 2:
        mask = db_labels != -1
        sil_db = silhouette_score(X_scaled[mask], db_labels[mask])
        print(f"Silhouette (DBSCAN, excluindo ruído): {sil_db:.3f}")
    profile["DBSCAN_LABEL"] = db_labels

    return profile, {"model": km_final, "scaler": scaler, "best_k": best_k,
                     "dbscan": dbscan, "n_clusters_dbscan": n_clusters_db}


def _plot_clusters(profile: pd.DataFrame, X_scaled: np.ndarray,
                   feature_cols: list, n_clusters: int):
    """Visualização dos clusters via PCA 2D."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter PCA
    scatter = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=profile["CLUSTER"], cmap="Set2", s=60, alpha=0.8, edgecolors="black", linewidth=0.5
    )
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)")
    axes[0].set_title("Clusters de Aeroportos (PCA 2D)")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")

    # Anotar aeroportos relevantes
    for _, row in profile.iterrows():
        idx = row.name
        axes[0].annotate(row["ORIGIN_AIRPORT"], (X_pca[idx, 0], X_pca[idx, 1]),
                         fontsize=5, alpha=0.6)

    # Heatmap do perfil por cluster
    cluster_means = profile.groupby("CLUSTER")[feature_cols].mean()
    # Normalizar para heatmap
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    sns.heatmap(cluster_means_norm, annot=cluster_means.round(1).values,
                fmt="", cmap="YlOrRd", ax=axes[1], linewidths=0.5)
    axes[1].set_title("Perfil Médio por Cluster (valores reais, cores normalizadas)")
    axes[1].set_ylabel("Cluster")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "figures" / "clustering_results.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/clustering_results.png")


def run_pca_analysis(df: pd.DataFrame) -> dict:
    """PCA para redução de dimensionalidade e análise de variância explicada."""
    numeric_cols = [
        "DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE", "SCHEDULED_TIME",
        "ELAPSED_TIME", "AIR_TIME", "TAXI_OUT", "TAXI_IN",
        "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"
    ]

    X = df[numeric_cols].dropna()

    # Amostra para PCA (performance)
    if len(X) > 200_000:
        X = X.sample(n=200_000, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Plot variância explicada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color="steelblue", alpha=0.7, label="Individual")
    axes[0].plot(range(1, len(cumvar) + 1), cumvar, "ro-", label="Acumulada")
    axes[0].axhline(0.90, color="gray", linestyle="--", alpha=0.5, label="90% variância")
    axes[0].set_xlabel("Componente Principal")
    axes[0].set_ylabel("Variância Explicada")
    axes[0].set_title("Variância Explicada por Componente")
    axes[0].legend()

    # Componentes com 90% de variância
    n_90 = np.argmax(cumvar >= 0.90) + 1
    print(f"\nComponentes para 90% da variância: {n_90}")

    # Loadings do PC1 e PC2
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=numeric_cols,
        columns=["PC1", "PC2"]
    )
    loadings_sorted = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=True).index)
    loadings_sorted.plot(kind="barh", ax=axes[1], color=["steelblue", "coral"])
    axes[1].set_title("Loadings - PC1 e PC2")
    axes[1].set_xlabel("Loading")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "figures" / "pca_analysis.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/pca_analysis.png")

    return {"pca": pca, "n_components_90": n_90, "loadings": loadings}


def run_anomaly_detection(df: pd.DataFrame) -> dict:
    """Detecção de anomalias em voos com Isolation Forest."""
    feature_cols = [
        "DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE", "SCHEDULED_TIME",
        "TAXI_OUT", "TAXI_IN",
    ]
    X = df[feature_cols].dropna()

    if len(X) > 200_000:
        X = X.sample(n=200_000, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1
    )
    labels = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)

    n_anomalies = (labels == -1).sum()
    n_normal = (labels == 1).sum()
    pct_anomalies = n_anomalies / len(labels) * 100

    print(f"\nDetecção de Anomalias (Isolation Forest):")
    print(f"  Normais: {n_normal:,} ({100 - pct_anomalies:.1f}%)")
    print(f"  Anomalias: {n_anomalies:,} ({pct_anomalies:.1f}%)")

    X_result = X.copy()
    X_result["ANOMALY"] = labels
    X_result["ANOMALY_SCORE"] = scores

    # Perfil das anomalias vs normais
    profile = X_result.groupby("ANOMALY")[feature_cols].mean().round(2)
    profile.index = profile.index.map({-1: "Anomalia", 1: "Normal"})
    print(f"\nPerfil médio:")
    print(profile.to_string())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Score distribution
    axes[0].hist(scores, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    threshold = np.percentile(scores, 5)
    axes[0].axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.3f})")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Distribuição do Anomaly Score")
    axes[0].legend()

    # Anomalias em scatter 2D (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    normal_mask = labels == 1
    axes[1].scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                    alpha=0.1, s=3, color="steelblue", label="Normal")
    axes[1].scatter(X_pca[~normal_mask, 0], X_pca[~normal_mask, 1],
                    alpha=0.3, s=10, color="red", label="Anomalia")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Anomalias em Espaço PCA 2D")
    axes[1].legend()

    # Compare delay distribution: normal vs anomaly
    axes[2].hist(X_result.loc[normal_mask, "ARRIVAL_DELAY"].clip(-60, 200),
                 bins=60, alpha=0.6, color="steelblue", label="Normal", density=True)
    axes[2].hist(X_result.loc[~normal_mask, "ARRIVAL_DELAY"].clip(-60, 200),
                 bins=60, alpha=0.6, color="red", label="Anomalia", density=True)
    axes[2].set_xlabel("Atraso na Chegada (min)")
    axes[2].set_ylabel("Densidade")
    axes[2].set_title("Distribuição de Atraso: Normal vs Anomalia")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "figures" / "anomaly_detection.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  >> Salvo: outputs/figures/anomaly_detection.png")

    return {
        "model": iso_forest,
        "n_anomalies": n_anomalies,
        "pct_anomalies": pct_anomalies,
        "profile": profile,
    }