"""Criação de variáveis derivadas (feature engineering)."""

import pandas as pd
import numpy as np


# Feriados federais dos EUA em 2015
US_HOLIDAYS_2015 = [
    "2015-01-01",  # New Year
    "2015-01-19",  # MLK Day
    "2015-02-16",  # Presidents Day
    "2015-05-25",  # Memorial Day
    "2015-07-03",  # Independence Day (observed)
    "2015-07-04",  # Independence Day
    "2015-09-07",  # Labor Day
    "2015-11-11",  # Veterans Day
    "2015-11-26",  # Thanksgiving
    "2015-12-25",  # Christmas
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features temporais derivadas."""
    out = df.copy()

    # Período do dia baseado no horário de partida programado
    dep_hour = out["SCHEDULED_DEPARTURE"] // 100
    conditions = [
        (dep_hour >= 5) & (dep_hour < 12),
        (dep_hour >= 12) & (dep_hour < 17),
        (dep_hour >= 17) & (dep_hour < 21),
    ]
    choices = ["manha", "tarde", "noite"]
    out["PERIODO_DIA"] = np.select(conditions, choices, default="madrugada")

    # Hora da partida (contínua)
    out["DEP_HOUR"] = dep_hour

    # Estação do ano
    season_map = {
        12: "inverno", 1: "inverno", 2: "inverno",
        3: "primavera", 4: "primavera", 5: "primavera",
        6: "verao", 7: "verao", 8: "verao",
        9: "outono", 10: "outono", 11: "outono",
    }
    out["ESTACAO"] = out["MONTH"].map(season_map)

    # É fim de semana? (6=Sábado, 7=Domingo)
    out["IS_WEEKEND"] = (out["DAY_OF_WEEK"] >= 6).astype(int)

    # É feriado?
    holidays = pd.to_datetime(US_HOLIDAYS_2015)
    out["IS_HOLIDAY"] = out["FL_DATE"].isin(holidays).astype(int)

    # Proximidade de feriado (1 dia antes ou depois)
    holiday_proximity = pd.DatetimeIndex([
        d for h in holidays
        for d in [h - pd.Timedelta(days=1), h, h + pd.Timedelta(days=1)]
    ])
    out["NEAR_HOLIDAY"] = out["FL_DATE"].isin(holiday_proximity).astype(int)

    return out


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de congestionamento operacional."""
    out = df.copy()

    # Voos na mesma hora + aeroporto de origem (proxy de congestionamento)
    out["FLIGHTS_SAME_HOUR_ORIGIN"] = out.groupby(
        ["ORIGIN_AIRPORT", "FL_DATE", "DEP_HOUR"]
    )["FLIGHT_NUMBER"].transform("count")

    # Popularidade da rota (total de voos na mesma rota)
    out["ROUTE_POPULARITY"] = out.groupby(
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    )["FLIGHT_NUMBER"].transform("count")

    return out


def add_delay_target(df: pd.DataFrame, threshold_min: int = 15) -> pd.DataFrame:
    """Cria target binário para classificação (atraso >= threshold)."""
    out = df.copy()
    out["IS_DELAYED"] = (out["ARRIVAL_DELAY"] >= threshold_min).astype(int)
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    df = add_time_features(df)
    df = add_congestion_features(df)
    df = add_delay_target(df)
    return df