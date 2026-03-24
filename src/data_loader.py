"""Carregamento e limpeza dos dados de voos."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw_data():
    """Carrega os três CSVs brutos."""
    flights = pd.read_csv(DATA_DIR / "raw" / "flights.csv", low_memory=False)
    airlines = pd.read_csv(DATA_DIR / "raw" / "airlines.csv")
    airports = pd.read_csv(DATA_DIR / "raw" / "airports.csv")
    return flights, airlines, airports


def clean_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """Limpa o dataframe de voos: remove cancelados, trata nulos."""
    df = flights.copy()

    # Remover voos cancelados (não possuem informação de atraso)
    df = df[df["CANCELLED"] == 0].copy()
    df.drop(columns=["CANCELLATION_REASON"], inplace=True)

    # Remover voos desviados
    df = df[df["DIVERTED"] == 0].copy()

    # Preencher colunas de atraso com 0 onde nulo (atraso inexistente = 0)
    delay_cols = [
        "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"
    ]
    df[delay_cols] = df[delay_cols].fillna(0)

    # Remover linhas onde DEPARTURE_DELAY ou ARRIVAL_DELAY são nulos
    df.dropna(subset=["DEPARTURE_DELAY", "ARRIVAL_DELAY"], inplace=True)

    # Criar coluna de data
    df["FL_DATE"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        )
    )

    df.reset_index(drop=True, inplace=True)
    return df


def enrich_with_names(flights: pd.DataFrame, airlines: pd.DataFrame,
                      airports: pd.DataFrame) -> pd.DataFrame:
    """Junta nomes das companhias e aeroportos ao dataframe de voos."""
    df = flights.merge(
        airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
        on="AIRLINE", how="left"
    )
    df = df.merge(
        airports[["IATA_CODE", "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"]].rename(
            columns={
                "IATA_CODE": "ORIGIN_AIRPORT",
                "AIRPORT": "ORIGIN_AIRPORT_NAME",
                "CITY": "ORIGIN_CITY",
                "STATE": "ORIGIN_STATE",
                "LATITUDE": "ORIGIN_LAT",
                "LONGITUDE": "ORIGIN_LON",
            }
        ),
        on="ORIGIN_AIRPORT", how="left"
    )
    df = df.merge(
        airports[["IATA_CODE", "AIRPORT", "CITY", "STATE", "LATITUDE", "LONGITUDE"]].rename(
            columns={
                "IATA_CODE": "DESTINATION_AIRPORT",
                "AIRPORT": "DEST_AIRPORT_NAME",
                "CITY": "DEST_CITY",
                "STATE": "DEST_STATE",
                "LATITUDE": "DEST_LAT",
                "LONGITUDE": "DEST_LON",
            }
        ),
        on="DESTINATION_AIRPORT", how="left"
    )
    return df


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pipeline completo: carrega, limpa e enriquece."""
    flights, airlines, airports = load_raw_data()
    flights = clean_flights(flights)
    flights = enrich_with_names(flights, airlines, airports)
    return flights, airlines, airports