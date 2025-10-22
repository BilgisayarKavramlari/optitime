"""Reference time series datasets bundled with OptiProphet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from importlib import resources


@dataclass(frozen=True)
class Dataset:
    """Metadata describing a built-in dataset."""

    filename: str
    description: str
    frequency: str
    start: str
    end: str


_DATASETS: Dict[str, Dataset] = {
    "air_passengers": Dataset(
        filename="air_passengers.csv",
        description="Monthly totals of international airline passengers (1949-1960).",
        frequency="MS",
        start="1949-01-01",
        end="1960-12-01",
    ),
    "shampoo_sales": Dataset(
        filename="shampoo_sales.csv",
        description="Monthly shampoo sales in millions of units (1901-1903).",
        frequency="MS",
        start="1901-01-01",
        end="1903-12-01",
    ),
    "us_acc_deaths": Dataset(
        filename="us_acc_deaths.csv",
        description="Monthly accidental deaths in the United States (1973-1978).",
        frequency="MS",
        start="1973-01-01",
        end="1978-12-01",
    ),
}


def available_datasets() -> List[str]:
    """Return the list of dataset identifiers bundled with the library."""

    return sorted(_DATASETS.keys())


def dataset_info(name: str) -> Dataset:
    """Return metadata for a dataset."""

    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available options: {available}.")
    return _DATASETS[name]


def load_dataset(name: str) -> pd.DataFrame:
    """Load a bundled dataset as a pandas DataFrame."""

    info = _DATASETS.get(name)
    if info is None:
        available = ", ".join(sorted(_DATASETS.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available options: {available}.")

    package = __name__
    path = resources.files(package).joinpath(info.filename)
    with path.open("rb") as fh:
        df = pd.read_csv(fh, parse_dates=["ds"])
    return df.sort_values("ds").reset_index(drop=True)


__all__ = ["available_datasets", "dataset_info", "load_dataset", "Dataset"]
