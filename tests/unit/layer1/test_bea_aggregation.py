from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fetch_bea_io import COVERAGE_FLOOR, aggregate_use_table, main


def _mini_table() -> pd.DataFrame:
    # 4 industries. I3 is split 50/50 across ENERGY and MATERIALS.
    codes = ["I1", "I2", "I3", "I4"]
    z = np.array(
        [
            [10.0, 4.0, 6.0, 2.0],
            [3.0, 8.0, 5.0, 1.0],
            [2.0, 2.0, 4.0, 2.0],
            [1.0, 1.0, 1.0, 3.0],
        ]
    )
    df = pd.DataFrame(z, index=codes, columns=codes)
    df.loc["Gross Output"] = [40.0, 30.0, 20.0, 10.0]
    df.attrs["concordance"] = {"I1": "ENERGY", "I2": "MATERIALS", "I4": "AUTOS"}
    df.attrs["splits"] = {"I3": {"ENERGY": 0.5, "MATERIALS": 0.5}}
    return df


def test_split_distributes_rows_and_columns() -> None:
    df = _mini_table()
    z, go, coverage = aggregate_use_table(
        df,
        df.attrs["concordance"],
        df.attrs["splits"],
        codes=("ENERGY", "MATERIALS", "AUTOS"),
    )
    # I3 column 20 of GO split 10/10; I1=40 ENERGY, I2=30 MATERIALS, I4=10 AUTOS
    assert go[0] == pytest.approx(40.0 + 10.0)
    assert go[1] == pytest.approx(30.0 + 10.0)
    assert go[2] == pytest.approx(10.0)
    # I3→I3 cell 4 splits into four 1.0 blocks (0.5*0.5*4)
    # ENERGY←I3 row contribution to ENERGY col: 0.5 * 0.5 * 4 = 1
    assert coverage == pytest.approx(1.0)
    assert z.shape == (3, 3)
    # ENERGY sells to ENERGY: I1-I1 (10) + I1-I3*0.5 (3) + I3-I1*0.5 (1) + I3-I3*0.25 (1)
    assert z[0, 0] == pytest.approx(10.0 + 3.0 + 1.0 + 1.0)


def test_coverage_below_floor_exits(tmp_path: Path, monkeypatch) -> None:
    df = _mini_table()
    # drop most of the concordance so coverage collapses
    df.attrs["concordance"] = {"I4": "AUTOS"}
    df.attrs["splits"] = {}
    path = tmp_path / "use.xlsx"
    # write values only; attrs are passed in-memory via monkeypatch
    df.to_excel(path)
    monkeypatch.setattr("fetch_bea_io._load_xlsx", lambda _p: df)
    rc = main(["--xlsx", str(path), "--coverage-floor", str(COVERAGE_FLOOR)])
    assert rc == 1


def test_missing_workbook_is_friendly() -> None:
    rc = main([])
    assert rc == 2
    rc = main(["--xlsx", "/no/such/use.xlsx"])
    assert rc == 2
