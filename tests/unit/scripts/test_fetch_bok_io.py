from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fetch_bok_io import (
    COVERAGE_FLOOR,
    DEFAULT_CONCORDANCE,
    HOWTO_URL_BOK,
    aggregate_bok_use_table,
    load_bok_concordance,
    main,
)
from marketsim.layer1.build_io import CODES

ROOT = Path(__file__).resolve().parents[3]


def test_concordance_rows_sum_to_one() -> None:
    doc = yaml.safe_load(DEFAULT_CONCORDANCE.read_text())
    assert doc["codes_ref"] == "marketsim.layer1.build_io.CODES"
    assert isinstance(doc["sources"], list)
    assert len(doc["sources"]) >= 18
    for entry in doc["sources"]:
        weights = entry["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-9, entry["id"]
        unknown = set(weights) - set(CODES)
        assert not unknown, f"{entry['id']} -> {unknown}"


def test_load_concordance_uses_codes_not_a_second_list() -> None:
    concordance, splits = load_bok_concordance(DEFAULT_CONCORDANCE, classification="bok33")
    assert concordance["농림수산품"] == "AGRIFOOD"
    assert splits["컴퓨터, 전자 및 광학기기"]["SEMIS"] == pytest.approx(0.55)
    # BOK-33 "10" is fabricated metal; KSIC "10" is food — namespace by classification.
    assert concordance["10"] == "MATERIALS"
    ksic_c, ksic_s = load_bok_concordance(DEFAULT_CONCORDANCE, classification="ksic")
    assert ksic_c["10"] == "AGRIFOOD"
    assert abs(sum(ksic_s["제조업"].values()) - 1.0) < 1e-9


def test_aggregate_bok_use_table_korean_names_and_split() -> None:
    codes = ["농림수산품", "컴퓨터, 전자 및 광학기기", "자동차"]
    z = np.array(
        [
            [4.0, 2.0, 1.0],
            [1.0, 6.0, 3.0],
            [0.0, 1.0, 5.0],
        ]
    )
    df = pd.DataFrame(z, index=codes, columns=codes)
    df.loc["총산출"] = [20.0, 40.0, 10.0]
    concordance, splits = load_bok_concordance(DEFAULT_CONCORDANCE, classification="bok33")
    z18, go, coverage = aggregate_bok_use_table(df, concordance, splits)
    assert z18.shape == (18, 18)
    assert coverage == pytest.approx(1.0)
    energy = CODES.index("AGRIFOOD")
    semis = CODES.index("SEMIS")
    autos = CODES.index("AUTOS")
    # 총산출 40 on electronics × SEMIS share 0.55; autos 10 stays on AUTOS.
    assert go[energy] == pytest.approx(20.0)
    assert go[semis] == pytest.approx(40.0 * 0.55)
    assert go[autos] == pytest.approx(10.0)
    # electronics→electronics 6 splits by 0.55×0.55 onto SEMIS×SEMIS
    assert z18[semis, semis] == pytest.approx(6.0 * 0.55 * 0.55)


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_no_xlsx_exits_2_and_prints_howto(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert HOWTO_URL_BOK in err
    assert "config/kr/README.md" in err
    assert main(["--xlsx", "/no/such/bok.xlsx"]) == 2


def test_script_does_not_touch_the_network() -> None:
    text = (ROOT / "scripts" / "fetch_bok_io.py").read_text()
    for needle in ("urllib", "requests", "httpx", "urlopen", "httplib"):
        assert needle not in text
    # Import + --help / no-xlsx paths must not open a socket.
    rc = main([])
    assert rc == 2


def test_coverage_below_floor_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codes = ["농림수산품", "미분류산업"]
    df = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=codes, columns=codes)
    df.loc["총산출"] = [5.0, 95.0]
    path = tmp_path / "bok.xlsx"
    df.to_excel(path)
    monkeypatch.setattr("fetch_bok_io._load_xlsx", lambda _p, sheet=None: df)
    rc = main(["--xlsx", str(path), "--coverage-floor", str(COVERAGE_FLOOR)])
    assert rc == 1
