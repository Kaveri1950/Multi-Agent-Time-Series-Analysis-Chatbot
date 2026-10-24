"""Skip tests that need the cached price panel and fitted model when they are absent.

A fresh clone (and CI) has no data/cache/; those tests are skipped there,
with the reason, rather than failing on a missing file.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NEEDED = [ROOT / "data" / "cache" / "quant" / "price_panel.parquet",
          ROOT / "data" / "cache" / "models" / "volatility_ens_log.joblib"]


@pytest.fixture(autouse=True)
def _private_answer_log(tmp_path, monkeypatch):
    """Tests that call the chatbot must not write into the real audit log."""
    from audit import answer_log
    monkeypatch.setattr(answer_log, "DB_PATH", tmp_path / "audit.db")


def pytest_collection_modifyitems(config, items):
    missing = [p.name for p in NEEDED if not p.exists()]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"needs cached data ({', '.join(missing)}); "
                                   "run `python -m quant.data` and `python -m quant.production_model --fit`")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
