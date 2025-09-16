# conftest.py
import pytest




@pytest.fixture(autouse=True)
def patch_global(monkeypatch):
    monkeypatch.setattr("emulator.emulator.AUTOMATIC_LOADING_AND_SAVING", False)
    from emulator.emulator import AUTOMATIC_LOADING_AND_SAVING
    assert AUTOMATIC_LOADING_AND_SAVING is False