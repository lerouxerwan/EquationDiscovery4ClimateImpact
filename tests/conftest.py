# conftest.py
import pytest




@pytest.fixture(autouse=True)
def patch_global(monkeypatch):
    monkeypatch.setattr("emulator.utils_emulator.Config.automatic_loading_and_saving", False)
    from emulator.emulator import Config
    assert Config.automatic_loading_and_saving is False