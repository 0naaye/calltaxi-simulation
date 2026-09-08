"""pytest 공통 설정 — src/ 를 import 경로에 넣는다."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: 원본 대용량 CSV를 읽는 테스트(첫 실행 1~2분)")


@pytest.fixture(scope="session")
def data_dir():
    import load
    if not load.DATA_DIR.exists():
        pytest.skip(f"데이터 디렉터리 없음: {load.DATA_DIR}")
    return load.DATA_DIR
