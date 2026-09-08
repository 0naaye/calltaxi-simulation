"""
test_smoke.py — 코드 점검 기본 테스트

코드가 논리대로 도는지 확인하는 극단 케이스.
실제 테스트는 코딩 진행하며 채운다.

TODO: src/idle.py 테스트가 없다. load / travel_time / metrics 는 전용 테스트
      파일이 있지만 idle 은 아직 없다. 최소한 아래는 고정할 것.
        - build_gaps: 차량별 마지막 운행은 구간을 만들지 않는다(n = 운행 − 차량 수)
        - build_gaps: 다음 '배차'를 잇는다 — 다음 '승차'가 아니다
        - hourly_concurrency: 자정을 넘는 구간이 두 시간 칸으로 갈린다
        - 음수 구간(다음 배차 < 직전 하차)이 분포에서 빠진다
"""
import pytest


@pytest.mark.skip(reason="구현 후 활성화")
def test_travel_time_non_negative():
    """이동시간은 음수가 될 수 없다."""
    pass


@pytest.mark.skip(reason="구현 후 활성화")
def test_zero_calls():
    """콜 0건이어도 시뮬이 터지지 않는다."""
    pass


@pytest.mark.skip(reason="구현 후 활성화")
def test_single_vehicle():
    """차량 1대여도 정상 동작한다."""
    pass
