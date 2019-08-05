from pytorch_rl.utils import RunningStats
import pytest
import math

def test_running_stats():
    rs = RunningStats()
    rs.push(0)
    rs.push(1)
    rs.push(5)

    assert rs.mean == 2.
    assert pytest.approx(rs.variance) == 7.
    assert pytest.approx(rs.standard_deviation) == math.sqrt(70)