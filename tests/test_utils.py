from pytorch_rl.utils import RunningStats, RingBuffer
import pytest
import math

def test_running_stats():
    rs = RunningStats()
    rs.push(0)
    rs.push(1)
    rs.push(5)

    assert rs.mean == 2.
    assert pytest.approx(rs.variance) == 7.
    assert pytest.approx(rs.standard_deviation) == math.sqrt(7)

def test_ring_buffer():
    rb = RingBuffer(100)
    for i in range(200):
        rb.add(i)
        assert rb.last() == i, 'Ringbuffer last is incorrect'
    assert rb.max() == 199, 'Ringbuffer max is incorrect'
    assert rb.min() == 100, 'Ringbuffer min is incorrect'
    assert rb.mean() == 149.5, 'Ringbuffer mean is incorrect'
    assert rb.last() == 199, 'Ringbuffer last is incorrect'
    rb.add(5)
    assert rb.last() == 5, 'Ringbuffer last is incorrect'
