import memory_buffer

def test_memory_buffer_append():
    mb = memory_buffer.MemoryBuffer(10)
    for i in range(100):
        mb.add_memory(i)

    assert mb.memories == list(range(90, 100))

def test_memory_buffer_sample():
    mb = memory_buffer.MemoryBuffer(10)
    for i in range(100):
        mb.add_memory(i)

    sample = mb.sample_minibatch(batch_size=5)

    assert len(sample) == 5