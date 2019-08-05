import random

class MemoryBuffer:
  def __init__(self, buffer_length):
    self.memories = [None] * buffer_length
    self.next_insert_location = 0
    self.current_length = 0
    self.max_length = buffer_length

  def add_memory(self, memory):
    self.memories[self.next_insert_location] = memory
    self.next_insert_location += 1
    self.next_insert_location %= self.max_length
    if self.current_length < self.max_length:
        self.current_length += 1
    
  def sample_minibatch(self, batch_size):
    indices = random.sample(range(self.current_length), batch_size)
    return [self.memories[index] for index in indices]