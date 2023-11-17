from collections import deque, namedtuple
import random

class ReplayBuffer:
  def __init__(self,capacity):
    self.buffer = deque(maxlen = capacity) #erases the last item to add new item

  def __len__(self):
    return len(self.buffer) #length of the buffer

  def append(self, experience):
    self.buffer.append(experience) #insert data into the list

  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size) #randomly select data based on the size of the batch ex: batch_size = 3 gives 3 random samples
    