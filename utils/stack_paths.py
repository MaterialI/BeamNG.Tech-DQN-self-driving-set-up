import math


class Checkpointpath:
    def __init__(self, checkpoint1):
        self.stack = [checkpoint1]  # lon lat time in each checkpoint
        self.stack_len = 1

    def push(self, checkpoint):
        self.stack.append(checkpoint)
        self.stack_len += 1

    def pop_el(self):
        element = None
        if self.stack_len > 0:
            element = self.stack.pop()
            self.stack_len -= 1
        return element

    def get(self, longitude, latitude, time):
        if self.stack_len == 0:
            return None, False
        else:

            dist = math.sqrt(
                (self.stack[-1][0] - longitude) ** 2
                + (self.stack[-1][1] - latitude) ** 2
            )
            print(dist)
            if dist <= 10:
                return self.pop_el(), True
            else:
                return self.stack[-1], False
