import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    def __init__(self):
        pass

    def plotMovingAverage(self, values, window):
        values_moving_average = np.convolve(np.array(values), np.ones(window), mode="valid")/window
        self.plot(values_moving_average)

    def plot(self, values):
        plt.plot(range(len(values)), values)
    
    def show(self):
        plt.show()