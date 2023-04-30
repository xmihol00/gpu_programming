#!/usr/bin/python
# Authors: Martin Wistauder <mwistauder@student.tugraz.at>
# Date: 01.03.2022
# Version: 1.0

import struct, sys
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Constants
usage = "Usage: signal_viewer.py {path to run}"
example = "Example: signal_viewer.py ./generated/output/CPUParallelMatrix8/run_0/"

# ------------------------------------------------------------
# Functions

def load_signal(filename):
    signal = []
    with open(filename, 'rb') as fin:
        while True:
            chunk = fin.read(4)
            if not chunk:
                break
            signal.append(struct.unpack('f', chunk)[0])
    return signal

def show_signal(signals):
    def next_linestyle(state):
            if state:
                return "solid"
            else:
                return "dashed"

    state = True
    for data in signals:
        plt.plot(data, linestyle=next_linestyle(state))
        plt.title("Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        state = not state
    plt.show()

# ------------------------------------------------------------
# Main

if __name__ == "__main__":
    print(usage)
    print(example)

    # ------------------------------------------------------------
    # Arg handling
    target_run = "./tc/output/GPU/"
    target_signal = "0"
    if len(sys.argv) >= 2:
        target_run = sys.argv[1]
    else:
        print("Applying default parameters.")
    if len(sys.argv) >= 3:
        target_signal = sys.argv[2]

    print("Plotting run:", target_run)

    # ------------------------------------------------------------
    # Plot signals
    show_signal([load_signal(target_run + "/input/" + target_signal + ".signal"),
                 load_signal(target_run + "/" + target_signal + ".signal")])
