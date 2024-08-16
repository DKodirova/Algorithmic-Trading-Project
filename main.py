import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import sa_pairs_trading
import lstm
import warnings
warnings.filterwarnings('ignore')


def measure_memory_usage(func, n_values):
    memory_usage_list = []
    for n in n_values:
        mem_usage = memory_usage((func, (n,)))
        max_mem = max(mem_usage) - min(mem_usage)
        memory_usage_list.append(max_mem)
    return memory_usage_list

def time_intensive_function(func, n):
    start_time = time.time()
    func(n)
    end_time = time.time()
    return end_time - start_time

def measure_time_complexity(func, n_values):
    time_usage_list = []
    for n in n_values:
        elapsed_time = time_intensive_function(func, n)
        time_usage_list.append(elapsed_time)
    return time_usage_list

def plot_and_fit(n_values, memory_usage, time_usage, graph_name):
    n_values = np.array(n_values)
    memory_usage = np.array(memory_usage)
    time_usage = np.array(time_usage)

    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_values, memory_usage, 'o', label='Measured Memory Usage')
    
    plt.title("Memory Usage Analysis")
    plt.xlabel("Input Size (n)")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_values, time_usage, 'o', label='Measured Time Usage')
    
    plt.title("Time Usage Analysis")
    plt.xlabel("Input Size (n)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout() 
    plt.savefig(graph_name, dpi=300)
    return

def main():
    
    n_values = list(range(10, 510, 10))
    
    print("measuring memory usage for strat1")
    memory_usage_list = measure_memory_usage(sa_pairs_trading.strategy1, n_values)
    
    print("measuring time usage for strat1")
    time_usage_list = measure_time_complexity(sa_pairs_trading.strategy1, n_values)
    
    plot_and_fit(n_values, memory_usage_list, time_usage_list, "images/complexity_strat1")
        
    n_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.5, 8, 8.5, 9, 9.5, 10]

    print("measuring memory usage for strat2")
    memory_usage_list  = measure_memory_usage(lstm.strategy2, n_values)
    
    print("measuring time usage for strat2")
    time_usage_list = measure_time_complexity(lstm.strategy2, n_values)
    
    plot_and_fit(n_values, memory_usage_list, time_usage_list, "images/complexity_strat2")

if __name__ == "__main__":
    main()
