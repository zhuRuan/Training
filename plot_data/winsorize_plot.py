import matplotlib.pyplot as plt

def winsorize_plot(data, vertical_lines=[]):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    _df = ax.hist(data, 100, density=True)
    color_list = ['red', 'green', 'yellow', 'black', 'gold', 'gray']
    count = 0
    for value in vertical_lines:
        ax.bar(value, 0.1, width=0.05, color=color_list[count], alpha=1)
        count += 1
    plt.show()