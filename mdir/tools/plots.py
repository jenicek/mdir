import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_curve(data, ax):
    if len(data) == 1:
        ax.plot(*zip(*data), marker='.')
        ax.set_xticks([data[0][0]])
    else:
        ax.plot(*zip(*data))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_distribution(histograms, ax):
    PERCENTILES = [0, 0.07, 0.16, 0.31, 0.5, 0.69, 0.84, 0.93, 1]
    GRID_COLOR = (0.7, 0.7, 0.7)
    CURVE1_COLOR = (1, 165/255, 115/255)
    CURVE2_COLOR = (226/255, 115/255, 70/255) # "bold"

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    if not histograms:
        return
    elif len(histograms) == 1:
        ax.set_xticks([histograms[0][0]])
        ax.set_xlim(histograms[0][0]-0.01, histograms[0][0]+0.01)
        histograms = [(histograms[0][0]-0.001,) + histograms[0][1:], (histograms[0][0]+0.001,) + histograms[0][1:]]

    nhists = len(histograms)

    xlabels = []
    percentiles = []
    for histogram in histograms:
        xlabels.append(histogram[0])
        perc = np.interp(np.array(PERCENTILES)*np.sum(histogram[2]), np.cumsum(histogram[2]), histogram[1])
        percentiles.append(perc)
    percentiles = list(zip(*percentiles))
    assert len(percentiles) % 2 == 1

    for i in range(len(percentiles) // 2):
        if i % 2 == 0:
            ax.plot(xlabels, percentiles[i], color=CURVE1_COLOR)
            ax.plot(xlabels, percentiles[-i-1], color=CURVE1_COLOR)
        else:
            color = CURVE1_COLOR if i <= 1 else CURVE2_COLOR
            ax.fill_between(xlabels, percentiles[i], percentiles[-i-1], color=color+(0.4,))
            ax.plot(xlabels, percentiles[i], color=CURVE2_COLOR, linewidth=1)
            ax.plot(xlabels, percentiles[-i-1], color=CURVE2_COLOR, linewidth=1)
    ax.plot(xlabels, percentiles[len(percentiles) // 2], color=CURVE2_COLOR)


def plot_histogram(histograms, ax):
    N_YTICKS = 3
    N_YLINES = 30
    TICK_COLOR = (0.7, 0.7, 0.7)
    CURVE_COLOR = (1, 1, 1, 0.4)
    cmap = lambda x: ((200 + 55*x) / 255, (65 + 100*x) / 255, (25 + 90*x) / 255)

    ax.yaxis.tick_right()
    ax.set_ylim(0, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color(TICK_COLOR)
    ax.xaxis.set_tick_params(width=0.5, colors=TICK_COLOR)
    ax.yaxis.set_tick_params(width=0.5, colors=TICK_COLOR)

    if not histograms:
        ax.set_xticks([])
        ax.set_yticks([])
        return

    nhists = len(histograms)
    max_y = max(np.max(x[2]) for x in histograms)

    ticks = []
    for i, histogram in enumerate(histograms):
        offset = 0 if nhists == 1 else (nhists-i-1)/(nhists-1)
        ticks.append((offset, histogram[0]))
        ax.fill_between(histogram[1], offset, histogram[2]/max_y + offset, facecolor=cmap(i/nhists), edgecolor=CURVE_COLOR, zorder=2)

    if len(ticks) > 1:
        n_ylines = nhists // max(nhists // N_YLINES, 1)
        ticks = sorted(ticks[int(i / (n_ylines-1) * (len(ticks)-1))] for i in range(n_ylines))
    for offset, _ in ticks:
        ax.axhline(offset, color=TICK_COLOR, linewidth=0.5, zorder=1)

    if len(ticks) > 1:
        n_yticks = len(ticks) if len(ticks) < 2*N_YTICKS-1 else N_YTICKS
        ticks = sorted(ticks[int(i / (n_yticks-1) * (len(ticks)-1))] for i in range(n_yticks))
    ax.set_yticks(list(zip(*ticks))[0])
    ax.set_yticklabels(list(zip(*ticks))[1])


if __name__ == "__main__":
    x = np.arange(0.0, np.pi, 0.01)
    fig, axes = plt.subplots(6, 9)
    axes = [x for y in axes for x in y]
    acc = [(i, x+i/20, np.clip(np.sin(x) + np.cumsum((np.random.rand(x.size)-0.5) / 20), 0, 1)) for i in range(200)]
    print(acc)
    for N, ax in enumerate(axes):
        plot_histogram(acc[:N*2], ax)
        # plot_distribution(acc[:N*2], ax)

    plt.show()
