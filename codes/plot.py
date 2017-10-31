import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def show_xy(x, y, filename='plot.png'):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.plot(points)
    ax.plot(x, y, label='valid METEOR')
    fig.savefig(filename)
    plt.close(fig)


def compare_xy(x1, y1, x2, y2, filename='plot.png'):

    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x1, y1, 'b-', label='train loss')
    ax.plot(x2, y2, 'r-', label='valid loss')
    plt.legend()
    fig.savefig(filename)
    plt.close(fig)
