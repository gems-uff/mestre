import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec

def get_figure_size(number_graphs, graphs_per_line):
    N = number_graphs
    cols = graphs_per_line
    rows = int(math.ceil(N / cols))

    length_x_axis = 30
    length_y_axis = 15

    fig_height = 12.

    height = length_y_axis * rows
    width = length_x_axis  * cols

    plot_aspect_ratio= float(width)/float(height)
    return (fig_height  * plot_aspect_ratio, fig_height )

#TODO: remove counts for values with zero elements
def count_plot_categorical(columns, df, grid_plots_per_line=2, log=False):
    sns.set(font_scale=1)
    N = len(columns)
    cols = grid_plots_per_line
    rows = int(math.ceil(N / cols))
    fig = plt.figure(figsize=get_figure_size(len(columns), grid_plots_per_line))
    gs = gridspec.GridSpec(rows, cols)
    for n in range(N):
        column = columns[n]
        ax = fig.add_subplot(gs[n])
        ax.set_xticklabels(list(df[column].unique()), rotation=45)
        g = sns.countplot(df[column], color='gray', ax=ax)
        if log:
            g.set_yscale("log")
    fig.tight_layout()

def hist_plot(columns, df, number_columns=2):
    sns.set(font_scale=2)

    i = j = 0
    cols = number_columns
    rows = math.ceil(len(columns)/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30,30))
    for column in columns:
        df.hist(column=column, ax=ax[i][j], bins=25)
        if j == cols-1:
            j = 0
            i+=1
        else:
            j+=1

    plt.tight_layout()
    plt.show()

def hist_plot_by_category(columns, df, by):
    for column in columns:
        sns.set(font_scale=1)
        g = sns.FacetGrid(df, col=by)
        g.map(sns.histplot, column)
        g.fig.suptitle(column)
        g.fig.subplots_adjust(top=0.8)

def violin_by_category(columns, df, category, number_columns=2):
    sns.set(font_scale=2)
    cols = number_columns
    rows = math.ceil(len(columns)/cols)
    fig = plt.figure(figsize=get_figure_size(len(columns), number_columns))
    gs = fig.add_gridspec(rows, cols)
    i = j = 0
    for column in columns:
        ax = fig.add_subplot(gs[i, j])
        g = sns.violinplot(x=category, y=column, data=df)
        g.set_xticklabels(g.get_xticklabels(),rotation=45)
        if j == cols-1:
            j = 0
            i+=1
        else:
            j+=1
    plt.tight_layout()

def scatter_by_category(columns, df, category, number_columns=2):
    
    sns.set(font_scale=1.3)
    cols = number_columns
    rows = math.ceil(len(columns)/cols)
    fig = plt.figure(figsize=get_figure_size(len(columns), number_columns))
    gs = fig.add_gridspec(rows, cols)
    i = j = 0
    for column in columns:
        ax = fig.add_subplot(gs[i, j])
        g = df.plot.scatter(x=column, y=category, ax=ax)
        g.set_xticklabels(g.get_xticklabels(),rotation=45)
        if j == cols-1:
            j = 0
            i+=1
        else:
            j+=1
    plt.tight_layout()