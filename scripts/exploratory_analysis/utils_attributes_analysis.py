import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec

def get_figure_size(number_graphs, graphs_per_line):
    N = number_graphs
    cols = graphs_per_line
    rows = int(math.ceil(N / cols))

    length_x_axis = 32
    length_y_axis = 28

    fig_height = 20.

    height = length_y_axis * rows
    width = length_x_axis  * cols

    plot_aspect_ratio= float(width)/float(height)
    return (fig_height  * plot_aspect_ratio, fig_height )

def count_plot_categorical(columns, df, grid_plots_per_line=2, log=False, log_base=10):
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
        plt.xticks(
            rotation=45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='medium'  
        )
        labels = list(df[column].value_counts().reset_index(name="count").query("count > 0")["index"])
        count = df[column].value_counts().reset_index(name="count").query("count > 0")
        chart = count.plot(kind='bar', color='gray', legend=None, ax=ax)
        chart.set_xticklabels(labels, rotation=45, horizontalalignment='right')
        ax.set_title(column)
        ax.set_ylabel(f'Count')
        if log:
            plt.yscale("log", base=log_base)
    fig.tight_layout()

def hist_plot(columns, df, number_columns=2, log=False, log_base=10):
    sns.set(font_scale=2)

    i = j = 0
    cols = number_columns
    rows = math.ceil(len(columns)/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30,30))
    for column in columns:
        df.hist(column=column, ax=ax[i][j], bins=25)
        ax[i][j].set_ylabel(f'Count')
        if log:
            ax[i][j].set_yscale('log', basey=log_base)
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
    sns.set(font_scale=1.2)
    cols = number_columns
    rows = math.ceil(len(columns)/cols)
    fig = plt.figure(figsize=get_figure_size(len(columns), number_columns))
    gs = fig.add_gridspec(rows, cols)
    i = j = 0

    new_df = df.copy()
    new_df[category] = "Overall"
    new_df = pd.concat([df,new_df])

    for column in columns:
        ax = fig.add_subplot(gs[i, j])
        g = sns.violinplot(x=category, y=column, data=new_df)
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
        if j == cols-1:
            j = 0
            i+=1
        else:
            j+=1
    plt.tight_layout()