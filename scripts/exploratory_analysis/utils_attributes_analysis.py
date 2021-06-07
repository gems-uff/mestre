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

# this version considers the discretization based on log 10 values (using strings)
# e.g. Tens, Hundreds, and so on.
def count_plot_categorical(columns, df, labels_order, grid_plots_per_line=2, log=False, log_base=10):
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
        count = df[column].value_counts().reset_index(name="count").query("count > 0")
        count['index'] = count['index'].astype("category")
        count['index'].cat.set_categories(labels_order, inplace=True)
        count = count.sort_values(['index'])
        chart = count.plot(kind='bar', color='gray', legend=None, ax=ax)
        labels = list(count["index"])
        chart.set_xticklabels(labels, rotation=45, horizontalalignment='right')
        ax.set_title(column)
        ax.set_ylabel(f'Count')
        if log:
            plt.yscale("log", base=log_base)
    fig.tight_layout()

# this version considers the discretization based on log2 values.
def count_plot_categorical_new(columns, df, grid_plots_per_line=2, log=False, log_base=10):
    sns.set(font_scale=1)
    N = len(columns)
    cols = grid_plots_per_line
    rows = int(math.ceil(N / cols))
    fig = plt.figure(figsize=get_figure_size(len(columns), grid_plots_per_line))
    gs = gridspec.GridSpec(rows, cols)
    df = df.copy()
    for n in range(N):
        column = columns[n]
        ax = fig.add_subplot(gs[n])
        count = df[column].value_counts().reset_index(name="count").query("count > 0")
        count = count.sort_values('index', ascending=True)
        count.loc[count['index']==-2, 'index'] = 'NA'
        count.loc[count['index']==-1, 'index'] = 'Zero'
        chart = count.plot(kind='bar', x='index', y='count', color='gray', legend=None, ax=ax)
        ax.set_title(column)
        ax.set_ylabel(f'Count')
        ax.set_xlabel(f'log2({column})')
        if log:
            ax.set_ylabel(f'Log{log_base}(Count)')
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

# about the density function value being greater than 1:
#   https://stats.stackexchange.com/questions/4220/can-a-probability-distribution-value-exceeding-1-be-ok
def density_by_category(columns, df, category, number_columns):
    sns.set(font_scale=1)
    i = j = 0
    cols = number_columns
    rows = math.ceil(len(columns)/cols)
    fig = plt.figure(figsize=get_figure_size(len(columns), number_columns))
    gs = fig.add_gridspec(rows, cols)
    g = df.groupby(category)
    for column in columns:
        ax = fig.add_subplot(gs[i, j])
        for group in g.groups.keys():
            group_values = g.get_group(group)
            sns.kdeplot(data=group_values[column], label=group, ax=ax, shade=True)
        if j == cols-1:
            j = 0
            i+=1
        else:
            j+=1
    ax = fig.get_axes()[0]
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)    
    plt.tight_layout()

def relative_percentage_by_group_plot(columns, df, group_name, log=False):
    sns.set_context("paper", font_scale=1.5)
    df = df.copy()
    for column in columns:
        x,y = column, group_name
        df1 = df.groupby(x, observed=True)[y].value_counts(normalize=True)
        df1 = df1.mul(100)
        df1 = df1.rename('percent').reset_index()

        if log:
            df1.loc[df1[column]==-2, column] = 'NA'
            df1.loc[df1[column]==-1, column] = 'Zero'

        g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, height=6, aspect = 15/6)
        g.ax.set_ylim(0,100)
        g.ax.set_title(f"Relative distribution per discretized level of {column} attribute")
        
        if not log:
            for p in g.ax.patches:
                txt = str(p.get_height().round(2)) + '%'
                txt_x = p.get_x() 
                txt_y = p.get_height()
                if not math.isnan(txt_y):
                    g.ax.text(txt_x,txt_y,txt)
        else:
            g.ax.set_xlabel(f"log2({column})")

        plt.tight_layout()
        plt.show()

def count_plot_by_category(columns, df, group_name, log=False):
    sns.set(font_scale=1)
    sns.set_context("paper", font_scale=1.5)
    df1 = df.copy()
    for column in columns:
        plt.figure(figsize=(10,6))
        
        if log:
            values = list(df1[column].unique())
            if -2 in values: values.remove(-2)
            if -1 in values: values.remove(-1)
            
            values = sorted(values)
            values.insert(0, 'NA')
            values.insert(1, 'Zero')
            df1.loc[df1[column]==-2, column] = 'NA'
            df1.loc[df1[column]==-1, column] = 'Zero'

            df1[column] = pd.Categorical(df1[column], 
                categories=values, ordered=True)
        
        g = sns.countplot(x=column, hue=group_name, data=df1)
        g.set_yscale("log")
        g.set_title(f"Count distribution per discretized level of {column} attribute")
        g.set_xticklabels(g.get_xticklabels(), rotation=40)
        plt.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left')
        
        if log:
            g.set_xlabel(f"log2({column})")

        plt.tight_layout()
        plt.show()