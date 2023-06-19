#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib.patches import Ellipse
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import statannot
import sys
import utils

def setupplot():
    #matplotlib.use('Agg')
    linewidth = 0.25
    matplotlib.rcParams['grid.color'] = 'lightgray'
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["lines.linewidth"] = linewidth
    matplotlib.rcParams["figure.figsize"] = (4, 4)
    matplotlib.rcParams["axes.linewidth"] = linewidth
    matplotlib.rcParams['axes.facecolor'] = 'none'
    matplotlib.rcParams['xtick.major.width'] = linewidth
    matplotlib.rcParams['ytick.major.width'] = linewidth
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['axes.axisbelow'] = True

def clustermap(df, sig=None, figsize=(4,5), **kwargs):
    g = sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        yticklabels=True,
        xticklabels=True,
        **kwargs,
    )
    if not sig is None:
        for i, ix in enumerate(g.dendrogram_row.reordered_ind):
            for j, jx in enumerate(g.dendrogram_col.reordered_ind):
                text = g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    "*" if sig.iloc[ix, jx] else "",
                    ha="center",
                    va="center",
                    color="black",
                )
                text.set_fontsize(8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right")
    return g

def heatmap(df, sig=None, ax=None):
    if ax is None: fig, ax= plt.subplots()
    pd.set_option("use_inf_as_na", True)
    if not sig is None:
        df = df[(sig < 0.05).sum(axis=1) > 0]
        sig = sig.loc[df.index]
    g = sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
    )
    for tick in g.get_yticklabels(): tick.set_rotation(0)
    if not sig is None:
        annot=pd.DataFrame(index=sig.index, columns=sig.columns)
        annot[(sig < 0.0005) & (df > 0)] = '+++'
        annot[(sig < 0.005) & (df > 0)] = '++'
        annot[(sig < 0.05) & (df > 0)] = '+'
        annot[(sig < 0.0005) & (df < 0)] = '---'
        annot[(sig < 0.005) & (df < 0)] = '--'
        annot[(sig < 0.05) & (df < 0)] = '-'
        annot[sig >= 0.05] = ''
        for i, ix in enumerate(df.index):
            for j, jx in enumerate(df.columns):
                text = g.text(
                    j + 0.5,
                    i + 0.5,
                    annot.values[i,j],
                    ha="center",
                    va="center",
                    color="black",
                )
                text.set_fontsize(8)
    plt.setp(g.get_xticklabels(), rotation=40, ha="right")
    return g

def pointheatmap(df, ax=None, size_scale=300):
    df.columns.name='x'
    df.index.name='y'
    vals = df.unstack()
    vals.name='size'
    fvals = vals.to_frame().reset_index()
    x, y, size= fvals.x, fvals.y, fvals['size']
    if ax is None: fig, ax= plt.subplots()
    x_labels = x.unique()
    y_labels = y.unique()
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * size_scale,
    )
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    plt.grid()
    return ax

def spindleplot(df, x='PC1', y='PC2', ax=None, palette=None):
    if palette is None: palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    if ax is None: fig, ax= plt.subplots()
    centers = df.groupby(df.index).mean()
    centers.columns=['nPC1','nPC2']
    j = df.join(centers)
    j['colours'] = palette
    i = j.reset_index().index[0]
    for i in j.reset_index().index:
        ax.plot(
            j[['PC1','nPC1']].iloc[i],
            j[['PC2','nPC2']].iloc[i],
            linewidth = 1,
            color = j['colours'].iloc[i],
            zorder=1,
            alpha=0.3
        )
        ax.scatter(j.PC1.iloc[i], j.PC2.iloc[i], color = j['colours'].iloc[i], s=3)
    for i in centers.index:
        ax.text(centers.loc[i,'nPC1']+0.01,centers.loc[i,'nPC2'], s=i, zorder=3)
    ax.scatter(centers.nPC1, centers.nPC2, c='black', zorder=2, s=20, marker='+')
    return ax

def polar(df):
    palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    ndf = df.loc[~df.index.str.contains('36'), df.columns.str.contains('Raw')].groupby(level=0).mean()
    data = ndf.T.copy().to_numpy()
    angles = np.linspace(0, 2*np.pi, len(ndf.columns), endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = ndf.columns.to_list()
    loopcategories = ndf.columns.to_list()
    loopcategories.append(df.columns[0])
    alldf = pd.DataFrame(data=data, index = loopcategories, columns=ndf.index).T
    allangles = pd.Series(data=angles, index=loopcategories)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for color in alldf.index.unique():
        plotdf = alldf.loc[alldf.index==color]
        ax.plot(allangles, plotdf.T, linewidth=1, color = palette[color])
    plt.title('Radial Line Graph')
    ax.set_xticks(allangles[:-1])
    ax.set_xticklabels(categories)
    ax.grid(True)

def abund(df, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    df.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def bar(*args, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    df = args[0].copy()
    if df.columns.shape[0] > 20:
        df['other'] = df[df.mean().sort_values().iloc[21:].index].sum(axis=1)
    df = df[df.median().sort_values(ascending=False).head(20).index]
    mdf = df.melt()
    kwargs['ax'] = sns.boxplot(data=mdf, x=mdf.columns[0], y='value', showfliers=False, boxprops=dict(alpha=.25))
    sns.stripplot(data=mdf, x=mdf.columns[0], y='value', size=2, color=".3", ax=kwargs['ax'])
    kwargs['ax'].set_xlabel(mdf.columns[0])
    kwargs['ax'].set_ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def box(**kwargs):
    try:
        stats = kwargs['stats']
        del kwargs['stats']
        stats = stats.loc[stats]
        if stats.sum() > 0:
            stats.loc[stats] = 0.05
    except:
        pass
    try: ax = kwargs['ax']
    except: fig, ax = plt.subplots(figsize=(4, 4))
    try:
        s = kwargs['s']
        del kwargs['s']
    except:
        pass
    sns.boxplot(showfliers=False, showcaps=False, **kwargs)
    try: del kwargs['palette']
    except: pass
    try:
        if kwargs['hue']:
            kwargs['dodge'] = True
    except: pass
    sns.stripplot(s=2, color="0.2", **kwargs)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    try:
        statannot.add_stat_annotation(
            ax,
            data=kwargs['data'],
            x=kwargs['x'],
            y=kwargs['y'],
            box_pairs=stats.index,
            perform_stat_test=False,
            pvalues=stats,
            text_format='star',
            verbose=0,
        )
    except: pass
    return ax

def volcano(lfc, pval, fcthresh=1, pvalthresh=0.05, annot=False, ax=None):
    if not ax: fig, ax= plt.subplots()
    lpval = pval.apply(np.log10).mul(-1)
    ax.scatter(lfc, lpval, c='black', s=0.5)
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(-1 * np.log10(pvalthresh), color='red', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(fcthresh, color='red', linestyle='-')
    ax.axvline(-fcthresh, color='red', linestyle='-')
    ax.set_ylabel('-log10 p-value')
    ax.set_xlabel('log2 fold change')
    ax.set_ylim(ymin=-0.1)
    x_max = np.abs(ax.get_xlim()).max()
    ax.set_xlim(xmin=-x_max, xmax=x_max)
    sigspecies = lfc.abs().gt(fcthresh) & lpval.abs().gt(-1 * np.log10(pvalthresh))
    sig = pd.concat([lfc.loc[sigspecies], lpval.loc[sigspecies]], axis=1) 
    sig.columns=['x','y']
    if annot: [ax.text(sig.loc[i,'x'], sig.loc[i,'y'], s=i) for i in sig.index]
    return ax

def aucroc(model, X, y, ax=None, colour=None):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    if ax is None: fig, ax= plt.subplots(figsize=(4, 4))
    y_score = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    AUC = auc(fpr, tpr)
    if colour is None:
        ax.plot(fpr, tpr, label=r"AUCROC = %0.2f" % AUC )
    else:
        ax.plot(fpr, tpr, color=colour, label=r"AUCROC = %0.2f" % AUC )
    ax.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    return ax

def newcurve(df, mapping, ax=None):
    from scipy import stats
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None: fig, ax= plt.subplots(figsize=(4, 4))
    df = df.apply(np.log1p).T
    df.loc['other'] = df[~df.index.isin(mapping.index)].sum()
    df = df.drop(df[~df.index.isin(mapping.index)].index).T
    grid = np.linspace(df.index.min(), df.index.max(), num=500)
    df1poly = df.apply(lambda x: np.polyfit(x.index, x.values, 3), axis=0).T
    df1polyfit = df1poly.apply(lambda x: np.poly1d(x)(grid), axis=1)
    griddf = pd.DataFrame( np.row_stack(df1polyfit), index=df1polyfit.index, columns=grid)
    griddf.clip(lower=0, inplace=True)
    if griddf.shape[0] > 20:
        griddf.loc["other"] = griddf.loc[
            griddf.T.sum().sort_values(ascending=False).iloc[21:].index
        ].sum()
    griddf = griddf.loc[griddf.T.sum().sort_values().tail(20).index].T
    griddf.sort_index(axis=1).plot.area(stacked=True, color=mapping.to_dict(), ax=ax)
    plt.tick_params(bottom = False)
    #plt.xlim(0, 35)
    #plt.legend( title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small")
    #plt.ylabel("Log(Relative abundance)")
    #plotdf = df.cumsum(axis=1).stack().reset_index()
    #sns.scatterplot(data=plotdf.sort_values(0), x='Days after birth', y=0, s=10, linewidth=0, ax=ax)
    return ax


def dendrogram(df):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt
    distance_matrix = np.array([[0, 1, 2, 3],
                                [1, 0, 4, 5],
                                [2, 4, 0, 6],
                                [3, 5, 6, 0]])
    Z = linkage(distance_matrix, method='average')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.title('Dendrogram')
    return ax

def upset(dfdict):
    from upsetplot import UpSet, from_contents
    intersections = from_contents(dfdict) 
    upset = UpSet(intersections)
    upset.plot()

def changeplot(subject, fcthresh=1, pvalthresh=0.0000000005):
    import pandas as pd
    changes = pd.read_csv(f'../results/{subject}changes.csv', index_col=0)
    data = pd.read_csv(f'../results/{subject}.csv', index_col=[0,1])
    f.setupplot()
    increase = changes.loc[
            (changes['MWW_q-value'].lt(pvalthresh)) & (changes[changes.columns[changes.columns.str.contains('Log2')]].gt(0).iloc[:,0])
            , 'MWW_q-value'].index
    order = data[increase].groupby(level=1).median().iloc[0].sort_values(ascending=False).index
    plotdf = data[order].stack().to_frame('Value').reset_index()
    fig, ax = plt.subplots(figsize=(len(increase),2))
    f.box(data=plotdf, x='level_2', y='Value', hue='ARM', ax=ax)
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
    plt.savefig(f'../results/{subject}decreasechangeplot.svg')
    plt.show()
    decrease = changes.loc[
            (changes['MWW_q-value'].lt(pvalthresh)) & (changes[changes.columns[changes.columns.str.contains('Log2')]].lt(0).iloc[:,0])
            , 'MWW_q-value'].index
    order = data[decrease].groupby(level=1).median().iloc[0].sort_values(ascending=False).index
    plotdf = data[order].stack().to_frame('Value').reset_index()
    fig, ax = plt.subplots(figsize=(len(decrease),2))
    f.box(data=plotdf, x='level_2', y='Value', hue='ARM', ax=ax)
    [ax.axvline(x+.5,color='k') for x in ax.get_xticks()]
    plt.savefig(f'../results/{subject}decreasechangeplot.svg')
    plt.show()

def networkplot(G, group=None):
    import matplotlib.pyplot as plt
    from itertools import count
    import networkx as nx
    if group:
        nx.set_node_attributes(G, group, "group")
        groups = set(nx.get_node_attributes(G, 'group').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = G.nodes()
        colors = [mapping[G.nodes[n]['group']] for n in nodes]
    pos = nx.spring_layout(G)
    #pos= nx.spring_layout(G, with_labels=True, node_size=50)
    ax = nx.draw_networkx_edges(G, pos, alpha=0.2)
    #nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
    ax = nx.draw_networkx_nodes(G, pos, node_color=group, node_size=20, cmap=plt.cm.jet)
    #nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    plt.colorbar(ax)
    return ax

def venn(df1, df2, df3):
    from matplotlib_venn import venn3
    from itertools import combinations
    def combs(x): return [c for i in range(1, len(x)+1) for c in combinations(x,i)]
    comb = combs([df1, df2, df3])
    result = []
    for i in comb:
        if len(i) > 1:
            result.append(len(set.intersection(*(set(j.columns) for j in i))))
        else:
            result.append(len(i[0].columns))
    venn3(subsets = result)

if __name__ == '__main__':
    setupplot()
    parser = argparse.ArgumentParser(description='Plot - Produces a plot of a given dataset')
    parser.add_argument('subject')
    parser.add_argument('-m', '--mult')
    parser.add_argument('-p', '--perm')
    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    output = change(**args)
    print(*output)
