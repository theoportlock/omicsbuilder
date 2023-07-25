#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Make multibox for species abundance

from itertools import combinations
from itertools import count
import pickle
from matplotlib.patches import Ellipse
from matplotlib_venn import venn3
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from upsetplot import UpSet, from_contents
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statannot

def setupplot():
    #matplotlib.use('Agg')
    linewidth = 0.25
    matplotlib.rcParams['grid.color'] = 'lightgray'
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["lines.linewidth"] = linewidth
    matplotlib.rcParams["figure.figsize"] = (3, 3)
    matplotlib.rcParams["axes.linewidth"] = linewidth
    matplotlib.rcParams['axes.facecolor'] = 'none'
    matplotlib.rcParams['xtick.major.width'] = linewidth
    matplotlib.rcParams['ytick.major.width'] = linewidth
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['axes.axisbelow'] = True

def clustermap(subject, sig=None, figsize=(4,4), corr=False,  **kwargs):
    if not sig is None:
        sig = pd.read_csv(f'../results/{sig}.tsv', sep='\t', index_col=0)
    if corr:
        stackdf = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=[0,1])
        sig = stackdf.sig.unstack().fillna(0)
        df = stackdf.rho.unstack().fillna(0)
    else:
        df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
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

def heatmap(subject, sig=None, ax=None):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    if ax is None: fig, ax= plt.subplots()
    pd.set_option("use_inf_as_na", True)
    if not sig is None:
        sig = pd.read_csv(f'../results/{sig}.tsv', sep='\t', index_col=0)
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

def pointheatmap(subject, ax=None, size_scale=300):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
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

def spindle(subject, x='PC1', y='PC2', ax=None, palette=None):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
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

def polar(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
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
    return ax

def circos(subject, **kwargs):
    edges = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    varcol = kwargs.get('varcol')
    col = pd.read_csv(f'../results/{varcol}.csv', index_col=0).rename_axis('source')
    for i, element in col.groupby('datatype', sort=False):
        col.loc[element.index,'ID'] = element.reset_index().reset_index().set_index('source')['index'].astype(int)
    col['ID'] = col.ID.astype('int')
    data = edges.join(col[['ID','datatype']], how='inner').rename(columns={'datatype':'datatype1', 'ID':'index1'})
    data = data.reset_index().set_index('target').join(col[['ID','datatype']], how='inner').rename_axis('target').rename(columns={'datatype':'datatype2', 'ID':'index2'}).reset_index()
    # remove internal correlations
    data = data.loc[data.datatype1 != data.datatype2]
    Gcircle = pycircos.Gcircle
    Garc = pycircos.Garc
    circle = Gcircle()
    for i, row in col.groupby('datatype', sort=False):
        arc = Garc(arc_id=i,
                   size=row['ID'].max(),
                   interspace=30,
                   label_visible=True)
        circle.add_garc(arc)
    circle.set_garcs()
    for i, row in data.iterrows():
        circle.chord_plot(start_list=(row.datatype1, row.index1-1, row.index1, 500), end_list=(row.datatype2, row.index2-1, row.index2, 500),
        facecolor=plt.cm.get_cmap('coolwarm')(row.weight)) 
    return circle

def abund(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    df.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def bar(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
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

def hist(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    kwargs['col'] = 'sig' if not kwargs.get('col') else kwargs.get('col')
    kwargs['ax'] = sns.histplot(data=df[kwargs['col']])
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def scatter(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    #kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    sns.regplot(data=df, x=kwargs.get('x'), y=kwargs.get('y'), ax=kwargs.get('ax'))
    return kwargs['ax']

def box(subject, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t')
    if not kwargs.get('x'): kwargs['x'] = df.columns[0]
    if not kwargs.get('y'): kwargs['y'] = df.columns[1]
    if not kwargs.get('ax'): kwargs['ax'] = plt.subplots()[1]
    stats=None
    if kwargs.get('stats'):
        stats = pd.read_csv(f'../results/{kwargs.get("stats")}.tsv', sep='\t', index_col=0)
        del kwargs['stats']
        stats = stats.loc[kwargs.get('y'), 'sig']
        if stats.sum() > 0:
            stats.loc['sig'] = 0.05
    sns.boxplot(data=df, showfliers=False, showcaps=False, **kwargs)
    if kwargs.get('palette'): del kwargs['palette'] 
    if kwargs.get('hue'): kwargs['dodge'] = True
    sns.stripplot(data=df, s=2, color="0.2", **kwargs)
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=40, ha="right")
    if stats:
        statannot.add_stat_annotation(
            ax,
            data=df,
            x=kwargs['x'],
            y=kwargs['y'],
            box_pairs=stats.index,
            perform_stat_test=False,
            pvalues=stats,
            text_format='star',
            verbose=0,
        )
    return kwargs['ax']

def multibox(subject, **kwargs):
    # takes a filtered abundance table by 
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    if not kwargs.get('figsize'): kwargs['figsize'] = (6,5)
    fig, ax = plt.subplots(nrows=1, ncols=len(df.columns), figsize=kwargs.get('figsize'), sharey=True)
    i, j = 0, df.columns[0]
    for i, j in enumerate(df.columns):
        #stats = sig.loc[speciessig]
        boxkwargs = {
                'data':df[j].to_frame(),
                'x':df[j].index,
                'y':j,
                #'palette':kwargs.get(colours),
                'ax':ax[i]
                }
        sns.boxplot(showfliers=False, showcaps=False, **boxkwargs)
        sns.stripplot(s=2, color="0.2", **boxkwargs)
        plt.setp(ax[i].get_xticklabels(), rotation=40, ha="right")
    return ax

def volcano(subject, change='lfc', sig='sig', fc=1, pval=0.05, annot=False, ax=None, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    if not ax: fig, ax= plt.subplots()
    lfc = df[change]
    pvals = df[sig] 
    lpvals = pvals.apply(np.log10).mul(-1)
    ax.scatter(lfc, lpvals, c='black', s=0.5)
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(-1 * np.log10(pval), color='red', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(fc, color='red', linestyle='-')
    ax.axvline(-fc, color='red', linestyle='-')
    ax.set_ylabel('-log10 q-value')
    ax.set_xlabel('log2 fold change')
    ax.set_ylim(ymin=-0.1)
    x_max = np.abs(ax.get_xlim()).max()
    ax.set_xlim(xmin=-x_max, xmax=x_max)
    sigspecies = lfc.abs().gt(fc) & lpvals.abs().gt(-1 * np.log10(pval))
    sig = pd.concat([lfc.loc[sigspecies], lpvals.loc[sigspecies]], axis=1) 
    sig.columns=['x','y']
    if annot: [ax.text(sig.loc[i,'x'], sig.loc[i,'y'], s=i) for i in sig.index]
    return ax

def aucroc(subject, ax=None, colour=None, **kwargs):
    # Need to sort this so model is the pickle
    #subject, ax, colour = 'metabARMaucroc',None,None
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    AUC = auc(df.fpr, df.tpr)
    if ax is None: fig, ax= plt.subplots()
    if colour is None:
        ax.plot(df.fpr, df.tpr, label=r"AUCROC = %0.2f" % AUC )
    else:
        ax.plot(df.fpr, df.tpr, color=colour, label=r"AUCROC = %0.2f" % AUC )
    ax.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    return ax

def curve(subject, mapping=None, ax=None, **kwargs):
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    from scipy import stats
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None: fig, ax= plt.subplots()
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

def dendrogram(subject, **kwargs):
    # need to fix
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
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

def upset(subject, dfdict=None):
    # Need to fix
    df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    intersections = from_contents(dfdict) 
    upset = UpSet(intersections)
    upset.plot()
    return upset

def networkplot(subject, group=None):
    # Need to fix
    G = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
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

def lefsebar(subject, **kwargs):
    os.system(f'lefse_plot_res.py ../results/{subject}LEFSE_scores.txt ../results/{subject}LEFSE_features.pdf --format pdf')
    return None

def lefseclad(subject, **kwargs):
    os.system(f'lefse_plot_cladogram.py ../results/{subject}LEFSE_scores.txt ../results/{subject}LEFSE_clad.pdf --format pdf')
    return None

def lefsefeat(subject, **kwargs):
    os.system(f'mkdir ../results/{subject}_biomarkers_raw_images')
    os.system(f'lefse_plot_features.py ../results/{subject}LEFSE_format.txt ../results/{subject}LEFSE_scores.txt ../results/{subject}_biomarkers_raw_images/ --format svg')
    return None

def venn(subject, **kwargs):
    # Untested
    DF1 = pd.read_csv(f'../results/{df1}.tsv', sep='\t', index_col=0)
    DF2 = pd.read_csv(f'../results/{df2}.tsv', sep='\t', index_col=0)
    DF3 = pd.read_csv(f'../results/{df3}.tsv', sep='\t', index_col=0)
    def combs(x): return [c for i in range(1, len(x)+1) for c in combinations(x,i)]
    comb = combs([DF1, DF2, DF3])
    result = []
    for i in comb:
        if len(i) > 1:
            result.append(len(set.intersection(*(set(j.columns) for j in i))))
        else:
            result.append(len(i[0].columns))
    ax = venn3(subsets = result)
    return ax

def plot(subject, plottype, logx=False, logy=False, **kwargs):
    available={
        'clustermap':clustermap,
        'heatmap':heatmap,
        'pointheatmap':pointheatmap,
        'spindle':spindle,
        'polar':polar,
        'abund':abund,
        'bar':bar,
        'hist':hist,
        'box':box,
        'volcano':volcano,
        'aucroc':aucroc,
        'lefsebar':lefsebar,
        'lefseclad':lefseclad,
        'lefsefeat':lefsefeat,
        'curve':curve,
        'dendrogram':dendrogram,
        'scatter':scatter,
        'upset':upset,
        'networkplot':networkplot,
        'venn':venn
        }
    if not kwargs.get('figsize'): kwargs['figsize'] = (3,3)
    if not kwargs.get('ax'): kwargs['ax'] = plt.subplots(figsize=kwargs.get('figsize'))[1]
    kwargs.pop('figsize')
    ax = available.get(plottype)(subject, **kwargs)
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    plt.savefig(f'../results/{subject}{plottype}.svg')
    return ax

'''
def expvsobs
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkblue', marker='o')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray', linewidth=2)
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], color='red', alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression Model')
    plt.savefig('../results/expvsobs.svg')
    plt.show()
'''

if __name__ == '__main__':
    setupplot()
    parser = argparse.ArgumentParser(description='Plot - Produces a plot of a given dataset')
    parser.add_argument('plottype')
    parser.add_argument('subject')
    parser.add_argument('--logx', action='store_true')
    parser.add_argument('--logy', action='store_true')
    args, unknown = parser.parse_known_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    kwargs = eval(unknown[0]) if unknown != [] else {}
    output = plot(**args|kwargs)
    plt.show()
