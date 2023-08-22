#!/usr/bin/env/ python

from itertools import combinations, count, permutations
from matplotlib_venn import venn3
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import distance
from scipy.stats import levene, mannwhitneyu, shapiro, spearmanr
from skbio import stats
from skbio.diversity.alpha import pielou_e, shannon
from skbio.stats.composition import ancom, clr
from skbio.stats.composition import multiplicative_replacement as mul
from skbio.stats.distance import permanova
from skbio.stats.distance import permanova
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, mean_absolute_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_som.som import SOM
from statsmodels.stats.multitest import fdrcorrection
from upsetplot import UpSet, from_contents
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import shap
import skbio
import statannot
import subprocess
import sys
import umap

# Load
def load(subject):
    return pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)

# Save
def save(df, subject):
    df.to_csv(f'../results/{subject}.tsv', sep='\t')

# Prediction
def classifier(df, **kwargs):
    model = RandomForestClassifier(n_jobs=-1, random_state=1, oob_score=True)
    X, y = df, df.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    performance = classification_report(y_true=y_test, y_pred=y_pred) + '\n' \
        'AUCROC=' + str(roc_auc_score(y_test, y_prob)) + '\n\n' +\
        pd.DataFrame(confusion_matrix(y_test, y_pred)).to_string() + '\n' +\
        'oob score=' + str(model.oob_score_)
    print(performance)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y_test[0])
    aucrocdata = pd.DataFrame(pd.concat([pd.Series(fpr), pd.Series(tpr)],axis=1)).set_axis(['fpr','tpr'], axis=1)
    return model, performance, aucrocdata

def regressor(df, **kwargs):
    model = RandomForestRegressor(n_jobs=-1, random_state=1, oob_score=True)
    X, y = df, df.index
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    performance = "Mean Absolute Error:" + str(mae) + '\n\n' + \
        "R-squared score:" + str(r2) + '\n' + \
        'oob score =' + str(model.oob_score_)
    print(performance)
    with open(f'../results/{subject}performance.txt', 'w') as of: of.write(performance)
    return model, performance

def networkpredict(df):
    i = df.columns[0]
    shaps = pd.DataFrame(index=df.columns)
    for i in df.columns:
        tdf = df.set_index(i)
        X,y = tdf, tdf.index
        #X_train, X_test, y_train, y_test =  train_test_split(X,y)
        model = RandomForestRegressor(random_state=1, n_jobs=-1)
        #model.fit(X_train, y_train)
        model.fit(X, y)
        shaps[i] = SHAP_reg(df, model)
        # careful running this as need to sort out pandas
    edges = to_edges(shaps, thresh=0.01)
    return edges

def to_edges(df, thresh=0.5, directional=True):
    import pandas as pd
    df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
    edges = df.stack().to_frame()[0]
    nedges = edges.reset_index()
    edges = nedges[nedges.target != nedges.source].set_index(['source','target']).drop_duplicates()[0]
    if directional:
        fin = edges.loc[(edges < 0.99) & (edges.abs() > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source').sort_values('weight')
    else:
        fin = edges.loc[(edges < 0.99) & (edges > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source').sort_values('weight')
    return fin

def predict(analysis, df, **kwargs):
    available={
        'regressor':regressor,
        'classifier':classifier,
        }
    output = available.get(analysis)(df, **kwargs)
    return output

# Plotting
def setupplot(agg=False):
    if agg: matplotlib.use('Agg')
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

def clustermap(df, sig=None, figsize=(4,4), corr=False, **kwargs):
    pd.set_option("use_inf_as_na", True)
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

def heatmap(df, sig=None, ax=None, **kwargs):
    pd.set_option("use_inf_as_na", True)
    if ax is None: fig, ax= plt.subplots()
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

def pointheatmap(df, ax=None, size_scale=300, **kwargs):
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

def spindle(df, x='PC1', y='PC2', ax=None, palette=None, **kwargs):
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

def polar(df, **kwargs):
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

def circos(df, **kwargs):
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

def abund(df, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    if df.columns.shape[0] > 20:
        df['others'] = df[df.mean().sort_values(ascending=False).iloc[21:].index].sum(axis=1)
    df = df.loc[:, df.mean().sort_values(ascending=False).iloc[:20].index]
    df.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def bar(df, **kwargs):
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

def hist(df, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    kwargs['col'] = 'sig' if not kwargs.get('col') else kwargs.get('col')
    kwargs['ax'] = sns.histplot(data=df[kwargs['col']])
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def scatter(df, **kwargs):
    #kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    sns.regplot(data=df, x=kwargs.get('x'), y=kwargs.get('y'), ax=kwargs.get('ax'))
    return kwargs['ax']

def box(df, **kwargs):
    df = df.reset_index()
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

def multibox(df, **kwargs):
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

def volcano(df, change='lfc', sig='sig', fc=1, pval=0.05, annot=False, ax=None, **kwargs):
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

def aucroc(df, ax=None, colour=None, **kwargs):
    #df, ax, colour = df,None,None
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

def curve(df, mapping=None, ax=None, **kwargs):
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
    return ax

def dendrogram(df, **kwargs):
    df = df.unstack()
    Z = linkage(df, method='average')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.title('Dendrogram')
    return ax

def upset(df, dfdict=None, **kwargs):
    intersections = from_contents(dfdict) 
    upset = UpSet(intersections, **kwargs)
    upset.plot()
    return upset

def networkplot(*args, group=None, **kwargs):
    G = nx.read_gpickle(f'../results/kwargs.get(subject).gpickle')
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

def lefsebar(*args, **kwargs):
    os.system(f'lefse_plot_res.py ../results/kwargs.get(subject)LEFSE_scores.txt ../results/kwargs.get(subject)LEFSE_features.pdf --format pdf')
    return None

def lefseclad(*args, **kwargs):
    os.system(f'lefse_plot_cladogram.py ../results/kwargs.get(subject)LEFSE_scores.txt ../results/kwargs.get(subject)LEFSE_clad.pdf --format pdf')
    return None

def lefsefeat(*args, **kwargs):
    os.system(f'mkdir ../results/{subject}_biomarkers_raw_images')
    os.system(f'lefse_plot_features.py ../results/kwargs.get(subject)LEFSE_format.txt ../results/kwargs.get(subject)LEFSE_scores.txt ../results/kwargs.get(subject)_biomarkers_raw_images/ --format svg')
    return None

def venn(*args, **kwargs):
    DF1 = pd.read_csv(f'../results/kwargs.get(df1).tsv', sep='\t', index_col=0)
    DF2 = pd.read_csv(f'../results/kwargs.get(df2).tsv', sep='\t', index_col=0)
    DF3 = pd.read_csv(f'../results/kwargs.get(df3).tsv', sep='\t', index_col=0)
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

def expvsobs(df, **kwargs):
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkblue', marker='o')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray', linewidth=2)
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [y_test[i], y_pred[i]], color='red', alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression Model')
    return None

def plot(plottype, df, logx=False, logy=False, **kwargs):
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
        'multibox':multibox,
        'dendrogram':dendrogram,
        'scatter':scatter,
        'upset':upset,
        'networkplot':networkplot,
        'venn':venn,
        'expvsobs':expvsobs
        }
    setupplot()
    if not kwargs.get('figsize'): kwargs['figsize'] = (3,3)
    if not kwargs.get('ax'): kwargs['ax'] = plt.subplots(figsize=kwargs.get('figsize'))[1]
    kwargs.pop('figsize')
    ax = available.get(plottype)(df, **kwargs)
    if logx: plt.xscale('log')
    if logy: plt.yscale('log')
    return ax

# Merge
def merge(datasets=None, type='inner', append=None, filename=None):
    if append:
        outdf = pd.concat(datasets, axis=0, join=type)
    else:
        outdf = pd.concat(datasets, axis=1, join=type)
    return outdf

# Filter 
def filter(df, min_unique=None, gt=None, lt=None, column=None, filter_df=None, filter_df_axis=0, absgt=None, rowfilt=None, colfilt=None, nonzero=None, prevail=None, abund=None):
    if filter_df is not None:
        if filter_df_axis == 1:
            df = df.loc[:, filter_df.index]
        else:
            df = df.loc[filter_df.index]
    if colfilt:
        df = df.loc[:, df.columns.str.contains(colfilt, regex=True)]
    if rowfilt:
        df = df.loc[df.index.str.contains(rowfilt, regex=True)]
    if prevail:
        df = df.loc[:, df.agg(np.count_nonzero, axis=0).gt(df.shape[0]*prevail)]
    if abund:
        df = df.loc[:, df.mean().gt(abund)]
    if column and lt:
        df = df.loc[df[column].lt(lt)]
    elif lt:
        df = df.loc[:, df.abs().lt(lt).any(axis=0)]
    if column and gt:
        df = df.loc[df[column].gt(gt)]
    elif gt:
        df = df.loc[:, df.abs().gt(gt).any(axis=0)]
    if column and absgt:
        df = df.loc[df[column].abs().gt(absgt)]
    if nonzero:
        df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) !=0]
    return df

# Explain
def SHAP_interact(df, model, **kwargs):
    X = df.copy()
    explainer = shap.TreeExplainer(model)
    inter_shaps_values = explainer.shap_interaction_values(X)
    vals = inter_shaps_values[0]
    for i in range(1, vals.shape[0]): vals[0] += vals[i]
    final = pd.DataFrame(vals[0], index=X.columns, columns=X.columns)
    final = final.stack().sort_values().to_frame('SHAP_interaction')
    final.index = final.index.set_names(['source', 'target'], level=[0,1])
    return final

def SHAP_bin(df, model, **kwargs):
    # work on this to get std shap
    X = df.copy()
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
        np.abs(shaps_values.values[:, :, 0]).mean(axis=0),
        index=X.columns
    )
    corrs = [spearmanr(shaps_values.values[:, x, 1], X.iloc[:, x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    final = final.sort_values()
    return final

def SHAP_reg(X, model):
    import numpy as np
    from scipy.stats import spearmanr
    import pandas as pd
    import shap
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
            np.abs(shaps_values.values).mean(axis=0),
            index=X.columns
            )
    corrs = [spearmanr(shaps_values.values[:,x], X.iloc[:,x])[0] for x in range(len(X.columns))]
    shaps = pd.DataFrame(shaps_values.values, columns=X.columns, index=X.index)
    shaps = shaps.loc[:,shaps.sum() != 0]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return final

def explain(analysis, subject, **kwargs):
    available={
        'SHAP_bin':SHAP_bin,
        'SHAP_interact':SHAP_interact
        }
    output = available.get(analysis)(df, model, **kwargs)
    return output

# Describe
def describe(df, pval=0.05, corr=None, change=None, sig=None, **kwargs):
    # CHANGED
    if change and sig:
        changed = 'sig changed = ' +\
            str(df[sig].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df[sig].lt(pval).sum()/df.shape[0] * 100)) + '%)'
        # INCREASED
        increased = 'sig increased = ' +\
            str(df.loc[(df[sig].lt(pval)) & (df[change].gt(0)),sig].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df[sig].lt(pval)) & (df[change].gt(0)),sig].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        # DECREASED
        decreased = 'sig decreased = ' +\
            str(df.loc[(df[sig].lt(pval)) & (df[change].lt(0)),sig].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df[sig].lt(pval)) & (df[change].lt(0)),sig].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        summary = pd.DataFrame([changed, increased, decreased], columns=[subject])
    else:
        summary = df.describe(include='all').T.reset_index()
    if corr:
        changed = 'sig correlated = ' +\
            str(df['sig'].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df['sig'].lt(pval).sum()/df.shape[0] * 100)) + '%)'
        # INCREASED
        increased = 'sig positively correlated = ' +\
            str(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].gt(0).iloc[:,0]), 'sig'].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].gt(0).iloc[:,0]), 'sig'].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        # DECREASED
        decreased = 'sig negatively correlated = ' +\
            str(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].lt(0).iloc[:,0]), 'sig'].lt(pval).sum()) +\
            '/' +\
            str(df.shape[0]) +\
            ' (' +\
            str(round(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].lt(0).iloc[:,0]), 'sig'].lt(pval).sum()/df.shape[0] * 100)) +\
            '%)'
        summary = pd.DataFrame([changed, increased, decreased], columns=[subject])
    return summary

# Corr
def corrpair(df1, df2, FDR=True, min_unique=10):
    df1 = df1.loc[:, df1.nunique() > min_unique]
    df2 = df2.loc[:, df2.nunique() > min_unique]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    pvaldf.fillna(1, inplace=True)
    if FDR:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    return cordf, pvaldf

def sparcc(subject):
    ncor = pd.read_csv('/home/theop/SparCC/sparcc_output/cor_sparcc_tapetes.csv', index_col=0)
    ncor.index, ncor.columns = cor.index, cor.columns
    thresh = 0.03
    edges = f.to_edges(ncor, thresh=thresh)
    edges.sort_values('weight').to_csv('../results/edges.csv')
    return edges

def corr(df, mult=True):
    def to_edges(df):
        df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
        edges = df.stack().to_frame()[0]
        nedges = edges.reset_index()
        edges = nedges[nedges.target != nedges.source].set_index(['source','target'])[0]
        return edges
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    if mult:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    outdf = to_edges(cordf).to_frame('rho').join(to_edges(pvaldf).to_frame('sig')).sort_values('rho')
    return outdf

# Compare - TODO

# Change
def shapiro(df, **kwargs):
    output = pd.DataFrame()
    for col in df.columns: 
        for cat in df.index.unique():
            output.loc[col,cat] = shapiro(df.loc[cat,col])[1]
    return output

def levene(df, **kwargs):
    output = pd.Series()
    for col in df.columns: 
        output[col] = levene(*[df.loc[cat,col] for cat in df.index.unique()])[1]
    return output

def ANCOM(df, **kwargs):
    combs = list(combinations(df.index.unique(), 2))
    if kwargs.get('perm'): combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
            [ancom(pd.concat([df.loc[i[0]], df.loc[i[1]]]), pd.concat([df.loc[i[0]], df.loc[i[1]]]).index.to_series())[0]['Reject null hypothesis'] for i in combs],
            columns = df.columns,
            index = combs,
            )
    return outdf

def LEFSE(df, subject, **kwargs):
    ndf = df.T
    ndf.index.name = 'class'
    ndf = ndf.T.reset_index().T
    ndf.to_csv(f'../results/{subject}LEFSE_data.txt', sep='\t')
    os.system(f'lefse_format_input.py ../results/{subject}LEFSE_data.txt ../results/{subject}LEFSE_format.txt -f r -c 1 -u 1 -o 1000000')
    os.system(f'lefse_run.py ../results/{subject}LEFSE_format.txt ../results/{subject}LEFSE_scores.txt -l 2 --verbose 1')

def mww(df, **kwargs):
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    combs = list(combinations(df.index.unique(), 2))
    if kwargs.get('perm'): combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
        [mannwhitneyu(df.loc[i[0]], df.loc[i[1]])[1] for i in combs],
        columns = df.columns,
        index = combs
        ).T
    if kwargs.get('mult'):
        outdf = pd.DataFrame(
            fdrcorrection(outdf.values.flatten())[1].reshape(outdf.shape),
            columns = outdf.columns,
            index = outdf.index
            )
    outdf.columns = outdf.columns.str.join('/')
    outdf = outdf.add_prefix('mww_sig(').add_suffix(')')
    outdf = outdf.replace([np.inf, -np.inf], np.nan)
    return outdf

def lfc(df, **kwargs):
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    combs = list(combinations(df.index.unique(), 2))
    if kwargs.get('perm'): combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(np.array(
        [df.loc[i[0]].mean().div(df.loc[i[1]].mean()) for i in combs]),
        columns = df.columns,
        index = combs
        ).T.apply(np.log2)
    outdf.columns = outdf.columns.str.join('/')
    outdf = outdf.add_prefix('log2(').add_suffix(')')
    outdf = outdf.replace([np.inf, -np.inf], np.nan)
    return outdf

def prevail(df, **kwargs):
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    basemean = df.mean().to_frame('basemean')
    means = df.groupby(level=0).mean().T
    means.columns = means.columns + '_Mean'
    baseprevail = df.agg(np.count_nonzero, axis=0).div(df.shape[0]).to_frame('baseprevail')
    prevail = df.groupby(level=0, axis=0).apply(lambda x: x.agg(np.count_nonzero, axis=0).div(x.shape[0])).T
    prevail.columns = prevail.columns + '_Prev'
    output = pd.concat([basemean,means,baseprevail,prevail], join='inner', axis=1)
    return output

def std(df, **kwargs):
    df.index, df.columns = df.index.astype(str), df.columns.astype(str)
    basestd = df.std().to_frame('basestd')
    stds = df.groupby(level=0).std().T
    stds.columns = stds.columns + '_Std'
    output = pd.concat([basestd,stds], join='inner', axis=1)
    return output

def change(df, analysis=['prevail','lfc','mww'], **kwargs):
    df = df.sort_index()
    available={
        'prevail':prevail,
        'mww':mww,
        'lfc':lfc,
        'ancom':ANCOM,
        'std':std,
        }
    output = []
    i = analysis[0]
    for i in analysis:
        output.append(available.get(i)(df, **kwargs))
    out = pd.concat(output, join='inner', axis=1)
    return out

# Calculate
def diversity(df, **kwargs):
    def Richness(df, axis=1): return df.agg(np.count_nonzero, axis=axis)
    def Evenness(df, axis=1): return df.agg(pielou_e, axis=axis)
    def Shannon(df, axis=1): return df.agg(shannon, axis=axis)
    diversity = pd.concat(
            [Evenness(df.copy()).to_frame('Evenness'),
             Richness(df.copy()).to_frame('Richness'),
             Shannon(df.copy()).to_frame('Shannon')],
            axis=1).sort_index().sort_index(ascending=False)
    return diversity

def fbratio(df, **kwargs):
    phylum = df.copy()
    phylum.columns = phylum.columns.str.extract('p__(.*)\|c')[0]
    taxoMsp = phylum.T.groupby(phylum.columns).sum()
    taxoMsp = taxoMsp.loc[taxoMsp.sum(axis=1) != 0, taxoMsp.sum(axis=0) != 0]
    taxoMsp = taxoMsp.T.div(taxoMsp.sum(axis=0), axis=0)
    FB = taxoMsp.Firmicutes.div(taxoMsp.Bacteroidetes)
    FB.replace([np.inf, -np.inf], np.nan, inplace=True)
    FB.dropna(inplace=True)
    FB = FB.reset_index().set_axis(['Host Phenotype', 'FB_Ratio'], axis=1).set_index('Host Phenotype')
    return FB

def pbratio(df, **kwargs):
    genus = df.copy()
    genus.columns = genus.columns.str.extract('g__(.*)\|s')[0]
    taxoMsp = genus.T.groupby(genus.columns).sum()
    taxoMsp = taxoMsp.loc[taxoMsp.sum(axis=1) != 0, taxoMsp.sum(axis=0) != 0]
    taxoMsp = taxoMsp.T.div(taxoMsp.sum(axis=0), axis=0)
    pb = taxoMsp.Prevotella.div(taxoMsp.Bacteroides)
    pb.replace([np.inf, -np.inf], np.nan, inplace=True)
    pb.dropna(inplace=True)
    pb = pb.reset_index().set_axis(['Host Phenotype', 'PB_Ratio'], axis=1).set_index('Host Phenotype')
    return pb

def pca(df, **kwargs):
    scaledDf = StandardScaler().fit_transform(df)
    pca = PCA()
    results = pca.fit_transform(scaledDf).T
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
    return df[['PC1', 'PC2']]

def pcoa(df, **kwargs):
    ndf = df.copy()
    Ar_dist = distance.squareform(distance.pdist(ndf, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    results = PCoA.samples.copy()
    ndf['PC1'], ndf['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values
    return ndf[['PC1', 'PC2']]

def nmds(df, **kwargs):
    BC_dist = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    mds = MDS(n_components = 2, metric = False, max_iter = 500, eps = 1e-12, dissimilarity = 'precomputed')
    results = mds.fit_transform(BC_dist)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def tsne(df, **kwargs):
    results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def top20(df, **kwargs):
    if df.shape[1] > 20:
        df['other'] = df[df.sum().sort_values(ascending=False).iloc[19:].index].sum(axis=1)
    df = df[df.sum().sort_values().tail(20).index]
    return df

def som(df, **kwargs):
    som = SOM(m=3, n=1, dim=2)
    som.fit(df)
    return som

def umap(df, **kwargs):
    scaledDf = StandardScaler().fit_transform(df)
    reducer = umap.UMAP()
    results = reducer.fit_transform(scaledDf)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def calculate(analysis, df, **kwargs):
    available={
        'diversity':diversity,
        'fbratio':fbratio,
        'pbratio':pbratio,
        'pca':pca,
        'pcoa':pcoa,
        'nmds':nmds,
        'tsne':tsne,
        'top20':top20,
        'som':som,
        'umap':umap,
        }
    output = available.get(analysis)(df, **kwargs)
    return output

# Scale
def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def standard(df):
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df.T),
            index=df.T.index,
            columns=df.T.columns).T
    return scaledDf

def minmax(df):
    scaledDf = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def log(df):
    return df.apply(np.log1p)

def clr(df):
    return pd.DataFrame(clr(df), index=df.index, columns=df.columns)

def mult(df):
    return pd.DataFrame(mul(df), index=df.index, columns=df.columns)

def scale(analysis, df):
    available={
        'norm':norm,
        'standard':standard,
        'minmax':minmax,
        'log':log,
        'clr':clr,
        'mult':mult,
        }
    output = available.get(analysis)(df)
    return output

# Variance
def PERMANOVA(df, pval=True, full=False, **kwargs):
    #with open(f'../results/{df1}variance.txt','w') as of: of.write(output.to_string())
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    result = permanova(DM_dist, df.index)
    if full:
        return result
    if pval:
        return result['p-value']
    else:
        return result['test statistic']

def explainedvariance(df1, df2, pval=True, **kwargs):
    # how does df1 explain variance in df2 where df2 is meta (only categories)
    # should rework this one to include in calculate but hard
    # 4.32 is significant
    # -np.log(0.05)
    target = DF2.columns[0]
    output = pd.Series()
    for target in DF2.columns:
        tdf = DF1.join(DF2[target].fillna('missing'),how='inner').set_index(target)
        if all(tdf.index.value_counts().lt(10)):
            continue
        if tdf.index.nunique() <= 1:
            continue
        output[target] = PERMANOVA(tdf, pval=pval)
    if pval:
        power = -output.apply(np.log2)
    else:
        power = output
    power = power.to_frame(df1)
    return power

# Stratify
def stratify(df, meta, level):
    metadf = df.join(meta[level].dropna(), how='inner').reset_index(drop=True).set_index(level)
    return metadf

# Splitter
def splitter(df, column, df2='meta', **kwargs):
    metadf = pd.DataFrame()
    for level in meta[column].unique():
        merge = df.join(meta.loc[meta[column] == level, column], how='inner').drop(column, axis=1)
        metadf[level] = merge
    return metadf

# Misc
def taxofunc(msp, taxo, short=False):
    import pandas as pd
    m, t = msp.copy(), taxo.copy()
    t['superkingdom'] = 'k_' + t['superkingdom']
    t['phylum'] = t[['superkingdom', 'phylum']].apply(lambda x: '|p_'.join(x.dropna().astype(str).values), axis=1)
    t['class'] = t[['phylum', 'class']].apply(lambda x: '|c_'.join(x.dropna().astype(str).values), axis=1)
    t['order'] = t[['class', 'order']].apply(lambda x: '|o_'.join(x.dropna().astype(str).values), axis=1)
    t['family'] = t[['order', 'family']].apply(lambda x: '|f_'.join(x.dropna().astype(str).values), axis=1)
    t['genus'] = t[['family', 'genus']].apply(lambda x: '|g_'.join(x.dropna().astype(str).values), axis=1)
    t['species'] = t[['genus', 'species']].apply(lambda x: '|s_'.join(x.dropna().astype(str).values), axis=1)
    mt = m.join(t, how='inner')
    df = pd.concat([mt.set_index(t.columns[i])._get_numeric_data().groupby(level=0).sum() for i in range(len(t.columns))])
    df.index = df.index.str.replace(" ", "_", regex=True).str.replace('/.*','', regex=True)
    if short:
        df.index = df.T.add_prefix("|").T.index.str.extract(".*\|([a-z]_.*$)", expand=True)[0]
    df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    return df
