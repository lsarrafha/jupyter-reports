##############################################################
##############################################################
############ Parkinson's Disease Analysis Support ############
##############################################################
##############################################################
###################### Author: Lily Sarrafha
###################### Affiliation: Ma'ayan Laboratory
###################### Icahn School of Medicine at Mount Sinai

##############################################################
############# 1. Load libraries
##############################################################

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import sklearn
import matplotlib.pyplot as plt, pandas as pd
import numpy as np, seaborn as sns, scipy.stats as ss, plotly.graph_objs as go
from clustergrammer_widget import *
from sklearn.decomposition import PCA
from IPython.display import display, Markdown
from rpy2.robjects import r, pandas2ri

##############################################################
############# 2. Prepare functions
##############################################################

import sys
sys.path.append('scripts')

def apply_voom(dataframe):
    # Connect to R
    r.source('scripts/code_library.R')
    pandas2ri.activate()

    # Convert to R
    dataframe_r = pandas2ri.py2ri(dataframe)

    # Run
    signature_dataframe_r = r.apply_voom(dataframe_r)

    # Convert to pandas and sort
    signature_dataframe = pandas2ri.ri2py(signature_dataframe_r)

    # Return
    return signature_dataframe

def generate_figure_legend(figure_number, description):
    display(Markdown('**Figure {figure_number} -** {description}'.format(**locals())))

def plot_2D_scatter(x, y, text='', title='', xlabel='', ylabel='', color='#6495ed', colorscale='Reds'):
    trace = go.Scattergl(
        x = x,
        y = y,
        mode = 'markers',
        hoverinfo = 'text',
        text = text,
        marker = dict(
            color=color,
            colorscale=colorscale,
            line = dict(
                width = 1,
                color = '#404040')
        )
    )
    data = [trace]
    layout = go.Layout(title = title)
    fig = dict(data=data, layout=layout)
    fig['layout']['xaxis'] = dict(title=xlabel)
    fig['layout']['yaxis'] = dict(title=ylabel)
    iplot(fig)

##############################################################
########## 3. Sample sum values (bar chart)
##############################################################

def sample_barchart(dataframe):
    display(Markdown('**Bar Chart** <br> First, we calculate the sum of the raw counts for each sample and plot the results in a bar chart.'))
    sample_sum=dataframe.sum(axis=0)
    plt.rcParams['figure.figsize']
    plt.title('Sample Sum Values', fontsize=20)
    plt.xlabel('Samples', fontsize=15)
    plt.ylabel('Sum', fontsize=15)
    sample_sum.plot.bar(figsize=[12,8], color='#6495ed')

##############################################################
########## 4. Gene median distribution (histogram)
##############################################################

def gene_histogram(dataframe):
    display(Markdown('**Histogram** <br> Then, we calculate the median values for each gene in log scale and plot the results in a histogram.'))
    gene_median=dataframe.median(axis=1)
    np.log10(gene_median+1).plot(kind='hist', bins=50, color='#6495ed', log=True, figsize=[12,8])
    plt.rcParams['figure.figsize']
    plt.title('Gene Median Distribution', fontsize=20)
    plt.xlabel('Gene Median', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)

##############################################################
############# 5. Clustermap (for voom results)
##############################################################

def plot_clustermap(dataframe, z_score=0, cmap=sns.color_palette("RdBu_r"), nr_genes=500):
    display(Markdown('**Clustermap** <br> We can visualize the 500 most variable genes in a clustermap.'))
    voom_dataframe = apply_voom(dataframe)
    top_genes = voom_dataframe.var(axis=1).sort_values(ascending=False).index.tolist()[:nr_genes]
    voom_dataframe = voom_dataframe.loc[top_genes]
    sns.clustermap(voom_dataframe, z_score=z_score, cmap=cmap)

##############################################################
############# 6. Correlation heatmap (for voom results)
##############################################################

def plot_correlation_heatmap(dataframe, cmap=sns.color_palette("RdBu_r", 500), correlation_axis=0, correlation_method='spearman', nr_genes=500):
    display(Markdown('**Gene Co-expression Heatmap** <br> We can also represent the 500 most variable genes in the dataset using a correlation heatmap.'))
    voom_dataframe = apply_voom(dataframe)
    top_genes = voom_dataframe.var(axis=1).sort_values(ascending=False).index.tolist()[:nr_genes]
    voom_dataframe = voom_dataframe.loc[top_genes]

    # correlation
    if correlation_axis:
        dataframe = voom_dataframe.corr(method=correlation_method)
    else:
        dataframe = voom_dataframe.T.corr(method=correlation_method)

    sns.clustermap(dataframe, z_score=None, cmap=cmap)

##############################################################
############# 7. Clustergrammer (for voom results)
##############################################################
# Note: Did not include clustergrammer in the notebooks because the widget would not save properly.

def plot_clustergrammmer(dataframe, filter_rows=True, filter_rows_by='var', filter_rows_n=500, normalize=True, nr_genes=500):
    display(Markdown('**Clustergrammer** <br> Another way of demonstrating the data is a clustergrammer.'))
    voom_dataframe = apply_voom(dataframe)
    top_genes = voom_dataframe.var(axis=1).sort_values(ascending=False).index.tolist()[:nr_genes]
    voom_dataframe = voom_dataframe.loc[top_genes]
    net = Network(clustergrammer_widget)
    net.load_df(voom_dataframe)
    net.normalize()
    net.cluster()
    return net.widget()

##############################################################
############# 8. 3D PCA plot (for voom results)
##############################################################

def plot_pca_3d(dataframe, size=20, color_by_categorical=None, color_by_continuous=None, colorscale="Viridis", showscale=True, colors=['#6495ed', '#324a76', '#671999', '#810038', '#6379ff', '#94789c', '#8ad08d'], nr_genes=500):
    display(Markdown('**3D PCA Plot** <br> Finally, we can use a 3D PCA plot to examine the clustering of the data based on gene expression.'))
    width=900
    height=600
    voom_dataframe = apply_voom(dataframe)
    top_genes = voom_dataframe.var(axis=1).sort_values(ascending=False).index.tolist()[:nr_genes]
    voom_dataframe = voom_dataframe.loc[top_genes]
    data_zscore=voom_dataframe.apply(ss.zscore, 1)
    pca=PCA(n_components=3)
    pca.fit(data_zscore)
    var_explained = ['PC'+str((i+1))+'('+str(round(e*100, 1))+'% var. explained)' for i, e in enumerate(pca.explained_variance_ratio_)]
    
    if str(color_by_categorical) == 'None':
        if str(color_by_continuous) == 'None':
            marker = dict(size=size)
        else:
            marker = dict(size=size, color=color_by_continuous, colorscale=colorscale, showscale=showscale)
        trace = go.Scatter3d(x=pca.components_[0],
                             y=pca.components_[1],
                             z=pca.components_[2],
                             mode='markers',
                             hoverinfo='text',
                             text=data_zscore.columns,
                             marker=marker)
        data = [trace]
    else:
        # Get unique categories
        unique_categories = color_by_categorical.unique()

        # Define empty list
        data = []
            
        # Loop through the unique categories
        for i, category in enumerate(unique_categories):

            # Get the color corresponding to the category
            category_color = colors[i]

            # Get the indices of the samples corresponding to the category
            category_indices = [i for i, sample_category in enumerate(color_by_categorical) if sample_category == category]
            
            # Create new trace
            trace = go.Scatter3d(x=pca.components_[0][category_indices],
                                 y=pca.components_[1][category_indices],
                                 z=pca.components_[2][category_indices],
                                 mode='markers',
                                 hoverinfo='text',
                                 text=data_zscore.columns[category_indices],
                                 name = category,
                                 marker=dict(size=size, color=category_color))
            
            # Append trace to data list
            data.append(trace)  
        
    layout = go.Layout(hovermode='closest',height=height,scene=dict(xaxis=dict(title=var_explained[0]),
                                                                            yaxis=dict(title=var_explained[1]),
                                                                            zaxis=dict(title=var_explained[2]),
                                                                    ),margin=dict(l=0,r=0,b=0,t=0))
    fig = go.Figure(data=data, layout=layout)

    iplot(fig)

##############################################################
############# 11. DEG Calculations
##############################################################

import sys
sys.path.append('scripts')

def compute_degs(dataframe, method, samples, controls, constant_threshold=10, filter_low_expressed=False, min_counts=10):

    # Filter lowly expressed genes
    if filter_low_expressed:
        dataframe = dataframe.loc[[index for index, value in dataframe.sum(axis=1).iteritems() if value > min_counts]]
        print(dataframe.shape)
    # Connect to R
    r.source('scripts/code_library.R')
    pandas2ri.activate()

    # Create design dict
    sample_dict = {'samples': samples, 'controls': controls}

    # Create design dataframe
    design_dataframe = pd.DataFrame({group_label: {sample:int(sample in group_samples) for sample in dataframe.columns} for group_label, group_samples in sample_dict.items()})

    # Convert to R
    dataframe_r = pandas2ri.py2ri(dataframe)
    design_dataframe_r = pandas2ri.py2ri(design_dataframe)

    # Run
    if method == 'CD':
        signature_dataframe_r = r.apply_characteristic_direction(dataframe_r, design_dataframe_r, constant_threshold)
    elif method == 'limma':
        signature_dataframe_r = r.apply_limma(dataframe_r, design_dataframe_r)
    else:
        raise ValueError('Wrong method supplied.  Must be limma or CD.')
 
    # Convert to pandas and sort
    signature_dataframe = pandas2ri.ri2py(signature_dataframe_r)

    # Return
    return signature_dataframe

##############################################################
############# 13. MA plot for limma
##############################################################

def plot_MA(dataframe, title):
    display(Markdown('**MA Plot** <br> The results from limma analysis can be visualized using an MA plot, in which the average expression values are on the x-axis and the logFC values on the y-axis.'))
    plot_2D_scatter(dataframe['AveExpr'], dataframe['logFC'],
                    text = ['<span style="font-size: 12pt; color: white; text-decoration: underline; text-align: center; font-weight: 600;">'+gene_symbol+'</span>'+'<br>logFC='+str(round(rowData['logFC'], ndigits=2))+'<br>p value='+"{:.2E}".format(rowData['adjPVal'])+'<br>Avg Exp='+str(round(rowData['AveExpr'], ndigits=2)) for gene_symbol, rowData in dataframe.iterrows()], 
                    title = title, xlabel='AveExpr', ylabel='logFC')

##############################################################
############# 14. Volcano plot for limma
##############################################################

def plot_volcano(dataframe, title):
    display(Markdown('**Volcano Plot** <br> We can also display the same results using a Volcano plot, in which the logFC values are on the x-axis and the log10-transformed adjusted p-values on the y-axis.'))
    limma_log = -np.log10(dataframe['adjPVal'])
    plot_2D_scatter(dataframe['logFC'], limma_log, 
                    text = ['<span style="font-size: 12pt; color: white; text-decoration: underline; text-align: center; font-weight: 600;">'+gene_symbol+'</span>'+'<br>logFC='+str(round(rowData['logFC'], ndigits=2))+'<br>p value='+"{:.2E}".format(rowData['adjPVal'])+'<br>Avg Exp='+str(round(rowData['AveExpr'], ndigits=2)) for gene_symbol, rowData in dataframe.iterrows()],
                    title = title, xlabel='logFC', ylabel='adjPVal(log10)')

##############################################################
############# 15. MA plot for CD
##############################################################
# Note: Did not include this plot due to problems with the CD code.

def plot_cd(dataframe, title):
    display(Markdown('**Volcano Plot** <br> We can also display the same results using a Volcano plot, in which the logFC values are on the x-axis and the log10-transformed adjusted p-values on the y-axis.'))
    limma_log = -np.log10(dataframe['adjPVal'])
    plot_2D_scatter(dataframe['CD'], dataframe['logFC'], 
                    text = ['<span style="font-size: 12pt; color: white; text-decoration: underline; text-align: center; font-weight: 600;">'+gene_symbol+'</span>'+'<br>logFC='+str(round(rowData['logFC'], ndigits=2))+'<br>p value='+"{:.2E}".format(rowData['adjPVal'])+'<br>Avg Exp='+str(round(rowData['AveExpr'], ndigits=2)) for gene_symbol, rowData in dataframe.iterrows()], 
                    title = title, xlabel='logFC', ylabel='adjPVal(log10)')

##############################################################
############# 16. Run Enrichment Analysis
##############################################################

import json
import requests

def extract_genesets(dataframe, limma=True, sort_by='CD', nr_genes=500, logFC=2, p_value=0.05): 
    if limma == True:
        upregulated = dataframe.query('logFC > {logFC} & adjPVal < {p_value}'.format(**locals())).index.tolist()
        downregulated = dataframe.query('logFC < -{logFC} & adjPVal < {p_value}'.format(**locals())).index.tolist()
    else:
        ranked_genes = dataframe['CD'].sort_values(ascending=False).index.tolist()
        upregulated = ranked_genes[:nr_genes]
        downregulated = ranked_genes[-nr_genes:]
    genesets = {'upregulated': upregulated, 'downregulated': downregulated}
    return genesets

def submit_enrichr_geneset(genesets):
    # Initialize results
    enrichr_results = {}

    # Loop through genesets
    for label, geneset in genesets.items():
        # Run Enrichr for upregulated and downregulated genes
        ENRICHR_URL = 'http://amp.pharm.mssm.edu/Enrichr/addList'
        genes_str = '\n'.join(geneset)
        payload = {
            'list': (None, genes_str),
        }

        response = requests.post(ENRICHR_URL, files=payload)
        if not response.ok:
            raise Exception('Error analyzing gene list')

        data = json.loads(response.text)
        enrichr_results[label] = data

    return enrichr_results

# Write a for loop for the userListId in the dictionary
def dict_forloop(enrichr_results):
    subset_dict = {downregulated: {upregulated: v2 for upregulated, v2 in v1.items() if upregulated == 'userListId'} for downregulated, v1 in enrichr_results.items()}
    for downregulated, v1 in subset_dict.items():
        userlistid = ""
        userlistid += downregulated
        for upregulated, v2 in v1.items():
            userlistid = userlistid + " " + str(v2)
        print(userlistid)
 
# Extract the enrichment results from Enrichr
import qgrid
def get_enrichment_results(user_list_id, gene_set_libraries=['GO_Biological_Process_2017b', 'GO_Molecular_Function_2017b', 'GO_Cellular_Component_2017b']):
    ENRICHR_URL = 'http://amp.pharm.mssm.edu/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'
    results = []
    for gene_set_library in gene_set_libraries:
        response = requests.get(
            ENRICHR_URL + query_string % (user_list_id, gene_set_library)
         )
        if not response.ok:
            raise Exception('Error fetching enrichment results')

        data = json.loads(response.text)
        enrichmentDataframe = pd.DataFrame(data[gene_set_library], columns=['rank', 'term_name', 'pvalue', 'zscore', 'combined_score', 'overlapping_genes', 'FDR', 'old_pvalue', 'old_FDR'])
        enrichmentDataframe = enrichmentDataframe.loc[:,['term_name','zscore','combined_score','pvalue', 'FDR', 'overlapping_genes']]
        enrichmentDataframe['geneset_library'] = gene_set_library
        results.append(enrichmentDataframe)
    resultDataframe = pd.concat(results).sort_values('pvalue')
    # widget = qgrid.QGridWidget(df=resultDataframe.set_index('term_name').drop(['zscore', 'combined_score'], axis=1))
    return resultDataframe 

# Combine the 3 codes above
def run_enrichr(dataframe):
    # Extract genesets
    genesets = extract_genesets(dataframe)

    # Submit genesets to enrichr
    enrichr_results = submit_enrichr_geneset(genesets)
    display(Markdown('Link to Enrichr results for upregulated geneset: http://amp.pharm.mssm.edu/Enrichr/enrich?dataset={shortId}'.format(**enrichr_results['upregulated'])))
    display(Markdown('Link to Enrichr results for downregulated geneset: http://amp.pharm.mssm.edu/Enrichr/enrich?dataset={shortId}'.format(**enrichr_results['downregulated'])))

    # Define an empty list
    my_results = []

    # Loop through the enrichr IDs (the long ones)

    for key, value in enrichr_results.items():
        # For each ID, get the enrichment results, add column with geneset label and assign each geneset to 'upregulated' or 'downregulated'
        results = get_enrichment_results(enrichr_results[key]['userListId'])
        # results.insert(6, 'geneset', enrichr_results[key]['userListId'])
        results['geneset'] = key
        # Append the enrichment results to the empty list defined above
        my_results.append(results)
    
    # Concatenate the two using pd.concat()
    resultDataframe = pd.concat(my_results).set_index('term_name').drop(['zscore', 'combined_score'], axis=1)

    # Return datafram
    return resultDataframe

def display_qgrid(dataframe):
    # Display a widget using Qgrid as a result
    widget = qgrid.QGridWidget(df=dataframe)
    return widget
