import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

def imputation(df):
    """ Missing value imputation.
    
    Parameters:
    -----------
    df: Pandas dataframe
    
    Yields:
    -------
    df_new: Pandas dataframe
    """
    
    print("Start missing value imputation ...")
    genes = df.columns[4:-1]
    #ave = {g:df[g].sum()/df[g].count() for g in genes} # compute average gene's non-missing values
    # fill missing values with gene's average exp levels
    df_new = df.copy(deep = True)
    for i in df_new.index:
        ave = df.loc[i, genes].sum()/df.loc[i, genes].count()
        #print(df.loc[i, genes].count())
        df_new.loc[i,df_new.loc[i].isnull()] = ave
    #print(df_new)
    return df_new

def quantile_normalization(df1, df2):
    """ Quantile normalization cross platforms
    Parameters:
    -----------
    df1: first Pandas dataframe after imputation
    df2: second Pandas dataframe after imputation
    Yields:
    -------
    df1_new: Pandas dataframe
    df2_new: Pandas dataframe
    """
    
    print("Start quantile normalization ...")
    genes = df1.columns[4:-1]
    df = pd.concat([df1[genes], df2[genes]])
    
    sorted_matrix = np.array([sorted(row[genes].tolist()) for _,row in df.iterrows()])
    quantiled_ave = {i:j for i, j in enumerate(list(np.mean(sorted_matrix,axis = 0)))} #{rank: value}
    df1_new = []
    for i,row in df1.iterrows():
        vals = list(row[genes])
        order = {}
        for j,k in enumerate(np.argsort(vals)):
            order.update({vals[k]:j})
        ranked_genes = [quantiled_ave[order[v]] for v in vals]
        df1_new.append(list(row.iloc[:4])+ranked_genes+list(row.iloc[-1:]))
    df1_new = pd.DataFrame(df1_new, columns = list(df1.columns))

    df2_new = []
    for i,row in df2.iterrows():
        vals = list(row[genes])
        order = {}
        for j,k in enumerate(np.argsort(vals)):
            order.update({vals[k]:j})
        ranked_genes = [quantiled_ave[order[v]] for v in vals]
        df2_new.append(list(row.iloc[:4])+ranked_genes+list(row.iloc[-1:]))
    df2_new = pd.DataFrame(df2_new, columns = list(df2.columns))
    return df1_new, df2_new

def preprocess_in_vivo(df):
    """ Preprocess raw in vivo dataset
    """
    if "ClearanceRate_binary" not in list(df.columns.values):
        # preprocess prediction features
        df_new = imputation(df)
        # binarize prediction labels
    
        assert (list(sorted(set(df_new['ClearanceRate']))) == ['Fast', 'Slow']), "Clearance Rate in in vivo dataset is not correct!"
        df_new['ClearanceRate_binary'] = [0 if x == 'Fast' else 1 for x in df_new['ClearanceRate']]
        df_new = df_new.drop(columns = ['ClearanceRate'])
    else: # we've already imputed and binarized
        df_new = df

    return df_new

def preprocess_in_vitro(df):
    """ Preprocess raw in vitro dataset
    """
    df_new = imputation(df)
    return df_new

def data_preparation(df_invivo, df_invitro, no_quantile = False, common_genes = None, validate_in_vivo=False):
    """ Preprocess of in vivo and in vitro datasets
    Parameters:
    -----------
    df_invivo: Pandas Dataframe
        in vivo dataset of gene expression levels
        extra columns:  'Sample_Names', 'Country', 'Asexual.stage..hpi.', 'Kmeans.Grp'
        label: 'ClearanceRate'
    df_invitro: Pandas Dataframe
        in vitro dataset of gene expression levels
        extra columns: 'Sample_Name', 'Isolate', 'Timepoint', 'Treatment', 'BioRep'
        label: 'DHA_IC50'
    if_quantile: boolean
        if quantile normalization or not
    common_genes: list
        if specified, use common_genes only as feature set. else use the whole genesets shared by both in vivo and in vitro dataset.
    Yields:
    -------
    df_invivo: Pandas Dataframe
        processed in vivo dataset
    df_invitro: Pandas Dataframe
        processed in vitro dataset
    """
    # remove samples without labels
    if not validate_in_vivo:
        df_invivo = df_invivo.dropna(subset = ['ClearanceRate']) if "ClearanceRate" in list(df_invivo.columns.values) else df_invivo # if already preprocessed
        df_invitro = df_invitro.dropna(subset = ['DHA_IC50'])
    
    # find common genes
    if common_genes == None:
        if not validate_in_vivo:
            common_genes = sorted(list(set(df_invivo.columns[4:-1])&set(df_invitro.columns[5:-1])))
        else:
            common_genes = sorted(list(set(df_invivo.columns[4:-1])&set(df_invitro.columns[4:-1])))
        print("Shared genes between in vivo and in vitro datasets: ",len(common_genes))
    
    # select common columns; attach four extra informatio columns at front
    if not validate_in_vivo:
        if "ClearanceRate_binary" not in list(df_invivo.columns.values):
            df_invivo = df_invivo[['Sample_Names', 'Country', 'Asexual.stage..hpi.', 'Kmeans.Grp']+common_genes+['ClearanceRate']]
        else:
            df_invivo = df_invivo[['Sample_Names', 'Country', 'Asexual.stage..hpi.', 'Kmeans.Grp']+common_genes+['ClearanceRate_binary']]
        if "Isolate" in list(df_invitro.columns.values):
            df_invitro = df_invitro[['Isolate', 'Timepoint', 'Treatment', 'BioRep']+common_genes+['DHA_IC50']]
        else:
            df_invitro = df_invitro[common_genes+['DHA_IC50']]
        
    else: # WARNING: We are testing on the same set here!
        df_invivo = df_invivo[['Sample_Names', 'Country', 'Asexual.stage..hpi.', 'Kmeans.Grp']+common_genes+['ClearanceRate_binary']]
        df_invitro = df_invitro[['Sample_Names', 'Country', 'Asexual.stage..hpi.', 'Kmeans.Grp']+common_genes+['ClearanceRate_binary']]
    
    # preprocess in invo and in vitro datasets respectively since they contain differernt labels
    if not validate_in_vivo:
        df_invivo = preprocess_in_vivo(df_invivo)
        df_invitro = preprocess_in_vitro(df_invitro)
    else:
        df_invivo = preprocess_in_vivo(df_invivo)
        df_invitro = preprocess_in_vivo(df_invitro)
    
    # quantile normalization on both datasets
    if no_quantile:
        pass
    else:
        df_invivo, df_invitro = quantile_normalization(df_invivo, df_invitro)
    
    return df_invivo, df_invitro
