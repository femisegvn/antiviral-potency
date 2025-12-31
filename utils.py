import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from typing import Tuple

def canonicalise_smiles(smiles):
    '''
    Converts (CX)SMILES into `rdkit.mol` object and back into (CX)SMILES

    Arguments
    ----------
    smiles: array-like, list of smiles strings to 'clean'

    Returns
    ----------
    smiles: array-like, list of canonicalised smiles
    '''
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    smiles = [Chem.MolToCXSmiles(m) for m in mols]
    return smiles

def train_test_overlap(df, smiles_col="CXSMILES"):
    '''
    Identifies instances of train-test overlap -- where CXSMILES in the training set also appear
    in the test set 

    Arguments
    ----------
    df: pd.DataFrame, input df containing both train and test data
    smiles_col: str, name of df column name containing SMILES, default == "CXSMILES"

    Returns
    -------
    overlap: set of CXSMILES that appear in both the train set and the test set
    '''

    df_copy = df[[smiles_col, "Set"]].copy()

    train_smiles = set(df_copy.loc[df_copy["Set"] == "Train", smiles_col])
    test_smiles  = set(df_copy.loc[df_copy["Set"] == "Test",  smiles_col])

    overlap = train_smiles & test_smiles

    return overlap


def merge_duplicates(df, smiles_col = "CXSMILES",
                     pic50_cols = ("pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)")):
    """
    Merge duplicate CXSMILES into a single row per unique CXSMILES.

    - For pIC50 columns: uses the mean across duplicates (NaNs ignored).
    - For all other columns: keeps the first (or last) non-null value within the group.
      If values conflict, the chosen `keep` strategy applies.

    Arguments
    ----------
    df: pd.DataFrame, input df to be cleaned
    smiles_col: str, name of df column name containing SMILES, default == "CXSMILES"
    pic50_cols: tuple of str, name of columns containing numeric values to be merged

    Returns
    -------
    df: pd.DataFrame, cleaned df with no duplicate strings and merged data
    """

    # Identify "other" columns to carry through (e.g., Molecule Name, Set, etc.)
    other_cols = [c for c in df.columns if c != smiles_col and c not in pic50_cols]

    def _pick_non_null(series):
        s2 = series.dropna()
        return s2.iloc[0]

    agg = {}

    # Mean for pIC50 columns (NaNs ignored by default)
    for c in pic50_cols:
        agg[c] = "mean"

    # For all other columns, pick first non-null within the group
    for c in other_cols:
        agg[c] = _pick_non_null

    out = df.groupby(smiles_col, as_index=False, dropna=False).agg(agg)

    return out