import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
plt.rcParams["font.family"] = "Arial"

#------------------------------------
# DATA CLEANING
#------------------------------------

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


#------------------------------------
# DATA VISUALISATION
#------------------------------------


def plot_hist(df):

    train_color = '#7920d3'
    test_color = '#e69f00'

    mers_col = "pIC50 (MERS-CoV Mpro)"
    sars_col = "pIC50 (SARS-CoV-2 Mpro)"

    endpoints = [(mers_col, "MERS-CoV Mpro"), (sars_col, "SARS-CoV-2 Mpro")]

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, height_ratios = [4, 1], hspace=0.05, wspace=0.25)

    for i, (col, title) in enumerate(endpoints):
        ax = fig.add_subplot(gs[0,i])
        ax_box = fig.add_subplot(gs[1,i], sharex=ax)

        train = df.loc[df["Set"].eq("Train"), col].dropna()
        test  = df.loc[df["Set"].eq("Test"),  col].dropna()

        all_vals = pd.concat([train, test], ignore_index=True)
        bins = np.histogram_bin_edges(all_vals, bins=20)

        ax.hist(train, bins=bins, alpha=0.6, label=f"Train (n = {len(train)})", density=True, color = train_color, edgecolor="black", linewidth=0.8)
        ax.hist(test,  bins=bins, alpha=0.6, label=f"Test (n = {len(test)})", density=True, color = test_color, edgecolor="black", linewidth=0.8)

        # KDE curves
        x_min = np.min(np.r_[train, test])
        x_max = np.max(np.r_[train, test])
        x = np.linspace(x_min, x_max, 400)

        kde_train = gaussian_kde(train)  # you can tune bandwidth; see below
        kde_test  = gaussian_kde(test)

        ax.plot(x, kde_train(x), linewidth=1, color = train_color)
        ax.plot(x, kde_test(x),  linewidth=1, color = test_color)

        ax.set_title(f"pIC50 distribution: {title}")
        ax.set_ylabel("Density")
        ax.legend()

        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(axis="x", which="both", length=0)

        bp = ax_box.boxplot(
        [train, test],
        vert=False,
        labels=["Train", "Test"],
        patch_artist=True,
        showfliers=False)

        for key in ["boxes", "whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(1.0)

        if train_color is not None:
            bp["boxes"][0].set_facecolor(train_color)
            bp["boxes"][0].set_alpha(0.5)
        if test_color is not None:
            bp["boxes"][1].set_facecolor(test_color)
            bp["boxes"][1].set_alpha(0.5)

        ax_box.set_xlabel("pIC50")
        ax_box.set_yticks([1, 2])
        ax_box.set_yticklabels(["Train", "Test"])

    plt.tight_layout()
    plt.show()