import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors, QED
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
plt.rcParams["font.family"] = "Arial"

#------------------------------------
# DATA CLEANING
#------------------------------------

def canonicalise_smiles(smiles) -> list:
    '''
    Converts (CX)SMILES into `rdkit.mol` object and back into (CX)SMILES

    Arguments
    ----------
    smiles: array-like, list of smiles strings to 'clean'

    Returns
    ----------
    smiles: list, list of canonicalised smiles
    '''
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    smiles = [Chem.MolToCXSmiles(m) for m in mols]
    return smiles

def train_test_overlap(df: pd.DataFrame, smiles_col: str = "CXSMILES") -> set:
    '''
    Identifies instances of train-test overlap: where CXSMILES in the training set also appear
    in the test set 

    Arguments
    ----------
    df: pd.DataFrame, input df containing both train and test data
    smiles_col: str, name of df column name containing SMILES

    Returns
    -------
    overlap: set of CXSMILES that appear in both the train set and the test set
    '''

    df_copy = df[[smiles_col, "Set"]].copy()

    train_smiles = set(df_copy.loc[df_copy["Set"] == "Train", smiles_col])
    test_smiles  = set(df_copy.loc[df_copy["Set"] == "Test",  smiles_col])

    overlap = train_smiles & test_smiles

    return overlap


def merge_duplicates(df: pd.DataFrame, smiles_col: str = "CXSMILES",
                     pic50_cols: tuple[str] = ("pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)")) -> pd.DataFrame:
    """
    Finds the mean pIC50s of duplicate CXSMILES entries and mergesinto a single row per unique CXSMILES.

    Arguments
    ----------
    df: pd.DataFrame, input df to be cleaned
    smiles_col: str, name of df column name containing SMILES
    pic50_cols: tuple of str, name of columns containing numeric values to be merged

    Returns
    -------
    df: pd.DataFrame, cleaned df with no duplicate strings and merged data
    """

    # list of all df columns that aren't SMILES or pIC50 values
    other_cols = [c for c in df.columns if c != smiles_col and c not in pic50_cols]

    def _pick_non_null(series):
        s2 = series.dropna()
        return s2.iloc[0]

    agg = {}

    #aggregate based on mean of IC50 measurements
    for c in pic50_cols:
        agg[c] = "mean"

    # For all other columns, pick first non-null within the group
    for c in other_cols:
        agg[c] = _pick_non_null

    out = df.groupby(smiles_col, as_index=False, dropna=False).agg(agg)

    return out


#------------------------------------
# MODEL DEVELOPMENT
#------------------------------------

def get_desc_names() -> list:
    '''
    Returns
    --------
    descriptors: list of str, names of non-fragment RDKit Descriptors
    '''

    s = 'C'
    mol = Chem.MolFromSmiles(s)

    descriptors = []
    for desc in Descriptors.CalcMolDescriptors(mol).items():
        key, value = desc
        if not key.startswith("fr_"):
            descriptors.append(key)

    return descriptors

def get_frag_names() -> list:
    '''
    Returns
    --------
    fragments: list of str, names of RDKit fragments
    '''
    '''
    s = 'C'
    mol = Chem.MolFromSmiles(s)
    fragments = []
    for desc in Descriptors.CalcMolDescriptors(mol).items():
        key, value = desc
        if key.startswith("fr_"):
            fragments.append(key)
    '''
    fragments = ["fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO",
                     "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O",
                     "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1",
                     "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole",
                     "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
                     "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo",
                     "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo",
                     "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido",
                     "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan",
                     "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy",
                     "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho",
                     "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol",
                     "fr_phenol_noOrthoHbond", "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzin"
                     "", "fr_priamide", "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd",
                     "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan",
                     "fr_thiophene", "fr_unbrch_alkane", "fr_urea"]

    return fragments

def rdkit_all_desc_featuriser(mols) -> np.array:
    '''
    Featurises molecules using all available RDKit descriptors, except fragment-based descriptors

    Arguments
    ---------
    mols: array-like, list of rdkit Mol objects to featurise
    
    Returns
    ---------
    X: np.array, array of feature vectors corresponding to the molecules passed in
    '''

    X = []
    for mol in mols:
        x = []

        descriptors = Descriptors.CalcMolDescriptors(mol)
        for key, value in descriptors.items():

            if not key.startswith("fr"):
                x.append(value)

        X.append(np.array(x))
    X = np.array(X)
    return X

def featurise(mols, scheme: str = "morgan_fp", omit: str = None) -> np.array:
    '''
    Featurises molecules using a method of choice.

    Arguments
    ---------
    mols: array-like, list of rdkit Mol objects to featurise
    scheme: str, featurisation method to use, options; ["morgan_fp", "rdkit_desc", "rdkit_frags", "hybrid"]
    omit: str, featurisation method to omit when using "hybrid", options; ["morgan_fp", "rdkit_desc", "rdkit_frags"]

    Returns
    ---------
    X: np.array, array of feature vectors corresponding to the molecules passed in
    '''

    mols = [Chem.AddHs(mol) for mol in mols]

    def morgan_featuriser(mols, n_bits = 1024, radius = 3):
        '''
        Converts molecules into binary fingerprints using the Morgan circular fingerprint algorithm.
        n_bits: int, bit vector length
        radius: int, number of bonds to consider in atom neighbourhood 
        '''
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = n_bits, includeChirality = True)

        X = np.zeros((len(mols), n_bits), dtype=np.float32)
        for i, mol in enumerate(mols):
            fp = morgan_gen.GetFingerprint(mol)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i, :] = arr
        return X
    
    def rdkit_desc_featuriser(mols):
        '''
        Calculates 11 RDKit descriptors for each molecule, which have been selected using an F-test.
        See `select_features` and `main.ipynb` for report.
        '''

        X = []

        for mol in mols:
            x = []

            x.append(QED.qed(mol))
            x.append(Descriptors.MolWt(mol))
            x.append(Descriptors.NumValenceElectrons(mol))
            x.append(Descriptors.BertzCT(mol))
            x.append(Descriptors.Chi0v(mol))
            x.append(Descriptors.Kappa1(mol))
            x.append(Descriptors.LabuteASA(mol))
            x.append(Descriptors.SMR_VSA10(mol))
            x.append(Descriptors.EState_VSA3(mol))
            x.append(Descriptors.NumHeteroatoms(mol))
            x.append(Chem.Crippen.MolMR(mol))
            
            X.append(np.array(x))

        X = np.array(X)

        return X
    
    def rdkit_frags_featuriser(mols):
        '''
        Produces a one-hot encoded feature vector, counting the presence of molecule fragments
        in each molecule.
        '''
        X = []
        fragments = get_frag_names()
        
        for mol in mols:
            x = []
            for desc, value in Descriptors.CalcMolDescriptors(mol).items():
                if desc in fragments:
                    x.append(value)
            
            X.append(np.array(x))

        X = np.array(X)
        return X

    def hybrid_featuriser(mols, omit=omit):
        '''
        Combines feature vectors produced from `morgan_featuriser`, `rdkit_desc_featuriser`, and
        `rdkit_frags_featuriser`. A particular method can be left out, specified by `omit`.
        '''
        Xs = []

        if not omit == "morgan":
            X_fp = morgan_featuriser(mols)
            Xs.append(X_fp)
        
        if not omit == "rdkit_desc":
            X_desc = rdkit_desc_featuriser(mols)
            Xs.append(X_desc)
        
        if not omit == "rdkit_frags":
            X_frags = rdkit_frags_featuriser(mols)
            Xs.append(X_frags)


        X_hybrid = np.concatenate(Xs, axis=1)
        return X_hybrid

    if scheme == "morgan_fp":
        return morgan_featuriser(mols)

    elif scheme == "rdkit_desc":
        return rdkit_desc_featuriser(mols)

    elif scheme == "rdkit_frags":
        return rdkit_frags_featuriser(mols)

    elif scheme == "hybrid":
        return hybrid_featuriser(mols)


def select_features(X, y, k = 30) -> tuple[np.array, list[dict]]:
    '''
    Selects k most relevant features in X linked to y, using `sklearn.feature_selection.f_regression`

    Arguments
    ---------
    X: array-like, feature matrix of rdkit descriptors of shape (n_samples, n_features) 
    y: array-like, target values of shape (n_samples,)

    Returns
    ---------
    X_sel: np.array, reduced feature matrix of k most significant rdkit descriptors
    selection_report: list of dict, report of F-statistic and p-values for all seleted descriptors
    '''

    selector = SelectKBest(f_regression, k = k)
    selector.fit_transform(X, y)

    X_sel = selector.transform(X)

    idx = selector.get_support(indices = True)

    descriptors = get_desc_names()

    selected_descriptors = [descriptors[i] for i in idx]
    f_scores = selector.scores_[idx]
    p_values = selector.pvalues_[idx]

    selection_report = []
    for desc, f_score, p_value in zip(selected_descriptors, f_scores, p_values):
        selection_report.append({"Descriptor": desc,
                                  "F-Score": f_score,
                                  "p-value": p_value})

    
    return X_sel, selection_report



def cross_validation(X, y, model: str = "xgboost", endpoint_scaler = None, verbose: bool = False) -> list[dict]:
    '''
    Conducts a 5x5 CV protocol on inputs X and y
    5x5 CV:
    1. splits dataset into 5 equally sized, randomly arranged chunks
    2. evaluates performance with 4 chunks as training data, and 5th chunk as test data
    3. Repeat step 2 with a different chunk as test data each time
    4. Repeat steps 1-3 up to 5 iterations with a new random split each iteration.

    i.e. 5-fold cross validation, 5 times

    Arguments
    ----------
    X: array-like, feature matrix of shape (n_samples, n_features)
    y: array-like, vector containing target endpoints of shape (n_samples,)
    endpoint_scaler: scaler from sklearn.preprocessing, scaler used to transform y datapoints (so it can be reversed)
    verbose: bool, whether or not to print CV metrics when complete

    Returns
    ---------
    fold_metrics: list of dict, report of validation metrics from each fold
    '''

    rkf = RepeatedKFold(n_splits = 5, n_repeats = 5, random_state = 0) #random_state = 0 for reproducibility

    fold_metrics = []

    for fold, idx in enumerate(rkf.split(X), start=1):
        train_idx, test_idx = idx
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        feature_scaler = MaxAbsScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_test = feature_scaler.transform(X_test)

        if model == "xgboost":
            model = XGBRegressor()
        elif model == "MLP":
            model = MLPRegressor(hidden_layer_sizes=(128,128), activation='relu', learning_rate="adaptive")
        
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        if endpoint_scaler:
            pred = endpoint_scaler.inverse_transform(pred.reshape(-1, 1))
            y_test = endpoint_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mse = mean_squared_error(y_test, pred)
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)

        fold_metrics.append({"fold": fold, "mse": mse, "mae": mae, "r2": r2})

    
    if verbose:
        mses = np.array([m["mse"] for m in fold_metrics])
        maes  = np.array([m["mae"]  for m in fold_metrics])
        r2s   = np.array([m["r2"]   for m in fold_metrics])

        print(f"5x5 CV Results:")
        print(f"R^2 : mean={r2s.mean():.3f}  std={r2s.std(ddof=1):.3f}")
        print(f"MAE : mean={maes.mean():.3f}  std={maes.std(ddof=1):.3f}")
        print(f"MSE: mean={mses.mean():.3f}  std={mses.std(ddof=1):.3f}")

    return fold_metrics


def test_model(X_train, X_test, y_train, y_test, model = "xgboost", endpoint_scaler=None) -> tuple[np.array]:
    '''
    Trains and tests a model on pre-split train and test data

    Arguments
    ---------
    X_train: array-like, Training data feature matrix
    X_test: array-like, Test data feature matrix
    y_train: array-like, Training data target values
    y_test: array-like, Test data target values

    Returns
    ---------
    test: np.array, Test data target values (identical to y_test unless scaled)
    pred: np.array, Test data predicted values
    '''

    feature_scaler = MaxAbsScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    if model == "xgboost":
        model = XGBRegressor()
    elif model == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(128,128), activation='relu', learning_rate="adaptive")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if endpoint_scaler:
        # inverse the endpoint scaler
        pred = endpoint_scaler.inverse_transform(pred.reshape(-1, 1))
        test = endpoint_scaler.inverse_transform(y_test.reshape(-1, 1))

    
    return test, pred


#------------------------------------
# DATA VISUALISATION
#------------------------------------


def hist_plot(df: pd.DataFrame, mers_col: str = "pIC50 (MERS-CoV Mpro)", sars_col: str = "pIC50 (SARS-CoV-2 Mpro)", kde: bool = False):
    '''
    Pair of histograms (with boxplots) for MERS and SARS pIC50 endpoints.

    Arguments
    ----------
    df: pd.DataFrame, dataframe containing SARS and MERS pIC50 values
    mers_col: str, df column containing MERS pIC50 values
    sars_col: str, df column containing SARS pIC50 values
    kde: bool, whether to display estimated distribution curves
    '''

    train_color = '#7920d3bb'
    test_color = '#e69f00bb'

    endpoints = [(mers_col, "MERS-CoV Mpro"), (sars_col, "SARS-CoV-2 Mpro")]

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, height_ratios = [4, 1], hspace=0.05, wspace=0.25)

    ax0 = None

    for i, (col, title) in enumerate(endpoints):
        if ax0 is None:
            ax = fig.add_subplot(gs[0,i])
            ax0 = ax
        else:
            ax = fig.add_subplot(gs[0,i], sharey = ax0)

        ax_box = fig.add_subplot(gs[1,i], sharex=ax)

        train = df.loc[df["Set"].eq("Train"), col].dropna()
        test  = df.loc[df["Set"].eq("Test"),  col].dropna()

        all_vals = pd.concat([train, test], ignore_index=True)
        bins = np.histogram_bin_edges(all_vals, bins=20)

        ax.hist(train, bins=bins, label=f"Train (n = {len(train)})", density=True, color = train_color, edgecolor="black", linewidth=0.8)
        ax.hist(test,  bins=bins, label=f"Test (n = {len(test)})", density=True, color = test_color, edgecolor="black", linewidth=0.8)

        if kde:
            x_min = np.min(np.r_[train, test])
            x_max = np.max(np.r_[train, test])
            x = np.linspace(x_min, x_max, 400)

            kde_train = gaussian_kde(train) 
            kde_test  = gaussian_kde(test)

            ax.plot(x, kde_train(x), linewidth=1, color = '#7920d3')
            ax.plot(x, kde_test(x),  linewidth=1, color = '#e69f00')
        
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
        showfliers=False, 
        widths = 0.6)

        for key in ["boxes", "whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(1.0)

        if train_color is not None:
            bp["boxes"][0].set_facecolor(train_color)
        if test_color is not None:
            bp["boxes"][1].set_facecolor(test_color)

        ax_box.set_xlabel("pIC50")
        ax_box.set_yticks([1, 2])
        ax_box.set_yticklabels(["Train", "Test"])
        ax_box.invert_yaxis()

    plt.tight_layout()
    plt.show()

def scatter_plot(y_test, pred, save_to: str = None, title: str = None):
    '''
    Produces a scatter plot of experimental pIC50 values (x-axis) and the predicted pIC50 values (y-axis).

    Arguments
    ----------
    y_test: array-like, experimental pIC50 values
    pred: array-like, predicted pIC50 values
    save_to: str, path to where figure will be saved
    '''
    mse = mean_squared_error(y_test, pred)
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)

    plt.scatter(y_test, pred, marker = 'o', color='#7920d3bb', edgecolors='black', linewidths=0.5)

    lims = [
        min(min(y_test), min(pred)),  # min of both axes
        max(max(y_test), max(pred))   # max of both axes
    ]

    plt.plot(lims, lims, 'k-', alpha=1, label = 'x = y', lw = 1) 
    if title:
        plt.title(f'{title}')
    plt.ylabel('Predicted pIC50 (-Log[mol/L])')
    plt.xlabel('Experimental pIC50 (-Log[mol/L])')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.text(0.02,0.95,f"R² = {r2:.3f}",transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02,0.90,f"MAE = {mae:.3f}",transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(0.02,0.85,f"MSE = {mse:.3f}",transform=plt.gca().transAxes, verticalalignment='top')

    plt.legend()
    if save_to:
        plt.savefig(f"{save_to}")

    plt.show()

def bar_plot(mers_results: pd.DataFrame, sars_results: pd.DataFrame, save_to: str = None):
    '''
    Bar chart for plotting cross-validation results (R^2 and MAE) from testing multiple models and featurisation
    methods.

    Arguments
    ----------
    mers_results: pd.DataFrame, results from MERS CV, df with headers "scheme", "model", "r2", and "mae" 
    sars_results: pd.DataFrame, results from SARS CV, df with headers "scheme", "model", "r2", and "mae"
    save_to: str, path to where figure will be saved
    '''

    fig, axes = plt.subplots(2, 2, figsize = (10, 6), constrained_layout = True, sharey = "row")

    palette = ["#7920d3bb",'#e69f00bb']

    sns.barplot(data = mers_results, x = "scheme", y="r2", hue="model",palette = palette,
                errorbar = 'sd', ax = axes[0, 0], edgecolor = 'black', linewidth=1, err_kws={"linewidth": 0.8},
                capsize=0.2)

    sns.barplot(data = mers_results, x = "scheme", y="mae", hue="model",palette = palette, 
                errorbar = 'sd', ax = axes[1,0], edgecolor = 'black', linewidth=1, err_kws={"linewidth": 0.8},
                capsize=0.2)

    sns.barplot(data = sars_results, x = "scheme", y="r2", hue="model",palette = palette, 
                errorbar = 'sd', ax = axes[0, 1], edgecolor = 'black', linewidth=1, err_kws={"linewidth": 0.8},
                capsize=0.2)

    sns.barplot(data = sars_results, x = "scheme", y="mae", hue="model",palette = palette, 
                errorbar = 'sd', ax = axes[1,1], edgecolor = 'black', linewidth=1, err_kws={"linewidth": 0.8},
                capsize=0.2)

    axes[0, 0].set_title("pIC50 MERS: CV R²")
    axes[0, 1].set_title("pIC50 SARS: CV R²")
    axes[1, 0].set_title("pIC50 MERS: CV MAE")
    axes[1, 1].set_title("pIC50 SARS: CV MAE")

    #axes[0, 0].set_xlabel("Scheme")
    axes[0, 0].set_ylabel("R²")

    #axes[1, 0].set_xlabel("Scheme")
    axes[1, 0].set_ylabel("MAE")

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.tick_params(axis="x")

    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if save_to:
        plt.savefig(f"{save_to}")

    plt.show()

def feature_report(report: list[dict], title: str = None):
    '''
    Horizontal barchart, displaying the F-Score of various descriptors, sorted by most significant

    Arguments
    ----------
    report: list of dict, contains descriptors and their F-statistics (second output from `select_features`)
    title: str, figure title
    '''

    df = pd.DataFrame(report)

    df["F-Score"] = df["F-Score"].astype(float)
    df["p-value"] = df["p-value"].astype(float)

    
    df = df.sort_values("F-Score", ascending=False)

    colors = ['#7920d3bb']*len(df)
    
    fig, ax = plt.subplots(figsize=(6, max(4, 0.2 * len(df))))
    ax.barh(df["Descriptor"], df["F-Score"], edgecolor='black', linewidth=0.8, color = colors)
    ax.invert_yaxis()

    ax.set_xlabel("F-Score")
    ax.set_ylabel("Descriptor")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()