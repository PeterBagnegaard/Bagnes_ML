# -*- coding: utf-8 -*-
# Basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
# Select best features
from sklearn.feature_selection import SelectKBest, f_classif
# Processing 
from sklearn.preprocessing import StandardScaler
# Optimization algorithms
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
# Machine learning algorithms
from sklearn.decomposition import PCA 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



def get_allowed_headers():
    return ['actualInteractionsPerCrossing', 'averageInteractionsPerCrossing', 'correctedActualMu', 'correctedAverageMu', 'correctedScaledActualMu', 'correctedScaledAverageMu', 'NvtxReco', 'p_nTracks', 'p_pt_track', 'p_eta', 'p_phi', 'p_charge', 'p_qOverP', 'p_z0', 'p_d0', 'p_sigmad0', 'p_d0Sig', 'p_EptRatio', 'p_dPOverP', 'p_z0theta', 'p_etaCluster', 'p_phiCluster', 'p_eCluster', 'p_rawEtaCluster', 'p_rawPhiCluster', 'p_rawECluster', 'p_eClusterLr0', 'p_eClusterLr1', 'p_eClusterLr2', 'p_eClusterLr3', 'p_etaClusterLr1', 'p_etaClusterLr2', 'p_phiClusterLr2', 'p_eAccCluster', 'p_f0Cluster', 'p_etaCalo', 'p_phiCalo', 'p_eTileGap3Cluster', 'p_cellIndexCluster', 'p_phiModCalo', 'p_etaModCalo', 'p_dPhiTH3', 'p_R12', 'p_fTG3', 'p_weta2', 'p_Reta', 'p_Rphi', 'p_Eratio', 'p_f1', 'p_f3', 'p_Rhad', 'p_Rhad1', 'p_deltaEta1', 'p_deltaPhiRescaled2', 'p_TRTPID', 'p_TRTTrackOccupancy', 'p_numberOfInnermostPixelHits', 'p_numberOfPixelHits', 'p_numberOfSCTHits', 'p_numberOfTRTHits', 'p_numberOfTRTXenonHits', 'p_chi2', 'p_ndof', 'p_SharedMuonTrack', 'p_E7x7_Lr2', 'p_E7x7_Lr3', 'p_E_Lr0_HiG', 'p_E_Lr0_LowG', 'p_E_Lr0_MedG', 'p_E_Lr1_HiG', 'p_E_Lr1_LowG', 'p_E_Lr1_MedG', 'p_E_Lr2_HiG', 'p_E_Lr2_LowG', 'p_E_Lr2_MedG', 'p_E_Lr3_HiG', 'p_E_Lr3_LowG', 'p_E_Lr3_MedG', 'p_ambiguityType', 'p_asy1', 'p_author', 'p_barys1', 'p_core57cellsEnergyCorrection', 'p_deltaEta0', 'p_deltaEta2', 'p_deltaEta3', 'p_deltaPhi0', 'p_deltaPhi1', 'p_deltaPhi2', 'p_deltaPhi3', 'p_deltaPhiFromLastMeasurement', 'p_deltaPhiRescaled0', 'p_deltaPhiRescaled1', 'p_deltaPhiRescaled3', 'p_e1152', 'p_e132', 'p_e235', 'p_e255', 'p_e2ts1', 'p_ecore', 'p_emins1', 'p_etconeCorrBitset', 'p_ethad', 'p_ethad1', 'p_f1core', 'p_f3core', 'p_maxEcell_energy', 'p_maxEcell_gain', 'p_maxEcell_time', 'p_maxEcell_x', 'p_maxEcell_y', 'p_maxEcell_z', 'p_nCells_Lr0_HiG', 'p_nCells_Lr0_LowG', 'p_nCells_Lr0_MedG', 'p_nCells_Lr1_HiG', 'p_nCells_Lr1_LowG', 'p_nCells_Lr1_MedG', 'p_nCells_Lr2_HiG', 'p_nCells_Lr2_LowG', 'p_nCells_Lr2_MedG', 'p_nCells_Lr3_HiG', 'p_nCells_Lr3_LowG', 'p_nCells_Lr3_MedG', 'p_pos', 'p_pos7', 'p_poscs1', 'p_poscs2', 'p_ptconeCorrBitset', 'p_ptconecoreTrackPtrCorrection', 'p_r33over37allcalo', 'p_topoetconeCorrBitset', 'p_topoetconecoreConeEnergyCorrection', 'p_topoetconecoreConeSCEnergyCorrection', 'p_weta1', 'p_widths1', 'p_widths2', 'p_wtots1', 'p_e233', 'p_e237', 'p_e277', 'p_e2tsts1', 'p_ehad1', 'p_emaxs1', 'p_fracs1', 'p_DeltaE', 'p_E3x5_Lr0', 'p_E3x5_Lr1', 'p_E3x5_Lr2', 'p_E3x5_Lr3', 'p_E5x7_Lr0', 'p_E5x7_Lr1', 'p_E5x7_Lr2', 'p_E5x7_Lr3', 'p_E7x11_Lr0', 'p_E7x11_Lr1', 'p_E7x11_Lr2', 'p_E7x11_Lr3', 'p_E7x7_Lr0', 'p_E7x7_Lr1']

class Bagnes_ML:
    
    def __init__(self, K=13, fraction_of_dataset = 1):
        """
        Get data from file
        Select important headers
        Preprocess data
        
        """      
        self.get_collision_data(fraction_of_dataset)
        self.get_k_best_headers(K)
        self.Preprosessing()

    def read_from_file(self, string):
        if string == "train":
            file_location = "\\Users\\Bagne\\Desktop\\Big Data Analysis\\Small_Project\\Data\\train.h5"
        elif string == "test":
            file_location = "\\Users\\Bagne\\Desktop\\Big Data Analysis\\Small_Project\\Data\\test.h5"
        with h5py.File(file_location, "r") as hf :
            data            = hf[string][:]
        return pd.DataFrame(data)   

    def get_collision_data(self, fraction_of_dataset):  
        """
        Returns X, y, X_unknown  ...  (no y_test!)
        """
        trainingset = self.read_from_file("train").sample(frac = fraction_of_dataset)
        self.X_verification = self.read_from_file("test")
        self.X, self.y= trainingset[get_allowed_headers()], trainingset[['Truth', 'p_truth_E']]#, trainingset['Truth'], trainingset['p_truth_E']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.y_train_class, self.y_test_class = self.y_train['Truth'], self.y_test['Truth']
        self.y_train_reg, self.y_test_reg = self.y_train['p_truth_E'], self.y_test['p_truth_E']
    
    def get_k_best_headers(self, K):    
        selector = SelectKBest(f_classif, k=K)
        selector.fit(self.X, self.y['Truth']) 
        best_features = np.array(get_allowed_headers())[selector.get_support(indices=True)]
        self.X = self.X[best_features]
    
    def Preprosessing(self):
        """
        X_preprosess = (X - u) / s
        where u is the mean of the training samples and 
        s is the standard deviation of the training samples
        Removes p_truth_E from y
        """
        self.X_preprosess = StandardScaler().fit_transform(self.X)

    def Principle_Components(self, n_components = 40, show_evr = False):
        self.pca = PCA(n_components=n_components)
        self.X_pc = self.pca.fit_transform(self.X_train)     




    def DecisionTree_CrossVal(self, max_depth, min_samples_leaf):
        """
        
        """
        
        estimator = DecisionTreeClassifier(max_depth=int(max_depth), 
                                           min_samples_leaf=int(min_samples_leaf))
                                           #random_state=42,)
        
        cval = cross_val_score(estimator, self.X_train, self.y_train_class, scoring='accuracy', cv=self.n_folds)
        
        return cval.mean()
    
    def optimize_DecisionTree(self, pars, return_type='parameters', n_folds=5, n_iter=5, verbose = 2, init_points=3):
        """
        Apply Bayesian Optimization to Decision Tree parameters
        pars: dict of parameter constraints
        return_type: can be 'parameters' or 'trained model'
        n_folds: folds in cross validation     
        n_iter: simulation steps
        init_points: initial random simulation steps
        """
        
        self.n_folds = n_folds # The class needs to know this for BayesianOptimization to work
        
        if estimator_type == 'decision tree':
            f = self.DecisionTree_CrossVal
            estimator = DecisionTreeClassifier
        
        optimizer = BayesianOptimization(f=f,#self.DecisionTree_CrossVal, 
                                         pbounds=pars, 
                                         #random_state=42, 
                                         verbose=verbose)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
        if return_type == 'parameters':
            return optimizer.max
        elif return_type == 'trained model':
            clf = estimator(best_params.max['params'])
            return clf.fit(self.X_train, self.y_train_class)


#%%
"""
Instantiate Machine Learning object
"""
plt.close('all')
ml = Bagnes_ML(10, 0.1)
#%%
parameters_BayesianOptimization = {"max_depth": (1, 100), "min_samples_leaf": (1, 100)}

best_params = ml.optimize_DecisionTree(parameters_BayesianOptimization)



#%%

# In[25]:


parameters_BayesianOptimization = {"max_depth": (1, 100), "min_samples_leaf": (1, 100)}

Bayesian_Optimization = optimize_DecisionTree(ml.X_train, ml.y_train_class, parameters_BayesianOptimization, n_iter=5)

best_param = Bayesian_Optimization.max['params']
score = Bayesian_Optimization.max['target']

print(best_param)
print(score)







#%%
"""
Optimize hyperparameters 
using Grid Search
"""
# Imputs
from sklearn.neighbors import KNeighborsClassifier
n_params = np.arange(1, 20)

# Inside function
n_scores = list()
n_scores_err = list()

for n in n_params:
    estimator = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(estimator, ml.X, ml.y_class, cv = 10)
    n_scores.append(scores.mean())
    n_scores_err.append(scores.std() * 2)
    print(n, "of", n_params[-1])


#%%
plt.errorbar(n_params, n_scores, n_scores_err, marker='o', lw=5)
plt.title("Hyperparameter optimization KN classifier")
plt.xlabel("k: nbr of neighbours")
plt.ylabel("Cross validation score")






