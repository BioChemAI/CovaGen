import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from  rdkit import Chem
from rdkit.Chem import AllChem
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import MinMaxScaler
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('/workspace/codes/toxicity/tpot_tox.csv', sep='COLUMN_SEPARATOR', dtype=np.float64)
csv1 = pd.read_csv('/workspace/codes/toxicity/Acute_Toxicity_mouse_intraperitoneal_LD50.csv')
# with open()
csv1 = csv1.dropna(subset=['Canonical SMILES'])
smi_list=csv1['Canonical SMILES'].tolist()
tag_list=np.array(csv1['mouse_intraperitoneal_LD50'].tolist())
cleaned_list = [x for x in smi_list if isinstance(x,str)]
cleaned_list = [smiles.split('.')[0] for smiles in cleaned_list]
molecules = [Chem.MolFromSmiles(smiles) for smiles in cleaned_list]
	#
	# 计算MACCS指纹
maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules]
fp_array = np.array([list(fp) for fp in maccs_fps])
np.save('./data/tox_maccs_arr.npy',fp_array)
np.save('./data/tox_maccs_tag.npy',tag_list)

# features = np.load
# features = tpot_data.drop('target', axis=1)
# features = np.array(features['features'].tolist())
# targ = np.array(tpot_data['target'])
training_features, testing_features, training_target, testing_target = \
            train_test_split(fp_array, tag_list, random_state=None)

# Average CV score on the training set was: -0.20702482038930295
# exported_pipeline = make_pipeline(
#     RobustScaler(),
#     ExtraTreesRegressor(bootstrap=False, max_features=0.25, min_samples_leaf=1, min_samples_split=5, n_estimators=100,random_state=114)
# )
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=5, min_samples_split=2, n_estimators=100)),
    MinMaxScaler(),
    RandomForestRegressor(bootstrap=True, max_features=0.9500000000000001, min_samples_leaf=2, min_samples_split=3, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(testing_target, results)
r2 = r2_score(testing_target, results)
# with open('tpot_toxicity_best_model.pkl', 'wb') as file:
#     pickle.dump(exported_pipeline, file)
print('done')