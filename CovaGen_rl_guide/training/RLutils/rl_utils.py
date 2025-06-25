from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors

def calc_qed_score(z,t,scoring_model):
	'''
	Calculate QED score for the latent vector z at timestep t(if given).
	Args:
		z: Tensor, the latent vector
		t: Timestep t.
		scoring_model: The model to give the QED score prediction

	Returns:
		A float QED score.
	'''
	qed_score = scoring_model.predict_qed(z,t)
	return qed_score
