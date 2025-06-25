'''
A unified method to calculate reconstruction accuracy for arbitrary (char) VAE model
This would request straight up decode method and encode method for the VAE.
outline: input string - vae - output string
'''

def reconstruct(vae,smiles):
	'''
	Takes in a list of smile strings, output their reconstructions.
	Args:
		vae: VAE model
		smiles: A list of smile strings

	Returns:
		recons: Reconstructed smiles
	'''

	recons = 1
	return recons
