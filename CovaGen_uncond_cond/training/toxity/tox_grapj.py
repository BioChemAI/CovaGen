from rdkit import Chem
import networkx as nx
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch.utils.data.dataloader import default_collate
def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        return Batch.from_data_list(batch)
    return default_collate(batch)

class GraphClassifier(nn.Module):
	def __init__(self, in_channels, hidden_channels, num_classes):
		super(GraphClassifier, self).__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, num_classes)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1)


# Create the model
in_channels = 1  # Number of atom features (e.g., atomic number)
hidden_channels = 64
num_classes = 2  # Binary classification
model = GraphClassifier(in_channels, hidden_channels, num_classes)

def smiles_to_molecule(smiles):
	molecule = Chem.MolFromSmiles(smiles)
	return molecule


def molecule_to_graph(molecule):
	num_atoms = molecule.GetNumAtoms()
	atom_features = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
	adjacency_matrix = Chem.GetAdjacencyMatrix(molecule)
	return atom_features, adjacency_matrix


def build_graph(atom_features, adjacency_matrix):
	graph = nx.Graph()
	num_atoms = len(atom_features)

	for i in range(num_atoms):
		graph.add_node(i, features=atom_features[i])

	for i in range(num_atoms):
		for j in range(i + 1, num_atoms):
			if adjacency_matrix[i, j] > 0:
				graph.add_edge(i, j)

	return graph


def graph_to_pytorch_data(graph):
	edge_index = torch.tensor(list(graph.edges)).t().contiguous()
	x = torch.tensor([graph.nodes[node]['features'] for node in graph.nodes], dtype=torch.float)

	data = Data(x=x, edge_index=edge_index)
	return data





class MolecularDataset(Dataset):
	def __init__(self, smiles_list, labels):
		self.smiles_list = smiles_list
		self.labels = labels

	def __len__(self):
		return len(self.smiles_list)

	def __getitem__(self, idx):
		smiles = self.smiles_list[idx]
		label = self.labels[idx]
		data = smiles_to_graph(smiles)
		# molecule = smiles_to_molecule(smiles)
		# atom_features, adjacency_matrix = molecule_to_graph(molecule)
		# graph = build_graph(atom_features, adjacency_matrix)
		# data = graph_to_pytorch_data(graph)
		return data, label

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    atom_features = []

    # Get atom features
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    # Get bond information
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # Assume undirected graph
        edge_attr.append(bond.GetBondTypeAsDouble())
        edge_attr.append(bond.GetBondTypeAsDouble())  # Duplicate for undirected edge

    atom_features = torch.tensor(atom_features, dtype=torch.float32).reshape(1,-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_atoms)

    return data

def train_epoch(model, dataloader, optimizer, criterion):
	model.train()
	total_loss = 0.0
	for data, label in dataloader:
		data = data.to(device)
		label = label.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
	return total_loss / len(dataloader)
if __name__ == '__main__':
	csv1 = pd.read_csv('/workspace/codes/toxicity/Hepato.csv')
	# with open()
	csv1 = csv1.dropna(subset=['Canonical SMILES'])
	smi_list=csv1['Canonical SMILES'].tolist()
	labels=csv1['Toxicity Value'].tolist()
	cleaned_list = [x for x in smi_list if isinstance(x,str)]
	smiles_list = [smiles.split('.')[0] for smiles in cleaned_list]
# Example dataset
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	criterion = nn.NLLLoss()
	dataset = MolecularDataset(smiles_list, labels)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

	# Train the model
	num_epochs = 10
	for epoch in range(num_epochs):
		loss = train_epoch(model, dataloader, optimizer, criterion)
		print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
