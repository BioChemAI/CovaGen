import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch_geometric.nn import NNConv, Set2Set
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,r2_score
import matplotlib.pyplot as plt

csv1 = pd.read_csv('/workspace/codes/toxicity/Hepato.csv')
# with open()
csv1 = csv1.dropna(subset=['Canonical SMILES'])
smi_list=csv1['Canonical SMILES'].tolist()
labels=csv1['Toxicity Value'].tolist()
cleaned_list = [x for x in smi_list if isinstance(x,str)]
smiles_list = [smiles.split('.')[0] for smiles in cleaned_list]
smiles = pd.Series(smiles_list)
ys  = np.array(labels)

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"Br", "C", "Cl", "F", "H", "I", "N", "O"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

class MoleculesDataset(InMemoryDataset):
	def __init__(self, root, transform=None):
		super(MoleculesDataset, self).__init__(root, transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return 'data.csv'

	@property
	def processed_file_names(self):
		return 'data.pt'

	def download(self):
		# Download to `self.raw_dir`.
		pass

	def process(self):
		datas = []
		for smile, y in zip(smiles, ys):
			mol = Chem.MolFromSmiles(smile)
			mol = Chem.AddHs(mol)

			embeddings = []
			for atom in mol.GetAtoms():
				embeddings.append(atom_featurizer.encode(atom))
			embeddings = torch.tensor(embeddings, dtype=torch.float32)

			edges = []
			edge_attr = []
			for bond in mol.GetBonds():
				edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
				edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

				edge_attr.append(bond_featurizer.encode(bond))
				edge_attr.append(bond_featurizer.encode(bond))

			edges = torch.tensor(edges).T
			edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

			y = torch.tensor(y, dtype=torch.long)

			data = Data(x=embeddings, edge_index=edges, y=y, edge_attr=edge_attr)
			datas.append(data)

		# self.data, self.slices = self.collate(datas)
		torch.save(self.collate(datas), self.processed_paths[0])
max_nodes = 128
dataset = MoleculesDataset(root= "data")

train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,valid_size, test_size],generator=torch.Generator().manual_seed(1))


test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# for i in train_loader:
#     print(i.edge_attr.shape)

class NNConvNet(nn.Module):
    def __init__(self,node_feature_dim, edge_feature_dim, edge_hidden_dim):
        super(NNConvNet,self).__init__()
        #第一层
        edge_network1 = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_feature_dim * node_feature_dim)
        )
        self.nnconv1 = NNConv(node_feature_dim, node_feature_dim, edge_network1, aggr="mean")

        # # 第二层
        # edge_network2 = nn.Sequential(
        #     nn.Linear(edge_feature_dim, edge_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(edge_hidden_dim, 32 * 16)
        # )
        # self.nnconv2 = NNConv(32, 16, edge_network2, aggr="mean")


        self.relu = nn.ReLU()
        self.set2set = Set2Set(24, processing_steps=3)
        self.fc2 = nn.Linear(2 * 24, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self,data):
        x, edge_index, edge_attr,batch = data.x, data.edge_index, data.edge_attr,data.batch
        x = self.nnconv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.nnconv1(x, edge_index, edge_attr)
        x = self.set2set(x, batch)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


node_feature_dim, edge_feature_dim, edge_hidden_dim = 24 , 6 ,32

# conv = NNConvNet(node_feature_dim, edge_feature_dim, edge_hidden_dim)
#
# for i in train_loader :
#     print(conv(i))

num = 100
lr = 0.005

model = NNConvNet(node_feature_dim, edge_feature_dim, edge_hidden_dim)


optimizer = torch.optim.SGD(model.parameters(),lr)
criterion = torch.nn.CrossEntropyLoss()

#loss
tra_loss = []

tra_acc = []


val_acc = []



for e in tqdm(range(num)):
    print('Epoch {}/{}'.format(e + 1, num))
    print('-------------')
    model.train()
    epoch_loss = []

    train_total = 0
    train_correct = 0

    train_preds = []
    train_trues = []
    #train
    for data in train_loader:
        y = data.y
        optimizer.zero_grad()
        out = model(data)

        # print(y.dtype, out.dtype)

        loss = criterion(out, y)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.data)

        _, predict = torch.max(out.data, 1)
        train_total += y.shape[0] * 1.0
        train_correct += int((y == predict).sum())

        #评估指标
        train_outputs = out.argmax(dim=1)

        train_preds.extend(train_outputs.detach().cpu().numpy())
        train_trues.extend(y.detach().cpu().numpy())

    epoch_loss = np.average(epoch_loss)
    # scheduler.step(epoch_loss)

    # early_stopping(epoch_loss, model)
    # # 若满足 early stopping 要求
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     # 结束模型训练
    #     break

    #f1_score
    sklearn_f1 = f1_score(train_trues, train_preds)
    print('train Loss: {:.4f} Acc:{:.4f} f1:{:.4f}'.format(epoch_loss,train_correct/train_total,sklearn_f1))
    tra_loss.append(epoch_loss)
    tra_acc.append(train_correct/train_total)
    print(confusion_matrix(train_trues, train_preds))

    #----------------------------------------------------------valid------------------------------------------------------------
    correct = 0
    total = 0

    valid_preds = []
    valid_trues = []

    with torch.no_grad():
        model.eval()
        for data in val_loader:
            labels = data.y
            outputs = model(data)


            _, predict = torch.max(outputs.data, 1)
            total += labels.shape[0] * 1.0
            correct += int((labels == predict).sum())

            valid_outputs = outputs.argmax(dim=1)
            valid_preds.extend(valid_outputs.detach().cpu().numpy())
            valid_trues.extend(labels.detach().cpu().numpy())

        sklearn_f1 = f1_score(valid_trues, valid_preds)

        print('val Acc: {:.4f} f1:{:.4f}'.format(correct / total,sklearn_f1))
        val_acc.append(correct / total)

        print(confusion_matrix(valid_trues, valid_preds))


#test
correct = 0
total = 0
test_preds = []
test_trues = []
with torch.no_grad():
    model.eval()
    for data in test_loader:

        labels = data.y
        outputs = model(data)

        _, predict = torch.max(outputs.data, 1)
        total += labels.shape[0] * 1.0
        correct += int((labels == predict).sum())

        test_outputs = outputs.argmax(dim=1)
        test_preds.extend(test_outputs.detach().cpu().numpy())
        test_trues.extend(labels.detach().cpu().numpy())

    sklearn_f1 = f1_score(test_trues, test_preds)

    print('test Acc: {:.4f} f1:{:.4f}'.format(correct/total,sklearn_f1))





plt.plot(tra_loss, label="Train Loss")
plt.plot(val_acc, label="Val acc")
plt.plot(tra_acc, label="Train acc")
# plt.yscale('log')
plt.legend()
plt.show()