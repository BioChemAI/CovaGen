import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('/workspace/codes/toxicity/hepato_emb.pkl','rb')as f:
	data_list = pickle.load(f)
with open('/workspace/codes/toxicity/hepato_tag.pkl','rb')as f:
	label_list = pickle.load(f)
# 划分训练集和测试集
# 准备数据（假设你的数据和标签分别是data_list和label_list）
X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.2, random_state=42)

# 转换数据为PyTorch的Tensor
# X_train_tensor = torch.FloatTensor(X_train)
# y_train_tensor = torch.FloatTensor(y_train)
# X_test_tensor = torch.FloatTensor(X_test)
# y_test_tensor = torch.FloatTensor(y_test)

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self,input_dim,hid_dim):
        super().__init__()
        self.qed_sa = nn.Sequential(nn.Linear(input_dim,hid_dim),
                                 nn.ELU(),
                                 nn.BatchNorm1d(hid_dim),
                                 nn.Linear(hid_dim,32),
                                 nn.ELU(),
                                 nn.BatchNorm1d(32),
                                 ###
                                 nn.Linear(32,2),
                                 nn.ReLU()
                                 )
    def forward(self,smile_vec):
        out = self.qed_sa(torch.tensor(smile_vec))
        return out

model = Net(input_dim=128,hid_dim=64)

# 定义损失函数和优化
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
	optimizer.zero_grad()
	outputs = model(X_train)
	y_train = torch.tensor(y_train)
	loss = F.cross_entropy(outputs, y_train, reduction="none")
	loss.backward()
	optimizer.step()

# 在测试集上进行预测
with torch.no_grad():
	y_pred_tensor = model(X_test)
	y_pred = (y_pred_tensor > 0).numpy()

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")