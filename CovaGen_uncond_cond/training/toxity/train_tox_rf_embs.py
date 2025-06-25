import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


with open('/workspace/codes/toxicity/acute_emb_jingmai.pkl','rb')as f:
	data_list = pickle.load(f)
with open('/workspace/codes/toxicity/acute_value_jingmai.pkl','rb')as f:
	label_list = pickle.load(f)

if __name__ == '__main__':

# 划分训练集和测试集
#     csv1 = pd.read_csv('/workspace/codes/toxicity/Hepato.csv')
#         # with open()
#     csv1 = csv1.dropna(subset=['Canonical SMILES'])
#     data_list=csv1['Canonical SMILES'].tolist()
#     cleaned_list = [x for x in data_list if isinstance(x,str)]
#     cleaned_list = [smiles.split('.')[0] for smiles in cleaned_list]
#     label_list=csv1['Toxicity Value'].tolist()
#     hot_list = []
#     for i in data_list:
#         hot = smiles_to_one_hot(i).flatten()
#         hot_list.append(hot)
	new_data_list = []
	for i in data_list:
		new_data_list.append(np.array(i))
	X_train, X_test, y_train, y_test = train_test_split(new_data_list, label_list, test_size=0.1, random_state=42)

	# 创建随机森林模型
	clf = RandomForestClassifier(n_estimators=1000, random_state=42)

	# 在训练集上训练模型
	clf.fit(X_train, y_train)

	# 在测试集上进行预测#
	y_pred = clf.predict(X_test)

	# 计算准确率
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {accuracy:.2f}")
	# with open('/workspace/codes/toxicity/rf_hot.pkl', 'wb') as model_file:
	#     pickle.dump(clf, model_file)