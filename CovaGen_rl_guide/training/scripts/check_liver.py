import pickle
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

def evaluate_generated_molecules(smiles_list, model_path="xgb.pkl"):
    # Helper: 转换成 RDKit MACCS 指纹
    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)

    smiles_list = list(set(smiles_list))
    # 1. 处理 SMILES → 指纹（分类器用 numpy，Tanimoto 用 RDKit 原指纹）
    fps_rdkit = []
    fps_np = []
    valid_indices = []
    valid_smiles = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps_rdkit.append(fp)
            arr = np.array([int(fp[i]) for i in range(1, fp.GetNumBits())], dtype=int)
            fps_np.append(arr)
            valid_indices.append(i)
            valid_smiles.append(smi)

    if not fps_np:
        raise ValueError("无有效分子，无法评估。")

    # 2. 加载模型
    with open("/workspace/codes/lddd_ddpo/guided_diffusion/rf_maccs_model2.pkl", "rb") as f:
        model = pickle.load(f)


    # 3. 预测毒性
    X = np.stack(fps_np)
    y_pred = model.predict(X)
    non_toxic_ratio = (y_pred == 0).mean()

    # 4. 多样性计算（基于 Tanimoto）
    def compute_tanimoto_diversity(fps):
        n = len(fps)
        if n < 2:
            return 0.0
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                sims.append(sim)
        return 1.0 - np.mean(sims)

    diversity_score = compute_tanimoto_diversity(fps_rdkit)
    unique_smiles_count = len(set(valid_smiles))
    total_valid = len(valid_smiles)
    uniqueness_ratio = unique_smiles_count / total_valid

    # 输出
    print(f"有效分子数量：{total_valid}")
    print(f"无毒预测比例：{non_toxic_ratio:.3f}")
    print(f"多样性得分（1 - 平均 Tanimoto）：{diversity_score:.3f}")
    print(f"唯一性得分（unique SMILES 比例）：{uniqueness_ratio:.3f}")

    return non_toxic_ratio, diversity_score, uniqueness_ratio

with open("/workspace/codeeval/trysave_ddpo_uncond/base_notox", "rb") as f:
# with open("/workspace/codeeval/trysave_ddpo_uncond/hepato_noguide_test_norl", "rb") as f:
# with open("/workspace/codeeval/trysave_ddpo_uncond/1725_fulllm.pkl", "rb") as f:
    raw = pickle.load(f)
nls = [smi for sublist in raw for smi in sublist]

evaluate_generated_molecules(nls, model_path="xgb.pkl")