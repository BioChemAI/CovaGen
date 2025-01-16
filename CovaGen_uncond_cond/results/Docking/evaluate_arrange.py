"""说明：这是对liutx的采样10000分子的evaluate的开始。liutx是对crossdock中sbdd划分出的100个test分子进行分别采样100个分子。
采样最后得到一个依顺序的100分子的smiles的pkl文件，由此文件处理成一个json文件，这个json文件可以直接输入到下一个脚本：evaluate.collect_retry.py
进行处理，使用楷豪的含有蛋白质的crossdock数据集，获得符合rFragLi的evaluate流程的核心文件 collect.pkl。注意，evaluate.collect_retry.py中
使用的index参数并非由rFragLi预处理获得的index.json，而是liutx输出出的一个包含蛋白质路径的json文件（也就是说，现在evaluate.collect_retry.py
中的index-json参数不使用了，取而代之的是--index-protein /dataset/crossdock/rotein_filepath.json这一参数）（实际上有了之前经本脚本处理出
的smiles的json，应该也能调整成使用index-json的）。获得了collect.pkl后就可以进行以下的各种evaluate。
docking-score:注意采用v6的docker。
qed_sa:注意运行前先运行evaluate.collect.smiles_db.py来从之前的collect.pkl中获得用于qed_sa脚本的smiles_db.json文件。
retro_star:运行缓慢，注意用song2的rFragLi文件夹中配好的数据和环境运行，且应该和docking_score一起跑。
novelty_diversity:注意！这个需要rFragLi中预处理出的两个文件：index.json和frag_ligand_sbdd.pkl!现在在song2的rFragLi文件夹中的save/下有处理好的。
在获得上面的所有结果后运行evaluate.analysis.overall.py，获得最终汇总的所有指标。
注意evaluate.analysis.overall.py中114行有更改，为两个ref_smiles。运行正常。
diff采样出向量结果-> diff/shuli.py 整理 -> diff/decing.py 解码 -> rFragLi/pkltodicjson.py 再次整理 -> evaluate.collect_retry.py获得核心数据
-> 一系列指标 -> overall得到结果。
"""

#default save_pth /workspace/codeeval/smi_toeval/lastbeforecollect/bef_rnnattn_clip_100000.pkl

#对于一部分的sample那我就先decode再跑这个？
import pickle
import json
import argparse
from rdkit import Chem



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis.')
    parser.add_argument("--decoded",type=str)
    parser.add_argument("--save_pth",type=str)
    args = parser.parse_args()
    # with open("/dataset/crossdock/protein_filepath.pkl","rb") as f1:
    #     k=pickle.load(f1)
    with open("/workspace/codes/Docking/testdata_path/protein_filepath.pkl","rb") as f1:
        pro_pth=pickle.load(f1)
    with open(args.decoded,"rb") as f1:
        smis = pickle.load(f1)
    lst100 = []
    lstall = []
    for i in range(0,len(smis),100):
        smi_invalid = smis[i:i+100]#
        smi_valid = []
        for j in smi_invalid:
            mol = Chem.MolFromSmiles(j)
            if mol is not None:
                smi_valid.append(j)
        lenval = len(smi_valid)
        print(f"{i}th, valid amount:",lenval)
        lst100.append(smi_valid)
    len2 = len(lst100)
    print(len2)

    #for esm, this is the current removal
    # indices_to_remove = [37, 64, 87]
    # my_list = [lst100[i] for i in range(len(lst100)) if i not in indices_to_remove]

    for i,data in enumerate(lst100):
        dic = {}
        # protein_pocket_name = index_path.split(".")[0][:-9]
        pro_now = pro_pth[i][0]
        protein_pocket_name = pro_now.split(".")[0][:-9]
        dirname = protein_pocket_name.split("/")[0]
        codename = protein_pocket_name.split("/")[1]
        dic["code"] = codename
        dic["smiles"] = data
        dic["dirfirst"] = dirname
        lstall.append(dic)
    with open(args.save_pth,"w") as f2:
        json.dump(lstall,f2)
    print("Done")