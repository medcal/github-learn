# github-learn
you can learn from it
sk-oITQFs32lAPbTc5IE53a82F2F3174324A4A7521559F9B721
用中文
user@ubuntu:~$ ssh dev
Warning: Permanently added '[115.190.90.101]:39318' (ED25519) to the list of known hosts.
Last login: Sat Oct 25 01:01:42 2025 from 125.35.71.202
root@di-20251023145617-qqh76:~# cd ..
root@di-20251023145617-qqh76:/# ls
bin   ebs   lib    libx32      mnt  proc  sbin  tmp  vepfs
boot  etc   lib32  media       nix  root  srv   usr  vepfs-readonly
dev   home  lib64  mlplatform  opt  run   sys   var  workspace
root@di-20251023145617-qqh76:/# cd vepfs-readonly
root@di-20251023145617-qqh76:/vepfs-readonly# cd problem2

我有一台虚拟开发及，有一张A100的显卡，显存为80G，用sshdev访问
完成第二题
蛋白质溶解性预测
给定一个已经训练好的ESM-100M的预训练模型（checkpoint）
在此基础上构建蛋白质溶解的完整微调过程
模型的输入为蛋白的氨基酸序列，输出为溶解性序列
定义为2分类，可容不可溶解
数据集已经准备并划分完毕
提供标准化接口
from solubilitydataset import ProtrinSolubilityLMDBDataset
ds_train,ds_valid,ds_test =ProtrinSolubilityLMDBDataset.load(lmdb_path)
每个样本包含sequence与对应的溶解性label
数据划分为training valid test
微调时，要调用ESM-100M的序列编码器，将每个蛋白质序列转化为隐藏表示，在通过池化层如mean-pool，或cls表征获得全局特征，输入到一个分类或回归头预测
ESM使用是可从vepfs-readonly/problem2/ESM-checkpoint/加载模型权重，通过自带的alphabet对象获得tokenizeer和批处理转换器batch_converter
后者可以将蛋白质列表转为模型输入的token张量
下面展示加载esm并提取表征
cpkt_path为需要load的文件路径，esm_model为建在权重的模型
alphabet为esm-2的词表
import esm
import torch
from pathlib import path
## load the checkpoint
data = torch.load(ckpt_path,map_location="cpu")
model_name = path(ckpt_path).stem
# built
esm_model,alpabet = esm.pretrained.load_model_and_alphabet_core(
model_name,data,none
)
下面展示esm2的数据格式转化
batch_converter = alphabet.get_batch_converter()
data =[
("protein","MKTVRQERLKSIVRILERSKEPVSGAQ..."),
("protein","MKTVRQERLKSIVRILERSKEPVSGAQ..."),
]
banch_labels,banch_strs,banch_tokens = batch_converter(data)
下面展示提取esm-2的特征向量
results =model(banch_tokens,repr_layers[30],return_contacts=True)
token_representations =results[''representation''][30]
token_representations的形状为[banch_size,seq_len,hidden_dim]表示每个氨基酸的向量表征
在下游任务中氨基酸的向量表征平均池化，用cls token表征全局，在输入到分类或回归头中进行微调

莫新微调
输入：用ESM-100M的tokenizer/embedding
2表征汇聚cls向量
3,任务头，回归或分类
优化器adamW
optional混合精度：amp下做前向与反转；unscale后再梯度裁减记录是否出现inf/Nan
指标
二分类accuracy

文件描述
位于/vepfs-readonly/ problem2目录下
esm的repo已经下载好，在/vepfs-readonly/ problem2/esm/
solubility_mutetest.lmdb:数据文件夹
ESM-checkpoint/：ESM预训练模型权重

本题要提交一个存储测试及结果的文件在/vepfs/ problem2/目录下
提交以下格式的json，民命为test_result.json.
在json的test_result字段中，key是蛋白名称，可以从提供的dataset中获取
fasta sequence为预测蛋白的氨基酸序列，pred label是模型训练的溶解性
{
''name'':''yourname",
''test_result'':{
''protein_name'':[''fasta squence'',pre label]
}
}
2,一份readme.me的说明文档
3,所有用到的代码文件
4,训练日至train。log




我有一台虚拟开发及，有一张A100的显卡，显存为80G，用sshdev访问
先完成第一题，problem1
vepfs-readonly为止读，在Vepfs上读写，存储文件

第一体内容是基于大模型的UI界面元素定位输入指令，可以输出空间位置bbox
基于Qwen3-VL-4B-Instruct
可输入图片文本
输出文本
训练数据
没有微调的模型定位效果一般，希望通过微调提升
用开元数据集UGROUND
我们用小的UGROUND子集
可以找load_example.py查看训练数据
评测指标
有ios】windows等平台的测试及包含截图，指令，和bbox
也可以找load_example.py查看测试数据
用point accuracy评价指标
任务步奏
python eval_qwen_v1_uground.py
--model ../Qwen3-VL-4B-Instruct\
--dataset ../uground-test-data\
--output /vepfs/before_finetune.jsonl
用脚本加载模型，数据上测评，将结果保存到problem的jsonl上
2,在训练脚本train_qwen_v1_uground.py中用UGROUND





用中文
user@ubuntu:~$ ssh dev
Warning: Permanently added '[115.190.90.101]:39318' (ED25519) to the list of known hosts.
Last login: Sat Oct 25 01:01:42 2025 from 125.35.71.202
root@di-20251023145617-qqh76:~# cd ..
root@di-20251023145617-qqh76:/# ls
bin   ebs   lib    libx32      mnt  proc  sbin  tmp  vepfs
boot  etc   lib32  media       nix  root  srv   usr  vepfs-readonly
dev   home  lib64  mlplatform  opt  run   sys   var  workspace
root@di-20251023145617-qqh76:/# cd vepfs-readonly
root@di-20251023145617-qqh76:/vepfs-readonly# cd problem3

我有一台虚拟开发及，有一张A100的显卡，显存为80G，用sshdev访问
完成第三题
有机化合物机器学习力场训练
用模型EquiformerV2学习有机分子
阿司匹林的分子运动规律，学习在看到新分子时，快速预测他门的物理特性，机器学习力场mlff
通过训练模型，让AI预测分子在不同构型下的能量和各个原子的受力
输入：一个阿司匹林的分子，包括分子中每个原子的元素种类（用原子序数表示），以及每个原子位置的三维空间坐标
输出分子中每个原子的受力（每个原子受力是一个三维数组），以及分子整体的能量

基础模型
用EquiformerV2，不提供权重
训练和测试数据
10000个阿司匹林的分子存在aspirin_train.npz
另外分别10000个作验证和测试
在aspirin_valid.npz    aspirin_test.npz 
可以用inspect_aspirin_example.py脚本查看数据集中样本
python inspect_aspirin_example.py  \
-- npz aspirin_train.npz\
--index 1000
评价指标
mae

任务步奏
1.实现损失函数，在训练脚本train_equiformer_v2_md17_aspirin.py中
完善energyforceloss类的实现设计损失，在energyforceloss类的forward方法中计算lossenergy能量损失和受力损失
loss总=w1*loss1+w2*loss2
w权重通过--we和wf确定
2.运行训练脚本制定训练集aspirin_train.npz训练并在验证机上验证将参数保存在/vepfs/problem3/
选择最优的epoch命名checkpoint_best.pt

下面是一个训练脚本执行命令
python train_equiformer_v2_md17_aspirin.py \
-- train_data_path  aspirin_train.npz\
-- valid_data_path  aspirin_valid.npz\
--save_dir  ./checkpoints/
3.运行评估脚本-- eval_equiformer_v2_md17_aspirin.py
加载2中的权重对测试集预测嫩量和受力
结果输出到problem3/test_prediction.pt中
评估和保存逻辑已经实现，测试集的能量和受力不提供
下面是评估脚本
python ；eval_equiformer_v2_md17_aspirin.py \
-- test_data_path  aspirin_train.npz\
-- result_output_path  /vepfs/problem3/test_prediction.pt\
--checkpoint_path    ./checkpoints/checkpoint_best.pt

文件描述在/vepfs-readonly/ problem3/目录下
aspirin_train.npz
aspirin_valid.npz
aspirin_test.npz 
Asprin-Force-Field/train_equiformer_v2_md17_aspirin.py 
Asprin-Force-Field/eval_equiformer_v2_md17_aspirin.py 
Asprin-Force-Field/inspect_aspirin_example.py 
提交内容，要在/vepfs/ problem3/下
1测试集的权重文件以checkpoint_best.pt命名
2,一份readme.me的说明文档
3,修改后的代码目录Asprin-Force-Field
4,测试结果test_prediction.pt
