# OOD检测任务：
训练模型让模型具有检测分布外数据的能力，比如针对猫狗的分类器对于输入非猫狗的数据要求模型能够响应判断X为OOD数据
# 几个基线方法：
主要分为两种：

- OE方法（异常暴露）提前引入异常数据，模型输出类别为n+1类，将OOD数据作为第n+1类分类
  - NPOS
  - CIDER
  - NPOS
  - VOS
  - DINO
  - DreamOOD
  - FodFom
  - TagFog
- 事后方法
  - MSP(基线)
  - Vim
  - Energy
  - ReAct
  - LogitNorm
  - CSI
  - SSD+
### 通常OE做法是可以结合事后方法达到更好的效果
- 两种类方法，前者是从模型本身出发提前引入异常重构模型权重让模型对OOD样本有更好的鲁棒性而不会产生过高的置信度
- 后者是对预训练的分类模型的输出的中间层权重进行分析通常不会重新训练模型，会对特征值做一些处理（剪裁、归一化、退火或者额外引入轻量分支）避免模型学到ID无关知识
## 我的方法是生成异常样本
- 异常样本最好和原始数据有相关性不能是噪声避免模型学到过于简单的规律，也就是原始数据的边缘样本以帮助模型收紧决策边界
- 主要参考的**两类方法**（利用Stable diffusion生成异常样本）
    - FodFom
    - DreamOOD
    两者都是先提取类别的语义嵌入，对嵌入做边缘采样或者低似然采样，再把嵌入送到diffusion模型中生成异常样本
    语义方向不可控（比如可能会生成艺术风格的鸟类）我的观点是未必适合真实世界，我的方法可以根据需要定向产生特定风格的异常样本
# OOD样本生成的大致思路：
1. 用预训练好的CLIPVisionModelWithProjection（'h94/IP-Adapter'适配diffusion的）提取每个类别的图像特征
2. 对每个类别的聚类进行边缘采样
   - **采样算法(embeds_sampler)**
   - 遍历每个类
       - PCA降维（保留0.9方差）
       - 计算这个类中所有降为后样本的余弦距离，1-cosine_dist作为样本之间的距离
       - 取前k个最近的样本的嵌入
       - 将这k个嵌入降序排序
         - l_percent = 0
         - r_percent = 0.05
         - l_threshold = mean_knn[l_percent]
         - r_threshold = mean_knn[r_percent]
       - 根据下面这两个公式计算每个样本被采样的概率，knn越大越稀疏的embeds被采样的概率越大
         - density = torch.exp(-self._min_max_scale(all_knn_means) / temperature)
         - prob = density / (density.sum() + 1e-8)
       - 然后根据上面的概率随机采样n个样本计算均值再加上一个noise作为新的采样向量，共candidate_batch组采样向量
         - 然后计算这个新的向量PCA后在原始的嵌入里面的knn距离
         - 保留下在l和r范围之间的向量
         - 计算命中率
       - 命中率过低就逐渐增加noise，过高就减少noise,(hit_min,hit_max提前定义)
       - 采样够了samples数量就停止
3. 用下面这几个预训练好的模型替换sd的组件去生成尽可能符合原数据集风格的OOD样本，直接把类名作为negative_prompt传入sdpipe强迫生成远离原本类别的样本
` sd_model="stable-diffusion-v1-5/stable-diffusion-v1-5",
ip_adapter="h94/IP-Adapter",
adapter_weight_name="ip-adapter_sd15_light.bin",
vae_model = "stabilityai/sd-vae-ft-mse", `
4. 有的时候可能会生成一些far ood，如果需要extreme near的ood可以把Ontology或者一些具有因果关系的特征作为正向提示词
- 比如dog可以写成quadruped（四足生物）
- water bottle可以写成Cylinder object或者container

## 生成数据以后就正常按照FodFom的方法进行评估
## 评估准则
FPR95
AUROC
衡量ID和OOD的



# 模型结构
半监督的Resnet,主干采用resnet18（CIFAR10、100），resnet50（ImageNet100）
### 三个模块：
- encoder
- classfier
- projection（用于对比学习，增加ID类别内紧凑度）

### 损失函数（同FodFom，但是为ID样本和OOD样本的交叉熵和监督损失增加了权重，fake OOD样本总数和每个类别样本不均衡的情况）

### 推理
推理配置也和FodFom一样，采用react+energy两种特征处理的事后方法
### 具体的实验配置参考FodFom