# 内容

## 数据

### 癌症基因组图谱

癌症基因组图谱（The Cancer Genome Atlas， TCGA）是一始于2005年的项目——使用基因组测序和生物信息学对癌症相关的基因突变进行编目。TCGA主要致力于应用高通量基因组分析技术，通过了解癌症的遗传基础，以提高医疗工作者诊断，治疗和预防它的能力。

TCGA受隶属于美国国立卫生研究院（National Institutes of Health，NIH）的国家癌症研究所（National Cancer Institute， NCI）的癌症基因组学中心和国家人类基因组研究监管。至2015年，TCGA完成了33种不同肿瘤类型的基因组表征和序列分析，包括10种罕见癌症。["Cancers Selected for Study". The Cancer Genome Atlas – National Cancer Institute. Retrieved 2015-11-02.]同时，NCI开启了基因组数据共享（Genomic Data Commons，GDC）研究计划与TCGA并行。GDC为癌症研究界提供统一的数据存储库，以便在癌症基因组研究中共享数据，以支持精准医学。当前，大量癌症图像分析相关的研究应用其中的影像数据。

本文的影像数据均来自GDC中TCGA-BRCA项目的数据,可由此<https://portal.gdc.cancer.gov/projects/TCGA-BRCA>下载。

### 切片选择

TCGA-BRCA提供快速冷冻（Flash Frozen）和福尔马林固定石蜡包埋（Formalin-Fixed Paraffin-Embedded, FFPE）这两种类型的切片影像。

快速冷冻样品通常在手术期间在冷冻液中制作，用来帮助外科医生确定肿瘤的边界是否清洁（即，肿瘤是否已被完全切除）。快速冷冻是一种快速且“简单”的过程，但经常会使组织受损，使其大致外观多孔，很可能致使图像中与肿瘤相关的关键信息丢失；FFPE切片是诊断医学的金标准。FFPE的制作需要先将样本固定在甲醛中，然后将其嵌入石蜡块中进行切割。它呈现的组织更为完整，使其更适合应用于计算分析。

### 人工识别特征

## 实验内容

我们先以人工识别特征建立Cox HP模型得到参照基本线（baseline）。之后，我们对初始切片图进行区域分割及预筛选。在训练得到分类器后，对区域再次进行筛选。将筛选区域作为样本内容，采用一系列方法训练得到生存模型。最后对生存模型进行评价和优化。

### 人工识别特征建立Cox HP模型

比例风险模型（Proportional hazards models， PH models）是统计学中生存模型的一类。生存模型通过分析在某事件发生之前经过的时间与可能与该时间量相关联的一个或多个协变量的关系能建立“时间-事件”模型。在比例风险模型中，协变量中单位增加的效应与相应的危险率是乘积关系。例如，服用药物可能会使一个人发生中风的危险率降低一半。其他类型的生存模型，如加速失效时间模型（accelerated failure time models），不能表现比例风险。
（ Breslow, N. E. (1975). "Analysis of Survival Data under the Proportional Hazards Model". International Statistical Review / Revue Internationale de Statistique. 43 (1): 45–57. doi:10.2307/1402659. JSTOR 1402659.
）

在Cox比例风险模型（Cox Proportional hazards model，Cox PH model）（Cox 1972）常被医学统计用于研究患者存活时间和一个或多个预测变量之间的关联。它和Kaplan-Meier曲线和logrank测试不同——Kaplan-Meier曲线和logrank测试是单变量分析，它们探究一个因素与生存时间的关联，但忽略了其他因素的影响；同时它们需要的变量类型是分类的，如疗法A，疗法B或男性、女性，而不适用于定量的变量。Cox比例风险回归分析则不同，它适用于定量变量和分类变量。此外，Cox回归模型扩展了生存分析方法，能同时评估多个风险因素对生存时间的影响。
http://www.sthda.com/english/wiki/cox-proportional-hazards-model#references
(Cox DR (1972). Regression models and life tables (with discussion). J R Statist Soc B 34: 187–220)

本文应用基于Python的生存分析应用lifelines,<https://lifelines.readthedocs.io/en/latest/index.html>,进行Cox PH模型的建立。

### 区域分割及预筛选

影像数据分析的一大挑战是其包含的图像非常大，以至于其难以被读取至随机读取内存（random access memory, RAM），或者应用一般的流水线（pipeline）进行分析。以本研究所使用的TACG-BRCA数据为例，其以Aperio公司SVS文件为格式，759个样本的10倍镜成像平均包含5.38e+08个像素点，40倍成像平均图包含8.61e+09个像素点。如果在RGB模式，uint8数据格式下读取，40倍成像的图像将占用近26GB的RAM空间，此外，之后的运算还需要空间存储中间结果。而当前常规的计算机一般配备8至16GB的RAM。因此，直接读取文件进行分析是不可取的。同时，我们观察到10倍镜成像下，肿瘤细胞聚集分布在图中的某些区域，一个细胞仅占8*8的区域。此外，图中有大量的空白区域或重复的大块区域，直接使用全图是不明智的。从经验上讲，对空白区域可用简单的算法进行预筛选。

#### 显微放大倍数选择

10倍镜下细胞占8*8，包含一定的信息，虽相较于40倍镜有一定的损失，但相关的研究使用10倍（？？）得到较好的结果。故10倍镜放大倍数保留了足够的有效信息。

此外，后续我们NASNet分类器及迁移学习使用的Kaggle数据库（见方法）数据为10倍镜下成像。于是，我们在研究中也选用了10倍镜下的成像，以在计算机资源使用和有效信息保留上取得平衡。

#### openslide进行96*96区域切割

openslide是基于C语言的读取大图的开源工具。本文使用最新的3.4.1版。
（OpenSlide: A Vendor-Neutral Software Foundation for Digital Pathology
Adam Goode, Benjamin Gilbert, Jan Harkes, Drazen Jukic, M. Satyanarayanan
Journal of Pathology Informatics 2013, 4:27）
显而易见地，我们需要将整个图像分割为小的区域，对每个区域进行分析，再将分析结果进行整合。当前图像深度学习使用的图片一般会缩小为96*96的小图，以便于分析。

#### OpenCV预选

OpenCV，即开源计算机视觉库，是一个开源的应用于计算机视觉及机器学习的软件。OpenCV是计算机视觉的应用常用的平台。

在分割过程中，采用简单的灰度分析，即将小图由RGB模式转化为灰度模式，计算全图的平均灰度值，以此由一个平均灰度值代表整个小图，能去除空白的区域对区域进行预筛选。

### NASNet区域分类器

2017年5月，Google Brain设计的能产生人工智能(artificial intelligence, AI)的人工智能，AutoML。NASNet是Google的研究人员使用强化学习，以AutoML作为控制神经网络自动训练出来的神经网络模型。在2018年，NASNet已是图像识别领域的最佳模型。

#### NASNet分类模型架构

NASNet模型分为NASNet large模型和NASNet mobile模型。NASNet mobile是其中较小的模型，其硬件需求更小，但仍能实现强大的识别功能。

我们的模型的第一层由96\*96\*3的数据输入，之后则是NASNet mobile层。NASNet层会产生3组数据，下一层将3组数据合并，之后是一个随机丢失层（dropout），最后是sigmoid函数激活的1维全连接层。

该分类模型使用分类任务常用的二元交叉熵函数（binary cross entropy function）
$$
C = -\frac{1}{n} \sum_x [y \ln y+(1-y) \ln(1-y)]
$$
作为损失函数。

#### Kaggle数据库

Kaggle组织病理肿瘤识别（Histopathologic Cancer Detection, HCD)数据来自<https://www.kaggle.com/c/histopathologic-cancer-detection>。

Kaggle HCD数据库的源头是Camelyon16挑战数据库。该数据库包含400张40倍镜下的H&E染色切片全图。PCam数据库产生于将Camelyon16挑战数据库的40倍镜成像缩小为10倍成像，且分割为96*96的小区域。本文训练NASNet分类器使用的Kaggle数据库是Pcam数据库的子集。Kaggle数据库和Pcam数据库的差别在于Pcam数据库中的重复的图片全部被去除了。

Kaggle HCD数据库以在图像中心（32*32）区域是否包含至少一个像素肿瘤组织为评判标准将所有图像数据进行了分类。60%的图像为阴性，即图像中心没有肿瘤组织；40%的图像为阳性，即图像中心至少有一个像素的肿瘤组织。因此，该数据库也很适合于用于之后生存模型的迁移学习。

#### 区域采集量

生存模型的损失函数是受事件发生的时间影响的。因此，生存模型训练时，每次训练需要将数据按时间顺序输入，故一般图像识别为避免数据次序影响时所用的乱序（shuffle）方法在此处一定不能使用。同时，分批（batch）的数据输入策略也不应使用。合理的方法是每次将全部样本一起输入进行训练。这样，每次训练时，一个样本只能从待选区域里面选择一张作为代表。此时，有效区域数量一定，如果待选区域数量过多，那么选择到有效区域的几率将大大减小。故我们需要设置NASNet分类器的筛选阈值，以最大程度地提高训练时训练数据选择到有效区域的可能。

### 训练

在训练时，我们运用了迁移学习（Transfer learning）的方法。它是一种机器学习的研究方法——将解决一个问题的策略“知识”存储，然后应用于其他但相关的问题。比如识别轿车的模型能被尝试应用于识别卡车。
（West, Jeremy; Ventura, Dan; Warnick, Sean (2007). "Spring Research Presentation: A Theoretical Foundation for Inductive Transfer". Brigham Young University, College of Physical and Mathematical Sciences. Archived from the original on 2007-08-01. Retrieved 2007-08-05.）

#### SNAS生存模型架构

我们的模型的第一层由96\*96\*3的数据输入，之后则是NASNet层。NASNet层会产生3组数据，下一层将3组数据合并，之后是一个随机丢失层（dropout）后是一个全连接层，最后是基于全连接层的生存预测层。两个全连接层均使用L2函数
$$
L_2=\sum_{i=1}^{n}(y^{(i)}-\hat{y}^{(i)})^{2}
$$
作为核规范器（kernel regularizer）、运算规范器（activity regularizer），以Glorot均衡起始器（Glorot uniform initializer,或称Xavier uniform initializer）以促进损失函数收敛，防止过度拟合（overfitting），增强其普遍化的能力。（Understanding the difficulty of training deep feedforward neural networks）

从架构可见，我们将NASNet同样应用于生存模型中，设计得到我们的SNAS。NASNet含有近4.27e+07个参数。这些参数由学习Kaggle数据库得到。我们将NASNet层的参数冻结，以存储训练于Kaggle数据库能识别肿瘤区域的能力，实现迁移学习。

生存模型在机器学习中使用了一个特殊的损失函数（loss function），即负对数似然损失函数（negative log likelihood loss function）
$$
\boldsymbol{\mathcal{L}}=-\frac{1}{n}\sum_{i=1}^{n}\log(\hat{y}^{(i)})
$$

#### 随机区域选择

如我们在区域采集量中提到的生存模型训练的需求——数据需要按照时间顺序排序，同时损失函数会因数据选取（不同时间点，数量大小）而变化。于是我们每次选取一个区域代表样本，然后将所有样本输入进行训练。

由于我们没有相应的专业人员对我们的区域进行人工筛选或鉴定，此处我们只能采用随机选取的方法。

### 批量预测

应对肿瘤内异质性

#### 随机多区域选择

10，20，42区域选择

### 超参数（hyperparameter）优化及数据量影响

#### 模型大小

256,512选择

#### “数据增强”

augmentation内容

#### 数据量影响

0.5，0.8

## 结果

### Cox HP基础模型表现

ci

### NASNet分类模型表现

AUC

### 切片区域可采集量

099一般，特殊50下限

### SNAS生存模型表现

表现

### SNAS超参数优化

模型大小
数据增强

### 数据数据量影响

0.5，0.8

### 端到端系统

pipeline流水线
GitHub

## 讨论

### 乳腺癌特征

乳腺癌现状

### 模型情况

模型应用，

### 数据大小

数据现状

### 应用

健康产业，eHealth