---
title:  '以NASNet利用组织学及基因组预测乳腺癌存活预期'
# subtitle: "This is the subtitle"
author:
- 穆幼清
description: |
    This is a long
    description.

    It consists of two paragraphs
...

# 1 研究背景

在医学研究中，组织学成像为肿瘤诊断和治疗的提供了重要信息。
组织特征表现出分子层面改变所带来的总体影响。同时，组织能提供直观的视觉信息以帮助医疗人员判断癌症侵入性。然而，组织分析是高度主观、不能重复得到相同结果的。

计算机对组织学成像进行分析，不但能克服人工进行组织分析的缺陷，而且能提取人工进行组织分析忽略的信息。

# 2 研究内容

# 2.1 研究亮点

本文提出一种基于Nasnet的乳腺癌生存神经网络模型（SNAS）。它能分析乳腺癌组织学成像提供较为准确的"时间-事件"预测，即生存预测。该模型的预测能力能实现优于与使用专家分析同样图像提取的特征建立的生存分析模型的表现。同时，该模型能由文中搭建的全自动的由.svs病理切片图像到生存模型的"端到端"流水线（pipeline）生成。其中的所有环节均无需人为介入。此外，该模型大小中等，7.24e+07个参数，其中4.26e+07个参数为Nasnet以迁移Kaggle数据训练，故实际需要计算的参数量仅为2.97e+07个，一般能在个人计算机上无障碍地进行训练及使用。

# 2.2 研究流程

![图1 研究流程图](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\fig1.png)
研究流程如图我们从TCGA-BRCA收集到匹配专家识别数据且随访时间完整的759份样本。通过使用Kaggle HCD数据库训练得到NASNet分类器后，先对初始切片图进行区域分割及预筛选，再使用该分类器对区域进行二次筛选。接着，我们将筛选区域作为学习内容，采用包括迁移学习在内的一系列方法对生存模型进行训练。最后，对得到的生存模型进行相应的优化和评估以完成研究。

# 3 研究结果

![图2 SNAS生存模型架构](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\modelg.png)
SNAS生存模型的第一层由96\*96\*3的数据输入层，之后则是NASNet层。NASNet层会产生3组数据，合并层将3组数据合并；合并的数据经过一个随机丢失层后输入一个全连接层；最后是基于全连接层的生存预测层。

# 3.1 SNAS超参数

![图3 数据增强影响](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\fig8.png)
数据增强对不同大小模型的学习影响。比较40次模型训练的结果，无论其是大模型还是正常模型，不使用数据增强的表现都显著地优于使用数据增强的表现（正常模型的训练拟合差异显著性较弱，p=0.011）。

![图4 全连接层节点数量的影响](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\fig9.png)
全连接层节点数量的影响。正常模型和大模型的拟合表现相似；正常模型的预测能力显著优于大模型。选择256节点全连接层更优，因为其预测能力更好，同时消耗的计算资源更少。

# 3.2 SNAS生存模型表现

![图5 SNAS生存模型表现](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\result.png)
600次训练得到600个模型。其中拟合训练集最好的达到c index 0.61，拟合验证集中最好的达到c index 0.75，有78个优于同等条件下得到的Cox HP基础模型。随机训练得到预测能力优于Cox HP基础模型的概率是13%。

# 3.3 数据量影响

![图6 训练集大小与模型表现](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\fig11.png)
训练集大小与模型表现的关系。使用同样的架构，使用不同的数据量，经过40次训练得到相应的模型表现。训练的最终表现均不同。同时，随着训练时的数据量的增多，模型表现逐步变好。

# 3.4 结合基因组学

![图7 基因组学信息影响](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\fig12.png)
加入基因组学信息能显著增强模型的拟合能力，然而其预测能力有所减弱。我们猜想基因表达相互作用的情况相较于组织学特征可能更为特异。因此，在训练集群体中学习到的基因组学与生存间的规律不能很好地适用于其他群体，即测试集。

# 3.5 端到端系统

![图8 端到端系统](D:\Project\Survival-Analysis-by-Breast-Cancer-Slides\imgs\site.png)
本文的模型训练不需要任何的人力接入。因此，它是一个全自动的生存模型训练系统。该系统较好地解决了模型训练需要大量人力标注这一问题。以此，本文实现了从.svs文件到生存模型的流水线（end-to-end pipeline）。

# 4 参考文献

[1] JIAN R, SADIMIN E T, WANG D, et al. Computer aided analysis of prostate histopathology images Gleason grading especially for Gleason score 7; proceedings of the Engineering in Medicine & Biology Society, F, 2015 [C].
[2] NIAZI M K, YAO K, ZYNGER D, et al. Visually Meaningful Histopathological Features for Automatic Grading of Prostate Cancer [J]. IEEE Journal of Biomedical & Health Informatics, 2016, PP(99): 1-.
[3] FAUZI M F A, PENNELL M, SAHINER B, et al. Classification of follicular lymphoma: the effect of computer aid on pathologists grading [J]. Bmc Medical Informatics & Decision Making, 2015, 15(1): 1-10.
[4] WANG D, KHOSLA A, GARGEYA R, et al. Deep Learning for Identifying Metastatic Breast Cancer [J]. 2016, 
[5] LECUN Y, BENGIO Y, HINTON G. Deep learning [J]. Nature, 2015, 521(7553): 436.
[6] ZOPH B, VASUDEVAN V, SHLENS J, et al. Learning Transferable Architectures for Scalable Image Recognition [J]. 2017, 
[7] PRENTICE R L. Introduction to Cox (1972) Regression Models and Life-Tables [M]. 1992.
[8] STANLEY A P D, ANNIE X P D, LAPUERTA P, et al. Comparison of Predictive Accuracy of Neural Network Methods and Cox Regression for Censored Survival Data [J]. Computational Statistics & Data Analysis, 2000, 34(2): 243-57.
[9] KATZMAN J L, SHAHAM U, CLONINGER A, et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network [J]. Bmc Medical Research Methodology, 2016, 18(1): 24.
[10] MOBADERSANY P, YOUSEFI S, AMGAD M, et al. Predicting cancer outcomes from histology and genomics using convolutional networks [J]. Proceedings of the National Academy of Sciences of the United States of America, 2018, 115(13): 201717139.
[11] KONG J, COOPER L A, WANG F, et al. Integrative, multimodal analysis of glioblastoma using TCGA molecular data, pathology images, and clinical outcomes [J]. IEEE Transactions on Biomedical Engineering, 2011, 58(12): 3469-74.
[12] GUTMAN D A, COOPER L A D, HWANG S N, et al. MR Imaging Predictors of Molecular Profile and Survival: Multi-institutional Study of the TCGA Glioblastoma Data Set [J]. Radiology, 2013, 267(2): 560-9.
[13] SCHAUMBERG A J, RUBIN M A, FUCHS T J. H&E-stained Whole Slide Image Deep Learning Predicts SPOP Mutation State in Prostate Cancer [J]. 2017, 
[14] SALTZ J, R G, L H, et al. Spatial Organization and Molecular Correlation of Tumor-Infiltrating Lymphocytes Using Deep Learning on Pathology Images [J]. Cell Reports, 2018, 23(1): 181-93.
[15] AZIMZADEH O, BARJAKTAROVIC Z, AUBELE M, et al. Formalin-fixed paraffin-embedded (FFPE) proteome analysis using gel-free and gel-based proteomics [J]. Journal of Proteome Research, 2010, 9(9): 4710-20.
[16] HENG Y J, LESTER S C, TSE G M, et al. The molecular basis of breast cancer pathological phenotypes [J]. Journal of Pathology, 2016, 241(3): 375.
[17] LAWRENCE M S, PETAR S, PAZ P, et al. Mutational heterogeneity in cancer and the search for new cancer-associated genes [J]. Nature, 2013, 499(7457): 214-8.
[18] MERMEL C H, SCHUMACHER S E, HILL B, et al. GISTIC2.0 facilitates sensitive and confident localization of the targets of focal somatic copy-number alteration in human cancers [J]. Genome Biology, 2011, 12(4): R41-R.
[19] BRESLOW N E. Analysis of Survival Data under the Proportional Hazards Model [J]. International Statistical Review, 1975, 43(1): 45-57.
[20] EFRON B. Logistic Regression, Survival Analysis, and the Kaplan-Meier Curve [J]. Publications of the American Statistical Association, 1988, 83(402): 414-25.
[21] FREEDMAN L S. Tables of the number of patients required in clinical trials using the logrank test [J]. Statistics in Medicine, 1982, 1(2): 121–9.
[22] GOODE A, GILBERT B, HARKES J, et al. OpenSlide: A vendor-neutral software foundation for digital pathology [J]. J Pathol Inform, 2013, 4(1): 27.
[23] BRADSKI G R, KAEHLER A. Learning opencv, 1st edition [M]. 2008.
[24] GUYON I, BENNETT K, CAWLEY G, et al. Design of the 2015 ChaLearn AutoML challenge; proceedings of the International Joint Conference on Neural Networks, F, 2015 [C].
[25] PHAISANGITTISAGUL E. An Analysis of the Regularization Between L2 and Dropout in Single Hidden Layer Neural Network; proceedings of the International Conference on Intelligent Systems, F, 2017 [C].
[26] EHTESHAMI B B, VETA M, JOHANNES V D P, et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer [J]. Jama, 2017, 318(22): 2199.
[27] VEELING B S, LINMANS J, WINKENS J, et al. Rotation Equivariant CNNs for Digital Pathology [J]. 2018, 
[28] CHARALAMBOUS C C, BHARATH A A. A data augmentation methodology for training machine/deep learning gait recognition algorithms [J]. 2016, 
[29] HOYLE B, RAU M M, BONNETT C, et al. Data augmentation for machine learning redshifts applied to Sloan Digital Sky Survey galaxies [J]. Monthly Notices of the Royal Astronomical Society, 2018, 450(1): 305-16.
[30] PAN S J, QIANG Y. A Survey on Transfer Learning [J]. IEEE Transactions on Knowledge & Data Engineering, 2010, 22(10): 1345-59.
[31] MOUGENOT A, DARRASSE A, BLANC X, et al. Uniform Random Generation of Huge Metamodel Instances; proceedings of the Model Driven Architecture-foundations & Applications, European Conference, Ecmda-fa, Enschede, the Netherlands, June, F, 2009 [C].
[32] TNM Classification of Malignant Tumours 7e [J]. 2009, 
[33] KLEIN C A. The Metastasis Cascade [J]. Science, 2008, 321(5897): 1785-7.
[34] DOUGLAS A. The Hitchhiker's Guide to the Galaxy [J]. Br Med J, 1981, 283(6285): 173-8.
[35] CLAESEN M, DE MOOR B. Hyperparameter Search in Machine Learning [J]. 2015, 
[36] CATHERINE V P, SOMERFIELD M R, BAST R C, et al. Use of Biomarkers to Guide Decisions on Systemic Therapy for Women With Metastatic Breast Cancer: American Society of Clinical Oncology Clinical Practice Guideline [J]. Journal of Clinical Oncology Official Journal of the American Society of Clinical Oncology, 2016, 34(10): 1134.
[37] NG C K, MARTELOTTO L G, GAUTHIER A, et al. Intra-tumor genetic heterogeneity and alternative driver genetic alterations in breast cancers with heterogeneous HER2 gene amplification [J]. Genome Biology, 2015, 16(1): 107.
[38] YATES L R, KNAPPSKOG S, WEDGE D, et al. Genomic Evolution of Breast Cancer Metastasis and Relapse [J]. Cancer Cell, 2017, 32(2): 169-84.e7.
[39] STECK H, KRISHNAPURAM B, DEHING-OBERIJE C, et al. On ranking in survival analysis: Bounds on the concordance index; proceedings of the Advances in neural information processing systems, F, 2008 [C].
[40] POWERS D M. Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation [J]. 2011, 
[41] MUSTAFA M, NORNAZIRAH A, SALIH F, et al. Breast cancer: detection markers, prognosis, and prevention [J]. IOSR Journal of Dental and Medical Sciences, 2016, 15(8): 73-80.
[42] BENSON J R, JATOI I. The global breast cancer burden [J]. Future oncology, 2012, 8(6): 697-702.
[43] MITTRA I. Breast cancer screening in developing countries [J]. Preventive Medicine, 2011, 53(3): 121-2.
[44] FEB. World cancer report 2014 [J]. World Health Organization, 2015, 
[45] BRAY F, FERLAY J, SOERJOMATARAM I, et al. Global cancer statistics 2018: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries [J]. CA: a cancer journal for clinicians, 2018, 68(6): 394-424.
[46] BECK A H, SANGOI A R, LEUNG S, et al. Systematic Analysis of Breast Cancer Morphology Uncovers Stromal Features Associated with Survival [J]. Science Translational Medicine, 2011, 3(108): 108ra13.

# 5. 致谢

这篇论文的完成离不开孙逸仙纪念医院研究中心赵慧英老师的指导。她以丰富的研究经验以及敏锐的洞察力为我的研究提供了极有帮助的建议。同时，感谢骆观正老师作为校内指导老师，在管理审核上提供支持。

感谢家人，朋友，同学的陪伴。