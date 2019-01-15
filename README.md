# Survival-Analysis-by-Breast-Cancer-Slides

A backup for my undergraduate thesis.

## Construction

1.Baseline Data Preparation

I used the morphologic data created doctors manually to set up a Cox Proportional Hazard model. The outcome of this model will help me to judge whether the deep learning model is better than the professionals.

2.SCNN Data Preparation

I used the authentic National Institutes of Health (NIH)'s Harmonized Cancer Datasets as my data source. I aimed to do survival analysis based on images. For survival analysis, the indispensable part of information is the duration from the start to the occurrence of the event and whether the event is censoed, that is the event didn't happen during the certain period of time.

From my dataset, I found #TODO cases who have complete information that can satisfy the survial analysis requirement above. Also, for the sake of fair comparison, I decided to use the same samples which the baseline used. However, though in the original paper, there are 8?? cases. Only 779 cases overlap my selection. Besides, 9 cases are from male. Considering the gender may paly a large role in breast cancer (#TODO), I delete all the male cases. Then the gender factor will not influence both models.

The server I worked on is a RedHat system. I am not familiar with this kind of corporation system. It doesnt't support agt and some other Ubuntu command. Even worse, I didn't get the administrator permission. Or say, I am not in administrator group. As a result, I cannot install software by yum or rpm etc. commands. The only environment I can use is Anaconda.

Luckily, the GDC download software (GDC Data Transfer Tool) published by NIH have a python version. So, I can download the TCGA slices I needed automaticly. All slices are Formalin-Fixed Paraffin-Embedded (FFPE) from TCGA-BRCA.

3.SCNN Data Preprocession

TODO

## Progress

Search the cancer location on the server by other programs like the Baidu's ncrf or SCNN.

Using the location of cancer, I could get the patch in which only important cancar areas are.

Download the patches, I can use the computer with GPU to train my own model.

## Links

[Data Source]
[GDC](https://portal.gdc.cancer.gov/)

Download Tools
[gdc-data-transfer-tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

[Used on red-hat](https://gist.github.com/sbamin/7f33b26198a00ad6846d124b8ba8d2b4)

Slices Preparation
[download tcga digital pathology images ffpe](http://www.andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/)