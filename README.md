# Survival-Analysis-by-Breast-Cancer-Slides

A backup for my undergraduate thesis.

## Construction

### Baseline Data Preparation

I used the morphologic data created doctors manually to set up a Cox Proportional Hazard model. The outcome of this model will help me to judge whether the deep learning model is better than the professionals.

### SCNN Data Preparation

I used the authentic National Institutes of Health (NIH)'s Harmonized Cancer Datasets as my data source. I aimed to do survival analysis based on images. For survival analysis, the indispensable part of information is the duration from the start to the occurrence of the event and whether the event is censoed, that is the event didn't happen during the certain period of time.

From my dataset, I found #TODO cases who have complete information that can satisfy the survial analysis requirement above. Also, for the sake of fair comparison, I decided to use the same samples which the baseline used. However, though in the original paper, there are 8?? cases. Only 779 cases overlap my selection. Besides, 9 cases are from male. Considering the gender may paly a large role in breast cancer (#cite), I delete all the male cases. Then the gender factor will not influence both models.

The server I worked on is a RedHat system. I am not familiar with this kind of corporation system. It doesnt't support agt and some other Ubuntu command. Even worse, I didn't have the administrator permission. As a result, I cannot install software by yum or rpm etc. commands. The only environment I can use is Anaconda.

Luckily, the GDC download software (GDC Data Transfer Tool) published by NIH have a python version. So, I can download the TCGA slices I needed automaticly. All slices are Formalin-Fixed Paraffin-Embedded (FFPE) from TCGA-BRCA.

### Image Data Preprocession

The slices are .svs files which is the standard file format (#cite). Those files have pyramid-like magnitude, that is they have three or four magnitudes or powers, and in different magnitudes, the resolution they are in is different. Recent researchs using medical slices usually set the slides in
20X or 10 X power, which can achieve a beautiful balance between computing usage and outcome quality (#cite).

Here, I conduct my study using 10X power images. It's because I only have limited the computing resource, (#hardware). Also, similar researchs, like the competition in Kaggle in 2018, (#breast cancer competition), process images into 96 pixels * 96 pixels small 10X power, resulting an impressive performance of classification cancer type, whether it is metastatsis.

In 10X power, the images' resolution is range from #TODO to #TODO. Currently, if I simply make a whole image as input into the RAM(random access memory), it will occupy #TODO on average. However, typical computers usually have 32G RAM(#cite). Also, cancer cells can scatter around the images, since in 10X power, the cancer cells occupy #TODO pixels. Besides, large blank areas are in each slides, and tissues sometimes are recorded twice in one slides. Consequently, it's not realistic nor necessary to use the whole image. On top of that, each slide should be segmented into small images and rule out the images with little or replicated content.

One thing is worth noticing that the segmentation of large slides is usually completed manually (#cite). Experienced professors can identify the ROIs (regions of interest) empirically. However, due to the lack of outside expertise assistance, I couldn't rely on manual ROIs identication. First, I want to search the cancer location on the server by other models like Baidu's NCRF(#cite) or Google's Lyna(#cite). Unfortunately, NCRF run in a environment where computing resource is abundant and the running time is laxly considered, and Lyna hasn't open its source code before I finish my study. As a result, I use my home-made model adapted from one Kaggle's kernel in (#competition), which is based on Google's Nasnet to identify the ROIs.

To begin with, I segment whole slides in 10X power into 96 pixel * 96 pixels small .tiff images. The format .tiff is based on (CMYK), which is different from RGB (#cite). It's because the .svs files are created from individul .tiff files in the first place, and I used the Kaggle data that is in .tiff format to train the model for ROI identification. Different picture format use varing ways to represent the colour, so it's curcial to make sure the image format models are trained on is consistent with that the models will predict.

---

### home-made model

#### DATA PROFILE

The PCam dataset is derived from the Camelyon16 Challenge dataset which contains 400 H&E stained whole slide images using a 40x objective. This one uses 10x undersampling to increase the field of view, which gives the resultant pixel resolution of 2.43 microns.

All slides were inspected manually after scanning. The inspection was performed by an experienced technician (Q.M. and N.S. for UMCU, M.H. or R.vd.L. for the other centers) to assess the quality of the scan.

Litjens, G., Bandi, P., Ehteshami Bejnordi, B., Geessink, O., Balkenhol, M., Bult, P., Halilovic, A., Hermsen, M., van de Loo, R., Vogels, R. and Manson, Q.F., 2018. 1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset. GigaScience, 7(6), p.giy065.

The negative/positive ratio is not entirely 50/50, as the label mean is well below 0.5.

The ratio is closer to 60/40 meaning that there are 1.5 times more negative images than positives.

#### AUGMENTATION

horizontally flip 50% of all images
vertically flip 20% of all images
scale images to 80-120% of their size, individually per axis
translate by -20 to +20 percent (per axis)
rotate by -45 to +45 degrees
blur images with a sigma between 0 and 3.0
...​

#### NASNET

NASNet performs 1.2% better than all previously published results.

NASNet may be resized to produce a family of models that achieve good accuracies while having very low computational costs.

Zoph B , Vasudevan V , Shlens J , et al. Learning Transferable Architectures for Scalable Image Recognition[J]. 2017.

#### PERFORMANCE

TODO

### SCNN Architecture

Keras

Loss function

Completed model

### Result

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