# DocEE Dataset
DocEE: A Large-Scale and Fine-grained Benchmark for Document-level Event Extraction


# Introduction
DocEE, a new document-level event extraction dataset including 27,000+ events, 180,000+ arguments. DocEE has three features: large-scale manual annotations, fine-grained argument schema and application-oriented settings.DocEE focuses on the extraction of the main event, that is *one-event-per-document*. 

Our academic paper can be found here: https://tongmeihan1995.github.io/meihan.github.io/research/NAACL2022.pdf.

# Download DocEE
DocEE is now available at https://drive.google.com/drive/folders/1_cRnc2leAmOKT9Ma8koz6X8Ivl-_lapp?usp=sharing, which including three files:

- **DocEE-en.json**(All Data) 
- **normal_setting** (Train/Dev/Test under Normal Setting)
- **cross_domain_setting** (Train/Dev/Test under Cross-Domain Setting)

# Download DocEE-zh
DocEE-zh is now available, the dataset and its ontology can be downloaded from https://drive.google.com/drive/folders/15YDTsiTvt7qMC9itKoK5IyUAdcD8ezXB?usp=share_link. DocEE-zh contains 36,729 annotation data.

# Baseline
Baseline and evaluation indicators can refer to the project https://github.com/tongmeihan1995/DocEE-Application

# Event Schema of DocEE
To construct event schema, we gain insight from journalism, which divides events into **hard news** and **soft news** (Reinemann et al., 2012; Tuchman, 1973). 

**Hard news** is a social emergency that must be reported immediately, such as earthquakes, road accidents and armed conflicts. 

**Soft news** refers to interesting incidents related to human life, such as celebrity deeds, sports events and other entertainment-centric reports. 

Based on the hard/soft news theory and the category framework in (Lehman-Wilzig and Seletzky, 2010), we define a total of 59 event types, with 31 hard news event types and 28 soft news event types. We provides full event ontology in **Event Schema.md**.

# Example of DocEE
DocEE aims at **Event Classification** and **Event Arguments Extraction**. Here is an example of DocEE. 
![image](https://github.com/tongmeihan1995/DocEE/blob/main/image/dataset_display.png)

For each event argument, we annotate four keys:
- **start** (start_index)
- **end**(end_index)
- **type**(argument type)
- **text**(argument)
```
{'start': 82, 'end': 96, 'type': 'Date', 'text': 'Friday evening'}
```


# Statistics of DocEE
We are now the largest dataset for documnet-level event extraction.

| Datasets | #isDocEvent | #EvTyp. |#ArgTyp.| #Doc. | #ArgInst. | #ArgScat.|
| --- | --- | --- | --- | --- | --- | --- |
| ACE2005 | ✗ | 33 | 35 | 599 | 9,590 | 1.0 |
| KBP2016 | ✗ | 18 | 20 | 169 | 7,919 | 1.0 |
| KBP2017 | ✗ | 18 | 20 | 167 | 10,929 | 1.0 |
| MUC-4 | ✓ | 4 | 5 | 1,700 | 2,641 | 4.0 |
| WikiEvents | ✓ | 50 | 59 | 246 | 5,536 | 2.2 |   
| RAMS | ✓ | 139 | 65 | 9,124 | 21,237 | 4.8 |
| DocEE(ours) | ✓ | 59 | 356 | 27,485 | 180,528 |  10.2 | 


# How do I cite DocEE？
For now, cite the NAACL paper:
```
@article{tongdocee,
  title={DocEE: A Large-Scale and Fine-grained Benchmark for Document-level Event Extraction},
  author={Tong, Meihan and Xu, Bin and Wang, Shuai and Han, Meihuan and Cao, Yixin and Zhu, Jiangqi and Chen, Siyu and Hou, Lei and Li, Juanzi}
  journal={2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2022}
}
```
