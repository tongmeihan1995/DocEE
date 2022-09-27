# DocEE
DocEE: A Large-Scale and Fine-grained Benchmark for Document-level Event Extraction


# Introduction
DocEE, a new document-level event extraction dataset including 27,000+ events, 180,000+ arguments. DocEE has three features: large-scale manual annotations, fine-grained argument schema and application-oriented settings.

Our academic paper which describes DocEE in detail and provides full event ontology can be found here: https://tongmeihan1995.github.io/meihan.github.io/research/NAACL2022.pdf.

# Download DocEE
The dataset is now available at https://drive.google.com/drive/folders/1_cRnc2leAmOKT9Ma8koz6X8Ivl-_lapp?usp=sharing, which including three files:

- **DocEE-en.json**(All Data) 
- **normal_setting** (Train/Dev/Test under Normal Setting)
- **cross_domain_setting** (Train/Dev/Test under Cross-Domain Setting)

# Event Schema of DocEE
To construct event schema, we gain insight from journalism. Journalism typically divides events into hard news and soft news (Reinemann et al., 2012; Tuchman, 1973). Hard news is a social emergency that must be reported immediately, such as earthquakes, road accidents and armed conflicts. Soft news refers to interesting incidents related to human life, such as celebrity deeds, sports events and other entertainment-centric reports. Based on the hard/soft news theory and the category framework in (Lehman-Wilzig and Seletzky, 2010), we define a total of 59 event types, with 31 hard news event types and 28 soft news event types. Detailed information is shown in **Event Schema.md**.

# Example of DocEE
The figure below shows an example of DocEE. DocEE aims at **Event Classification** and **Event Arguments Extraction**. DocEE focuses on the extraction of the main event, that is *one-event-per-document*. 
![image](https://github.com/tongmeihan1995/DocEE/blob/main/image/dataset_display.png)

## Annotation Format of DocEE
For each event argument, there are four keys:
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
