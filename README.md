# DocEE
DocEE: A Large-Scale and Fine-grained Benchmark for Document-level Event Extraction


# Introduction
DocEE, a new document-level event extraction dataset including 27,000+ events, 180,000+ arguments. DocEE has three features: large-scale manual annotations, fine-grained argument schema and application-oriented settings.

Our academic paper which describes DocEE in detail and provides full event ontology can be found here: https://tongmeihan1995.github.io/meihan.github.io/research/NAACL2022.pdf.

# Dataset Statistic
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



# Fine-grained Argument Schema
DocEE focus on the extraction of the main event, that is *one-event-per-document*. 
![image](https://github.com/tongmeihan1995/DocEE/blob/main/image/dataset_display.png)

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
