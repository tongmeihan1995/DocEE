# DocEE
DocEE: A Large-Scale and Fine-grained Benchmark for Document-level Event Extraction


# Introduction
DocEE, a new document-level event extraction dataset including 27,000+ events, 180,000+ arguments. DocEE has three features: large- scale manual annotations, fine-grained argument types and application-oriented settings.

Our academic paper which describes DocEE in detail and provides full event ontology can be found here: https://tongmeihan1995.github.io/meihan.github.io/research/NAACL2022.pdf.

# Dataset Statistic
We are now the largest dataset for documnet-level event extraction.

| Datasets | #isDocEvent | #EvTyp. |#ArgTyp.| #Doc. | #Tok. | #Sent. | #ArgInst. | #ArgScat.|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACE2005 | KBP2016| KBP2017 | MUC-4 | WikiEvents | RAMS | DocEE(ours) |

✗  ✗
 ✓  ✓  ✓
 ✓

33 35 18 20 18 20 4 5 50 59
139 65 59 356

599 290k 169 94k 167 86k 1,700 495k 246 190k 9,124 957k
27,485 16,268k

15,789 5,295 4,839 21,928 8,544 34,536
749,568

9,590 1
7,919 1 10,929 1
2,641 4.0
5,536 2.2 21,237 4.8
180,528 10.2


# Dataset Display
DocEE focus on the extraction of the main event, that is *one-event-per-document*. 
![image](https://github.com/tongmeihan1995/DocEE/blob/main/image/dataset_display.pdf)

