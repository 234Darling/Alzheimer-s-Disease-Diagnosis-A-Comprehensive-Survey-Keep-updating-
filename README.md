# Alzheimer's Disease Diagnosis: A Comprehensive Survey (Keep updating)

## 🌟Introduction
This page compiles representative papers, codes, and datasets related to Alzheimer's Disease (AD) diagnosis. We aim to continuously update the latest research in this field to assist the community in tracking advancements. Contributions and questions are welcome—feel free to contact the relevant authors listed in the papers.

## 📚AD Diagnosis Papers
We have categorized papers on AD diagnosis based on their core methodologies and application scenarios. Detailed highlights of each paper can be explored through the links provided (where available).

### Main Methodologies
The mainstream methods for AD diagnosis primarily involve deep learning and can be roughly classified into the following categories:

#### Transformer-based Models
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Transformer-Based Multimodal Fusion for Early Diagnosis of Alzheimer's Disease Using Structural MRI And PET | Yuanwang Zhang; Kaicong Sun; Yuxiao Liu; Dinggang Shen | 2023 | 4 | [Github](https://github.com/Kateridge/TransMF_AD) | ADNI | ISBI |
| MAD-Former: A Traceable Interpretability Model for Alzheimer’s Disease Recognition Based on Multi-Patch Attention | Jiayu Ye; An Zeng; Yiqun Zhang; Jingliang Zhao; Qiuping Chen; Yang Liu | 2024 | - | [Github](https://github.com/yjy-97/MAD-Former) | ADNI, OASIS | JBHI |
| Understanding the Role of Self-Attention in a Transformer Model for the Discrimination of SCD From MCI Using Resting-State EEG | Elena Sibilano; Domenico Buongiorno; Michael Lassi; Antonello Grippo; Valentina Bessi; Sandro Sorbi; Alberto Mazzoni; Vitoantonio Bevilacqua; Antonio Brunetti | 2024 | 1 | [Github](https://github.com/LabInfInd/SCD_MCI_Transformer) | Private | JBHI |
| PVTAD: ALZHEIMER’S DISEASE DIAGNOSIS USING PYRAMID VISION TRANSFORMER APPLIED TO WHITE MATTER OF T1-WEIGHTED STRUCTURAL MRI DATA | Maryam Akhavan Aghdam; Serdar Bozdag; Fahad Saeed | 2024 | - | [Github](https://github.com/pcdslab/PVTAD) | ADNI | ISBI |
| DenseFormer-MoE: A Dense Transformer Foundation Model with Mixture of Experts for Multi-Task Brain Image Analysis | Ding R, Lu H, Liu M | 2025 | 1 | https://ieeexplore.ieee.org/document/10926590 | UKB, ADNI, PPMI | TMI |
| 3DNesT: A Hierarchical Local Self-Attention Model for Alzheimer's Disease Diagnosis | X Kang, Y Liu | 2023 | 0 | https://ieeexplore.ieee.org/document/10462762 | ADNI | ICNC |
| Trans-ResNet: Integrating Transformers and CNNs for Alzheimer’s disease classification | Li, C., Cui, Y., Luo, N., Liu, Y | 2022 | 35 | https://ieeexplore.ieee.org/document/9761549 | ADNI | ISBI |

#### Graph Convolutional Networks (GCN)
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Multi-Modal Diagnosis of Alzheimer’s Disease using Interpretable Graph Convolutional Networks | Houliang Zhou; Lifang He; Brian Y. Chen; Li Shen; Yu Zhang | 2024 | 0 | [Github](https://github.com/Houliang-Zhou/SGCN) | ADNI | TMI |
| A Graph Convolutional Network Based on Univariate Neurodegeneration Biomarker for Alzheimer’s Disease Diagnosis | Zongshuai Qu; Tao Yao; Xinghui Liu; Gang Wang | 2023 | 12 | https://ieeexplore.ieee.org/document/10149537 | ADNI | Journal of Translational Engineering in Health and Medicine |
| Interpretable modality-specific and interactive graph convolutional network on brain functional and structural connectomes | J **a, YH Chan, D Girish, JC Rajapakse | 2025 | 0 | https://www.sciencedirect.com/science/article/abs/pii/S136184152500057X | ADNI | MIA |

#### Convolutional Neural Networks (CNN)
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Fuzzy-VGG: A fast deep learning method for predicting the staging of Alzheimer's disease based on brain MRI | Zhaomin Yao; Wenxin Mao; Yizhe Yuan; Zhenning Shi; Gancheng Zhu; Wenwen Zhang; Zhiguo Wang; Guoxu Zhang | 2023 | 16 | https://www.sciencedirect.com/science/article/abs/pii/S0020025523007144 | KACD | Information Sciences |
| Conv-eRVFL: Convolutional Neural Network Based Ensemble RVFL Classifier for Alzheimer’s Disease Diagnosis | Rahul Sharma; Tripti Goel M Tanveer; P. N. Suganthan; Imran Razzak; R Murugan | 2023 | 23 | https://www.embs.org/jbhi/articles/imaging-informatics-9/ | ADNI | JBHI |
| LGG-NeXt: A Next Generation CNN and Transformer Hybrid Model for the Diagnosis of Alzheimer's Disease Using 2D Structural MRI | Bai, J., Zhang, Z., Yin, Y., **, W., Ali, T. A. A., **ong, Y., &**ao | 2025 | 0 | https://ieeexplore.ieee.org/document/10750309 | ADNI | JBHI |

#### Generative Adversarial Networks (GAN)
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Brain Status Transferring Generative Adversarial Network for Decoding Individualized Atrophy in Alzheimer’s Disease | Xingyu Gao; Hongrui Liu; Feng Shi; Dinggang Shen; Manhua Liu | 2023 | 4 | https://www.embs.org/jbhi/articles/imaging-informatics-9/ | ADNI, OASIS, AIBL | JBHI |
| CE-GAN: Community Evolutionary Generative Adversarial Network for Alzheimer's Disease Risk Prediction | Xia-An Bi; Zicheng Yang; Yangjun Huang; Ke Chen; Zhaoxu Xing; Luyun Xu; Zihao Wu; Zhengliang Liu; Xiang Li; Tianming Liu | 2024 | 3 | [Github](https://github.com/fmri123456/CE-GAN) | ADNI | TMI |
| IGUANe: A 3D generalizable CycleGAN for multicenter harmonization of brain MR images | Roca, V., Kuchcinski, G., Pruvo, J. P., Manouvriez, D., & Lopes, R | 2025 | 4 | https://www.sciencedirect.com/science/article/pii/S136184152400313X | ADNI | MIA |

#### Recurrent Neural Networks (RNN/LSTM)
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Multi-scale 3D Convolutional LSTM for Longitudinal Alzheimer's Disease Identification | Mengqing Liu; Xiao Shao; Liping Jiang; Kaizhi Wu | 2024 | 0 | https://ieeexplore.ieee.org/document/10635331 | Private | ISBI |
| Predicting Alzheimer’s Disease Progression Using a Versatile Sequence-Length-Adaptive Encoder-Decoder LSTM Architecture | Km Poonam; Rajlakshmi Guha; Partha P Chakrabarti | 2024 | 0 | https://ieeexplore.ieee.org/document/10495102 | ADNI | JBHI |

#### Self-Supervised Learning
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Detecting Early Risk of Alzheimer’s Disease Using Self-Supervised Multimodal Representation Learning | Jhon A. Intriago; Pablo A. Estevez; Jose A. Cortes-Briones; Cecilia A. Okuma; Fernando A. Henriquez; Patricia Lillo | 2023 | 2 | https://ieeexplore.ieee.org/document/10195072 | Private | Conference on Artificial Intelligence |
| Two-Stage Self-Supervised Cycle-Consistency Transformer Network for Reducing Slice Gap in MR Images | Zhiyang Lu; Jian Wang; Zheng Li; Shihui Ying; Jun Wang; Jun Shi; Dinggang Shen | 2023 | - | https://ieeexplore.ieee.org/document/10113164 | - | JBHI |
| Contrastive Learning for Prediction of Alzheimer’s Disease Using Brain 18F-FDG PET | Yonglin Chen; Huabin Wang; Gong Zhang; Xiao Liu; Wei Huang; Xianjun Han; Xuejun Li; Melanie Martin; Liang Tao | 2023 | 10 | https://ieeexplore.ieee.org/document/9999012 | ADNI | JBHI |

#### Multimodal Learning
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Alzheimer’s disease diagnosis from single and multimodal data using machine and deep learning models: Achievements and future directions | Ahmed Elazab; Chang miao Wang; Mohammed Abdelaziz; Jian Zhang; Jason Gu; Juan M. Gorriz; Yudong Zhang; Chunqi Chang | 2024 | 2 | https://www.sciencedirect.com/science/article/abs/pii/S0957417424016476 | - | Expert Systems with Applications |
| A Multimodal Deep Learning Approach for Automated Detection and Characterization of Distinctly Salient Features of Alzheimer's Disease | Ishaan Batta; Anees Abrol; Vince Calhoun | 2023 | 0 | https://ieeexplore.ieee.org/document/10230525 | ADNI | ISBI |
| Toward Robust Early Detection of Alzheimer's Disease via an Integrated Multimodal Learning Approach | Yifei Chen; Shenghao Zhu; Zhaojie Fang; Chang Liu; Binfeng Zou; Yuhe Wang; Shuo Chang; Fan Jia; Feiwei Qin; Jin Fan; Yong Peng; Changmiao Wang | 2024 | 0 | [Github](https://github.com/justlfc03/mstnet) | Private | arXiv |
| Enhanced Multimodal Low-Rank Embedding-Based Feature Selection Model for Multimodal Alzheimer’s Disease Diagnosis | Chen, Z., Liu, Y., Zhang, Y., Zhu, J., Li, Q., & Wu, X. | 2025 | 1 | https://ieeexplore.ieee.org/document/10684737 | ADNI | TMI |
| Hyperfusion: A hypernetwork approach to multimodal integration of tabular and medical imaging data for predictive modeling | **a, J., Chan, Y. H., Girish, D., & Rajapakse, J. C. | 2025 | 0 | https://www.sciencedirect.com/science/article/abs/pii/S1361841525000519 | ADNI | MIA |

#### Other Methods
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| Multi-Template Meta-Information Regularized Network for Alzheimer’s Disease Diagnosis Using Structural MRI | Kangfu Han; Gang Li; Zhiwen Fang; Feng Yang | 2023 | 3 | https://ieeexplore.ieee.org/document/10365189 | ADNI, NACC | TMI |
| Federated Domain Adaptation via Transformer for Multi-Site Alzheimer’s Disease Diagnosis | Baiying Lei; Yun Zhu; Enmin Liang; Peng Yang; Shaobin Chen; Huoyou Hu; Haoran Xie; Ziyi Wei | 2023 | 13 | https://ieeexplore.ieee.org/document/10198494 | ADNI | TMI |
| RClaNet: An Explainable Alzheimer’s Disease Diagnosis Framework by Joint Registration and Classification | Liang Wu; Shunbo Hu; Duanwei Wang; Changchun Liu; Li Wang | 2024 | 3 | [Github](https://github.com/LiangWUSDU/RClaNet) | ADNI,OASIS-3,AIBL,COVID-19 | JBHI |
| Visual-Attribute Prompt Learning for Progressive Mild Cognitive Impairment Prediction | Luoyao Kang; Haifan Gong; Xiang Wan; Haofeng Li | 2023 | 6 | https://arxiv.org/abs/2310.14158 | ADNI | MICCAI |
| Amyloid-β Deposition Prediction With Large Language Model Driven and Task-Oriented Learning of Brain Functional Networks | Liu Y, Liu M, Zhang Y | 2025 | 1 | https://pubmed.ncbi.nlm.nih.gov/40030867/ | ADNI,Huashan，OASIS | TMI |
| Longitudinal Alzheimer's Disease Progression Prediction With Modality Uncertainty and Optimization of Information Flow | Dao, D. P., Yang, H. J., Kim, J., & Ho, N. H.|2025|0|https://ieeexplore.ieee.org/document/10702601|ADNI|JBHI|
| BIGFormer: A Graph Transformer With Local Structure Awareness for Diagnosis and Pathogenesis Identification of Alzheimer's Disease Using Imaging Genetic Data|Zou, Q., Shang, J., Liu, J. X., & Gao, R.|2025|1|https://ieeexplore.ieee.org/document/10648828|ADNI|JBHI|
|Image-and-Label Conditioning Latent Diffusion Model: Synthesizing Aβ-PET From MRI for Detecting Amyloid Status|Ou, Z., Pan, Y.,**e, F., Guo, Q., & Shen, D. |2025|1|https://ieeexplore.ieee.org/document/10752348|ADNI|JBHI|

### Papers by Author "Y Liu"
| Title | Authors | Year | Citations | Code | Datasets | Journal/Conference |
|-------|---------|------|-----------|------|----------|-------------------|
| 3DNesT: A Hierarchical Local Self-Attention Model for Alzheimer's Disease Diagnosis | X Kang, Y Liu | 2023 | 0 | https://ieeexplore.ieee.org/document/10462762 | ADNI | ICNC |
| A systematic analysis of diagnostic performance for Alzheimer’s disease using structural MRI | J Wu, K Zhao, Z Li, D Wang, Y Ding, Y Wei, H Zhang, Y Liu | 2022 | 6 | https://pubmed.ncbi.nlm.nih.gov/38665142/ | - | Psychoradiology |
| Structure–function coupling reveals the brain hierarchical structure dysfunction in Alzheimer’s disease: A multicenter study | Y Sun, P Wang, K Zhao, P Chen, Y Qu, Z Li, S Zhong, B Zhou, J Lu, X Zhang, D Wang, Y Liu | 2024 | 0 | https://pubmed.ncbi.nlm.nih.gov/39072981/ | Private | Alzheimer's & Dementia |
| Convergent Neuroimaging and Molecular Signatures in Mild Cognitive Impairment and Alzheimer’s Disease: A Data‑Driven Meta‑Analysis with N = 3,118 | X Kang, D Wang, J Lin, H Yao, K Zhao, C Song, P Chen, Y Qu, H Yang, Z Zhang, B Zhou, Y Liu | 2024 | 0 | https://pubmed.ncbi.nlm.nih.gov/38824231/ | MCAD, ADNI, EDSD | Neuroscience Bulletin |
| Coupling of the spatial distributions between sMRI and PET reveals the progression of Alzheimer’s disease | K Zhao, J Lin, M Dyrba, D Wang, T Che, H Wu, J Wang, Y Liu, S Li | 2023 | 5 | https://pubmed.ncbi.nlm.nih.gov/37334010/ | ADNI | Network Neuroscience |
| Delineating the Heterogeneity of Alzheimer’s Disease and Mild Cognitive Impairment Using Normative Models of the Dynamic Brain Functional Networks | Y Huo, R **g, P Li, P Chen, J Si, G Liu, Y Liu | 2024 | 2 | https://pubmed.ncbi.nlm.nih.gov/38857821/ | Cam-CAN, MCAD, ADNI | Biological Psychiatry |
| Editorial: Neuroimaging Biomarkers and Cognition in Alzheimer’s Disease Spectrum | J Chen, S Wang, R Chen, Y Liu | 2022 | 3 | https://pubmed.ncbi.nlm.nih.gov/35283754/ | - | Frontiers in Aging Neuroscience |
| Four distinct subtypes of Alzheimer's disease based on restingstate connectivity biomarkers | P Chen, H Yao, BM Tijms, P Wang, D Wang, C Song，Y Liu | 2023 | 43 | https://pubmed.ncbi.nlm.nih.gov/36137824/ | MCADI | Biological Psychiatry |
| Altered large-scale dynamic connectivity patterns in Alzheimer's disease and mild cognitive impairment patients: A machine learning study | R**g, P Chen, Y Wei, J Si, Y Zhou, D Wang, C Song, H Yang, Z Zhang, H Yao, X Kang, Y Liu | 2023 | 7 | https://pubmed.ncbi.nlm.nih.gov/36988434/ | YN, MACD, ADNI | Human Brain Map |
| Learning with Domain-Knowledge for Generalizable Prediction of Alzheimer’s Disease from Multi-site Structural MRI | Y Zhou, Y Liu, F Zhou, Y Liu, L Tu | 2023 | 2 | https://link.springer.com/chapter/10.1007/978-3-031-43904-9_44 | - | Cham: Springer Nature Switzerland |
| MACROSCALE BRAIN STRUCTURAL NETWORK COUPLING IS RELATED TO AD PROGRESSION | Y Sun, P Chen, Y Liu, K Zhao | 2024 | 0 | https://ieeexplore.ieee.org/document/10635655 | - | ISBI |
| Mapping cerebral atrophic trajectory from amnestic mild cognitive impairment to Alzheimer’s disease | X Wei, X Du, Y Xie, X Suo, X He, H Ding，Y Liu | 2023 | 10 | https://pubmed.ncbi.nlm.nih.gov/35368064/ | ADNI | Cerebral Cortex |
| Multipredictor risk models for predicting individual risk of Alzheimer’s disease | Hou, X. H., Suckling, J., Shen, X. N., Liu, Y. | 2023 | 2 | https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-023-04646-x | ADNI | Journal of translational medicine |
| A neuroimaging biomarker for Individual Brain-Related Abnormalities In Neurodegeneration (IBRAIN): a cross-sectional study | Zhao, K., Chen, P., Alexander-Bloch, A., Wei, Y., Y Liu | 2023 | 11 | https://pubmed.ncbi.nlm.nih.gov/37954904/ | ADNI, MCADI, OASIS | EClinicalMedicine |
| Predicting Conversion to Mild Cognitive Impairment in Cognitively Normal with Incomplete Multi-modal Neuroimages | Y Sun, Y Liu, B Liu | 2024 | 0 | https://ieeexplore.ieee.org/document/9802479 | ADNI | ICBCB |

## 📊Datasets

### Public Datasets
| Dataset | Resolution | Classes/Findings | Collected By | # of Samples | Link (if available) |
|---------|------------|------------------|--------------|--------------|---------------------|
| ADNI (Alzheimer's Disease Neuroimaging Initiative) | Various (depends on imaging modality) | AD, MCI (Mild Cognitive Impairment), Normal, etc. | Multiple research institutions and organizations | Large (ongoing collection) | - |
| OASIS (Open Access Series of Imaging Studies) | Various | AD, healthy aging, cognitive impairment | University of Pennsylvania | - | - |
| OASIS-3 | Various | AD, MCI, Normal, etc. | University of Pennsylvania | - | - |
| AIBL (Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing) | Various | AD, MCI, Normal | Australian research institutions | - | - |
| NACC (National Alzheimer's Coordinating Center) | - | AD, MCI, related neurodegenerative diseases | National Institute on Aging (NIA) | - | - |
| KACD (Korean Alzheimer's Disease Dataset) | - | AD staging-related categories | Korean research institutions | - | - |
| UKB (UK Biobank) | Various | Large number of health conditions including neurodegenerative diseases | UK Biobank Organization | - | - |
| PPMI (Parkinson's Progression Markers Initiative) | Various | Parkinson's disease and related neurodegenerative conditions (used for comparative studies) | International Parkinson's research community | - | - |
| MCAD (Multicenter Alzheimer's Disease) | - | AD, MCI | Multiple clinical centers | - | - |
| EDSD (European Dementia Services Development Group) | - | Dementia spectrum disorders | European institutions | - | - |
| Cam-CAN (Cambridge Centre for Ageing and Neuroscience) | Various | Normal aging, cognitive decline | University of Cambridge | - | - |
| Montgomery (TB dataset, used in related neuroimaging studies) | 4020×4892 | Normal and TB (relevant for comorbidity studies) | Montgomery County Department of Health and Human Services | 138 | https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery |
| Shenzhen (TB dataset, used in related studies) | 3000×3000 | Normal and TB (relevant for comorbidity studies) | Shenzhen No. 3 People’s Hospital, Guangdong Medical College | 662 | https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen |
| RSNA-Pneumonia-CXR (used in related medical imaging studies) | Random | Pneumonia, infiltration, consolidation (relevant for comorbidity studies) | RSNA and STR | 15,000 | https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data |

### Private Datasets
| Dataset | Resolution | Classes/Findings | Collected By | # of Samples | Link |
|---------|------------|------------------|--------------|--------------|------|
| - | - | AD-related categories | Nanfang Hospital, China (implied in similar studies) | - | - |
| - | - | AD, MCI, Normal | Private clinical centers (mentioned in individual papers) | - | - |
| YN dataset | - | AD, MCI | - | - | - |
| MACD dataset | - | AD, MCI | - | - | - |

### Specialized Datasets
| Dataset | Resolution | Purpose | Collected By | # of Samples | Link (if available) |
|---------|------------|---------|--------------|--------------|---------------------|
| TADPOLE (Toolkit for Analysis of Dynamics in PET, MRI and Other Longitudinal Data) | Various | AD progression prediction | - | - | - |
| Huashan dataset | - | Amyloid-β deposition-related categories | Huashan Hospital | - | - |

## 📝Acknowledgements
Thanks to all researchers and institutions who contributed to the papers, datasets, and codes included in this survey. Special recognition to the authors of the compiled papers for their valuable work in advancing Alzheimer's disease diagnosis research. If you find any missing datasets or papers, please contact the relevant authors to update the list.
