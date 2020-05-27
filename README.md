
# PyTorch (+ AllenNLP) Reimplementation of [SPIGOT](https://arxiv.org/abs/1805.04658v1) Syntactic-then-semantic Parser

This repo tries to reimplement a pipeline system of syntactic and semantic parsers, trained end-to-end, using the technique, called "SPIGOT", proposed in ACL2018 paper [Backpropagating through Structured Argmax using a SPIGOT](https://arxiv.org/abs/1805.04658v1) by Peng et al.

## [SemEval 2015](http://alt.qcri.org/semeval2015/task18/) Broad-Coverage Semantic Dependency Parsing

(from their paper)

|Model| - | - |
|:---:|:---:|:---:|
|-| - | - |
|-| - | - |

(reproduced result by me)

|Model| - | - |
|:---:|:---:|:---:|
|-| - | - |
|-| - | - |

## Running the code

Please refer to `requirements.txt` for the versions of libraries used in the reproduction.

```sh
pip install torch allennlp allennlp-models git+https://github.com/masashi-y/allennlp_spigot
```

For training,

```sh
allennlp train --include-package spigot --serialization-dir results configs/syntactic_then_semantic_dependencies.jsonnet
```




## Citation Information

```
@inproceedings{peng-etal-2018-backpropagating,
    title = "Backpropagating through Structured Argmax using a {SPIGOT}",
    author = "Peng, Hao  and
      Thomson, Sam  and
      Smith, Noah A.",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1173",
    doi = "10.18653/v1/P18-1173",
    pages = "1863--1873",
    abstract = "We introduce structured projection of intermediate gradients (SPIGOT), a new method for backpropagating through neural networks that include hard-decision structured predictions (e.g., parsing) in intermediate layers. SPIGOT requires no marginal inference, unlike structured attention networks and reinforcement learning-inspired solutions. Like so-called straight-through estimators, SPIGOT defines gradient-like quantities associated with intermediate nondifferentiable operations, allowing backpropagation before and after them; SPIGOT{'}s proxy aims to ensure that, after a parameter update, the intermediate structure will remain well-formed. We experiment on two structured NLP pipelines: syntactic-then-semantic dependency parsing, and semantic parsing followed by sentiment classification. We show that training with SPIGOT leads to a larger improvement on the downstream task than a modularly-trained pipeline, the straight-through estimator, and structured attention, reaching a new state of the art on semantic dependency parsing.",
}
```
