# Multi-Domain Image Retrieval

A code repository for the following publication:

**No Fear of the Dark:** Image Retrieval under Varying Illumination Conditions  
[Tomas Jenicek][jenicek] and [Ondřej Chum][chum]  
In International Conference on Computer Vision (ICCV), 2019

Related:&nbsp; [project website][daynight],&nbsp; [paper pdf][arxiv]

This codebase builds on top of [cirtorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch). Its patched version is distributed together with the code and its dataset format is honored.


## Getting Started

Clone repository, install dependencies:

```
git clone https://github.com/jenicek/mdir.git
cd mdir
pip3 install -r requirements.txt
```

Optionally, two environment variables may be specified:

- `CIRTORCH_ROOT` -- path for downloaded datasets, default is mdir top folder
- `CUDA_VISIBLE_DEVICES` -- gpu to be used for inference (training), default is index 0

Test inference:

```
cd mdir/examples/iccv19
./eval.py test
```

Datasets will be downloaded and stored in `CIRTORCH_ROOT`, trained models will be loaded from remote urls each time (see yaml scenario files). You should get following output:

```
    roxford.5k medium    39.06
    rparis.6k medium     58.94
    247tokyo.1k          72.0
```

The `test` argument is a shortcut for a yaml scenario which fully defines the evaluation. If multiple yaml scenarios are provided, they will be overlayed in the order they were provided. Shortcut `test` is equal to arguments `eval.yml eval_test.yml`.


## Evaluation

In order to evaluate trained models from the ICCV19 paper, the following scenarios are provided in `mdir/examples/iccv19`

- `eval_clahe.yml` (shortcut `clahe`) -- "CLAHE N/D" method
- `eval_composition.yml` (shortcut `composition`) -- "U-Net jointly N/D" method

Configuration common for both scenarios is in `eval.yml`


## Training

Currently, convenient scripts are provided only for evaluation. For training, only the code is provided. The repository is under development; training scripts will appear soon.


<!-- References -->

[daynight]: http://cmp.felk.cvut.cz/daynightretrieval/
[arxiv]: https://arxiv.org/pdf/1908.08999.pdf
[jenicek]: http://cmp.felk.cvut.cz/~jenicto2
[chum]: http://cmp.felk.cvut.cz/~chum
