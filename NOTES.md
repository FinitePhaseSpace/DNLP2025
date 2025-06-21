Following Resources were used to cross check and fix existing issues in our implementation:

* https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb
* https://github.com/Montinger/Transformer-Workbench/blob/main/transformer-from-scratch/model_defs.py#L55

While comparing those existing implementations we also found errors in those existing implementations:
* Both implementations do not share the memory for both embedding and output linear layer, even though this is clearly specified in the paper!