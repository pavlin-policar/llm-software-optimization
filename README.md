# Companion repository to "TBA"

## openTSNE

The benchmark script `benchmark-scripts/run_tsne.py` runs on the Zheng et al.
(2017) 10x Genomics mouse brain data set, which can be downloaded from
https://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz

```bash
curl -O https://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz
```

The script compares the optimized openTSNE in `openTSNE/` against the original
openTSNE v1.0.4, which can be cloned from GitHub:

```bash
git clone --branch v1.0.4 --depth 1 https://github.com/pavlin-policar/openTSNE.git openTSNE-original
```

Both are built in place with OpenMP:

```bash
(cd openTSNE && python setup.py build_ext --inplace)
(cd openTSNE-original && python setup.py build_ext --inplace)
```

```bash
python benchmark-scripts/run_tsne.py run --original openTSNE-original --optimized openTSNE --data 10x_mouse_zheng.pkl.gz
```
