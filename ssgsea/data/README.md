# TCGA-BRCA expression and GO biological-process sets

The expression matrix and the gene sets are not in this repository. Download and align them with:

```bash
python download_script.py
```

Run that from this directory, or pass the script path from anywhere. The script needs only Python 3. It writes two files here and deletes the downloads:

- `tcga-brca.star_tpm.tsv` — genes by samples. The first column is `gene_symbol` (HGNC). Remaining columns are TCGA sample barcodes. Values are `log2(tpm+1)` as published by UCSC Xena. Rows are restricted to symbols that remain in at least one gene set below. When several Ensembl IDs share a symbol, their values are averaged.
- `c5.go.bp.v2026.1.Hs.json` — MSigDB C5 GO Biological Process, release 2026.1.Hs, as `{set name: [symbols]}`. Each list keeps only symbols present in the expression table, and sets that become empty are dropped.

The benchmark size window (overlap 15 to 500) is not applied here. From `llm-software-optimization/`:

```bash
python benchmark-scripts/run_ssgsea.py run --data ssgsea/data/tcga-brca.star_tpm.tsv --genesets ssgsea/data/c5.go.bp.v2026.1.Hs.json
```

## Sources

Expression is the GDC TCGA Breast Cancer (BRCA) STAR TPM matrix from the [UCSC Xena portal](https://xena.ucsc.edu/), file [TCGA-BRCA.star_tpm](https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.star_tpm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net). Xena averages measurements from the same sample and stores `log2(tpm+1)`. Gene rows are versioned Ensembl IDs from GENCODE v36. The script maps them with Xena's `gencode.v36.annotation.gtf.gene.probemap`.

The cohort is [The Cancer Genome Atlas](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) breast cancer data. Quantification follows the [GDC mRNA analysis pipeline](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/).

Gene sets are the human [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp) C5 GO Biological Process collection, release 2026.1.Hs (`c5.go.bp.v2026.1.Hs.json`). Downloading that JSON from the GSEA site requires a free account. If the download is redirected to the login page, the script uses the public Broad Institute copy of the same file and says so.
