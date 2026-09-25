"""Download TCGA-BRCA STAR TPM and MSigDB GO biological-process sets.

Writes two files next to this script and deletes the downloads:

- ``tcga-brca.star_tpm.tsv`` — genes by samples, HGNC symbols, ``log2(tpm+1)``
- ``c5.go.bp.v2026.1.Hs.json`` — ``{set name: [symbols]}`` restricted to those genes

The GSEA download link requires an account. If it redirects to the login page,
the script uses the public Broad Institute copy of the same JSON.
"""

from __future__ import annotations

import gzip
import json
import os
import ssl
import urllib.error
import urllib.request
from array import array
from os.path import abspath, dirname, join

HERE = dirname(abspath(__file__))

EXPRESSION_URL = (
    "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_tpm.tsv.gz"
)
PROBEMAP_URL = (
    "https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
    "gencode.v36.annotation.gtf.gene.probemap"
)
SIGNATURE_URL = (
    "https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp"
    "?filePath=/msigdb/release/2026.1.Hs/c5.go.bp.v2026.1.Hs.json"
)
SIGNATURE_MIRROR_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2026.1.Hs/c5.go.bp.v2026.1.Hs.json"
)

EXPRESSION_GZ = join(HERE, "TCGA-BRCA.star_tpm.tsv.gz")
PROBEMAP_PATH = join(HERE, "gencode.v36.annotation.gtf.gene.probemap")
SIGNATURE_RAW = join(HERE, "c5.go.bp.v2026.1.Hs.full.json")
EXPRESSION_OUT = join(HERE, "tcga-brca.star_tpm.tsv")
SIGNATURE_OUT = join(HERE, "c5.go.bp.v2026.1.Hs.json")

INTERMEDIATES = (EXPRESSION_GZ, PROBEMAP_PATH, SIGNATURE_RAW)
USER_AGENT = "openSSGSEA-download/1.0"


class LoginRequired(Exception):
    """The signature host sent the request to an account login page."""


def log(message: str) -> None:
    print(message, flush=True)


def download(url: str, dest: str) -> None:
    """Stream ``url`` to ``dest``. Raise ``LoginRequired`` on an HTML login page."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    temporary = dest + ".partial"
    try:
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            if "login" in final_url.lower() or "text/html" in content_type.lower():
                raise LoginRequired(final_url)
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            next_report = 50 * 1024 * 1024
            with open(temporary, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if total and written >= next_report:
                        log(f"  {written // (1024 * 1024)} / {total // (1024 * 1024)} MiB")
                        next_report += 50 * 1024 * 1024
        os.replace(temporary, dest)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def download_signatures(dest: str) -> None:
    try:
        log(f"downloading signatures from {SIGNATURE_URL}")
        download(SIGNATURE_URL, dest)
    except (LoginRequired, urllib.error.HTTPError) as error:
        log(
            "MSigDB requires a free account for that download "
            f"({error}). Using the Broad Institute copy of the same file."
        )
        download(SIGNATURE_MIRROR_URL, dest)


def load_probemap(path: str) -> dict[str, str]:
    """Map versioned Ensembl gene IDs to HGNC symbols."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:2] != ["id", "gene"]:
            raise ValueError(f"unexpected probe map header: {header[:2]}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"short probe map record at line {line_number}")
            gene_id, symbol = fields[0], fields[1]
            if not gene_id or not symbol or symbol == ".":
                continue
            mapping.setdefault(gene_id, symbol)
    if not mapping:
        raise ValueError("probe map contains no gene identifiers")
    return mapping


def load_signature_symbols(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("signature file is not a JSON object of gene sets")
    gene_sets: dict[str, list[str]] = {}
    for name, record in payload.items():
        if not isinstance(record, dict) or "geneSymbols" not in record:
            raise ValueError(f"gene set {name!r} has no geneSymbols")
        symbols = record["geneSymbols"]
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise ValueError(f"gene set {name!r} has a malformed geneSymbols list")
        gene_sets[name] = symbols
    return gene_sets


def accumulate_expression(
    path: str,
    id_to_symbol: dict[str, str],
    universe: set[str],
) -> tuple[list[str], list[str], dict[str, array], dict[str, int]]:
    """Sum ``log2(tpm+1)`` rows that map into ``universe``. Return samples, order, sums, counts."""
    sums: dict[str, array] = {}
    counts: dict[str, int] = {}
    order: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if not header or header[0] != "Ensembl_ID":
            raise ValueError(f"unexpected expression header: {header[:1]}")
        samples = header[1:]
        if not samples:
            raise ValueError("expression file has no samples")
        width = len(samples)
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != width + 1:
                raise ValueError(
                    f"expression row {line_number} has {len(fields) - 1} values, expected {width}"
                )
            symbol = id_to_symbol.get(fields[0])
            if symbol is None or symbol not in universe:
                continue
            values = [float(value) for value in fields[1:]]
            existing = sums.get(symbol)
            if existing is None:
                sums[symbol] = array("d", values)
                counts[symbol] = 1
                order.append(symbol)
            else:
                for index, value in enumerate(values):
                    existing[index] += value
                counts[symbol] += 1
    if not order:
        raise ValueError("no expression rows mapped onto the signature symbols")
    return samples, order, sums, counts


def align(
    gene_sets: dict[str, list[str]],
    expressed: set[str],
) -> tuple[dict[str, list[str]], set[str]]:
    """Drop signature members missing from the matrix, then genes left in no set."""
    aligned: dict[str, list[str]] = {}
    for name, symbols in gene_sets.items():
        kept = [symbol for symbol in symbols if symbol in expressed]
        if kept:
            aligned[name] = kept
    if not aligned:
        raise ValueError("no gene set overlaps the expression symbols")
    used = {symbol for symbols in aligned.values() for symbol in symbols}
    return aligned, used


def write_expression(
    path: str,
    samples: list[str],
    order: list[str],
    sums: dict[str, array],
    counts: dict[str, int],
    used: set[str],
) -> int:
    written = 0
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("gene_symbol\t" + "\t".join(samples) + "\n")
        for symbol in order:
            if symbol not in used:
                continue
            count = counts[symbol]
            row = sums[symbol]
            if count == 1:
                values = (format(value, ".10g") for value in row)
            else:
                values = (format(value / count, ".10g") for value in row)
            handle.write(symbol + "\t" + "\t".join(values) + "\n")
            written += 1
    os.replace(temporary, path)
    return written


def write_signatures(path: str, gene_sets: dict[str, list[str]]) -> None:
    temporary = path + ".partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(gene_sets, handle, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def remove_intermediates() -> None:
    for path in INTERMEDIATES:
        if os.path.exists(path):
            os.remove(path)


def main() -> None:
    os.makedirs(HERE, exist_ok=True)
    try:
        log(f"downloading expression from {EXPRESSION_URL}")
        download(EXPRESSION_URL, EXPRESSION_GZ)
        log(f"downloading probe map from {PROBEMAP_URL}")
        download(PROBEMAP_URL, PROBEMAP_PATH)
        download_signatures(SIGNATURE_RAW)

        id_to_symbol = load_probemap(PROBEMAP_PATH)
        gene_sets = load_signature_symbols(SIGNATURE_RAW)
        universe = {symbol for symbols in gene_sets.values() for symbol in symbols}
        log(f"mapping expression onto {len(universe)} signature symbols")
        samples, order, sums, counts = accumulate_expression(
            EXPRESSION_GZ, id_to_symbol, universe
        )
        aligned, used = align(gene_sets, set(order))
        n_genes = write_expression(EXPRESSION_OUT, samples, order, sums, counts, used)
        write_signatures(SIGNATURE_OUT, aligned)
        remove_intermediates()
    except BaseException:
        for path in (EXPRESSION_OUT + ".partial", SIGNATURE_OUT + ".partial"):
            if os.path.exists(path):
                os.remove(path)
        raise

    duplicated = sum(1 for symbol in order if symbol in used and counts[symbol] > 1)
    log(
        f"wrote {n_genes} genes x {len(samples)} samples to {os.path.basename(EXPRESSION_OUT)}"
    )
    log(
        f"wrote {len(aligned)} gene sets to {os.path.basename(SIGNATURE_OUT)} "
        f"({duplicated} symbols averaged across Ensembl IDs)"
    )


if __name__ == "__main__":
    main()
