# FLEURS example: data sources and licence

The files in this directory are the inputs and outputs of the FLEURS example search (README.md, section "FLEURS
example"). No audio is included.

## Sources

- **FLEURS** (Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, SLT 2022,
  arXiv 2205.12446): Hugging Face `google/fleurs`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, configs
  `en_us` (dev and test: audio and transcripts) and `cmn_hans_cn` (dev and test transcripts). Licence CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). `fleurs/fetch_fleurs.sh` lists every file with its size and sha256.
- **FLoRes-200** (NLLB Team, *No Language Left Behind: Scaling Human-Centered Machine Translation*, 2022,
  arXiv 2207.04672; built on FLoRes-101, Goyal et al., TACL 2022): `flores200_dataset.tar.gz` from
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (25,585,843 bytes, sha256
  `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`). Licence CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). FLEURS sentences are FLoRes devtest sentences; the English line
  (`eng_Latn`), the Chinese reference (`zho_Hans`) and the source article (`metadata_devtest.tsv`) come from here.

**Licence of these files.** Every file below holds FLoRes-200 text (the English sentence, the `zho_Hans` reference
line, or glossary entries and translations made from them), so the files are shared under **CC BY-SA 4.0**, with
attribution to FLoRes-200 (NLLB Team 2022; Goyal et al. 2022) and FLEURS (Conneau et al. 2022). The code that made
them is MIT (`LICENSE`).

## Models

| Model | Revision | Used for |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | glossary terms (`terms`), vLLM 0.30.0 |
| Systran/faster-whisper-large-v3 | edaa852ec7e145841d8ffdb056a99866b5f0a478 | word times of the gold transcript (`align`) |
| Qwen/Qwen2.5-Omni-7B | ae9e1690543ffd5c0221dc27f79834d0294cba00 | streaming translation (`stream`) |

## Files and the command that wrote each

The stage names are those of `fleurs/run.sh`; the commands are the ones it runs.

| File | Rows | Written by |
|---|---|---|
| `terms_raw.jsonl` | 820 (one per FLoRes line of a kept article) | `terms`: `python -m fleurs.terms --lines OUT/prep/lines.jsonl --model Qwen2.5-7B-Instruct --out terms_raw.jsonl` |
| `sentences.jsonl` | 234 sentences, 225 decision items (`first: true`) from 161 articles | `items`: `python -m fleurs.items --lines OUT/prep/lines.jsonl --terms terms_raw.jsonl --out OUT/items --min-dur 8.0` |
| `align.jsonl` | 225 | `align`: `python -m stglossary.align --sentences sentences.jsonl --corpus FLEURS --subset first --model faster-whisper-large-v3 --out align.jsonl` |
| `stream.jsonl` | 675 = 225 items x 3 arms (correct, irrelevant, wrong) | `stream`: `python -m fleurs.stream3 decode --mode speech --family omni25 --model Qwen2.5-Omni-7B --sentences sentences.jsonl --corpus FLEURS --align align.jsonl --chunk 1.0 --prompt instructed --subset first --out stream.jsonl` |
| `scan.json` | 111 examples | `scan`: `python -m fleurs.scan --stream stream.jsonl --align align.jsonl --sentences sentences.jsonl --out scan.json` |

`prep` (`python -m fleurs.prep --corpus FLEURS --out OUT/prep`) reads only the corpus, so its output is not stored
here; it is rebuilt from the corpus in seconds.

`terms_raw.jsonl`, `align.jsonl` and `stream.jsonl` were passed through `stglossary/export.py` (`python -m
stglossary.export jsonl SRC DST --model ID@REVISION`), which removes the `commit` field and replaces the local model
path in `model` by the id and revision above; nothing else is changed. `sentences.jsonl` is unchanged
(sha256 `886ddd06209c344fd5b8c911615440645f9ab8a58c06eef7831fa57d57349156`, the `items_sha256` of every streaming row).
The streaming rows' `align_sha256` is the hash of the alignment file before the export. `scan.json` was then
recomputed from the files in this directory; its examples are identical to those of the scan of the original files,
and its `*_sha256` fields are the hashes of the files here.

The streaming rows carry `runs: "main"` in their setting although only three arms were decoded: `fleurs/stream3.py`
replaces the run set of `stglossary.stream` and leaves the rest of its code unchanged.
