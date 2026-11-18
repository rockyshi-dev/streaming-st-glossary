# Data sources and licence

## Corpus text: ACL 60/60

`sentences.jsonl` (224 gold-segmented sentences of ACL 60/60 dev and eval that contain a glossary term, from 10
talks; the 116 decision items, one sentence per term and talk, are marked `first: true`; with the three glossaries
of each item) is derived from the ACL 60/60 evaluation sets:

- Elizabeth Salesky, Kareem Darwish, Mohamed Al-Badrashiny, Mona Diab, Jan Niehues. 2023. *Evaluating Multilingual
  Speech Translation Under Realistic Conditions with Resegmentation and Terminology.* IWSLT 2023.
  https://aclanthology.org/2023.iwslt-1.2/
- Download: https://aclanthology.org/attachments/2023.iwslt-1.2.dataset.zip (948,169,976 bytes, sha256
  `5f2a3855b5f442c83e6461c32e8a8deb6c2b053518b02b957eb4686bacfce7cc`).

**Licence.** The dataset's own README does not name a licence; it asks that users cite the paper above (done here).
The dataset is published as an attachment of the paper in the ACL Anthology, whose footer states that materials
published at ACL venues in 2016 or later are licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/), and a paper in the same IWSLT 2023 proceedings describes the ACL 60/60 data as CC BY 4.0
(https://aclweb.org/anthology/2023.iwslt-1.28.pdf). The derived text in this directory is therefore shared under
CC BY 4.0 with attribution to Salesky et al. (2023). Other pages: https://iwslt.org/2023/multilingual,
https://github.com/IWSLT-23/60_60_data (the shared-task repository, which carries no licence file).

No audio is included. `sentences.jsonl` stores, per sentence, the relative path of its segmented wav inside the
extracted zip (`wav`); the full re-decode reads the audio from there.

`python -m stglossary.build_items CORPUS_ROOT OUT_DIR --seed 20261117` rebuilds `sentences.jsonl` byte for byte
(sha256 `3c84b591e61f1e69eb97e80d3bb8a12759d003fa8c38f59625e81465599050d5`) from the text files of the zip
(`2/acl_6060/{dev,eval}/text/` and `2/intermediate_files/terminology_glossary.csv`) and the hand labels in
`stglossary/labels.tsv` and `stglossary/pair_rejects.tsv`.

## Model outputs

All decodes are greedy, batch size 1, at most 256 new tokens, with the `instructed` prompt (`stglossary/prompts.py`).
Translations are model outputs on the CC BY 4.0 source text above and are shared under the same terms.

| File | System | Rows |
|---|---|---|
| `offline.jsonl` | Qwen2.5-Omni-7B, audio in, whole segment | 696 = 116 items x 6 decodes |
| `streaming.jsonl` | Qwen2.5-Omni-7B, audio in, LA-2 on 1.0 s steps, sentence-final mark held back | 696 |
| `streaming-before-holdback.jsonl` | the same, without the hold-back; only key, duration and per-step texts | 696 |
| `align.jsonl` | Whisper large-v3 (faster-whisper) forced alignment of the gold transcript: word times | 116 |
| `panel/text-llm.jsonl` | Qwen2.5-7B-Instruct (vLLM), gold transcript in; `extra` rows are repeat passes, not counted | 2688 |
| `panel/qwen2-audio-text.jsonl` | Qwen2-Audio-7B-Instruct, gold transcript in | 1344 = 224 sentences x 6 |
| `panel/qwen25-omni-text.jsonl` | Qwen2.5-Omni-7B, gold transcript in | 1344 |

The panel files cover all 224 sentences; every count reads the 116 decision sentences only. Each sentence has six
decodes: the glossary without the item's term (`none`, three times), with the correct entry
(`correct`), with an unrelated entry (`irrelevant`) and with a wrong target (`wrong`).

Models (Hugging Face id and revision, also stored in each row's `model` field):

| Model | Revision |
|---|---|
| Qwen/Qwen2.5-Omni-7B | ae9e1690543ffd5c0221dc27f79834d0294cba00 |
| Qwen/Qwen2-Audio-7B-Instruct | 0a095220c30b7b31434169c3086508ef3ea5bf0a |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| Systran/faster-whisper-large-v3 | edaa852ec7e145841d8ffdb056a99866b5f0a478 |

The rows were written by `stglossary/decode_hf.py`, `stream.py`, `align.py` and `decode_vllm.py`, then passed through
`stglossary/export.py`, which removes the `commit` field (a hash of the unpublished working tree the decodes ran from)
and replaces the local model path in `model` by the id and revision above. Nothing else is changed; in particular
`prompt` and `hyp` are byte-identical to what the model produced, which the hand-code files check (`hyp_sha`). The
streaming rows' `align_sha256` is the hash of the alignment file before this export.

## Hand codes

Every output whose translation contains the wrong form w' (arms `wrong` and `irrelevant`) was read by one person
and coded as a real use of w' as the translation of the term, or a false hit (another sense, part of a longer word,
a copy of the glossary). Rows were shuffled and shown without system, arm or condition.

- `coding/CODE.tsv`, `coding/KEY.tsv`: streaming vs offline, with the coder's guess of the condition per row
  (`condition_guess`), from which the blinding check is computed.
- `coding/CCODE.tsv`, `coding/CKEY.tsv`: spot check of 10 streaming `correct` outputs (does it use w for t?).
- `panel/CODE.tsv`, `panel/KEY.tsv`: the offline panel's codes for the three systems above.

`KEY.tsv` maps each opaque row id to its system and output (`hyp_sha` = first 16 hex digits of the output's
sha256); the counts refuse to run if an output no longer matches its code.
