# Wrong glossary entries in streaming speech translation

A speech LLM (Qwen2.5-Omni-7B) translates 116 English sentences into Chinese with a glossary that holds one wrong
entry, once offline on the whole sentence and once in streaming mode (LA-2 on growing 1.0 s audio prefixes); this
repository holds the code, the model outputs and the hand codes behind every number in the post. Streaming copies the
wrong entry into 12 outputs against 11 offline, and the 95% interval of the difference excludes a rise of 10% of the
items.

Post: https://rockyshi.dev/posts/streaming-speech-translation-wrong-glossary/

## Requirements

- Step 1 (recount): Python 3.10 or newer, standard library only. Any CPU; it runs in a few seconds.
- Step 2 (re-decode): one 24 GB GPU (the timings below are from an RTX 4090), the ACL 60/60 audio, and
  - `torch` 2.11, `transformers` 5.20.1, `librosa` 0.11, `accelerate` 1.16 (Qwen2.5-Omni and Qwen2-Audio decodes);
  - `faster-whisper` 1.2.1 with `ctranslate2` 4.8.3 (word alignment);
  - `vllm` 0.30.0 (the text-only LLM of the background table).

  These are the versions the published outputs were decoded with; vLLM and faster-whisper can live in their own
  environments, since each script imports only what it needs.

## Step 1: recount every number from the published outputs (CPU)

```bash
python3 -m stglossary.recount            # reads data/; --data DIR reads another directory with the same layout
```

Expected output:

```
items: 116 decision sentences from 10 talks; segments 2.1-22.0 s (median 7.4 s); glossary entries per item 7-20 (median 13)

1. streaming vs offline, Qwen2.5-Omni-7B
   items whose output inserts the wrong glossary form, wrong arm minus irrelevant arm (H): offline 11, streaming 12
   difference D = +1 of N = 116 (items up 3, down 2); 95% interval of D/N -0.035 .. 0.052; threshold k = 11.6 items
   a rise of k or more is excluded: upper bound 0.0521 < k/N 0.1000
   wrong     : w' in any streaming hypothesis 31, in the final output 12, committed before the speaker reached the term 0
   irrelevant: w' in any streaming hypothesis 23, in the final output 1, committed before the speaker reached the term 0
   latency, wrong arm: LAAL 2400 ms, AL 2219 ms
   spot check of the correct arm: 10 of 10 use w as the translation of t; condition guesses of the coder: balanced accuracy 0.39

2. streaming outputs that stop with 3 s or more of audio left
   without the sentence-end hold-back: 36 of 696
   with the hold-back: 1 of 696

3. offline panel, gold transcript in, instructed prompt: H
   Qwen2.5-7B-Instruct (text only)      H = 59
   Qwen2-Audio-7B-Instruct, text in     H = 4
   Qwen2.5-Omni-7B, text in             H = 7
```

How to read it:

- An item is one sentence with one glossary term t, whose correct Chinese form is w. The `wrong` glossary maps t to
  a wrong form w'; the `irrelevant` glossary instead holds an unrelated entry. H counts the items whose `wrong`
  output contains w', minus the items whose `irrelevant` output contains w' (so a w' the model would write anyway
  is not counted), over items kept because the model translates them at all, after removing hits hand-coded as
  false (w' in another sense, inside a longer word, or a copied glossary line).
- k = 10% of the items. A rise of k or more is excluded when the interval's upper bound is below k/N; streaming
  would be shown to raise insertion if D >= k and the interval's lower bound were above 0.
- "Committed before the speaker reached the term": w' committed by a prefix that ends before t's last word is
  spoken (word times from forced alignment of the gold transcript, `data/align.jsonl`).
- LAAL and AL are computed on the characters of the final output, against the reference length (AL) and the longer
  of output and reference (LAAL).
- A stop is a streaming output whose last step with real text ends 3 s or more before the audio does (every later
  step only punctuation or nothing). The hold-back keeps a sentence-final mark out of the committed text until the
  audio ends; forced after the assistant turn, a committed 。 makes the model end its answer.
- The bootstrap is clustered by talk (10 talks), with Webb six-point weights and 9,999 draws at a fixed seed, so the
  interval is printed exactly.

`--out result.json` also writes every intermediate count of part 1 (per-system counts, the items that moved up and
down, the early-commit lists, latency per arm).

The sentence that "about four of five repetitions would detect a true rise of 15 sentences" comes from a simulation
of this test, run before the decodes: `python3 -m stglossary.power > power.tsv` (needs numpy; 400 runs per
cell with 9,999 bootstrap draws each, tens of minutes on one CPU core).

Tests (no model, no GPU):

```bash
python3 tests/test_stream.py      # LA-2 commit policy, prefix grids, hold-back
python3 tests/test_contrast.py    # coding files, counts, decision rows, latency on synthetic rows
```

## Step 2: decode again (one 24 GB GPU)

1. Get the corpus (948 MB; the audio is not in this repository) and unzip it so that `$CORPUS/2/acl_6060/` exists:

   ```bash
   export CORPUS=/path/to/acl6060
   curl -L -o acl6060.zip https://aclanthology.org/attachments/2023.iwslt-1.2.dataset.zip
   sha256sum acl6060.zip   # 5f2a3855b5f442c83e6461c32e8a8deb6c2b053518b02b957eb4686bacfce7cc
   unzip -q acl6060.zip -d "$CORPUS"
   ```

   Optional: `python3 -m stglossary.build_items "$CORPUS" /tmp/items` rebuilds `data/sentences.jsonl` byte for
   byte from the corpus text and the hand labels in `stglossary/`.

2. Download the models at the revisions used, into a directory of your choice:

   ```bash
   export MODELS=/path/to/models
   hf download Qwen/Qwen2.5-Omni-7B --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 --local-dir $MODELS/Qwen2.5-Omni-7B
   hf download Systran/faster-whisper-large-v3 --revision edaa852ec7e145841d8ffdb056a99866b5f0a478 --local-dir $MODELS/faster-whisper-large-v3
   # background table only:
   hf download Qwen/Qwen2-Audio-7B-Instruct --revision 0a095220c30b7b31434169c3086508ef3ea5bf0a --local-dir $MODELS/Qwen2-Audio-7B-Instruct
   hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --local-dir $MODELS/Qwen2.5-7B-Instruct
   ```

3. Decode. Run from a clean clone of this repository: every row records the repository's commit and refuses to mix
   commits, and `runs/` is ignored by git. Each decode resumes where it stopped.

   ```bash
   S=data/sentences.jsonl; O7=$MODELS/Qwen2.5-Omni-7B
   python -m stglossary.align --sentences $S --corpus "$CORPUS" --subset first \
       --model $MODELS/faster-whisper-large-v3 --out runs/align.jsonl
   python -m stglossary.decode_hf --family omni25 --model $O7 --sentences $S --corpus "$CORPUS" --subset first \
       --mode speech --prompt instructed --runs main --out runs/offline.jsonl
   python -m stglossary.stream decode --mode speech --family omni25 --model $O7 --sentences $S --corpus "$CORPUS" \
       --align runs/align.jsonl --chunk 1.0 --prompt instructed --subset first --out runs/streaming.jsonl
   # background table (all 224 sentences; the counts read the 116 decision sentences):
   python -m stglossary.decode_hf --family omni25 --model $O7 --sentences $S --corpus "$CORPUS" \
       --mode text --prompt instructed --runs main --out runs/panel/qwen25-omni-text.jsonl
   python -m stglossary.decode_hf --family qa2 --model $MODELS/Qwen2-Audio-7B-Instruct --sentences $S --corpus "$CORPUS" \
       --mode text --prompt instructed --runs main --out runs/panel/qwen2-audio-text.jsonl
   python -m stglossary.decode_vllm --model $MODELS/Qwen2.5-7B-Instruct --sentences $S --prompt instructed \
       --out runs/panel/text-llm.jsonl
   ```

   | Step | Time on one RTX 4090 |
   |---|---|
   | alignment, 116 sentences | short (not timed separately) |
   | offline, 696 decodes | about 7 min |
   | streaming, 696 outputs, 4,000-odd prefix decodes | about 32 min |
   | Qwen2.5-Omni-7B, text in, 1,344 decodes | about 11 min |
   | Qwen2-Audio-7B-Instruct, text in, 1,344 decodes | about 27 min |
   | Qwen2.5-7B-Instruct with vLLM, 2,688 decodes | 2 to 3 min |

4. Compare and count. `stream compare` reports how many outputs are byte-identical to the published ones:

   ```bash
   python -m stglossary.stream compare --new runs/offline.jsonl --ref data/offline.jsonl --out runs/offline-cmp.json
   python -m stglossary.stream compare --new runs/streaming.jsonl --ref data/streaming.jsonl --out runs/streaming-cmp.json
   ```

   Greedy decoding on a GPU is close to deterministic but not guaranteed to be: another GPU, driver or library
   version can change a few outputs. When every output is identical, the published hand codes still apply; copy
   `data/` to a new directory, put the new files in place of the published ones and run
   `python3 -m stglossary.recount --data that/directory`. When an output with a w' hit has changed, the counts
   refuse the old codes for it (each code is tied to a hash of the output it was made on); write a new coding sheet
   with `python -m stglossary.sheets todo` (streaming vs offline) or `python -m stglossary.hits todo` (the panel),
   code it by hand as described in `data/SOURCE.md`, and count with `python -m stglossary.contrast count`.

The early-stop count without the hold-back is recounted from the published per-step texts
(`data/streaming-before-holdback.jsonl`); the decoder in this repository always holds the sentence-final mark back.

## FLEURS example

The post's example of a sentence written from the glossary comes from ACL 60/60 (dev/147, `wrong` arm): after about
seven words of audio, the model wrote a whole sentence out of glossary entries, repeated it at the next step, and
LA-2 committed it. `fleurs/` looks for the same thing in real outputs on everyday read speech: FLEURS `en_us`
(read Wikipedia sentences from FLoRes-200), with the same model, streaming policy, prompt and glossary arms. It is a
search for examples, not a rate estimate: no decision depends on how many are found. The rule was fixed before
anything was decoded.

### Search criterion

An output (one sentence, one arm) is an **example** when one committed step adds the glossary Chinese forms of at
least 2 distinct terms whose English term has not yet been spoken by that step's end, and that committed text
survives into the final output.

- The glossary is the block of the output's own arm.
- A step adds a form when the form occurs more often in the committed text after the step than before it
  (`stglossary.coding.spans`). Entries sharing one form count once.
- Not yet spoken: the forced-alignment end time of the term's first occurrence in the sentence is later than the
  step's audio end, or the term does not occur in the sentence.
- Survives: the final output starts with that step's committed text.
- Reported for each example: whether w' (the wrong arm's form of the probe term t) is among the early forms, and
  whether the step adds w' at all.

Details of the implementation: a block's English term counts as spoken when the sentence holds it or a spelling with
the same lemma (`stglossary.coding.lemma`); at the last step (the whole audio) every term in the sentence counts as
spoken, since alignment ends may exceed the duration by 0.05 s; a form nested inside another form added at the same
step counts once. On the ACL 60/60 streaming outputs of this repository the criterion flags dev/147 and dev/110 in
the `wrong` arm (`tests/test_fleurs.py`).

### Items

- Data: FLEURS `en_us` dev and test, matched to FLoRes-200 devtest by English text (every match is unique). One
  FLoRes article is one "talk". Articles about language technology, speech or computing are dropped (a pattern on
  topic and title, and a list of 8 titles in `fleurs/prep.py`): 222 articles are kept.
- Terms: Qwen2.5-7B-Instruct reads each English line with its FLoRes Chinese line and lists glossary terms, with the
  Chinese form copied from the translation and a class (`fleurs/terms.py`; raw outputs kept untouched in
  `data/fleurs/terms_raw.jsonl`).
- Items (`fleurs/items.py`, after `stglossary/build_items.py`): one glossary block per article, an item is
  (article, term) from a recorded sentence of 8-30 s, and w' (`wrong`, same class) and w'' (`irrelevant`, another
  class) are drawn from other articles. 225 items from 161 articles.

Differences from the ACL 60/60 items: terms and classes come from an LLM, not from the corpus terminology and hand
labels (no part-of-speech check, no hand pair-reject list); a block holds every specific term of the article's
FLoRes lines; sentences of 8-30 s only, one recording per sentence; the reference is FLoRes `zho_Hans`; arms
correct, irrelevant and wrong only (no `none` repeats).

### Recount from the published outputs (CPU)

```bash
python3 -m fleurs.scan --stream data/fleurs/stream.jsonl --align data/fleurs/align.jsonl \
    --sentences data/fleurs/sentences.jsonl --out scan.json | sed -n 1p
python3 -m fleurs.early_probe --stream data/fleurs/stream.jsonl --align data/fleurs/align.jsonl \
    --sentences data/fleurs/sentences.jsonl | sed -n 1,2p
python3 tests/test_fleurs.py
```

To rebuild the items from the stored FLoRes lines and glossary terms (about a minute on CPU), and check them:

```bash
python3 -m fleurs.items --lines data/fleurs/lines.jsonl --terms data/fleurs/terms_raw.jsonl \
    --out items-rebuilt --min-dur 8.0 > /dev/null
cmp items-rebuilt/sentences.jsonl data/fleurs/sentences.jsonl && echo identical
```

Expected output (each takes a few seconds):

```
scanned {'correct': 225, 'irrelevant': 225, 'wrong': 225}; not aligned 0; examples per arm {'correct': 42, 'wrong': 37, 'irrelevant': 32}
correct {'n': 225, 'final': 198, 'committed': 198, 'early': 34}
wrong {'n': 225, 'final': 44, 'committed': 44, 'early': 6}
3 tests passed
```

`scan` prints every example after its first line and writes them with their steps to `scan.json` (the published
copy is `data/fleurs/scan.json`). `early_probe` counts, per arm, the outputs whose final text holds the arm's form of
t (w in `correct`, w' in `wrong`), and how many committed it before t was spoken; it is description, not part of the
criterion.

### Run again (one 24 GB GPU)

Requirements as in Step 2, without Qwen2-Audio. The whole pipeline took about 55 minutes on one RTX 4090; most of it
is the streaming decode.

```bash
bash fleurs/fetch_fleurs.sh corpus/fleurs        # 488 MB; checks every file's sha256, extracts, writes SOURCE.json
export MODELS=/path/to/models                    # Qwen2.5-7B-Instruct, faster-whisper-large-v3, Qwen2.5-Omni-7B (Step 2)
export PYTHON_VLLM=/path/to/vllm-env/bin/python  # optional: interpreters of separate environments
export PYTHON_WHISPER=/path/to/whisper-env/bin/python
bash fleurs/run.sh all runs/fleurs 0             # stages prep, terms, items, align, stream, scan; GPU 0
python3 -m fleurs.early_probe --stream runs/fleurs/stream.jsonl --align runs/fleurs/align.jsonl \
    --sentences runs/fleurs/items/sentences.jsonl
```

Each stage can also be run alone (`bash fleurs/run.sh STAGE runs/fleurs [GPU]`); a stage whose output exists is
skipped, and the streaming decode resumes. Compare a new decode with
`python -m stglossary.stream compare --new runs/fleurs/stream.jsonl --ref data/fleurs/stream.jsonl --out cmp.json`.
The term extraction runs once over all lines and is not resumable. Greedy decoding on another GPU or library version
can change a few outputs, so a re-run may find a slightly different set of examples.

## Layout

```
stglossary/
  recount.py      step 1: every number of the post
  contrast.py     streaming vs offline: H, D, bootstrap interval, decision row, early commit, latency
  panel.py        decode loader, talk-clustered bootstrap, per-system counts, correct-arm spot check
  insertion.py    per-system counts (H, V, kept items) and bootstrap constants
  hits.py         echo cut, w' hits, coding sheets for the panel
  sheets.py       condition-blind coding sheets for streaming vs offline, blinding check
  coding.py       presence of a form in Chinese text; echo cut
  prompts.py      the prompt template and glossary blocks
  stops.py        early stops in streaming outputs
  stream.py       LA-2 streaming decoder
  decode_hf.py    offline decoder (transformers)
  decode_vllm.py  text-only LLM decoder (vLLM)
  align.py        forced word alignment of the gold transcript (faster-whisper)
  corpus.py, build_items.py, labels.tsv, pair_rejects.tsv   the item table from the corpus
  power.py        simulated power of the decision rule
  export.py       how data/ was made from the decoder outputs
  prov.py         git provenance of each decoded row
fleurs/
  fetch_fleurs.sh the FLEURS and FLoRes-200 files, checked by sha256
  run.sh          every stage of the FLEURS example
  prep.py, terms.py, items.py   FLEURS joined to FLoRes-200; glossary terms; items and the three arms
  stream3.py      stglossary.stream with the arms correct, irrelevant, wrong
  scan.py         the search criterion
  early_probe.py  early commits of the probe form per arm
tests/
data/             item table, decodes, hand codes; see data/SOURCE.md
data/fleurs/      the FLEURS example's items, terms, alignment, decodes and scan; see data/fleurs/SOURCE.md
```

## Licences

- Code: MIT (`LICENSE`).
- Data (`data/`): derived from ACL 60/60 (Salesky et al., IWSLT 2023), CC BY 4.0; see `data/SOURCE.md` for the
  sources and the attribution.
- FLEURS example (`data/fleurs/`): derived from FLoRes-200 (CC BY-SA 4.0) and FLEURS (CC BY 4.0), shared under
  CC BY-SA 4.0; see `data/fleurs/SOURCE.md`.
