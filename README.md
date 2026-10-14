# Context.exe

**How much context does AI actually need to understand language?**

NLP CA3 Build–Audit–Pitch mini project, SAII 2026-27. Team: Prashuk Jain, Aditya Pawar, Aarav Sahani, Palki Bhandari.

> "Yeah, that's great." Genuine or sarcastic? On its own, you can't tell.
> After *"My flight got cancelled for the third time."*, every human can.
> Does an NLP model get better when it sees the conversation?

We train two versions of the same sarcasm classifier, and the only difference between them is context:

- **Model A (context-free)** sees only the target sentence.
- **Model B (context-aware)** sees the previous message plus the target sentence.

Both are built twice, once as TF-IDF + Logistic Regression and once as fine-tuned DistilRoBERTa. The data is 13,000 real Reddit and Twitter threads from the [FigLang 2020 sarcasm shared task](https://github.com/EducationalTestingService/sarcasm).

## Results (official test set, 3,599 replies, 50% sarcastic)

| Model | Sentence only (A) | + previous message (B) | Gain (95% CI) |
|---|---|---|---|
| TF-IDF + LogReg | 63.0% | 63.5% | +0.4 pts (−1.0 to +1.9) |
| DistilRoBERTa | 69.1% | 69.5% | +0.4 pts (−0.8 to +1.7) |

**Key finding: context is a double-edged sword.** Adding context changed 15% of the verdicts. It fixed 281 mistakes and created 267 new ones, so the average barely moves.

A context-swap test shows the model really does read the context: with a random previous message, accuracy drops to 66.8%. What it extracts, though, mostly duplicates what the sentence already says.

**Explainability:** word occlusion shows which words, in the sentence or in the context, moved each decision. Same sentence, only the context differs:

| Previous message | Sentence | Model A | Model B |
|---|---|---|---|
| "My laptop crashed before I saved my assignment." | "Amazing." | 0.79 | 0.96 |
| "She surprised me with tickets to the concert." | "Amazing." | 0.79 | 0.58 |

**Limitations:**
- The models over-predict sarcasm: Model A flags 39% of genuine replies.
- The dataset has no replies shorter than 6 words.
- Much of this sarcasm needs world knowledge that one previous message can't supply.

## Run it in Google Colab (no installation)

[Open the notebook in Colab](https://colab.research.google.com/drive/1XHwCB_gswWPrDhJscbwjUe_eQigtSEGm)

1. Choose **Runtime → Change runtime type → T4 GPU**.
2. Choose **Runtime → Run all**.

The notebook downloads the dataset itself. It reruns the TF-IDF experiment (about 2 minutes), fine-tunes both transformers (about 15 minutes), and ends with a "type a line, get a result" demo.

## Dataset source

- **Name:** FigLang 2020 Sarcasm Detection Shared Task, published by Educational Testing Service (ETS).
- **Data:** https://github.com/EducationalTestingService/sarcasm
- **Paper:** Ghosh, D., Vajpayee, A., & Muresan, S. (2020). *A Report on the 2020 Sarcasm Detection Shared Task.* Proceedings of the Second Workshop on Figurative Language Processing, ACL, pp. 1–11. https://aclanthology.org/2020.figlang-1.1/
- **Size:** 13,000 replies (Reddit: 4,400 train + 1,800 test; Twitter: 5,000 train + 1,800 test), each labelled SARCASM or NOT_SARCASM, with the earlier messages of its conversation.

## Run it on your computer: type lines, get results

**Easiest way (no commands):**

1. Download the repository (**Code → Download ZIP**) and unzip it.
2. Install Python 3.9 or newer from python.org.
3. Double-click **`Run Context.exe (Mac).command`** or **`Run Context.exe (Windows).bat`**.
   - The first run installs the packages and downloads the fine-tuned models (306 MB). This takes about 10 minutes.
4. Choose **1** for the web app or **2** for the terminal app.

> On a Mac, if it says the file is from an unidentified developer: right-click → **Open**.

**From a terminal:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app/engine.py --download                 # fetch the fine-tuned models (once)

streamlit run app/app.py                        # web app in your browser
python app/cli.py                               # terminal app: type a previous message, then a sentence
python app/cli.py "Amazing." -c "My laptop crashed before I saved my assignment."
python app/cli.py --file examples.txt           # one per line:  previous message ||| sentence
```

| App | What you see |
|---|---|
| **Web app** (`app/app.py`) | Model A and Model B side by side with confidence. It shows whether context flipped the verdict, colours every word by its influence, and has a "same sentence, many contexts" view. |
| **Terminal app** (`app/cli.py`) | The same results as text: both verdicts, confidence bars and the words that mattered most. |

The models download automatically from the [v1.0 release](https://github.com/prashukjain17/NLP-Project/releases/tag/v1.0) and are stored in half precision (this changed 1 of 3,599 test predictions). If the download isn't possible, both apps fall back to the TF-IDF models, which ship with the repository.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/context_pipeline.py        # TF-IDF experiment: 4 context conditions, audit, explanations (~2 min)
python src/transformer_models.py      # (optional) re-train DistilRoBERTa A and B yourself (~1 h on an Apple M-series GPU)
python src/analyze_transformer.py     # transformer audit, context-swap test, occlusion examples
python docs_build/extra_figures.py && python docs_build/build_report.py \
  && python docs_build/build_model_card.py && python docs_build/build_pitch.py
```

`src/transformer_models.py` expects the base model in `models/distilroberta-base/`. Download `config.json`, `vocab.json`, `merges.txt`, `tokenizer.json` and `model.safetensors` from https://huggingface.co/distilroberta-base. It runs one model per process, and pads batches to multiples of 32, so it fits in 8 GB of memory.

## Repository layout

```
data/figlang/        FigLang 2020 Reddit + Twitter train/test (jsonl)
src/                 context_pipeline.py · transformer_models.py · analyze_transformer.py
app/                 app.py (web app) · cli.py (terminal app) · engine.py (shared model loading)
Run Context.exe …    one-click launchers for Mac and Windows
outputs/             metrics, audit tables, examples, figures
docs_build/          scripts that generate the report, model card and pitch deck
deliverables/        Context_exe_Report.docx · Context_exe_Model_Card.docx · Context_exe_Pitch.pptx
```

## Data artefacts we found and removed

- **A lone "_" label leak.** It appears in 79 sarcastic replies vs 4 non-sarcastic ones.
- **Uneven hashtags.** 54% of genuine tweets have one vs 20% of sarcastic tweets.
- **Platform placeholders.** `@USER` and `<URL>` tokens were stripped.
- **Tweet tokenisation.** Tweets were pre-tokenised (`don 't`); this was undone.

## References

- Ghosh, Vajpayee & Muresan (2020). *A Report on the 2020 Sarcasm Detection Shared Task.* FigLang @ ACL.
- Sanh et al. (2019). *DistilBERT.*
- Riloff et al. (2013). *Sarcasm as Contrast between a Positive Sentiment and Negative Situation.*
- Mitchell et al. (2019). *Model Cards for Model Reporting.*
