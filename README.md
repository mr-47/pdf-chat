# Small RAG app

PDF → split into chunks → create embeddings → store in FAISS → retrieve relevant chunks → answer.

LangChain's RAG docs describe this pattern, FAISS is for vector similarity search, and Sentence Transformers provide local embedding models.

## Install

Use `./run.sh`. It creates `.venv` and installs what the command needs on first
run, then executes your command inside that environment.

```bash
./run.sh cli.py paper.pdf          # CLI, core deps only
./run.sh -m streamlit run app.py   # web UI, also installs the UI extra
```

If you prefer to manage the environment yourself:

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt            # CLI
.venv/bin/pip install -r requirements-ui.txt         # CLI + web UI
```

A system Python install will not work on Debian/Ubuntu: `/usr/lib/python3.12`
is marked externally managed (PEP 668), so `pip install` into it is refused by
design. Use a virtualenv rather than `--break-system-packages`.

Two deliberate choices in the dependency setup:

- **No CUDA.** PyTorch is installed from the CPU-only index, because the
  default PyPI wheel bundles several GB of `nvidia-*` wheels that are never
  used. `torch` is intentionally absent from `requirements.txt` so that a
  plain `pip install -r requirements.txt` cannot reintroduce them.
- **Almost no LangChain.** Only `langchain-text-splitters` is kept, for the
  text chunking. The `langchain` meta-package pulled in `langgraph`, an agent
  framework this project never used, and `langchain-community` pulled in
  `langchain-classic`, `langsmith`, `sqlalchemy`, `aiohttp` and `httpx`, all
  to provide two classes that are a few lines each:

  | Was | Now | Why |
  | --- | --- | --- |
  | `PyPDFLoader` | `pypdf.PdfReader` | ~10 lines, and pypdf was already a dependency |
  | `FAISS.from_documents` | `VectorStore` in `rag.py` | exhaustive L2 over a few hundred chunks, ~70MB of native libs saved |

  Verified to produce identical rankings: 30/30 results across two documents
  match the FAISS implementation, with distances differing by at most 1e-6
  from float32 summation order.

Streamlit is an optional extra in `requirements-ui.txt` because the CLI does
not need it. Skipping it removes about 20 packages and 300 MB, mostly pyarrow
and pandas. `app.py` prints installation instructions if you try to run it
without them.

The embedding model (`all-MiniLM-L6-v2`) downloads once on first use, then is
cached in memory for the life of the process and on disk in
`~/.cache/huggingface`.

## Network access

After that first download, the app makes **no outbound requests at all**.

By default `huggingface_hub` sends HEAD requests on every run to check whether
a newer revision of the model exists upstream, which is why a fully cached
install still reached `huggingface.co`. `rag.py` now loads the model with
`local_files_only=True` and only falls back to the network when the model is
genuinely missing from the cache, so:

- first run downloads the model and prints a notice saying so
- every later run is fully offline, including with no network connection
- retrieval output is unchanged, because the cache already pins the revision

Set `PDFCHAT_ALLOW_DOWNLOAD=1` to let the app check the hub again, for example
after clearing `~/.cache/huggingface` to pick up a different model version.
`HF_HUB_OFFLINE=1` and `HF_TOKEN` are also honoured if you prefer the
standard variables.

## Suppressed log noise

transformers 5.x has roughly a hundred vision model modules that import
`torchvision` at module level without a backend guard, so importing any of
them without torchvision installed raises `ModuleNotFoundError`. This project
only does text embedding and never loads a vision model, so the traceback is
pure noise. Two independent guards remove it.

`.streamlit/config.toml` stops it at the source. The real trigger is not this
project: Streamlit's development file watcher walks `sys.modules` and calls
`hasattr(module, "__path__")` on every entry. Because transformers resolves
attributes through a lazy `_LazyModule`, that `hasattr` really does import the
submodule, and the watcher catches the resulting `ImportError` in a bare
`except` and logs a multi-line traceback. `server.fileWatcherType = "none"`
disables that scan, which also stops it importing thousands of unrelated
modules at every startup.

`rag.py` installs a logging filter as a backstop for every other entry point.
It matches on the word `torchvision` in the message or the raised exception and
nothing else. A log record can surface through whichever handler a library
happened to configure, and Streamlit sets up its logging *after* `rag.py` is
imported, so filtering only the handlers that exist at import time misses it.
The filter therefore sweeps every existing handler and wraps
`Logger.addHandler` so handlers created later are filtered too, and it also
covers the root logger and `logging.lastResort`. Verified for a handler
registered both before and after import: the torchvision record is dropped and
an unrelated `ValueError` on the same handler still prints in full.

The only cost is that Streamlit no longer hot reloads on file changes, so the
app needs a restart to pick up edits to `app.py`.

## CLI

Point the app at a PDF, then ask questions in the terminal.

```bash
./run.sh cli.py path/to/paper.pdf
```

The PDF is indexed once at startup, then the session stays open:

```
you> what is the main conclusion?
```

Each answer shows the matching passages with their page numbers and a
distance score (lower means closer to your question). In-session commands:

| Command    | Effect                        |
| ---------- | ----------------------------- |
| `/pages`   | list indexed page numbers     |
| `/chunks`  | show the chunk count          |
| `/help`    | show the command list         |
| `/exit`    | leave the session             |

Options:

```bash
./run.sh cli.py paper.pdf -k 6               # show 6 passages per question
./run.sh cli.py paper.pdf -q "who wrote it"  # ask once and exit
```

## Web UI

```bash
./run.sh -m streamlit run app.py
```

Upload one or more PDFs in the sidebar, then ask questions in the chat box.
Answers are rendered as chat turns with the source file, page number and
distance score for each passage. Retrieval spans every uploaded PDF and the
results are ranked together, so one question can pull from several documents.
Chat history survives reruns and can be cleared from the sidebar.

## What it does not do yet

There is no LLM call in either interface. Answers are the retrieved passages
themselves, not generated text. The prompt-building step needed to plug in a
model is where `rag.retrieve_matches` is consumed: `cli.py:answer` for the
terminal and `app.py:collect_matches` for the browser.

Two known limitations:

- **No relevance threshold.** The top-k passages are always returned, even
  when nothing in the PDF is related to the question.
- **Follow-up questions retrieve poorly.** History is stored for display, but
  a question like "and the budget?" has no referent when embedded. Fixing this
  needs query rewriting, not a UI change.

## Release notes
v0.1 - proves the main idea: load a PDF, index it, retrieve relevant chunks.
LangChain's RAG guides use retrieval plus generation as the standard pattern.

v0.2 - added a CLI that takes a PDF path and runs a question session from the
terminal, plus `requirements.txt`.

v0.3 - rewrote the Streamlit UI as a real chat interface with multi-PDF
support. Fixed two bugs: the index was never invalidated when a different PDF
was uploaded, and the embedding model was reloaded on every session.
`LocalEmbeddings` now subclasses LangChain's `Embeddings`, which repairs
retrieval on current langchain versions for both interfaces.

v0.4 - trimmed dependencies. Dropped the unused `langchain` meta-package
(which pulled in langgraph), moved Streamlit to an optional
`requirements-ui.txt`, and pinned installs to CPU-only PyTorch so CUDA wheels
cannot creep back in. `run.sh` installs whichever extra the command needs.

v0.5 - the embedding model is now loaded with `local_files_only=True` and falls
back to downloading only when it is not cached, removing the per-run
Hugging Face ETag checks. The app is offline after the first run.

v0.6 - removed `langchain-community` and `faiss-cpu`. `PyPDFLoader` is replaced
by `pypdf` directly and FAISS by a small exhaustive-L2 `VectorStore` in
`rag.py`. Rankings are unchanged. The environment went from 105 to 70 packages
and 1.9GB to 1.4GB.

v0.7 - suppressed the `ModuleNotFoundError: torchvision` traceback that
transformers 5.x emits from its vision modules, which this text-only project
never loads. The trigger was Streamlit's development file watcher, which
triggers those imports while scanning `sys.modules`; it is now disabled via
`server.fileWatcherType = "none"`, with a logging filter in `rag.py` covering
every other entry point. Only records mentioning torchvision are dropped.
Hot reload is disabled, so `app.py` changes need a restart.
