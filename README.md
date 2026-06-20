# Arabic ASR Streaming EDA + Local Whisper-Small

This repository contains an exploratory notebook for Arabic ASR dataset inspection before full benchmarking or fine-tuning.

## Files
- `arabic_asr_streaming_eda_whisper_small.ipynb`
- `requirements.txt`
- `outputs/`
- `outputs/samples/`
- `outputs/eda/`
- `models/`

## Run Instructions
1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
3. Optionally authenticate with Hugging Face (recommended for gated datasets/models):
   - `huggingface-cli login`
4. Launch Jupyter Lab:
   - `jupyter lab`
5. Open and run:
   - `arabic_asr_streaming_eda_whisper_small.ipynb`
6. Outputs are written under:
   - `outputs/`

## Notebook Scope
The notebook does the following:
- Streams small sample sets from multiple Arabic speech datasets with robust fallback logic.
- Captures local WAV + JSON sample metadata.
- Runs audio EDA and metadata/text EDA.
- Downloads `openai/whisper-small` into `./models/whisper-small`.
- Loads Whisper from local disk only and runs sanity-check transcription on captured samples.

It intentionally does **not** create the final benchmark leaderboard table yet.
