# Flickd Vibe Sync Engine

An Smart Tagging & Vibe Classification Engine system built during the **Flickd AI Hackathon (June 2025)** to enable scroll-native, video-first, vibe-led shopping for Gen Z. It detects fashion items in short videos, matches them with a product catalog, and classifies the video's "vibe" (e.g., _Coquette_, _Clean Girl_, _Cottagecore_).

## 🚀 Overview

The system processes videos to:

- Detect fashion items.
- Match them to a Shopify-style product catalog.
- Classify vibes using captions and audio transcripts.

It outputs structured JSON responses and offers 3 interaction modes:

- 🧠 CLI (`cli.py`)
- 🌐 Streamlit Web App (`streamlit_app.py`)
- 🔌 FastAPI Endpoint (`api.py`)

> It integrates OpenAI's **Whisper** for audio transcription and dynamically adjusts frame sampling based on FPS.

---

## 🎯 Features

- **Frame Extraction**: OpenCV-based with FPS-adjusted sampling (every 0.2 seconds).
- **Object Detection**: YOLOv8 to detect fashion categories.
- **Product Matching**: CLIP + FAISS with match categories:
  - `Exact`: > 0.9
  - `Similar`: 0.75–0.9
  - `No Match`: < 0.75
- **Vibe Classification**: spaCy / HuggingFace Transformers on combined captions + transcripts.
- **Audio Transcription**: MoviePy + Whisper.
- **Structured Output**: JSON files with vibes and detected products.
- **Bonus Streamlit App**: Glassmorphism + vibrant UI.
- **Robust Logging & Error Handling**.

---

## 🧰 Tech Stack

| Area        | Tools                        |
| ----------- | ---------------------------- |
| Programming | Python 3.11                  |
| CV          | OpenCV, YOLOv8 (Ultralytics) |
| Embeddings  | OpenAI CLIP                  |
| Search      | FAISS                        |
| Audio       | MoviePy, Whisper             |
| NLP         | spaCy, HuggingFace           |
| Interfaces  | CLI, Streamlit, FastAPI      |
| Others      | PyTorch, NumPy, FFmpeg       |

---

## 🏗 System Architecture

```text
   +------------------------ Input Files ------------------------+
   |       Video, Catalog CSV, Captions TXT                      |
   +-------------------------------------------------------------+
         |                   |                        |
         v                   v                        v
     [CLI]            [Streamlit App]             [FastAPI]
         |                   |                        |
         v                   v                        v
     Frame Extraction (OpenCV, FPS-adjusted every 0.2s)
         |
         v
     Object Detection (YOLOv8)
         |
         v
     Product Matching (CLIP + FAISS)
         |
         v
  +-------------------------+       +--------------------------+
  | Audio Extraction (WAV) |       |    Captions (TXT)        |
  +-------------------------+       +--------------------------+
         |                                 |
         v                                 v
     Audio Transcription (Whisper) -> Combine Text
         |
         v
     Vibe Classification (spaCy / Transformers)
         |
         v
     JSON Output (Detections, Matches, Vibes)
```

---

## 🔧 Setup Instructions

### ✅ Prerequisites

- Python 3.11
- FFmpeg
- Git
- (Recommended) Virtual Environment

### ⚙️ Installation

```bash
git clone https://github.com/NabidAkhtar/FlickdVibeSyncEngine.git
cd FlickdVibeSyncEngine

# Create virtual env
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate (Windows)

# Install requirements
pip install -r requirements.txt

# If PyTorch issues (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify
python -c "import cv2, numpy, ultralytics, clip, faiss, spacy, torch, whisper, moviepy"
```

---

## ⚙️ Usage

### 1. Command-Line (CLI)

```bash
python cli.py --video sample.mp4 --catalog catalog.csv --captions captions.txt
```

➡️ Output saved in `outputs/output_sample.mp4.json`.

---

### 2. Streamlit App

```bash
streamlit run streamlit_app.py
```

- URL: [http://localhost:8502](http://localhost:8502)
- Upload video, catalog CSV, and captions
- Press **Analyze Now 🚀**

---

### 3. FastAPI Endpoint

```bash
python api.py
```

- API URL: [http://localhost:8000](http://localhost:8000)
- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Example:

```python
import requests

files = {
  "video": open("sample.mp4", "rb"),
  "catalog": open("catalog.csv", "rb"),
  "captions": open("captions.txt", "rb")
}

res = requests.post("http://localhost:8000/process", files=files)
print(res.json())
```

---

## 🧪 Sample Output (JSON)

```json
{
  "video_id": "sample_video.mp4",
  "vibes": ["Coquette", "Evening"],
  "products": [
    {
      "type": "dress",
      "color": "black",
      "match_type": "similar",
      "matched_product_id": "prod_456",
      "confidence": 0.84
    }
  ]
}
```

---

## 📁 Project Structure

```
├── data
    ├── videos
        ├── 2025-05-22_08-25-12_UTC.jpg
        ├── 2025-05-22_08-25-12_UTC.mp4
         ...
    ├── images.csv
    ├── product_data.xlsx
    ├── vibes_list.json
├── outputs
    ├── 2025-05-22_08-25-12_UTC.json
    ├── 2025-05-27_13-46-16_UTC.json
    ├── 2025-05-28_13-40-09_UTC.json
    ├── 2025-05-28_13-42-32_UTC.json
    ├── 2025-05-31_14-01-37_UTC.json
    ├── 2025-06-02_11-31-19_UTC.json
├── src
    ├── models
        ├── detection.py
        ├── matching.py
    ├── processing
        ├── frames.py
        ├── vibe.py
    ├── cli.py
    ├── __init__.py
├── api.py
├── requirements.txt
├── streamlit.py
```

---

## 📊 Evaluation Files

- CLI: `outputs/output_*.json`
- Streamlit / FastAPI: `evaluation_*.json`

---

## 🎥 Loom Demo

[https://www.loom.com/share/82f28f643ed64e94a2fdf179ea04a96b?sid=d41d0ac5-0cc7-48c5-9d32-ee2e71faad0a]

**Showcases:**

- CLI run & output.
- Streamlit UI and analysis.
- API endpoint usage.
- Whisper-powered transcription.
- FPS-based sampling.

---

## 🧠 Technical Highlights

- **YOLOv8**: Detects fashion items.
- **CLIP + FAISS**: Semantic matching to catalog.
- **Whisper**: Audio transcription.
- **spaCy / Transformers**: Vibe classification from transcript + captions.

> ⚙️ Frame sampling = `int(fps * 0.2)`

---

## 🎨 UI Design

- Glassmorphism with blur effects
- Animations for transitions
- Colors: Coral `#ff6f61`, Golden Yellow `#feca57`

---

## 🧯 Troubleshooting

| Issue                         | Solution                                                                         |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `ModuleNotFoundError: 'clip'` | `pip install git+https://github.com/openai/CLIP.git`                             |
| PyTorch install fails         | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Whisper errors                | Ensure FFmpeg is installed and added to PATH                                     |

---

## 🧩 Future Improvements

- Real-time video stream processing.
- Scalable FAISS indexing.
- JWT-based API authentication.
- UI overlays for bounding boxes.

---

## 👤 Author

**Nabid Akhtar**

- GitHub: [NabidAkhtar](https://github.com/NabidAkhtar)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/nabid-akhtar-059a981bb?)

---

## 📄 License

This project is licensed under the **MIT License**.

> Built with ❤️ for the Flickd AI Hackathon, June 2025
