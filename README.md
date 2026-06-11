
# 🎵 Mood Music Recommender

---

## 🚀 Overview

The **Mood Music Recommender** bridges the gap between emotional state and audio stream. By combining advanced deep learning models with real-time audio analytics, this application scans user inputs (text or quick-select mood tokens) to curate hyper-personalized, context-aware music recommendations.

### ✨ Key Features

* 🎭 **Vibe Detection:** Instantly switch between predefined emotional states (Happy, Sad, Energetic, Chill) or type how you feel for custom processing.
* 🧠 **Hybrid ML Engine:** Leverages collaborative filtering and deep content-based audio analysis.
* ⚡ **Seamless Streaming:** Full integration with live streaming APIs for instant playback and playlist saving.
* 📊 **Interactive Dashboard:** Modern, glassmorphic dark-mode UI optimized for desktop and mobile web.

---

## 🛠 Tech Stack

The architecture is built using a robust, enterprise-grade stack separating high-performance machine learning inference from a lightning-fast web frontend:

| Layer | Technologies Used |
| --- | --- |
| **Frontend** | React, Next.js, TailwindCSS |
| **AI / ML Engine** | Python, TensorFlow, PyTorch |
| **Backend API** | FastAPI |
| **Database** | PostgreSQL |
| **Integrations** | Spotify API |

---

## 🧠 How It Works

```
   [ User Input / Vibe Selection ] 
                 │
                 ▼
     [ FastAPI Text/NLP Engine ] 
                 │
                 ▼
  [ PyTorch/TensorFlow Vector Embedding ] ───► [ Query PostgreSQL Vector DB ]
                 │                                        │
                 ▼                                        ▼
    [ Spotify API Track Hydration ] ◄─────────────────────┘
                 │
                 ▼
   [ Personalized Live Carousel ]

```

---

## 📦 Getting Started

### Prerequisites

Make sure you have Python 3.10+, Node.js 18+, and PostgreSQL installed.

### 1. Clone the repository

```bash
git clone https://github.com/vikramsingh-dev/mood-music-recommender.git
cd mood-music-recommender

```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

```

Create a `.env` file in the `backend/` directory:

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
DATABASE_URL=postgresql://user:password@localhost:5432/mood_db

```

Run the FastAPI server:

```bash
uvicorn main:app --reload

```

### 3. Frontend Setup

```bash
cd ../frontend
npm install
npm run dev

```

Open [http://localhost:3000](http://localhost:3000) in your browser to experience the application!

---

## 👨‍💻 Author

Developed with ❤️ by **Vikram Singh**.

* **Portfolio:** [@vikramsingh](https://vikramsingh.itsfolio.tech)
* **GitHub:** [@vikramsingh](https://github.com/babamosie333)

---

> ⭐ **Like this project?** Give it a star to show your support!