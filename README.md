# AI_agent_for_SmartFramingAdvice
# 🌾 Smart Farming Advisor (AI-based Agricultural Assistant)

The **Smart Farming Advisor** is an AI-powered agricultural guidance system that helps farmers make better decisions about crop management.  
It uses natural language understanding (NLP), translation, and intelligent retrieval to generate *dynamic*, context-aware farming advice — not static answers.
Deploy Link : https://aiagentforsmartframingadvice-kmckmhxznqyxybnoz8aans.streamlit.app/
---

## 🚀 Features

- 🧑‍🌾 **AI-Powered Question Answering**  
  Generates smart, situation-based advice for each query using FAISS and Sentence Transformers.

- 🌍 **Bilingual Support (English & Hindi)**  
  Automatically translates both user queries and AI-generated responses for inclusivity.

- 📊 **Smart Knowledge Retrieval**  
  Uses FAISS vector search to find the most relevant farming advice from your datasets.

- 🧾 **PDF Report Generator**  
  Converts AI advice into a downloadable PDF (supports Unicode text).

- 🔊 **Text-to-Speech (gTTS)**  
  Reads out generated advice for accessibility and ease of use.

- 📁 **Auto Dataset Detection**  
  Automatically detects CSV files in the app folder — no manual dataset selection required.

- 💻 **Streamlit Responsive UI**  
  A modern, bilingual web interface that works on both desktop and mobile screens.

---

## 🧱 Project Structure

AI_agent_for_SmartFramingAdvice/
│
├── app.py # Main Streamlit app
├── requirements.txt # List of dependencies
├── faiss_index.bin # Vector index (used by FAISS)
├── datasets/ # Folder containing your data files
│ ├── crop_data.csv
│ └── soil_data.csv
├── faiss_index/ # (if used) FAISS index storage
│ └── index files...
└── README.md # This documentation file


---

## ⚙️ Installation (Run Locally)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vishweshwari63/AI_agent_for_SmartFramingAdvice.git
cd AI_agent_for_SmartFramingAdvice

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

🌐 Deploying on Streamlit Cloud

Push your code to this GitHub repository.

Go to https://share.streamlit.io
.

Click “New app”.

Select your repo → AI_agent_for_SmartFramingAdvice.

Set:

Branch: main

Main file path: app.py

Click Deploy 🚀

📦 Requirements

Here’s an example of your requirements.txt file:

streamlit
pandas
numpy
sentence-transformers
faiss-cpu
deep-translator
gTTS
fpdf
chardet

🧑‍💻 Author

👩‍💻 Vishweshwari R
🎯 Developer of Smart Farming Advisor (AI Agent for Smart Farming)
📍 Anna University Regional Campus, Madurai

💡 Acknowledgements

Streamlit
 – Frontend Framework

Sentence Transformers
 – Embedding Model

FAISS
 – Similarity Search

Deep Translator
 – Language Translation

gTTS
 – Text-to-Speech

🧾 License

This project is open-source for educational and research purposes.
Feel free to fork, improve, and contribute! 🌱
Demo link : https://aiagentforsmartframingadvice-kmckmhxznqyxybnoz8aans.streamlit.app/
