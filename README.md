# VocalVet AI

**AI-Powered Bio-Acoustic Cattle Health Monitoring System**

VocalVet AI is a machine learning–based cattle health monitoring system that analyzes cow vocalizations to predict health risk levels. The system enables farmers and veterinarians to assess cattle condition using recorded or uploaded audio samples.

Built with Python, Streamlit, and Scikit-learn, the application provides real-time inference, history tracking, multilingual support, and downloadable health reports.

---

##  Features

 **AI Health Risk Prediction**
* Audio-based cattle health classification
* Binary risk detection: 🔴 High Risk | 🟢 Low Risk
* Pre-trained ML model (`model.pkl`)


 **Audio Input Options**
* Record cow sound directly
* Upload `.wav` audio files (up to 200MB)


 **Cow Management System**
* Add unique Cow IDs
* Select specific cow for analysis
* Track health history per cow


 **Multilingual Support**
* English, Hindi, Kannada, Tamil, Telugu, Punjabi


 **Health History Tracking**
* Date & time stamped predictions
* Status logs (High Risk / Low Risk)
* Reset history option


 **Doctor Report Download**
* Generate downloadable health report for veterinary review

---

## System Architecture

1. **Audio Input** (Record / Upload)
2. ↓ **Feature Extraction** (Librosa)
3. ↓ **ML Model** (Scikit-learn)
4. ↓ **Risk Prediction**
5. ↓ **UI Display + History Storage + Report Generation**

---

##  Tech Stack

| Component | Technology |
| --- | --- |
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Model** | Scikit-learn |
| **Audio Processing** | Librosa |
| **Model Serialization** | Joblib |
| **Data Handling** | Pandas, NumPy |

---

## 📂 Project Structure

```text
VocalVetAI/
│
├── app.py              # Main Streamlit application
├── model.pkl           # Trained ML model
├── predict.py          # Prediction logic
├── train_model.py      # Model training script
└── README.md

```

---

## Installation & Setup

1. **Clone Repository:** Clone the project repository to your local machine.
2. **Virtual Environment:** Create and activate a Python virtual environment.
3. **Install Dependencies:** Install the required packages listed for the project (including `streamlit`, `scikit-learn`, `librosa`, `joblib`, `numpy`, `pandas`, `soundfile`, and `matplotlib`).
4. **Run the Application:** Execute the main application file using Streamlit to launch the app in your web browser.

---

## How It Works

1. Add a Cow ID.
2. Record or upload cow vocal audio.
3. The system extracts audio features using Librosa.
4. The model predicts the risk level.
5. The result is displayed instantly.
6. History is saved with a timestamp.
7. A doctor report can be downloaded.

---

## Use Cases

* Early disease detection in cattle
* Farm health monitoring automation
* Veterinary decision support
* Remote livestock assessment

---

## Future Improvements

* Multi-class disease classification
* Cloud deployment
* Mobile app integration
* Real-time barn microphone monitoring
* IoT device integration
* Deep learning (CNN/RNN on spectrograms)

---

## Limitations

* Model accuracy depends on training dataset quality.
* Works best with clear, low-noise recordings.
* Binary classification only (current version).

---

##  Contribution

Contributions are welcome. Please fork the repository and submit a pull request.

---

##  Disclaimer

This system provides AI-assisted health risk predictions and should not replace professional veterinary diagnosis.

---
## App Preview


<img width="1920" height="1080" alt="Screenshot 2026-03-01 210214" src="https://github.com/user-attachments/assets/3edc96b6-e2a5-4974-b282-f747f97f0332" />



<img width="1920" height="1080" alt="Screenshot 2026-03-01 210259" src="https://github.com/user-attachments/assets/ee2110bb-b298-4b43-b878-0a4559e3d1af" />



<img width="1920" height="1080" alt="Screenshot 2026-03-01 210446" src="https://github.com/user-attachments/assets/e308f0cd-3cef-4242-97e2-9f0846c0d4e7" />

