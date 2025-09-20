# 🛡️ AirfieldShield

**AirfieldShield** is an AI-powered weather nowcasting and alerting system designed to help airfields predict and visualize severe weather events such as thunderstorms and gale winds.  

It combines **synthetic + AI/ML models** with a modern **interactive dashboard** that provides:  
- Real-time predictions  
- Storm overlays on a map  
- Risk trend charts  
- Alerts and data-source health monitoring  

---

## ✨ Features

- 🌩️ **Thunderstorm & Gale Prediction** using Random Forest, LSTM, CNN, and an Ensemble Meta-learner.  
- 🛰️ **Storm Visualization** with animated markers and polygons on an interactive map.  
- 📊 **Detailed Risk Trend Chart** (6h history with probabilities).  
- 🚨 **Active Alerts Dashboard** (with severity levels: RED / ORANGE / GREEN).  
- 🌦️ **Live Weather Summary** (temperature, humidity, pressure, wind, radar intensity).  
- ✅ **Data Source Health Check** (WeatherAPI, RadarFeed, Prediction Engine).  
- 🔄 **Auto-refresh every 10s** with manual refresh controls.  
- 🗺️ **Map Controls** (zoom to airfield, toggle storms).  

---

## 🛠️ Tech Stack

- **Backend:** Django REST Framework (Python)  
- **AI/ML Models:** Scikit-learn, TensorFlow/Keras, NumPy  
- **Frontend Dashboard:** HTML, Tailwind CSS, Chart.js, Leaflet.js  
- **Database:** SQLite (default) — can be upgraded to PostgreSQL  
- **Deployment Ready:** Can be containerized (Docker) or deployed to any Django-compatible hosting  

---

## 📂 Project Structure

airfieldshield/
│
├── core/ # Airfield, WeatherStation, Prediction models
├── ai_models/ # AI training & prediction logic
│ └── management/commands/
│ ├── train_ensemble.py # Train ensemble models (RF + LSTM + CNN)
│ ├── predict.py # Run predictions
│ └── pipeline.py # One-click training + prediction pipeline
├── alerts/ # Alert models and refresh logic
├── templates/ # Frontend templates (dashboard)
├── static/ # Static files (CSS, JS, etc.)
├── manage.py
└── README.md

yaml
Copy code

---

## 🚀 Getting Started

### 1️⃣ Clone Repository
git clone https://github.com/dhruvindave007/AIrfieldShield.git
cd AIrfieldShield

shell
Copy code

### 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

shell
Copy code

### 3️⃣ Install Requirements
pip install -r requirements.txt

shell
Copy code

### 4️⃣ Run Migrations
python manage.py makemigrations
python manage.py migrate

pgsql
Copy code

### 5️⃣ Create Superuser (Admin)
python manage.py createsuperuser

shell
Copy code

### 6️⃣ Train Models + Generate Predictions (One Pipeline)
python manage.py pipeline --samples 8000 --seq-len 30 --epochs 10 --batch 64

shell
Copy code

### 7️⃣ Run Development Server
python manage.py runserver

yaml
Copy code

Access at 👉 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)  

---

## 📊 Dashboard Overview

- **Storm Activity Map** → real-time animated storm cells  
- **Risk Trend Chart** → 6-hour rolling predictions for thunderstorm & gale wind  
- **Active Alerts Panel** → severity-based alerts (RED, ORANGE, GREEN)  
- **Latest Predictions** → probability progress bars with confidence levels  
- **Weather Summary** → temperature, humidity, pressure, wind, radar  

---

## 🧪 Example Commands

- Train only AI models:  
python manage.py train_ensemble --samples 5000 --seq-len 30 --epochs 8

diff
Copy code

- Run predictions:  
python manage.py predict

diff
Copy code

- Refresh alerts:  
python manage.py run_alerts

yaml
Copy code

---

## 📷 Screenshots

*(Replace with your actual screenshots later)*

- Dashboard Home  
- Storm Map with overlays  
- Risk Trend Chart  
- Alerts panel  

---

## 📌 Contribution

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/new-feature`)  
3. Commit changes (`git commit -m 'Add some feature'`)  
4. Push to branch (`git push origin feature/new-feature`)  
5. Create a Pull Request  

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute with attribution.  

---

## 👤 Author & Contact

**Dhruvin Krutarthkumar Dave**  
- 📍 Ahmedabad, Gujarat, India  
- 📧 Email: davedhruvin307@gmail.com  
- 📱 Phone: +91 9510151831  
- 🔗 LinkedIn: [www.linkedin.com/in/mrdhruvindave](https://www.linkedin.com/in/mrdhruvindave)  
- 🐙 GitHub: [dhruvindave007](https://github.com/dhruvindave007)  

---
