# 🔥 AI-Powered Web Application Firewall (WAF)

An intelligent Web Application Firewall that combines **signature-based detection** and **machine learning** to identify and block malicious web traffic in real time, with **Explainable AI (XAI)** for transparency.

---

## 🚀 Features

* 🛡️ **Signature-Based Detection**
  Detects common attacks like SQL Injection, XSS, LFI, and command injection using regex patterns.

* 🤖 **Machine Learning Detection**
  Uses a **Random Forest model** to identify anomalous and obfuscated payloads.

* 🧠 **Explainable AI (XAI)**
  Provides human-readable explanations for each decision (Blocked / Quarantined / Allowed).

* 📊 **Interactive Dashboard UI**
  Real-time testing panel to simulate attacks and view responses instantly.

* 📁 **Logging System**
  Tracks:

  * Blocked requests
  * Quarantined requests
  * Allowed traffic

* ⚙️ **Adaptive Thresholding**
  Dynamically adjusts detection sensitivity based on attack patterns.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn (Random Forest)
* **Frontend:** HTML, CSS, JavaScript
* **Other:** Regex, Feature Engineering, REST APIs

---

## 🧠 How It Works

1. Incoming request is captured
2. Signature engine checks for known attack patterns
3. ML model analyzes payload features:

   * Entropy (randomness)
   * Token patterns
   * Special characters
4. Risk score is calculated
5. Decision is made:

   * ✅ Allow
   * ⚠️ Quarantine
   * ❌ Block
6. AI generates explanation for the decision

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/lalualfaz/ai-firewall.git

# Navigate to project
cd ai-firewall

# Install dependencies
pip install flask scikit-learn numpy joblib

# Run the application
python app.py
```

---

## 🔐 Admin Access

* **URL:** http://127.0.0.1:8080/login
* **Username:** admin
* **Password:** adminpass

---

## 🧪 Demo (What You Can Test)

* SQL Injection attack → ❌ Blocked
* XSS attack → ❌ Blocked
* Normal request → ✅ Allowed
* High entropy payload → ⚠️ Quarantined

---


## ⚠️ Limitations

* Trained on synthetic dataset (not real production traffic)
* No real threat intelligence integration
* Designed for learning/demo purposes

---

## 🚀 Future Improvements

* Integration with real-world traffic datasets
* Deployment on cloud (AWS / Azure)
* SIEM integration (Splunk / ELK)
* Rate limiting and IP blocking

---

## 👨‍💻 Author

**Your Name**
Aspiring Cybersecurity Analyst

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
