# PRISM-X: Proactive Research Ideation Supervision with Multi-Agent Oversight

PRISM-X is a multi-agent pipeline for safer, more novel research idea generation. It comprises:

* **AEGIS X**: A parent-layer governance system with three agents (Guardian, Attacker, Observer) that oversee, test, and log behavior in real-time.
* **NOVA**: A child-layer agent that reads academic papers, ranks them, and generates structured research proposals.

Developed during a hackathon, this system enables transparent and trustworthy research ideation with explainable traces, audit logs, and novelty scoring.

---

## 🗂 Project Structure

```
├── aegis/                # Guardian, Attacker, and Observer agents
├── aegis-env/            # Python virtual environment folder (not tracked)
├── logs/                 # Action logs and JSON traces
├── metrics/              # Risk scores, novelty scores, clarity ratings
├── nova/                 # NOVA agent modules for idea generation
├── tutorials/            # Sample walkthroughs or notebooks
├── ui/                   # React + Tailwind-based frontend interface
├── run_demo.py           # Run the full AEGIS–NOVA demo pipeline
├── test_bedrock.py       # Bedrock integration and stress tests
├── holistic_ai_bedrock.py# Optional: Connect to Holistic AI platform
├── .env                  # Local environment variables (e.g., API keys)
├── requirements.txt      # Python dependencies
```

---

## 🛠 Setup Instructions

### 1. Create and Activate Python Virtual Environment

```bash
python -m venv aegis-env
source aegis-env/bin/activate   # Windows: aegis-env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root. Sample:

```
OPENAI_API_KEY=your-key-here
ARXIV_API_KEY=optional
```

---

## 🚀 Run the Demo Pipeline

Run the main AEGIS X + NOVA workflow:

```bash
python run_demo.py
```

This will:

* Accept a research query
* Trigger NOVA to generate a research idea
* Apply Guardian, Attacker, and Observer agents
* Output audit logs and final idea in `/logs` and `/metrics`

---

## 🖼️ Launch the Frontend Interface (Optional)

```bash
cd ui
npm install
npm run dev
```

This starts the AEGIS X UI with:

* Search box for research prompts
* System response panel
* Guardian and Attacker visualizations (bar charts)

---

## ✅ Run Tests

```bash
python test_bedrock.py
```

---

## 📁 Open the Final Submission

Please open the `final_submission` folder (if included in your project zip). It contains:

* The full AAMAS-style PDF paper
* Supporting images or diagrams
* Optional slides or summary sheets

---

## 📄 License

MIT License or institutional license if applicable.

---

## 👨‍💻 Authors

* Ravpreet Singh Gill (Project Lead)
* AEGIS X architecture, governance scoring
* NOVA ideation engine
* UI design and visual analytics

---

## 🔑 Keywords

`multi-agent systems`, `governance`, `AI safety`, `research automation`, `explainability`, `idea generation`, `LLM alignment`, `oversight`
