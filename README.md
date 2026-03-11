# StartupMatch

StartupMatch is a web app that matches students to startups based on their skills, interests, and experience. Users can type skills directly or upload an image of the skills section of a resume, and the system returns relevant startup matches using TF-IDF + cosine similarity.

## Data

The project uses an already-prepared enriched startup dataset committed to the repository. The underlying company data was constructed from:
- Y Combinator Directory (2023): 4,000+ YC companies https://www.kaggle.com/datasets/miguelcorraljr/y-combinator-directory
- Startups.csv: 700+ YC-backed startups from 2005–2014 https://www.kaggle.com/datasets/joebeachcapital/startups
- AI Company & Startup Funding Database https://www.kaggle.com/datasets/prajitdatta/ai-company-and-startup-funding-database


## Run Locally
### 1. Clone the Repo
```shell
git clone 
cd STARTUP-MATCH
```

### 1. Set up Python virtual environment
```shell
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies
```shell
pip install -r requirements.txt
```

### 3. Start Flask backend (in one terminal)
```shell
python src/app.py
```

### 4. In a NEW terminal, install and start React
```shell
cd frontend
npm install
npm run dev
```


The frontend runs at:
http://localhost:5173

## Notes
- Data preloaded into repo
- Once Flask and the frontend are running, the app should work immediately.
