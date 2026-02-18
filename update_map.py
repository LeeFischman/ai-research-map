import arxiv
import pandas as pd
import subprocess
import os
import re
import json
import time
import shutil
from datetime import datetime, timedelta, timezone

DB_PATH = "database.parquet"
LABELS_PATH = "manual_labels.csv"
STOPWORDS_PATH = "stop_words.csv"

# --- 1. DATA PREP ---
def load_stop_words():
    if os.path.exists(STOPWORDS_PATH):
        sw_df = pd.read_csv(STOPWORDS_PATH)
        return set(sw_df['word'].str.lower().tolist())
    return set()

STOP_WORDS = load_stop_words()

def scrub_text(text):
    words = text.split()
    cleaned = [w for w in words if w.lower().strip('.,()[]{}') not in STOP_WORDS]
    return " ".join(cleaned)

def fetch_results_with_retry(client, search):
    for i in range(5):
        try:
            return list(client.results(search))
        except Exception as e:
            time.sleep((2**i) * 30)
    return []

if __name__ == "__main__":
    if os.path.exists("docs"): shutil.rmtree("docs")
    os.makedirs("docs")

    client = arxiv.Client(page_size=100, delay_seconds=5)
    now = datetime.now(timezone.utc)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=7)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=200
    )
    
    results = fetch_results_with_retry(client, search)
    if results:
        data = []
        for r in results:
            # We preserve title/abstract for the user but scrub the 'text' column used for the AI
            data.append({
                "title": r.title,
                "summary": r.summary,
                "text": scrub_text(f"{r.title} {r.summary}"),
                "url": r.pdf_url,
                "id": r.entry_id.split('/')[-1]
            })
            
        df = pd.DataFrame(data)
        
        # Reputation logic (strictly for coloring)
        inst_pat = re.compile(r"\b(MIT|Stanford|CMU|Berkeley|Harvard|DeepMind|OpenAI|Anthropic|FAIR|Meta)\b", re.I)
        def get_rep(row):
            score = 3 if inst_pat.search(f"{row['title']} {row['summary']}") else 0
            if 'github.com' in row['summary'].lower(): score += 2
            return "Enhanced" if score >= 4 else "Standard"
        
        df['Reputation'] = df.apply(get_rep, axis=1)
        df.to_parquet(DB_PATH, index=False)

        # --- THE BRUTE FORCE LABEL FIX ---
        # We tell the tool: "Don't guess. Here is the label for the center of the map."
        # This ensures SOMETHING shows up even when stopwords are all gone.
        labels_df = pd.DataFrame([
            {"x": 0, "y": 0, "text": "Recent AI Research", "level": 0, "priority": 1}
        ])
        labels_df.to_csv(LABELS_PATH, index=False)

        print("🧠 Building Map with Manual Labels...")
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--labels", LABELS_PATH, # <--- FORCING THE LABEL
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # --- UI CONFIG ---
        config_path = "docs/data/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f: conf = json.load(f)
            conf["name_column"] = "title"
            conf.update({
                "color_by": "Reputation",
                "topic_label_column": None, # Disable the tool's broken auto-labels
                "column_mappings": {"title":"title", "Reputation":"Reputation", "url":"url"}
            })
            with open(config_path, "w") as f: json.dump(conf, f, indent=4)

        print("✨ Deployment Successful!")
