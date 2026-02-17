import arxiv
import pandas as pd
import subprocess
import os
import gc
import re
import json
import time
from datetime import datetime, timedelta, timezone

DB_PATH = "database.parquet"
os.makedirs("docs", exist_ok=True)

# --- 1. REPUTATION LOGIC ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard", "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    full_text = f"{row['label']} {row['text']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    authors = row.get('authors', [])
    if len(authors) >= 5: score += 1
    if any(k in full_text for k in ['github.com', 'huggingface.co']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 2. POST-BUILD INJECTION (The "Fail-Safe" Fix) ---
def finalize_site(docs_path):
    # Fix the labels by forcing the config.json
    config_path = os.path.join(docs_path, "data", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # This is the critical override:
        config["name_column"] = "label" 
        
        # Ensure the UI knows 'label' exists as a display field
        if "column_mappings" not in config:
            config["column_mappings"] = {}
        config["column_mappings"]["label"] = "label"
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            print("✅ Successfully forced 'label' column in config.json")

    # Fix the UI Pop-out
    index_file = os.path.join(docs_path, "index.html")
    if os.path.exists(index_file):
        ui_blob = """
        <div id="custom-overlay" style="position:fixed; top:20px; left:20px; z-index:999999;">
            <button id="info-toggle" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold;">⚙️ Menu</button>
            <div id="info-tab" style="display:none; margin-top:10px; width:260px; background:#111827; border:1px solid #374151; color:white; padding:20px; border-radius:12px; font-family:sans-serif;">
                <h2 style="margin:0; color:#60a5fa; font-size:18px;">AI Research Map</h2>
                <p style="font-size:13px;">By Lee Fischman</p>
                <hr style="border:0; border-top:1px solid #374151;">
                <p style="font-size:12px;">Color by 'Reputation' to see weighted labs.</p>
            </div>
        </div>
        <script>
            document.getElementById('info-toggle').onclick = function() {
                var tab = document.getElementById('info-tab');
                tab.style.display = (tab.style.display === 'none' || tab.style.display === '') ? 'block' : 'none';
            };
        </script>
        """
        with open(index_file, "r") as f:
            content = f.read()
        with open(index_file, "w") as f:
            f.write(content.replace("<body>", "<body>" + ui_blob))

# --- 3. EXECUTION ---
if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    results = list(client.results(search))
    if results:
        df = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1],
            "label": r.title,                 # This is our intended display title
            "text": f"{r.title}. {r.summary}", # This is the vector source
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d"),
            "authors": [a.name for a in r.authors]
        } for r in results])
        
        df = df.drop_duplicates(subset='id').reset_index(drop=True)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Building Map...")
        # We omit --labels to avoid the FileNotFoundError
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # Apply the logic that the tool failed to do itself
        finalize_site("docs")
        print("✨ Sync Complete!")
