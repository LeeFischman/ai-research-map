import arxiv
import pandas as pd
import subprocess
import os

def fetch_arxiv_data(limit=1000):
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    data = []
    for r in client.results():
        # SPECTER2 likes Title and Abstract joined
        full_text = f"{r.title}. {r.summary}"
        data.append({
            "title": r.title,
            "text_for_embedding": full_text,
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d")
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📥 Fetching latest cs.AI papers...")
    df = fetch_arxiv_data(1000)
    
    # Save as Parquet (efficient for large datasets)
    df.to_parquet("papers.parquet")
    
    print("🧠 Building Atlas with SPECTER2 (this may take a few minutes)...")
    # This runs the Apple Atlas CLI to generate the static site files
    subprocess.run([
        "embedding-atlas", "papers.parquet",
        "--text", "text_for_embedding",
        "--model", "allenai/specter2_base",
        "--export-application", "site.zip"
    ], check=True)
    
    # Unzip the generated site into the 'docs' folder for GitHub Pages
    os.makedirs("docs", exist_ok=True)
    os.system("unzip -o site.zip -d docs/")
    print("✅ Success! Site files generated in /docs")