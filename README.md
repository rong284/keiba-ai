# keiba-ai

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install playwright tqdm pandas beautifulsoup4 lxml
python -m playwright install --with-deps chromium
```

## Scraping
```bash
python src/data/scraping/scrape_race_data.py --year 2024 --concurrency 1 --skip
```

## Parse
```bash
python src/data/scraping/parse_race_html_to_tsv.py --year 2024
```
