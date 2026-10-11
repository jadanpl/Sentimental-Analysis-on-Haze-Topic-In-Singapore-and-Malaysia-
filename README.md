# Sentimental-Analysis-on-Haze-Topic-In-Singapore-and-Malaysia-
A data-driven initiative using Python on YouTube and Google Trends data.
# Haze Conversations in Malaysia and Singapore

A Python pipeline that collects public YouTube comments and Google Trends data about the Southeast Asian haze, then analyses **sentiment, emotions, topics, word associations, engagement and channel influence**, and compares Malaysia with Singapore.

> Snapshot used for the example results below: comments and searches collected in October 2026, while search interest in haze was climbing to its highest point of the past 12 months in both countries.

## Why this project

Official monitoring tells us how bad the air is. It does not tell us how people experience it, what worries them, or who they blame. This project listens to the public conversation to answer questions such as:

- Is the tone of the conversation different in Malaysia and Singapore?
- Are people angry, afraid, sad or hopeful?
- Which topics (health, government, burning, schools, travel, the economy) dominate each country?
- Which YouTube channels drive the haze discussion?
- Which kinds of comments get the most likes and replies?

## Key findings (October 2026 snapshot)

| Measure | Malaysia | Singapore |
| --- | --- | --- |
| Comments collected | 945 | 991 |
| English comments scored | 418 | 851 |
| Negative / neutral / positive | 50.5% / 19.6% / 29.9% | 40.4% / 16.9% / 42.7% |
| Comments containing an emotion word | 23% | 29% |
| Dominant emotion | Anger (42% of emotional comments) | Anger (43% of emotional comments) |
| Topics mentioned more | Cross-border blame, schools and work | Air quality readings, masks, travel and events, the economy |
| Channel with most comments | Francesca Chen (then Al Jazeera English) | CNA (then Al Jazeera English) |
| Discussion concentration | Spread across many channels | Concentrated in CNA and Al Jazeera English |

- Malaysia is significantly more negative than Singapore (two-proportion test, p < 0.001 for negative and p < 0.0001 for positive).
- The emotion mix is almost identical. Differences in sadness and gratitude are **not** statistically significant at the 5% level.
- In both countries, *palm* and *oil* are the strongest word pair, and both link to *companies*.
- Malaysia's data includes many Indonesian-language comments, so it partly reflects Indonesian viewers (see [Limitations](#limitations)).

Figures are read from the generated charts and are approximate. Your own run will differ.

## Features

| Analysis | What it shows | Output |
| --- | --- | --- |
| Dashboard | Google Trends interest, comments per month, sentiment counts, top words | `haze_<country>_dashboard.png` |
| Word clouds | Positive, neutral and negative comments, sized by how much each word *stands out* in that sentiment | `haze_<country>_wordclouds.png` |
| Sentiment over time | Average sentiment by month and monthly counts by sentiment | `haze_<country>_sentiment.png` |
| Emotions | Anger, fear, sadness, hope and gratitude, overall and by month | `haze_<country>_emotions.png` |
| Word-pair heatmap | How many comments contain each pair of the 20 most common words | `haze_<country>_wordpairs.png` |
| Engagement | Average likes and replies by sentiment and by emotion | `haze_<country>_engagement.png` |
| Channels | Channels whose haze videos attract the most comments and views | `haze_<country>_channels.png` |
| Country comparison | Sentiment share, emotion mix, topic shares with significance tests, distinctive words | `haze_comparison_malaysia_vs_singapore.png` |

Data files: `haze_youtube.csv` (comments), `haze_videos.csv`, `haze_channels.csv`, `haze_aspect_comparison.csv`, `haze_trends.csv`.

## How it works

1. **Collect.** The YouTube Data API searches each country (restricted to its region, past 12 months) with local search terms and returns the most relevant videos. For each video the script reads the top comments (with likes and replies) and records the channel, views and comment count. Google Trends (via `pytrends`) provides weekly search interest.
2. **Clean.** Links and mentions are removed, duplicates dropped, and each comment's language is detected with `langdetect`.
3. **Score sentiment.** VADER scores English comments from -1 to +1 and labels them negative, neutral or positive (cut-offs at -0.05 and +0.05). VADER's built-in word list does not know many haze-related words (for example *toxic*, *sue*, *corrupt*, *burning*), so the script adds about 50 of them in `HAZE_LEXICON`.
4. **Classify emotions.** Word lists (English and Malay) assign each comment a dominant emotion: anger, fear, sadness, hope or gratitude, or "none". Simple negation is handled ("no hope" counts as sadness).
5. **Tag topics.** Word lists tag comments with topics such as health, government and policy, burning and causes, cross-border blame, schools and work, air quality readings, travel and events, the economy, and masks and protection.
6. **Analyse and visualise.** Per-country charts are created, then a comparison chart. Topic differences between countries are tested with a two-proportion test.

## Getting started

### Requirements

- Python 3.9 or newer
- A YouTube Data API v3 key (see below)

### Install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Get a YouTube API key

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and create a project.
2. Enable **YouTube Data API v3** under *APIs & Services*.
3. Create an **API key** under *Credentials*.

### Set the key

Never put the key in the code or commit it. Set it as an environment variable:

```bash
# macOS / Linux
export YOUTUBE_API_KEY="your_key_here"

# Windows (Command Prompt)
set YOUTUBE_API_KEY=your_key_here

# Windows (PowerShell)
$env:YOUTUBE_API_KEY="your_key_here"
```

In a Jupyter notebook, run this in a first cell (and delete the cell before sharing the notebook):

```python
import os
os.environ["YOUTUBE_API_KEY"] = "your_key_here"
```

### Run

```bash
python haze_youtube_trends.py
```

or in Jupyter:

```python
%run haze_youtube_trends.py
```

Charts and CSV files are saved in the working directory.

## Configuration

Settings and word lists sit at the top of `haze_youtube_trends.py`:

| Setting | Purpose |
| --- | --- |
| `COUNTRIES` | Countries, YouTube region codes, search terms and Google Trends keywords |
| `VIDEOS_PER_QUERY`, `COMMENTS_PER_VIDEO` | How much to collect (more videos use more API quota) |
| `LOOKBACK_DAYS`, `TREND_TIMEFRAME` | Time window |
| `HAZE_LEXICON` | Extra words and scores for VADER (scale -4 to +4) |
| `EMOTION_WORDS`, `EMOTION_PHRASES` | Emotion word lists |
| `ASPECTS` | Topic word lists used in the country comparison |
| `CLOUD_MODE`, `CLOUD_STRONG`, `CLOUD_MIN_COUNT` | Word cloud behaviour |
| `HEATMAP_TOP_WORDS` | Number of words in the word-pair heatmap |

To study other countries, add an entry to `COUNTRIES`. Comments in other languages will need new word lists, or translation before scoring.

## Project structure

```
.
├── haze_youtube_trends.py   # the full pipeline
├── requirements.txt
├── README.md
└── images/                  # example charts used in this README
```

## Limitations

- **Not representative of the population.** YouTube commenters are a self-selected group, and the script collects the *top-ranked* comments, so popular and visible comments are over-represented. Likes and replies partly reflect visibility.
- **Sentiment covers English only.** VADER cannot score Malay, so Malaysia's sentiment results describe its English-speaking commenters.
- **Country is approximate.** The region setting biases which videos are found, but it does not guarantee the commenters live there. In the example run, many "Malaysia" comments were written in Indonesian.
- **Word lists are simple.** Emotion and topic detection are transparent but crude: they miss sarcasm, spelling variations and words not on the lists. In the example run, about three-quarters of comments contained no emotion word.
- **Small samples.** Differences between countries can be chance; the comparison chart marks only the topic differences that pass a statistical test.
- **Google Trends is relative.** Values run from 0 to 100 within each country and keyword set, so they cannot be compared directly across countries. `pytrends` is unofficial and may be rate limited.
- **One surge of attention.** The example results come from a single episode, not a long-term trend.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `quotaExceeded` or 403 from YouTube | The free daily quota (10,000 units) is used up. A search costs 100 units. Lower `VIDEOS_PER_QUERY` or wait until tomorrow. |
| Google Trends skipped or HTTP 429 | You were rate limited. Wait and re-run, or download the CSV from trends.google.com. The rest of the script still runs. |
| `ValueError: Only supported for TrueType fonts` | Upgrade the libraries (`pip install -U wordcloud pillow`) and restart the kernel. |
| Very few scored comments for a country | Most comments are probably not in English. Check the language table the script prints. |

## Ethics, data and terms

- Only publicly available comments are used, and the analysis reports aggregate patterns, not individuals.
- Follow the [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service) and Google Trends terms. Do not republish raw comment data; the provided `.gitignore` keeps the generated CSV files out of the repository.

## Roadmap

- Replace the emotion word lists with a trained emotion model
- Add topic modelling to find themes automatically
- Test a sentiment model that reads Malay directly
- Add significance tests for the sentiment and emotion panels of the comparison chart
- Extend the study to Indonesia and Thailand (translation needed)
- Compare conversation volume with official air quality readings (PSI / API)

## License

MIT. Add a `LICENSE` file before publishing.
