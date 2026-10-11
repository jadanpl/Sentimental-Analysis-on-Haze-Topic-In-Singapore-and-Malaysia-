ADD YOUR OWN YOUTUBE API
"""
Haze in Malaysia and Singapore: YouTube comments + Google Trends
-----------------------------------------------------------------
Setup:
    pip install google-api-python-client pytrends pandas matplotlib vaderSentiment langdetect wordcloud
    Get a YouTube Data API v3 key (Google Cloud Console > APIs & Services), then
    set it as an environment variable named YOUTUBE_API_KEY
    (or in a Jupyter cell: os.environ["YOUTUBE_API_KEY"] = "...").

What it produces
    Per country (malaysia / singapore):
        haze_<country>_dashboard.png    Trends, activity by month, sentiment, top words
        haze_<country>_wordclouds.png   positive / neutral / negative word clouds
        haze_<country>_sentiment.png    sentiment by month
        haze_<country>_emotions.png     anger / fear / sadness / hope / gratitude
        haze_<country>_wordpairs.png     word association heatmap (which words appear together)
        haze_<country>_engagement.png   likes and replies by sentiment and by emotion
        haze_<country>_channels.png     channels that drive the most discussion
    Both countries:
        haze_comparison_<a>_vs_<b>.png  sentiment, emotion, aspects and distinctive words
    Data: haze_youtube.csv (comments), haze_videos.csv, haze_channels.csv,
          haze_aspect_comparison.csv, haze_trends.csv

Notes
- YouTube search costs 100 quota units per call (default quota 10,000/day); comment, video
  and channel reads cost 1 unit per call.
- pytrends is UNOFFICIAL and can be rate limited; the script carries on without it.
- Sentiment (VADER) scores English comments only. VADER is extended with a haze-specific
  word list (HAZE_LEXICON) because it does not know words such as "toxic", "sue" or "corrupt".
- Emotion, aspect and network analysis use simple word lists (English + Malay), so they also
  work on Malay comments. They are transparent but crude: extend the lists for your data.
- Comments are the top comments by relevance, so engagement figures describe the most
  visible conversation, not every comment.
"""


import warnings
warnings.filterwarnings('ignore')
import os
os.environ["YOUTUBE_API_KEY"] = "ADD YOUR YOUTUBE API KEY"

import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import DetectorFactory, detect
from matplotlib import cm
from matplotlib.colors import Normalize
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS as WC_STOPWORDS
from wordcloud import WordCloud

DetectorFactory.seed = 0

COUNTRIES = {
    "Malaysia":  {"region": "MY", "yt_queries": ["jerebu", "haze Malaysia"],
                  "trend_kw": ["haze", "jerebu"]},
    "Singapore": {"region": "SG", "yt_queries": ["haze Singapore", "PSI haze Singapore"],
                  "trend_kw": ["haze"]},
}
VIDEOS_PER_QUERY = 25            # up to 50; a search call costs 100 quota units regardless
COMMENTS_PER_VIDEO = 100         # max 100 per request
LOOKBACK_DAYS = 365
TREND_TIMEFRAME = "today 12-m"

SENT_ORDER = ["negative", "neutral", "positive"]
SENT_COLORS = ["tab:red", "tab:gray", "tab:green"]

# ---------- Word lists you can edit ----------
# 1. Extra words for VADER (scale -4 to +4). VADER's built-in lexicon lacks these.
HAZE_LEXICON = {
    "toxic": -3.0, "sue": -2.0, "sues": -2.0, "sued": -2.0, "suing": -2.0, "lawsuit": -1.5,
    "corrupt": -3.0, "corruption": -3.0, "corrupted": -3.0, "hazardous": -3.0,
    "pollution": -1.8, "pollutants": -2.0, "polluted": -2.2, "smog": -1.5,
    "burning": -1.5, "burn": -1.2, "burnt": -1.5, "burned": -1.5, "fires": -1.5,
    "wildfire": -2.0, "wildfires": -2.0, "cough": -1.8, "coughing": -2.0, "asthma": -1.8,
    "suffocating": -3.0, "suffocate": -3.0, "choking": -2.5, "breathless": -2.0,
    "negligence": -2.5, "negligent": -2.5, "culprit": -2.0, "culprits": -2.0,
    "greedy": -2.5, "greed": -2.5, "incompetent": -2.5, "unacceptable": -2.2,
    "headache": -1.5, "deadly": -3.0, "lethal": -3.0, "unsafe": -2.0, "cancelled": -1.2,
    "carcinogenic": -3.0, "clearer": 1.2, "recovered": 1.0, "recovery": 1.0,
}

# 2. Emotions: each word belongs to ONE emotion. Hits are counted per comment; the emotion
#    with most hits is the comment's dominant emotion ("none" if no word matches).
EMOTIONS = ["anger", "fear", "sadness", "hope", "gratitude"]
EMO_COLORS = {"anger": "tab:red", "fear": "tab:purple", "sadness": "tab:blue",
              "hope": "tab:green", "gratitude": "gold", "none": "lightgray"}
EMOTION_WORDS = {
    "anger": "angry anger furious rage mad hate hated disgusting outrageous ridiculous useless stupid "
             "idiot idiots corrupt corruption greedy greed shameful blame irresponsible negligent "
             "negligence incompetent pathetic unacceptable sue sued suing lawsuit punish punished arrest "
             "accountable culprit culprits criminal criminals damn stop enough "
             "marah geram bengang bodoh rasuah korup tamak zalim sial celaka hentikan",
    "fear": "afraid scared fear fearful worried worry worrying anxious anxiety danger dangerous hazardous "
            "toxic unsafe deadly lethal threat risk panic asthma cough coughing breathing breathe "
            "breathless lungs cancer choking suffocating suffocate headache dizzy "
            "takut risau bimbang bahaya berbahaya cemas batuk asma kanser sakit demam",
    "sadness": "sad sadly sadness unfortunate unfortunately heartbreaking heartbroken tragic tragedy suffer "
               "suffering suffered pity sorry depressing depressed hopeless helpless cry crying tears "
               "despair grief painful victims victim died death dying "
               "sedih kasihan kesian menderita derita sayu pilu meninggal malang tragis mati",
    "hope": "hope hopefully hoped wish pray prayers praying prayer optimistic improve improving improvement "
            "better solution solutions recover recovery soon rain rains clearer together strength positive "
            "bless harap harapan doa berdoa semoga moga insyaallah pulih hujan selamat",
    "gratitude": "thank thanks thankful grateful gratitude appreciate appreciated appreciation kudos bravo "
                 "salute hero heroes syabas bersyukur alhamdulillah tq",
}
EMOTION_PHRASES = {
    "anger": ["fed up"],
    "fear": ["can't breathe", "cannot breathe", "sesak nafas", "tidak boleh bernafas", "tak boleh bernafas"],
    "hope": ["stay safe", "get well", "take care", "god bless", "insya allah", "mudah mudahan", "jaga diri"],
    "gratitude": ["well done", "great job", "terima kasih"],
}
NEGATORS = {"not", "no", "never", "dont", "don't", "cannot", "can't", "cant", "tak", "tidak",
            "bukan", "takde", "xde", "tiada"}
WORD_TO_EMO = {w: emo for emo, words in EMOTION_WORDS.items() for w in words.split()}

# 3. Aspects people talk about (used to compare Malaysia and Singapore)
ASPECTS = {
    "Health": "health cough coughing asthma breathing breathe lungs lung hospital doctor doctors respiratory "
              "illness disease cancer headache eyes throat sick kesihatan batuk asma doktor pernafasan nafas "
              "bernafas paru sesak demam sakit",
    "Government & policy": "government govt gov minister ministers ministry policy policies law laws enforcement "
                           "enforce accountable accountability corrupt corruption ban fine fines penalty "
                           "penalties sue lawsuit treaty kerajaan menteri kementerian undang rasuah saman dasar "
                           "penguatkuasaan",
    "Burning & causes": "burning burn burnt fires fire peat palm plantation plantations farmers companies "
                        "company slash wildfire wildfires kebakaran bakar pembakaran ladang sawit gambut syarikat",
    "Cross-border (Indonesia/ASEAN)": "indonesia indonesian indonesians neighbour neighbours neighboring "
                                      "neighbouring neighbor asean regional transboundary sumatra kalimantan "
                                      "riau jiran serantau indon",
    "Schools & work": "school schools students student teachers exam exams work office wfh closure closed "
                      "sekolah murid pelajar tutup kerja cuti",
    "Air quality readings": "psi aqi pm2 pm reading readings index nea ipu doe levels unhealthy hazardous "
                            "moderate",
    "Travel & events": "flight flights airport f1 grand prix race event events cancelled concert travel "
                       "tourism tourist tourists penerbangan kembara pelancong",
    "Economy": "cost costs money business businesses economy economic price prices tax afford ekonomi kos "
               "harga perniagaan duit wang",
    "Masks & protection": "mask masks n95 purifier purifiers hepa filter filters indoors indoor aircon "
                          "pelitup penapis",
}
ASPECT_WORDS = {a: set(w.split()) for a, w in ASPECTS.items()}

MALAY_STOPWORDS = (
    "saya kita kami anda awak dia mereka juga pun lagi sudah dah nak boleh kalau atau kerana sebab dalam pada "
    "ni tu la lah kat banyak sangat sgt ya tapi tetapi jadi macam lebih pula akan telah masih hanya semua "
    "setiap bila bagi oleh sama kena buat pergi je jer kan dengan dgn yg dr utk tak tk takde xde ada apa mana "
    "siapa kenapa sini situ yang dan di ke dari untuk ini itu tidak"
)
STOPWORDS = set(
    "the a an and or of to in on for is are was were it this that with at by from as be "
    "i we you they my our your have has had not no so but if will just can get like their its all should"
    "https http amp".split()
) | set(MALAY_STOPWORDS.split())

# Words that appear in almost every comment and carry no sentiment signal
CLOUD_EXTRA_STOPS = {
    # topic words
    "haze", "hazy", "jerebu", "kabut", "asap", "air", "burning", "burn", "burnt", "fire", "fires",
    "forest", "forests", "smoke", "pollution", "polluted", "quality", "video", "videos", "dust", "psi",
    # places
    "singapore", "singaporean", "malaysia", "malaysian", "asean", "sarawak", "penang", "johor",
    "kalimantan", "sumatra", "indonesia", "indonesian", "kuala", "lumpur",
    # generic filler
    "year", "years", "need", "every", "one", "now", "time", "even", "make", "also", "really",
    "going", "know", "see", "take", "must", "would", "could", "like", "get", "still", "already",
    "much", "many", "day", "days", "thing", "things", "way", "well", "want", "said", "say",
    "back", "come", "let", "dont", "don", "doesn", "didn", "isn", "just", "think", "yes", "yeah",
}

# Word clouds / word-pair heatmap
CLOUD_MODE = "distinctive"   # "distinctive" = words that stand out in each sentiment; "frequency" = plain counts
CLOUD_STRONG = 0.5           # only comments with |score| >= this go in the positive/negative clouds
CLOUD_MIN_COUNT = 2          # a word must appear at least this often to be shown
HEATMAP_TOP_WORDS = 20       # number of most common words shown in the word-pair heatmap
TOP_CHANNELS = 10


# ---------- 1a. YouTube ----------
def collect_youtube(api_key: str):
    """Returns (comments_df, videos_df)."""
    yt = build("youtube", "v3", developerKey=api_key)
    after = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    comment_rows, video_rows = [], {}

    for country, cfg in COUNTRIES.items():
        for q in cfg["yt_queries"]:
            print(f"YouTube [{country}] search: {q}")
            try:
                found = yt.search().list(
                    q=q, part="snippet", type="video", order="relevance",
                    maxResults=VIDEOS_PER_QUERY, regionCode=cfg["region"], publishedAfter=after,
                ).execute()["items"]
            except HttpError as e:
                print(f"  search failed (quota or key issue?): {e}")
                continue

            new_ids = []
            for v in found:
                vid, sn = v["id"]["videoId"], v["snippet"]
                if (country, vid) in video_rows:          # already found by another query
                    continue
                video_rows[(country, vid)] = {
                    "video_id": vid, "country": country, "title": sn["title"],
                    "channel_id": sn["channelId"], "channel_title": sn["channelTitle"],
                    "published_at": sn["publishedAt"], "views": 0, "video_likes": 0, "video_comments": 0,
                }
                new_ids.append(vid)

            for i in range(0, len(new_ids), 50):           # video statistics, 1 quota unit per call
                try:
                    stats = yt.videos().list(part="statistics", id=",".join(new_ids[i:i + 50])).execute()["items"]
                except HttpError as e:
                    print(f"  video statistics failed: {e}")
                    continue
                for s in stats:
                    st, row = s["statistics"], video_rows[(country, s["id"])]
                    row["views"] = int(st.get("viewCount", 0))
                    row["video_likes"] = int(st.get("likeCount", 0))
                    row["video_comments"] = int(st.get("commentCount", 0))

            for vid in new_ids:
                try:
                    threads = yt.commentThreads().list(
                        part="snippet", videoId=vid, order="relevance",
                        maxResults=min(100, COMMENTS_PER_VIDEO), textFormat="plainText",
                    ).execute()["items"]
                except HttpError:                           # comments disabled on this video
                    continue
                for c in threads:
                    s = c["snippet"]["topLevelComment"]["snippet"]
                    comment_rows.append({
                        "id": c["id"], "type": "comment", "country": country, "video_id": vid,
                        "channel_title": video_rows[(country, vid)]["channel_title"],
                        "created_at": s["publishedAt"], "text": s["textDisplay"],
                        "likes": s["likeCount"], "replies": c["snippet"].get("totalReplyCount", 0),
                    })

    videos = pd.DataFrame(list(video_rows.values()))
    if not videos.empty:                                   # channel subscriber counts
        subs, ids = {}, videos["channel_id"].unique().tolist()
        for i in range(0, len(ids), 50):
            try:
                for ch in yt.channels().list(part="statistics", id=",".join(ids[i:i + 50])).execute()["items"]:
                    st = ch["statistics"]
                    subs[ch["id"]] = None if st.get("hiddenSubscriberCount") else int(st.get("subscriberCount", 0))
            except HttpError as e:
                print(f"  channel statistics failed: {e}")
        videos["subscribers"] = videos["channel_id"].map(subs)
    return pd.DataFrame(comment_rows), videos


def clean(text: str) -> str:
    text = re.sub(r"http\S+|@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_detect(text: str) -> str:
    try:
        return detect(text) if len(text) > 20 else "unknown"
    except Exception:
        return "unknown"


# ---------- 1b. Emotion and aspect detection ----------
def classify_emotion(text: str):
    """Returns (dominant_emotion, counts_dict). Simple word-list method with basic negation."""
    low = text.lower()
    toks = re.findall(r"[a-z0-9']+", low)
    counts = dict.fromkeys(EMOTIONS, 0)
    for emo, phrases in EMOTION_PHRASES.items():
        counts[emo] += sum(p in low for p in phrases)
    for i, tok in enumerate(toks):
        emo = WORD_TO_EMO.get(tok)
        if not emo:
            continue
        if any(t in NEGATORS for t in toks[max(0, i - 2):i]):    # "no hope" -> sadness, other negated hits skipped
            if emo == "hope":
                counts["sadness"] += 1
            continue
        counts[emo] += 1
    best = max(EMOTIONS, key=lambda e: counts[e])               # ties go to the earlier emotion in EMOTIONS
    return (best if counts[best] > 0 else "none"), counts


def find_aspects(text: str):
    toks = set(re.findall(r"[a-z0-9']+", text.lower()))
    return [a for a, words in ASPECT_WORDS.items() if toks & words]


def enrich_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("id").copy()
    df["clean_text"] = df["text"].map(clean)
    df = df[df["clean_text"].str.len() > 0].copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_localize(None)
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    df["lang"] = df["clean_text"].map(safe_detect)

    # Sentiment: VADER + haze-specific words, English comments only
    df["scorable"] = df["lang"] == "en"
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(HAZE_LEXICON)
    df["sentiment"] = df.apply(
        lambda r: sia.polarity_scores(r["clean_text"])["compound"] if r["scorable"] else None, axis=1)
    df["sent_label"] = pd.cut(
        df["sentiment"], [-1.01, -0.05, 0.05, 1.01], labels=SENT_ORDER)

    # Emotion (English + Malay word lists) and aspects, all languages
    emo = df["clean_text"].map(classify_emotion)
    df["emotion"] = emo.map(lambda x: x[0])
    for e in EMOTIONS:
        df[f"emo_{e}"] = emo.map(lambda x, e=e: x[1][e])
    df["aspects"] = df["clean_text"].map(lambda t: ";".join(find_aspects(t)))
    return df


# ---------- 1c. Google Trends ----------
def collect_trends() -> pd.DataFrame:
    pt = TrendReq(hl="en-US", tz=480)
    frames = []
    for country, cfg in COUNTRIES.items():
        print(f"Trends [{country}]: {cfg['trend_kw']}")
        try:
            pt.build_payload(cfg["trend_kw"], timeframe=TREND_TIMEFRAME, geo=cfg["region"])
            d = pt.interest_over_time()
        except Exception as e:
            print(f"  skipped: {e}")
            continue
        if d.empty:
            continue
        d = d.drop(columns=["isPartial"], errors="ignore")
        out = pd.DataFrame({"date": d.index, "country": country, "interest": d.max(axis=1).values})
        frames.append(out)
        time.sleep(3)                         # be gentle with Google
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------- 2. Helpers ----------
def top_words(df: pd.DataFrame, n=20) -> pd.DataFrame:
    words = Counter()
    for t in df["clean_text"]:
        words.update(w for w in re.findall(r"[a-zA-Z]{3,}", t.lower()) if w not in STOPWORDS)
    return pd.DataFrame(words.most_common(n), columns=["word", "count"])


def _stops():
    return set(WC_STOPWORDS) | STOPWORDS | CLOUD_EXTRA_STOPS


def _doc_words(text: str, stops, limit=40):
    return sorted({t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in stops})[:limit]


def _annotate(ax, bars, ns):
    for rect, n in zip(bars, ns):
        ax.annotate(f"n={int(n)}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=9)


# ---------- 3. Per-country charts ----------
def plot_dashboard(country, yt, trends):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    if not trends.empty:
        trends.set_index("date")["interest"].plot(ax=ax[0, 0], color="tab:blue")
        ax[0, 0].set_ylim(0, 105)
    else:
        ax[0, 0].text(0.5, 0.5, "No Google Trends data\n(rate limited or skipped)",
                      ha="center", va="center", transform=ax[0, 0].transAxes)
    ax[0, 0].set_title("Google Trends interest (0-100)")

    if not yt.empty:
        yt.groupby("month").size().plot.bar(ax=ax[0, 1], color="tab:gray")
        ax[0, 1].set_title("YouTube comments collected per month")
        ax[0, 1].tick_params(axis="x", rotation=45)

        labels = yt["sent_label"].value_counts().reindex(SENT_ORDER).fillna(0)
        labels.plot.bar(ax=ax[1, 0], color=SENT_COLORS)
        ax[1, 0].set_title(f"Sentiment of English comments (n={int(labels.sum())})")
        ax[1, 0].tick_params(axis="x", rotation=0)

        tw = top_words(yt)
        if not tw.empty:
            tw.set_index("word")["count"].sort_values().plot.barh(ax=ax[1, 1])
        ax[1, 1].set_title("Top words")

    plt.suptitle(f"{country}: haze conversation on YouTube and Google Trends", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_dashboard.png", dpi=150)
    plt.show()
    plt.close(fig)


def _cloud_label(score):
    if pd.isna(score):
        return None
    if score >= CLOUD_STRONG:
        return "positive"
    if score <= -CLOUD_STRONG:
        return "negative"
    if abs(score) < 0.05:
        return "neutral"
    return None


def _count_words(texts, stops):
    c = Counter()
    for t in texts:
        c.update(w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in stops)
    return c


def distinctive_scores(texts_by_label, label, stops):
    """Weight each word by how much more it appears in this sentiment than in the others."""
    target = _count_words(texts_by_label[label], stops)
    rest = _count_words([t for l, ts in texts_by_label.items() if l != label for t in ts], stops)
    n_t, n_r = sum(target.values()), sum(rest.values())
    scores = {}
    for w, c in target.items():
        if c < CLOUD_MIN_COUNT:
            continue
        p_t = c / n_t
        p_r = (rest.get(w, 0) + 1) / (n_r + 1)
        s = p_t * math.log(p_t / p_r)
        if s > 0:
            scores[w] = s
    return scores


def plot_wordclouds(country, yt):
    d = yt.dropna(subset=["sentiment"]).copy()
    if d.empty:
        print(f"[{country}] no scored comments, skipping word clouds")
        return
    stops = _stops()
    styles = {"positive": "summer", "neutral": "winter", "negative": "autumn"}
    d["cloud_label"] = d["sentiment"].map(_cloud_label)
    texts_by_label = {l: d.loc[d["cloud_label"] == l, "clean_text"].tolist() for l in styles}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (label, cmap) in zip(axes, styles.items()):
        texts = texts_by_label[label]
        ax.axis("off")
        ax.set_title(f"{label.capitalize()} ({len(texts)} comments)", fontsize=14)
        wc = WordCloud(width=800, height=600, background_color="white", colormap=cmap,
                       stopwords=stops, collocations=False, max_words=80)
        try:
            if CLOUD_MODE == "distinctive":
                wc = wc.generate_from_frequencies(distinctive_scores(texts_by_label, label, stops))
            else:
                wc = wc.generate(" ".join(texts))
        except ValueError:
            ax.text(0.5, 0.5, "Not enough text", ha="center", va="center")
            continue
        ax.imshow(wc, interpolation="bilinear")

    mode = "words that stand out in each sentiment" if CLOUD_MODE == "distinctive" else "most frequent words"
    plt.suptitle(f"{country}: English YouTube comments by sentiment, {mode}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_wordclouds.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_sentiment(country, yt):
    scored = yt.dropna(subset=["sentiment"])
    if scored.empty:
        print(f"[{country}] no scored comments, skipping sentiment charts")
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    scored.groupby("month")["sentiment"].mean().plot(ax=ax[0], marker="o")
    ax[0].axhline(0, color="grey", lw=1)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("Average sentiment by month (-1 negative, +1 positive)")
    ax[0].tick_params(axis="x", rotation=45)

    (scored.groupby(["month", "sent_label"], observed=True).size().unstack(fill_value=0)
           .reindex(columns=SENT_ORDER, fill_value=0)
           .plot.bar(stacked=True, ax=ax[1], color=SENT_COLORS))
    ax[1].set_title("Comments per month by sentiment")
    ax[1].tick_params(axis="x", rotation=45)

    plt.suptitle(f"{country}: sentiment over time (English comments)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_sentiment.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_emotions(country, c):
    """Dominant emotion per comment (anger, fear, sadness, hope, gratitude), overall and by month."""
    order = EMOTIONS + ["none"]
    counts = c["emotion"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax[0].bar(order, counts.values, color=[EMO_COLORS[e] for e in order])
    for rect, n in zip(bars, counts.values):
        ax[0].annotate(f"{n / counts.sum() * 100:.0f}%", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                       ha="center", va="bottom", fontsize=10)
    ax[0].set_title(f"Dominant emotion (n={len(c)} comments; 'none' = no emotion word found)")

    monthly = (c.groupby(["month", "emotion"]).size().unstack(fill_value=0)
                .reindex(columns=order, fill_value=0))
    (monthly.div(monthly.sum(axis=1), axis=0) * 100).plot.bar(
        stacked=True, ax=ax[1], color=[EMO_COLORS[e] for e in order])
    ax[1].set_ylabel("% of comments")
    ax[1].set_title("Emotion mix by month")
    ax[1].tick_params(axis="x", rotation=45)

    plt.suptitle(f"{country}: emotions in haze comments", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_emotions.png", dpi=150)
    plt.show()
    plt.close(fig)


def plot_wordpairs(country, c):
    """Word association heatmap: how many comments contain both words of each pair."""
    stops = _stops()
    docs, word_docs, word_sent = [], Counter(), defaultdict(list)
    for text, sent in zip(c["clean_text"], c["sentiment"]):
        toks = _doc_words(text, stops)
        if len(toks) < 2:
            continue
        docs.append(toks)
        word_docs.update(toks)
        if pd.notna(sent):
            for w in toks:
                word_sent[w].append(sent)
    words = [w for w, n in word_docs.most_common(HEATMAP_TOP_WORDS) if n >= 2]
    if len(words) < 5:
        print(f"[{country}] not enough repeated words for a word-pair heatmap")
        return

    idx = {w: i for i, w in enumerate(words)}
    mat = np.zeros((len(words), len(words)))
    for toks in docs:
        present = [idx[w] for w in toks if w in idx]
        for i, j in combinations(present, 2):
            mat[i, j] += 1
            mat[j, i] += 1

    shown = np.where(np.tril(np.ones_like(mat, dtype=bool), k=-1), mat, np.nan)   # lower triangle only
    fig, ax = plt.subplots(figsize=(11, 9))
    heat = plt.get_cmap("YlOrRd").copy()
    heat.set_under("white")                                  # pairs that never appear together stay blank
    im = ax.imshow(np.ma.masked_invalid(shown), cmap=heat, vmin=0.5)
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=60, ha="right")
    ax.set_yticklabels(words)
    for i in range(len(words)):
        for j in range(i):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > np.nanmax(shown) * 0.6 else "black")
    # colour each word label by the average sentiment of English comments containing it
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        vals = word_sent.get(lab.get_text())
        m = np.mean(vals) if vals else 0
        lab.set_color("#c0392b" if m <= -0.15 else "#1e8449" if m >= 0.15 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="number of comments containing both words")
    ax.set_title(f"{country}: words that appear together in the same comment\n"
                 f"(label colour: red = mostly negative comments, green = mostly positive, black = mixed or unscored)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_wordpairs.png", dpi=150)
    plt.show()
    plt.close(fig)

    top = sorted(((mat[i, j], words[i], words[j]) for i in range(len(words)) for j in range(i)), reverse=True)[:8]
    print(f"[{country}] most frequent word pairs:", [(f"{a_}+{b_}", int(n)) for n, a_, b_ in top if n > 0])


def plot_engagement(country, c):
    """Average likes and replies by sentiment and by emotion."""
    scored = c.dropna(subset=["sent_label"])
    emo_order = EMOTIONS + ["none"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    def panel(axis, df, col, order, colors, metric, title):
        g = df.groupby(col, observed=True)[metric].agg(["mean", "count"]).reindex(order)
        bars = axis.bar(order, g["mean"].fillna(0), color=colors)
        _annotate(axis, bars, g["count"].fillna(0))
        axis.set_title(title)
        axis.set_ylabel(f"average {metric} per comment")

    panel(ax[0, 0], scored, "sent_label", SENT_ORDER, SENT_COLORS, "likes", "Likes by sentiment (English)")
    panel(ax[0, 1], scored, "sent_label", SENT_ORDER, SENT_COLORS, "replies", "Replies by sentiment (English)")
    ecol = [EMO_COLORS[e] for e in emo_order]
    panel(ax[1, 0], c, "emotion", emo_order, ecol, "likes", "Likes by emotion")
    panel(ax[1, 1], c, "emotion", emo_order, ecol, "replies", "Replies by emotion")

    plt.suptitle(f"{country}: which comments resonate? (top-relevance comments; n shown above bars)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_engagement.png", dpi=150)
    plt.show()
    plt.close(fig)

    print(f"\n[{country}] engagement (mean / median likes, mean replies):")
    print(scored.groupby("sent_label", observed=True).agg(
        n=("id", "count"), mean_likes=("likes", "mean"), median_likes=("likes", "median"),
        mean_replies=("replies", "mean")).round(2))
    print(c.groupby("emotion").agg(
        n=("id", "count"), mean_likes=("likes", "mean"), median_likes=("likes", "median"),
        mean_replies=("replies", "mean")).round(2))
    print(f"[{country}] most-liked comments:")
    cols = ["likes", "replies", "sent_label", "emotion", "clean_text"]
    top = c.nlargest(5, "likes")[cols].copy()
    top["clean_text"] = top["clean_text"].str.slice(0, 110)
    print(top.to_string(index=False))


def channel_table(country, videos_c, comments_c):
    """One row per channel: how much discussion its haze videos attract."""
    if videos_c.empty:
        return pd.DataFrame()
    ch = (videos_c.groupby("channel_title")
          .agg(videos=("video_id", "count"), views=("views", "sum"),
               total_comments=("video_comments", "sum"), subscribers=("subscribers", "max"))
          .reset_index())
    ch["comments_collected"] = ch["channel_title"].map(comments_c.groupby("channel_title").size()).fillna(0).astype(int)
    ch["avg_sentiment"] = ch["channel_title"].map(
        comments_c.dropna(subset=["sentiment"]).groupby("channel_title")["sentiment"].mean())
    ch["avg_likes_on_comments"] = ch["channel_title"].map(comments_c.groupby("channel_title")["likes"].mean())
    ch.insert(0, "country", country)
    return ch.sort_values("total_comments", ascending=False)


def plot_channels(country, ch):
    if ch.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    cmap, norm = plt.get_cmap("RdYlGn"), Normalize(-0.6, 0.6)

    top = ch.nlargest(TOP_CHANNELS, "total_comments").iloc[::-1]
    cols = [cmap(norm(s)) if pd.notna(s) else "lightgray" for s in top["avg_sentiment"]]
    ax[0].barh(top["channel_title"].str.slice(0, 32), top["total_comments"], color=cols)
    ax[0].set_title("Total comments on the channel's haze videos\n(colour = average sentiment of collected comments)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax[0], shrink=0.7, label="average sentiment (grey = no English comments)")

    top_v = ch.nlargest(TOP_CHANNELS, "views").iloc[::-1]
    ax[1].barh(top_v["channel_title"].str.slice(0, 32), top_v["views"], color="tab:blue")
    ax[1].set_title("Total views of the channel's haze videos")

    plt.suptitle(f"{country}: channels driving the haze discussion on YouTube", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"haze_{country.lower()}_channels.png", dpi=150)
    plt.show()
    plt.close(fig)

    print(f"\n[{country}] top channels by comments on haze videos:")
    print(ch.head(5)[["channel_title", "videos", "views", "total_comments", "subscribers", "avg_sentiment"]]
          .round(2).to_string(index=False))


# ---------- 4. Comparison of the two countries ----------
def two_prop_p(x1, n1, x2, n2):
    """Two-sided p-value for a difference between two proportions (normal approximation)."""
    if min(n1, n2) == 0:
        return 1.0
    p = (x1 + x2) / (n1 + n2)
    if p in (0, 1):
        return 1.0
    z = (x1 / n1 - x2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return math.erfc(abs(z) / math.sqrt(2))


def plot_comparison(yt):
    names = [c for c in COUNTRIES if (yt["country"] == c).any()]
    if len(names) < 2:
        print("Need data for two countries to compare.")
        return
    a, b = names[:2]
    A, B = yt[yt["country"] == a], yt[yt["country"] == b]
    colors = {a: "tab:blue", b: "tab:orange"}
    fig, ax = plt.subplots(2, 2, figsize=(17, 12))

    def share(df, col, order):
        s = df[col].value_counts().reindex(order).fillna(0)
        return s / s.sum() * 100 if s.sum() else s

    # 1. sentiment share
    sent = pd.DataFrame({a: share(A.dropna(subset=["sent_label"]), "sent_label", SENT_ORDER),
                         b: share(B.dropna(subset=["sent_label"]), "sent_label", SENT_ORDER)})
    sent.plot.bar(ax=ax[0, 0], color=[colors[a], colors[b]], rot=0)
    na, nb = A["sentiment"].notna().sum(), B["sentiment"].notna().sum()
    ax[0, 0].set_title(f"Sentiment share of English comments\n({a} n={na}, {b} n={nb})")
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("% of scored comments")

    # 2. emotion share
    ea, eb = A[A["emotion"] != "none"], B[B["emotion"] != "none"]
    emo = pd.DataFrame({a: share(ea, "emotion", EMOTIONS), b: share(eb, "emotion", EMOTIONS)})
    emo.plot.bar(ax=ax[0, 1], color=[colors[a], colors[b]], rot=0)
    ax[0, 1].set_title(f"Emotion mix of comments with an emotion word\n({a}: {len(ea)} of {len(A)}, {b}: {len(eb)} of {len(B)})")
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_ylabel("% of comments with an emotion")

    # 3. aspects (with significance test)
    rows = []
    for asp in ASPECTS:
        xa = int(A["aspects"].str.contains(asp, regex=False).sum())
        xb = int(B["aspects"].str.contains(asp, regex=False).sum())
        rows.append({"aspect": asp, f"{a}_n": xa, f"{b}_n": xb,
                     f"{a}_pct": xa / len(A) * 100, f"{b}_pct": xb / len(B) * 100,
                     "p_value": two_prop_p(xa, len(A), xb, len(B))})
    asp_df = pd.DataFrame(rows)
    asp_df.round(3).to_csv("haze_aspect_comparison.csv", index=False)
    plot_df = asp_df.assign(diff=asp_df[f"{a}_pct"] - asp_df[f"{b}_pct"]).sort_values("diff")
    ypos = np.arange(len(plot_df))
    ax[1, 0].barh(ypos - 0.2, plot_df[f"{a}_pct"], height=0.4, color=colors[a], label=a)
    ax[1, 0].barh(ypos + 0.2, plot_df[f"{b}_pct"], height=0.4, color=colors[b], label=b)
    ax[1, 0].set_yticks(ypos)
    ax[1, 0].set_yticklabels([f"{r.aspect}{' *' if r.p_value < 0.05 else ''}" for r in plot_df.itertuples()])
    ax[1, 0].set_xlabel("% of comments mentioning the aspect")
    ax[1, 0].set_title("What people talk about  (* = difference is statistically significant, p < 0.05)")
    ax[1, 0].legend()

    # 4. distinctive words
    # English comments only here: otherwise Malay words would dominate Malaysia's side and the
    # panel would show language, not topic (the aspect panel above already covers Malay words)
    stops = _stops()
    EA, EB = A[A["lang"] == "en"], B[B["lang"] == "en"]
    da, db = Counter(), Counter()
    for t in EA["clean_text"]:
        da.update(_doc_words(t, stops, limit=60))
    for t in EB["clean_text"]:
        db.update(_doc_words(t, stops, limit=60))
    scores = {}
    for w in set(da) | set(db):
        if da[w] + db[w] < 4:
            continue
        scores[w] = math.log((da[w] + 1) / (len(EA) + 2)) - math.log((db[w] + 1) / (len(EB) + 2))
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    pick = ranked[:10] + ranked[-10:] if len(ranked) > 20 else ranked
    ax[1, 1].barh([w for w, _ in pick], [s for _, s in pick],
                  color=[colors[b] if s < 0 else colors[a] for _, s in pick])
    ax[1, 1].axvline(0, color="grey", lw=1)
    ax[1, 1].set_title(f"English comments: words that stand out in {b} (left) vs {a} (right)")
    ax[1, 1].set_xlabel("log ratio of how often the word appears in each country's comments")

    plt.suptitle(f"{a} vs {b}: how the haze conversation differs", fontsize=17)
    plt.tight_layout()
    plt.savefig(f"haze_comparison_{a.lower()}_vs_{b.lower()}.png", dpi=150)
    plt.show()
    plt.close(fig)

    print("\nAspect comparison (share of comments mentioning each aspect):")
    print(asp_df.round(3).to_string(index=False))


# ---------- 5. Run ----------
def analyse_country(country, yt_c, tr_c, videos_c):
    print(f"\n===== {country} =====")
    plot_dashboard(country, yt_c, tr_c)
    if yt_c.empty:
        return pd.DataFrame()
    plot_wordclouds(country, yt_c)
    plot_sentiment(country, yt_c)
    plot_emotions(country, yt_c)
    plot_wordpairs(country, yt_c)
    plot_engagement(country, yt_c)
    ch = channel_table(country, videos_c, yt_c)
    plot_channels(country, ch)
    return ch


def main():
    yt, videos = pd.DataFrame(), pd.DataFrame()
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        raw, videos = collect_youtube(key)
        if not raw.empty:
            yt = enrich_comments(raw)
            yt.to_csv("haze_youtube.csv", index=False)
            videos.to_csv("haze_videos.csv", index=False)
            print(f"\nYouTube comments: {len(yt)}")
            print(yt["country"].value_counts())
            print("\nLanguages detected (top 4 per country):")
            print(yt.groupby("country")["lang"].value_counts().groupby(level=0).head(4))
            print("\nComments with a sentiment score (English only):")
            print(yt.groupby("country")["scorable"].agg(scored="sum", total="count"))
            print("\nDominant emotion by country (% of comments):")
            print((yt.groupby("country")["emotion"].value_counts(normalize=True) * 100).round(1).unstack(fill_value=0))
    else:
        print("YOUTUBE_API_KEY not set; skipping YouTube.")

    trends = collect_trends()
    if not trends.empty:
        trends.to_csv("haze_trends.csv", index=False)
        peak = trends.loc[trends.groupby("country")["interest"].idxmax()][["country", "date", "interest"]]
        print("\nPeak Google Trends week per country:\n", peak)

    channel_tables = []
    for country in COUNTRIES:
        yt_c = yt[yt["country"] == country] if not yt.empty else yt
        tr_c = trends[trends["country"] == country] if not trends.empty else trends
        v_c = videos[videos["country"] == country] if not videos.empty else videos
        ch = analyse_country(country, yt_c, tr_c, v_c)
        if not ch.empty:
            channel_tables.append(ch)
    if channel_tables:
        pd.concat(channel_tables).round(3).to_csv("haze_channels.csv", index=False)
    if not yt.empty:
        plot_comparison(yt)
    print("\nDone. Saved CSVs and charts for each country plus the comparison chart.")


if __name__ == "__main__":
    main()