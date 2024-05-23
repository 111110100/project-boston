from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from urllib.parse import quote
import feedparser
import requests
import nltk
import re
import sys


# Function to fetch and parse RSS feed
def fetch_rss_feed(url):
    return feedparser.parse(url)


# Function to extract article links from RSS feed entries
def get_article_links(feed, num=5):
    return [entry.link for entry in feed.entries][0:num]


# Function to fetch and parse article content
def fetch_article_content(url):
    print(f"Fetching: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            article_text = ' '.join([para.get_text() for para in paragraphs])
            return article_text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
    return ''


# Function to tokenize and clean text
def tokenize_and_clean(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words


# Function to check if a sentence is important
def is_important(sentence):
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    named_entities = nltk.ne_chunk(tagged_words, binary=True)
    for subtree in named_entities:
        if isinstance(subtree, nltk.Tree) and subtree.label() == 'NE':  # Check if the subtree is a named entity
            return True
    return False


# Function to generate a bullet list of important sentences
def generate_bullet_list(all_text):
    sentences = sent_tokenize(all_text)
    word_frequencies = Counter(tokenize_and_clean(all_text))
    top_words = set(word for word, freq in word_frequencies.most_common(50))
    
    bullet_list = []
    for sentence in sentences:
        if any(word in sentence for word in top_words) and is_important(sentence):
            if not re.search(r'\bREAD MORE\b', sentence, re.IGNORECASE) and not re.search(r'http\S+', sentence):
                bullet_list.append(f"- {sentence}")
        if len(bullet_list) >= 10:  # Limit the bullet list to first 10 relevant sentences
            break
    
    return '\n'.join(bullet_list)


def main(topic, hl, gl, ceid, num=10):
    encoded_topic = quote(topic)
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl={hl}&gl={gl}&ceid={ceid}&num={num}"
    feed = fetch_rss_feed(rss_url)
    links = get_article_links(feed, num)
    
    all_text = ''
    for link in links:
        article_text = fetch_article_content(link)
        all_text += ' ' + article_text
    
    bullet_list = generate_bullet_list(all_text)
    print(f"Generated Bullet List for topic {topic}:")
    print(bullet_list)

if __name__ == "__main__":
    # Check if we have the required parameters
    if len(sys.argv) < 3:
        print(f"{'#' * 30}")
        print(f"Requires two parameters: topic number-of-feeds-to-process")
        print("")
        print(f"Example: main.py celtics 3")
    else:
        # Download NLTK data files (only need to be run once)
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        main(sys.argv[1], "en-AU", "AU", "AU:en", int(sys.argv[2]))
