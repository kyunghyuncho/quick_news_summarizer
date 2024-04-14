# This file implements Headlines class for newsapi.org API.

import os
import time

import requests

from headlines import Headlines

class NewsAPIHeadlines(Headlines):
    def __init__(self, api_key: str = "your_newsapi_key_here"):
        self.api_key = api_key

    def _format_headlines(self, response, max_headlines: int = 100):
        headlines = []
        for aidx, article in enumerate(response):
            if aidx > max_headlines:
                break
            if "Removed" in article['title']:
                continue

            new_headline = f"{aidx+1}. {article['title'].strip()} "
            new_headline += f"(URL: {str(article['url']).strip()}) "
            new_headline += str(article['description']).strip().replace('\n', '...')
            new_headline += "\n"
            headlines.append(new_headline)

        headlines = "\n".join(headlines)

        return headlines

    def get_headlines(self, max_headlines: int = 100):
        url = f'https://newsapi.org/v2/top-headlines?country=us&language=en&apiKey={self.api_key}'
        response = requests.get(url)

        return self._format_headlines(response.json()['articles'], max_headlines)
