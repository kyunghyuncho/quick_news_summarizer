# this file uses Gradio to implement a news headline summarizer.
import gradio as gr

from newsapi_headlines import NewsAPIHeadlines
from gemma_summarizer import GemmaSummarizer
from llama3_summarizer import LlamaSummarizer

Summarizer = LlamaSummarizer

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def summarize_news(api_key: str, hf_token: str):
    print("Retrieving the headlines")
    headlines = NewsAPIHeadlines(api_key)
    news = headlines.get_headlines()
    print("Headlines:")
    print(news)

    print("Generating the summary")
    summarizer = Summarizer(hf_token, device)
    summary, _ = summarizer.summarize(news)
    print(summary)

    return news, summary

iface = gr.Interface(fn=summarize_news, 
                     inputs=[gr.Textbox(label="newsapi.org API token", placeholder="your token"), 
                             gr.Textbox(label="HF API token", placeholder="your token")], 
                     outputs=[gr.Textbox(label="Headlines"), 
                              gr.Textbox(label="Summary")],
                     title="News Headline Summarizer",
                     description="Summarize news headlines.")

iface.launch(share=True)