# AI-Powered Text Summarizer
# Author: Florentin Rusu

from transformers import pipeline

def summarize_text(text, max_length=130, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    article = """
    Artificial Intelligence (AI) is transforming industries worldwide. 
    From healthcare to finance, AI systems are being used to analyze data, 
    improve decision-making, and automate repetitive tasks. 
    While AI offers great opportunities, it also raises ethical questions 
    about bias, privacy, and the future of work.
    """
    print("Original Text:\n", article)
    print("\nSummary:\n", summarize_text(article))