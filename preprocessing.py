import re


def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[^а-яёА-ЯЁ ]", "", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    return text


def substitute_emoticons(text):
    text = re.sub(r"\){2,}", "<happy>", text)
    text = re.sub(r"\({2,}", "<sad>", text)
    return text
