# russian-reviews-sentiment-analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

I just wanted to try out :hugs: Huggingface's new [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html) and get up-to-date with recent [torchtext](https://pytorch.org/text/) releases, also I've stumbled upon on an interesting Russian reviews [dataset](https://github.com/sismetanin/rureviews).

So I decided to perform sentiment classification gradually increasing the complexity of classifiers and compare their performance.

## Data

You can see [dataset overview](https://github.com/sismetanin/rureviews#dataset-overview) in the original repository.

## Models

I chose TF-IDF and SVM as a baseline model, CNN (because it was used in the original article) and [DeepPavlov's RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational) (rubert-base-cased-conversational).

## Results

| Model                                            | Precision (macro) | Recall (macro) | F1-score (macro) |
| --------------------------------------------------- | ----------------- | -------------- | ---------------- |
| MNB, **article**                                    | 74.47             | 73.79          | 73.90            |
| CNN (without emoticons), **article**                | 74.71             | 74.54          | 74.31            |
| CNN (with emoticons), **article**                   | 75.63             | 75.31          | 75.45            |
| TF-IDF + SVM, **mine**                              |                   |                |                  |
| CNN (without emoticons), **mine**                   |                   |                |                  |
| RuBERT (rubert-base-cased-conversational), **mine** |                   |                |                  |
