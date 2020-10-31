# russian-reviews-sentiment-analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18u8LX5m3s18fYpVDP0UWlg9h0ZVmphe7?usp=sharing)

I just wanted to try out :hugs: Huggingface's new [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html) and get up-to-date with recent [torchtext](https://pytorch.org/text/) releases, also I've stumbled upon on an interesting Russian reviews [dataset](https://github.com/sismetanin/rureviews). So I decided to perform sentiment classification, gradually increasing the complexity of classifiers and compare their performance.

## Data

You can see [dataset overview](https://github.com/sismetanin/rureviews#dataset-overview) in the original repository.

## Models

I chose TF-IDF and SVM as a baseline model, CNN (because it was used in the original article) and [DeepPavlov's RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased-conversational) (rubert-base-cased-conversational), because I just wanted to try it out.

## Results

| Model                                               | Precision (macro) | Recall (macro) | F1-score (macro) |
| --------------------------------------------------- | ----------------- | -------------- | ---------------- |
| MNB (**article**)                                   | 74.47             | 73.79          | 73.90            |
| CNN (without emoticons) (**article**)               | 74.71             | 74.54          | 74.31            |
| CNN (with emoticons) (**article**)                  | 75.63             | 75.31          | 75.45            |
| TF-IDF + SVM (**mine**)                             | 74.44             | 74.46          | 74.45            |
| CNN (**mine**)                                      | 75.51             | 74.99          | 75.19            |
| RuBERT (rubert-base-cased-conversational) (**mine**)|                   |                |                  |

## References
1. Smetanin, S., & Komarov, M. (2019, July). Sentiment Analysis of Product Reviews in Russian using Convolutional Neural Networks. In 2019 IEEE 21st Conference on Business Informatics (CBI) (Vol. 1, pp. 482-486). IEEE.
