import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split


def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"                                        
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"                                        
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def convert_insulto_text(insultos):
    text = ''
    for insulto in insultos:
        text += insulto + ', '
    text = text.strip(', ')
    return text


def load_data(file, is_features):
    df = pd.read_json(file, lines=True)

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=1)
    df_train, df_eval = train_test_split(df_train, test_size=0.20, random_state=1)
    df_test.to_json("test.json", orient='records', lines=True)
    df_eval.to_json("eval.json", orient='records', lines=True)
    df_train.to_json("train.json", orient='records', lines=True)

    df["text"] = df.text.apply(preprocess)
    labels = df['label']
    # To labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels.values)

    texts = df['text']
    features = [[]] * len(df)
    if is_features:
        df['insultos'] = df['insultos'].apply(convert_insulto_text)
        insultos = df['insultos']
        list_of_tuples = list(zip(list(texts), list(insultos), list(labels), features))

        df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels', 'features'])
    else:
        list_of_tuples = list(zip(list(texts), list(labels)))
        df = pd.DataFrame(list_of_tuples, columns=['text', 'labels'])
    return df