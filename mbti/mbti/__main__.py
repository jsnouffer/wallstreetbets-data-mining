import pandas as pd

from .model import *
from .text_wrangle import *


def load_data():
    df = pd.read_csv("/home/jason/mbti_model/mbti_1.csv")
    # df = df.head(10)
    print(f"Input shape = {df.shape}")
    return df


df = load_data()
list_posts, list_personality = pre_process_text(
    df, remove_stop_words=True, remove_mbti_profiles=True
)
train(list_posts, list_personality)
