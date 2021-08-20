import pandas as pd
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords as st
import nltk
from nltk.corpus import webtext
from nltk.probability import FreqDist
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from sklearn import preprocessing
from wordcloud import WordCloud, ImageColorGenerator
import re
import plotly
from PIL import Image
import requests
from IPython.core.display import HTML
import random
from nlp_package_pv import *
import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

plotly.offline.init_notebook_mode(connected=True)
HTML(
    """
    <style>
    .output_png {
        display: table-cell;
        text-align: center;
        vertical-align: middle;
    }
    </style>
    """
)


def main():
    return HTML(
        """
    <style>
    .output_png {
        font-size: large ;
        display: table-cell;
        text-align: center;
        vertical-align: middle;
    }
    </style>
    """
    )


def rotate_n(df, number):
    x = []
    p = df.iloc[1:].values
    for i in range(len(p)):
        x.append([p[i, 2], p[i, 3]])
    t = x.copy()
    time = 1
    ho = len(x)
    times = [time] * ho
    for i in range(number - 1):
        time += 1
        t = deque(t)
        t.rotate(-1)
        t = list(t)
        x.extend(t)
        times.extend([time] * ho)
    x = np.array(x)
    dn = pd.DataFrame()
    words = []
    count = []
    for i in range(len(p)):
        words.append(p[i, 0])
        count.append(p[i, 1])
    words = words * number
    count = count * number
    dn["Words"] = words
    dn["Count"] = count
    dn["x"] = x[:, 0]
    dn["y"] = x[:, 1]
    dn["Time"] = times
    for i in range(1, len(set(times)) + 1):
        to_append = [df["Words"][0], df["Count"][0], df["x"][0], df["y"][0], i]
        a_series = pd.Series(to_append, index=dn.columns)
        dn = dn.append(a_series, ignore_index=True)

    return dn


def join_n_clear(data, name):
    tex = " ".join(data[name]).lower()
    tex = re.sub(r"(?is)[^a-zA-Z0-9 ]", "", tex)
    tex = re.sub(r"^[^ ]*", "", tex)
    return tex


def make_word_dictionary(tex, n=100):
    stop_words = set(st.words("english"))

    nltk.download("webtext")
    wt_words = webtext.words()
    data_analysis = nltk.FreqDist(wt_words)

    for i in stop_words:
        del data_analysis[i]
    data_analysis = {
        k: v
        for k, v in sorted(
            data_analysis.items(), key=lambda item: item[1], reverse=True
        )
    }
    key = list(data_analysis.keys())[:n]
    item = list(data_analysis.values())[:n]
    clear_output(wait=True)
    return data_analysis, key, item


def solar_plot(
    data,
    name,
    radius=10,
    rotations=20,
    templates="plotly_dark",
    top_higher=False,
    n_ratio=1.5,
    add_image=False,
    image_path=None,
):
    """In this plot you need to write the data frame name and the column name you need to work with and it will make a
    solar plot for you
    radius : Radius of the circle in which the words are moving
    rotations : Number of rotations the words are taking
    template : Plotly template you wanna use
    top_higher : If the top word has too high count we will normalize the count for all the words"""
    tex = join_n_clear(data, name)
    data_analysis, key, item = make_word_dictionary(tex)
    df = pd.DataFrame(columns=["Words", "Count", "x", "y"])
    indi = item[0]
    if top_higher:
        indi = indi / n_ratio
    to_append = [key[0], indi, 0, 0]
    a_series = pd.Series(to_append, index=df.columns)
    df = df.append(a_series, ignore_index=True)
    count = 1
    for j in range(4):
        x = radius * j / 4
        y = radius ** 2 - x ** 2
        to_append = [key[count], item[count], x, y]
        a_series = pd.Series(to_append, index=df.columns)
        df = df.append(a_series, ignore_index=True)
        count += 1
    for j in range(4):
        x = radius - radius * j / 4
        y = -1 * (radius ** 2 - x ** 2)
        to_append = [key[count], item[count], x, y]
        a_series = pd.Series(to_append, index=df.columns)
        df = df.append(a_series, ignore_index=True)
        count += 1
    for j in range(4):
        x = -1 * (radius * j / 4)
        y = -1 * (radius ** 2 - x ** 2)
        to_append = [key[count], item[count], x, y]
        a_series = pd.Series(to_append, index=df.columns)
        df = df.append(a_series, ignore_index=True)
        count += 1
    for j in range(4):
        x = -radius + (radius * j / 4)
        y = radius ** 2 - x ** 2
        to_append = [key[count], item[count], x, y]
        a_series = pd.Series(to_append, index=df.columns)
        df = df.append(a_series, ignore_index=True)
        count += 1
    df = rotate_n(df, rotations)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="Words",
        size="Count",
        size_max=40,
        color="Count",
        color_continuous_scale="solar",
        labels={"x": "", "y": ""},
        template=templates,
        animation_frame="Time",
    )
    fig.update_xaxes(
        range=[-radius - 6, radius + 6],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    fig.update_yaxes(
        range=[-(radius ** 2) - 18, radius ** 2 + 18],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    if add_image:
        try:
            fig.add_layout_image(
                dict(
                    x=-1 * 2 * radius - 12,
                    sizex=4 * radius + 24,
                    y=2 * radius ** 2 + 36,
                    sizey=4 * radius ** 2 + 72,
                    xref="x",
                    yref="y",
                    opacity=0.1,
                    layer="below",
                    sizing="stretch",
                    source=image_path,
                )
            )
        except:
            print("Please add a valid link to the image !!")
    fig.show()


def random_color_func(
    word=None,
    font_size=None,
    position=None,
    orientation=None,
    font_path=None,
    random_state=None,
):
    h = int(360.0 * 21.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)


def make_wordcloud(data, name, title="text", mask=None, image_path=None):
    """data : name of the dataframe ;
    name : name of the column ;
    mask : mask image ;
    image_path : path to the image for which you wanna use the colors can be link or from your directory
    """
    csfont = {"fontname": "Comic Sans MS"}
    tex = join_n_clear(data, name)
    wordcloud = WordCloud(
        background_color="white",
        color_func=random_color_func,
        max_words=1000,
        mask=mask,
        height=1500,
        width=1500,
    ).generate(tex)
    if image_path:
        if image_path[:4] == "http":
            im = Image.open(requests.get(image_path, stream=True).raw)
            im = im.resize((1500, 1500))
            im = np.array(im)
        else:
            im = Image.open(image_path)
            im = im.resize((1500, 1500))
            im = np.array(im)

        image_colors = ImageColorGenerator(im)
        #         plt.figure(figsize=[18,8])
        #         plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        #         plt.axis("off")
        mycloud = wordcloud.recolor(color_func=image_colors)
    else:
        #         plt.figure(figsize=[18,8])
        #         plt.imshow(wordcloud, interpolation="bilinear")
        #         plt.axis("off")
        mycloud = wordcloud
    #     plt.title(title,fontsize=40,**csfont)
    #     plt.savefig("Figure.png", format="png")
    #     plt.show()
    return mycloud


# Function to make random color in hexadecimal


def make_random_color(number_of_colors=1, typ=1):
    if typ == 1:
        color = [
            "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
            for i in range(number_of_colors)
        ]
        if number_of_colors == 1:
            return color[0]
        else:
            return color
    else:
        color = [
            "#"
            + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
            + ",0.8"
            for i in range(number_of_colors)
        ]
        if number_of_colors == 1:
            return color[0]
        else:
            return color


def length_n_color(data, column, color_column, template="white"):
    """This data required atleast 3 parameters :
    data : Dataframe name
    column : Text column name
    color_column name : Name of the column to choose colors from preferred to be integer
    """
    if template == "dark":
        plt.style.use("dark_background")
    plt.figure(figsize=(18, 8))
    df = data.copy()
    df.sort_values(color_column, inplace=True)
    df["Length"] = df[column].apply(lambda x: len(x.split()))
    df["Unique"] = df[column].apply(lambda x: len(set(x.split())))
    unique = df[color_column].unique()
    for i in unique:
        dn = df[df[color_column] == i]
        plt.loglog(
            dn["Length"].values, dn["Unique"], "o", label=i, color=make_random_color()
        )
    plt.legend(loc=2, prop={"size": 10}, numpoints=14, ncol=2)
    plt.xlabel("Text Length")
    plt.ylabel("Number of Unique words")
    plt.show()
    plt.style.use("_classic_test_patch")
    return main()


def get_top_n(data, word, n=2):
    dn = data[data["word"] == word]["next word"].value_counts()[:n].index
    return list(dn)


def parallel_word_chart(
    data_2,
    name,
    word=None,
    number_rows=2,
    number_cols=4,
    preprocess_text=False,
    color_scale=px.colors.sequential.Sunset,
):
    if preprocess_text:
        data = rem_stopwords_tokenize(data_2, name)
        data = lemmatize_all(data, name)
        data = make_sentences(data, name)
        data = preprocess_tweet_data(data, name)
    else:
        data = data_2.copy()

    text = join_n_clear(data, name)
    df = pd.DataFrame(columns=["word", "next word"])
    text = text.split()
    df["word"] = text[:-1]
    df["next word"] = text[1:]
    text = join_n_clear(data, name)
    if word:
        pass
    else:
        dic, key, item = make_word_dictionary(text)
        word = key[0]
    l = [word]
    total_words = number_rows ** (number_cols - 1)
    for i in range(number_rows ** (number_cols - 1)):
        l.extend(get_top_n(df, l[i], n=number_rows))

    df_list = []
    gp = []
    for i in range(number_cols + 1):
        gp.append(number_rows ** i - 1)
    for i in range(number_cols):
        po = []
        for j in range(gp[i], gp[i + 1]):
            po.extend([l[j]] * int(total_words / number_rows ** i))
        df_list.append(po)
    df_list = np.array(df_list)
    df_list = df_list.T
    dataframe = pd.DataFrame(data=df_list)
    if len(dataframe.columns) > 2:
        npp = 2
    else:
        npp = 1
    dataframe["color"] = pd.factorize(dataframe[npp])[0]
    fig = px.parallel_categories(
        dataframe,
        dimensions=[i for i in range(number_cols)],
        color_continuous_scale=color_scale,
        color="color",
    )
    fig.show()


def sunburst_plot(
    data_2,
    name,
    word=None,
    number_rows=2,
    number_cols=4,
    preprocess_text=False,
    color_scale=px.colors.sequential.Sunset,
):
    """
    Data: name of the dataframe
    name : name of the text column
    word : if there is a specific word you wanna make the plot for
    number_rows: number of the rows restricted to 3
    number_cols: number of the columns restricted to 6
    """
    if preprocess_text:
        data = rem_stopwords_tokenize(data_2, name)
        data = lemmatize_all(data, name)
        data = make_sentences(data, name)
        data = preprocess_tweet_data(data, name)
    else:
        data = data_2.copy()

    text = join_n_clear(data, name)
    df = pd.DataFrame(columns=["word", "next word"])
    text = text.split()
    df["word"] = text[:-1]
    df["next word"] = text[1:]
    text = join_n_clear(data, name)
    if word:
        pass
    else:
        dic, key, item = make_word_dictionary(text)
        word = key[0]
    l = [word]
    total_words = number_rows ** (number_cols - 1)
    for i in range(number_rows ** (number_cols - 1)):
        l.extend(get_top_n(df, l[i], n=number_rows))

    df_list = []
    gp = []
    for i in range(number_cols + 1):
        gp.append(number_rows ** i - 1)
    for i in range(number_cols):
        po = []
        for j in range(gp[i], gp[i + 1]):
            po.extend([l[j]] * int(total_words / number_rows ** i))
        df_list.append(po)
    df_list = np.array(df_list)
    df_list = df_list.T
    dataframe = pd.DataFrame(data=df_list)
    if len(dataframe.columns) > 2:
        npp = 2
    else:
        npp = 1
    dataframe["color"] = pd.factorize(dataframe[npp])[0]
    fig = px.sunburst(
        dataframe,
        path=[i for i in range(number_cols)],
        color_continuous_scale=color_scale,
        color="color",
    )
    fig.update_layout(
        annotations=[
            dict(
                text="Most probs next word distribution",
                x=0.5,
                y=1.15,
                font_size=24,
                showarrow=True,
                font_family="Arial Black",
                font_color="black",
            )
        ]
    )

    fig.show()


def Link_Plot(
    data_2,
    name,
    word="link",
    x_factor=2,
    n_iterations=10,
    preprocess_text=False,
    lemmatize_word=True,
):
    """
    This is a synonym word plot . This will try to find the closest related word in the text data
    data_2 : name of the dataframe
    name : name of the text data file
    word : word you wanna start with
    x_factor : This is a metric which tells how close the word might be
    n_iterations : Number of iterations you want the function to run
    preprocess_text : Check true if you wanna preprocess the text
    lemmatize word : Let it be true if you have checked preprocess_text
    """
    try:
        if lemmatize_word:
            word = lemmatize_single_word(word)
        if preprocess_text:
            data = rem_stopwords_tokenize(data_2, name)
            data = lemmatize_all(data, name)
            data = make_sentences(data, name)
            data = preprocess_tweet_data(data, name)
        else:
            data = data_2.copy()
        anr = []
        t = x_factor
        existing_word = [word]
        text = join_n_clear(data, name)
        text = text.split()
        word1 = np.array(text[:-2])
        word2 = np.array(text[1:-1])
        word3 = np.array(text[2:])
        words = np.vstack((word1, word2, word3)).T
        df = pd.DataFrame(
            columns=["First word", "Middle word", "Last word"], data=words
        )
        counts = 0
        # while n_iterations!=0 and count!=len(existing_word):
        while n_iterations != 0 and counts < len(existing_word):
            n_iterations -= 1
            word = existing_word[counts]
            df_word = df[df["Middle word"] == word]
            df_word.head()
            count = df_word.groupby(df_word.columns.tolist(), as_index=False).size()
            count.sort_values("size", ascending=False, inplace=True)
            count = count[count["size"] >= t]
            first_word = count["First word"].values
            last_word = count["Last word"].values
            bnr = []
            for i in range(len(first_word)):
                dn = df[
                    (df["First word"] == first_word[i])
                    & (df["Last word"] == last_word[i])
                    & (~df["Middle word"].isin(existing_word))
                ]
                dn = dn.groupby(dn.columns.tolist(), as_index=False).size()
                dn = dn[dn["size"] >= t]
                mid = dn["Middle word"].values
                size = dn["size"].values
                starter = np.array([existing_word[counts]] * len(mid))
                arr = np.vstack((starter, mid, size)).T
                arr = list(arr)
                bnr.extend(arr)
                if len(mid) > 0:
                    existing_word.extend(list(mid))
            counts += 1
            anr.extend(bnr)
        anr = np.array(anr)
        st = np.hstack((anr[:, 0], anr[:, 1]))
        dic = {j: i for i, j in enumerate(np.array(list(set(list(st)))))}
        for i in dic:
            anr = np.where(anr == i, dic[i], anr)
        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=list(dic.keys()),
                        color=make_random_color(len(dic), typ=1),
                    ),
                    link=dict(
                        source=anr[
                            :, 0
                        ],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                        target=anr[:, 1],
                        value=anr[:, 2],
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Closest Relatable Word",
            font=dict(size=20, color="black"),
            plot_bgcolor="#252525",
            paper_bgcolor="#caf7e3 ",
        )
        fig.show()
    except:
        print("Please choose a different word !!!!")
