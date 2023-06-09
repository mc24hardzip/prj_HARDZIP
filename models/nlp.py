import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from pprint import pprint
from gensim import corpora, models
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud


def okt_tokenize(data):
    okt = Okt()
    data["content"] = data["title"] + data["description"]
    data["token"] = ""
    for i in range(len(data)):
        data["token"].iloc[i] = okt.pos(data["content"].iloc[i])
    data.to_csv("okt_tokenized.csv", encoding="euc-kr", index_label="id")
    return data

stop_words = ['등', '및', '비', '실', '무엇', '요즘', '직접', '개', '변','분','수','락','내','개월','번','입', '곳','확인', '리','이장','현재','조절','모두','총','전용','후','대수','매우', '등등','사항','정말','몸','호실','이','시', '완료','각종', '평수','다수', '실제', '편','톤','중','보시', '완전','구', '제','더','자도','위','언제',
'매물', '등록', '위치', '직방', '방', '사진', '권', '전입', '층', '신고', '룸', '평', '시스템','부동산', '불가', '타입', '미사', '이용', '공간', '인접', '구조', '주변', 
'동안', '주하', '카', '만오', '이세', '조정은', '칸', '책정', '철', '종류', '기정', '분포', '화공', '제어', '이크', '상기', '료등', '욥', '리모', '두운', '아이디', '한눈', '순간', '원활', '규약', '정석', '진임', '평의', '질', '성의껏', '유사', '선물', '유명', '타시기', '더탑', '인증', '임박', '자율', '분석', '브', '멀티', '콤비', '대내', '방기', '로움', '계산서', '지출', '톡', '등록증', '푸른', '앤', '읍니', '가능', '포함', '업로드', '매일',
'박자', '빅사', '부공', '장인근', '두루', '고로', '패드', '얼마나', '평이', '액자', '주지', '임의', '실매', '옵션'
]


def strip_csv(data, stop_words):
    documents = []
    FEATURE_POS = ["Noun"]
    for row in data.token:
        morphs = []
        if type(row)==str:
            row = row.strip("[").strip("]").strip("(").strip(")").split("), (")
        for i in row:
            if type(i)==str:
                i = i.split(", ")
                word = i[0].strip("'")
                pos = i[-1].strip("'")
            else:
                word = i[0]
                pos = i[-1]               
            if pos not in FEATURE_POS:
                continue
            if word not in stop_words:
                morphs.append(word)
        documents.append(morphs)
    return documents


def topic_model(df, documents, num):
    dictionary = corpora.Dictionary(documents)
    corpus = []
    for document in documents:
        bow = dictionary.doc2bow(document)
        corpus.append(bow)

    lda_model = models.ldamodel.LdaModel(corpus, num_topics=num, id2word=dictionary)

    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

    pyLDAvis.save_html(vis_data, "LDA_topic{}.html".format(num))
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = lda_model.show_topic(topic_num, topn=10)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )
            else:
                break
    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    sent_topics_df.index = df.index
    text_df = pd.concat([df, sent_topics_df], axis=1)

    text_df.to_csv("LDA_topic_text.csv", encoding="euc-kr", index="id")

    return text_df


def wordCloud(text_df, stop_words, col, val):
    # for val in iter_value_list:
    print(col, val, "WordCloud")

    FEATURE_POS = ["Noun"]
    documents = []

    for row in text_df[text_df[col] == val].token:
        morphs = ""
        if type(row)==str:
            row = row.strip("[").strip("]").strip("(").strip(")").split("), (")
        for i in row:
            if type(i)==str:
                i = i.split(", ")
                word = i[0].strip("'")
                pos = i[-1].strip("'")
            else:
                word = i[0]
                pos = i[-1]   
            if pos not in FEATURE_POS:
                continue
            if word not in stop_words:
                morphs = morphs + word + " "
        documents.append(morphs)

    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(documents)
    dense = vecs.todense()
    lst1 = dense.tolist()
    df_tfidf = pd.DataFrame(lst1, columns=vectorizer.get_feature_names_out())

    Cloud = WordCloud(
        font_path="C:/Users/junel/Downloads/data/eda/malgunbd.ttf",
        relative_scaling=0.2,
        background_color="white",
        max_words=50,
    ).generate_from_frequencies(df_tfidf.T.sum(axis=1))
    plt.figure(figsize=(16, 8))
    plt.imshow(Cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    plt.savefig(f"{col}{val}_wordcloud.png")
    return

