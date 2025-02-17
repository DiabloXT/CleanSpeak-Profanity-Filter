import gc
import os
import time
import base64
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta

from gtts import gTTS
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import google.generativeai as genai

import streamlit as st
import streamlit_antd_components as sac

from backend.app_utils import BackendUtils, ImageUtils, AudioUtils
from backend.authentication import AuthenticationDB
from backend.profanity_filter import TextProfanityFilter, ImageProfanityFilter
from backend.profanity_filter import AudioProfanityFilter, VideoProfanityFilter


st.set_page_config(
    page_title="Profanity Filtering App",
    page_icon="assets/favicon/profanity_filter_app.png",
    initial_sidebar_state="expanded",
    layout="wide",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1.75rem;
                    padding-bottom: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

if "generate_results_cache" not in st.session_state:
    st.session_state.generate_results_cache = False

if "logged_in_username" not in st.session_state:
    st.session_state.logged_in_username = None

if "themes" not in st.session_state:
    st.session_state.themes = {
        "current_theme": "dark",
        "refreshed": True,
        "light": {
            "theme.base": "dark",
            "theme.backgroundColor": "#111111",
            "theme.primaryColor": "#64ABD8",
            "theme.secondaryBackgroundColor": "#181818",
            "theme.textColor": "#FFFFFF",
            "button_face": "🌜",
        },
        "dark": {
            "theme.base": "light",
            "theme.backgroundColor": "#fdfefe",
            "theme.primaryColor": "#64ABD8",
            "theme.secondaryBackgroundColor": "#f0f2f5",
            "theme.textColor": "#333333",
            "button_face": "🌞",
        },
    }


def change_streamlit_theme():
    previous_theme = st.session_state.themes["current_theme"]
    tdict = (
        st.session_state.themes["light"]
        if st.session_state.themes["current_theme"] == "light"
        else st.session_state.themes["dark"]
    )

    for vkey, vval in tdict.items():
        if vkey.startswith("theme"):
            st._config.set_option(vkey, vval)

    st.session_state.themes["refreshed"] = False

    if previous_theme == "dark":
        st.session_state.themes["current_theme"] = "light"

    elif previous_theme == "light":
        st.session_state.themes["current_theme"] = "dark"


if st.session_state.themes["refreshed"] == False:
    st.session_state.themes["refreshed"] = True
    st.rerun()


def set_generate_results_cache_to_true():
    st.session_state.generate_results_cache = True


def set_generate_results_cache_to_false():
    st.session_state.generate_results_cache = False


@st.cache_data(show_spinner=False)
def read_master_data(path):
    master_df = pd.read_csv(path)
    master_df.dropna(inplace=True)
    return master_df


@st.cache_data(show_spinner=False)
def search_tweets(search_string, num_results):
    df = read_master_data("data/processed_data_for_twitter_reporting.csv")

    filtered_df = df[
        df["cleaned_tweet"].str.contains(search_string, case=False, na=False)
    ]
    result_df = filtered_df.head(num_results).reset_index(drop=True)

    del df
    del filtered_df
    gc.collect()

    return result_df


@st.experimental_dialog("LogIn to CleanSpeak")
def user_login():
    username = st.text_input(
        "Username", placeholder="Username", label_visibility="collapsed"
    )
    password = st.text_input(
        "Password",
        placeholder="Password",
        label_visibility="collapsed",
        type="password",
    )

    if st.button("LogIn to CleanSpeak"):
        authdb = AuthenticationDB()

        if authdb.verify_user(username, password) == True:
            st.success(f"Welcome @{username}", icon="✔️")
            st.session_state.logged_in_username = True

            time.sleep(3)
            st.rerun()

        else:
            st.error(
                "Incorrect Credentials. Please check your Username.Password", icon="🚨"
            )
            time.sleep(3)
            st.rerun()


@st.experimental_dialog("Create a New Account")
def create_account():
    cola, colb = st.columns(2)
    with cola:
        fullname = st.text_input(
            "Full Name", placeholder="Full Name", label_visibility="collapsed"
        )
    with colb:
        username = st.text_input(
            "Username", placeholder="Username", label_visibility="collapsed"
        )

    email = st.text_input(
        "Email Id", placeholder="Email Id", label_visibility="collapsed"
    )

    cola, colb = st.columns(2)

    with cola:
        phone = st.text_input(
            "Phone No.", placeholder="Phone No.", label_visibility="collapsed"
        )

    with colb:
        password = st.text_input(
            "Password",
            placeholder="Password",
            label_visibility="collapsed",
            type="password",
        )

    tc_checkbox = st.checkbox(
        "I accept the terms and conditions of CleanSpeak streamlit app"
    )

    if st.button("Create an Account", disabled=not (tc_checkbox)):
        if (
            (len(fullname) > 0)
            and (len(username) > 0)
            and (len(email) > 0)
            and (len(phone) > 0)
            and (len(password) > 0)
        ):
            authdb = AuthenticationDB()
            authdb.add_user(username, fullname, email, phone, password)

            st.success("Account created successfully", icon="✔️")
            time.sleep(3)
            st.rerun()

        else:
            st.error("Some of your details are missing", icon="🚨")


@st.experimental_dialog("Tweet Details")
def view_tweet_details(user, date, time, tweet):
    st.markdown(f"<H3>Tweet by @{user}</H3>", unsafe_allow_html=True)

    st.markdown(
        f"<P>{tweet}<BR><BR><B>Posted on:</B> {date} at {time}</P>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    if st.session_state.logged_in_username == True:
        with st.sidebar:
            selected_menu_item = sac.menu(
                [
                    sac.MenuItem(
                        "Profanity Filters",
                        icon="grid",
                    ),
                    sac.MenuItem(
                        "Twitter Dashboard",
                        icon="clipboard-pulse",
                    ),
                    sac.MenuItem(" ", disabled=True),
                    sac.MenuItem(type="divider"),
                ],
                open_all=True,
            )

        if selected_menu_item == "Profanity Filters":
            with st.sidebar:
                profile_pic_container = st.container(height=365, border=False)
                if st.button("Log Out from CleanSpeak", use_container_width=True):
                    st.session_state.logged_in_username = False
                    st.rerun()

            st.markdown(
                "<BR><H2>💡 CleanSpeak: Profanity Filtering</H2>", unsafe_allow_html=True
            )
            st.markdown(
                "<P align='justify'>Welcome to our Profanity Filtering App! This tool detects and filters profanity in text, image, audio, and video inputs. It suggests nonprofane alternatives for text and audio and provides bounding boxes and timestamps for profane content in images and videos</P>",
                unsafe_allow_html=True,
            )

            tab_filter_type = sac.tabs(
                [
                    sac.TabsItem(label="Text Analysis"),
                    sac.TabsItem(label="Image Analysis"),
                    sac.TabsItem(label="Audio Analysis"),
                    sac.TabsItem(label="Video Analysis"),
                ],
                align="left",
                variant="outline",
            )

            if tab_filter_type == "Text Analysis":
                input_text = st.text_area(
                    "Enter your text here",
                    label_visibility="collapsed",
                    placeholder="Enter your text here",
                )

                temp_section_1, temp_section_2, _ = st.columns([1, 0.22, 2.4])

                with temp_section_1:
                    button_analyze_text = st.button(
                        "🛠️ Analyze this Text", use_container_width=True
                    )

                with temp_section_2:
                    st.button("🗑️", use_container_width=True)

                if button_analyze_text:
                    with st.spinner(
                        "Processing your input text for profane content. Please wait a moment..."
                    ):
                        text_profanity_filter = TextProfanityFilter()

                        profanity_scores = text_profanity_filter.detect_toxicity(
                            input_text
                        )

                    if type(profanity_scores[0]) is bool:
                        message = """
                        **No Profanity Detected**

                        Your text is clean and free of any profane content.
                        """
                        st.success(message, icon="✔️")

                    else:
                        if len(profanity_scores[1]) > 1:
                            message = f"""
                            **Profanity Detected**

                            Your input text has been flagged for containing {", ".join(profanity_scores[1][:-1])  + ", and " + profanity_scores[1][-1]} content.
                            """
                            st.error(message, icon="🚨")

                        else:
                            message = f"""
                            **Profanity Detected**

                            Your input text has been flagged for containing {profanity_scores[1][0]} content.
                            """

                        backend_utils = BackendUtils()

                        suggested_text = f"""
                        **Suggested Non-Profane Version:**

                        {str(backend_utils.extract_list_from_text(profanity_scores[0])[0])}
                        """
                        profile_pic_container.info(suggested_text, icon="ℹ️")

            elif tab_filter_type == "Image Analysis":
                input_image = st.file_uploader(
                    "Choose a file", label_visibility="collapsed"
                )

                temp_section_1, temp_section_2, _ = st.columns([1, 0.22, 2.4])

                with temp_section_1:
                    button_analyze_image = st.button(
                        "🛠️ Analyze this Image", use_container_width=True
                    )

                with temp_section_2:
                    st.button("🗑️", use_container_width=True)

                if button_analyze_image:
                    with st.spinner(
                        "Processing your input image for profane content. Please wait a moment..."
                    ):
                        with open(
                            f"assets/temp/{input_image.name}", "wb"
                        ) as image_file:
                            image_file.write(input_image.getbuffer())

                        image_profanity_filter = ImageProfanityFilter()
                        profanity_coordinates = image_profanity_filter.detect_toxicity(
                            f"assets/temp/{input_image.name}"
                        )

                    if type(profanity_coordinates) is bool:
                        message = """
                        **No Profanity Detected**

                        Your image is clean and free of any profane content.
                        """
                        st.success(message, icon="✔️")

                    else:
                        backend_utils = BackendUtils()
                        profanity_coordinates = backend_utils.extract_list_from_text(
                            profanity_coordinates
                        )

                        display_coordinates = [
                            profanity_coordinates[1],
                            profanity_coordinates[0],
                            profanity_coordinates[3],
                            profanity_coordinates[2],
                        ]

                        message = f"""
                        **Profanity Detected**

                        Your input image has been flagged for containing profane content at coordinates {profanity_coordinates}.
                        """
                        st.error(message, icon="🚨")

                        with profile_pic_container:
                            image_utils = ImageUtils()

                            buffer_image = image_utils.display_bounding_box(
                                f"assets/temp/{input_image.name}", profanity_coordinates
                            )
                            st.image(
                                buffer_image,
                                caption="Profane Content Detected",
                                use_column_width=True,
                            )

                        os.remove(f"assets/temp/{input_image.name}")

            elif tab_filter_type == "Audio Analysis":
                input_audio = st.file_uploader(
                    "Choose an audio file", label_visibility="collapsed"
                )

                temp_section_1, temp_section_2, _ = st.columns([1, 0.22, 2.4])

                with temp_section_1:
                    button_analyze_audio = st.button(
                        "🛠️ Analyze this Audio", use_container_width=True
                    )

                with temp_section_2:
                    st.button("🗑️", use_container_width=True)

                if button_analyze_audio:
                    with st.spinner(
                        "Processing your input audio for profane content. Please wait a moment..."
                    ):
                        with open(
                            f"assets/temp/{input_audio.name}", "wb"
                        ) as audio_file:
                            audio_file.write(input_audio.getbuffer())

                        audio_profanity_filter = AudioProfanityFilter()
                        is_profane = audio_profanity_filter.detect_toxicity(
                            f"assets/temp/{input_audio.name}"
                        )

                    if type(is_profane) is bool:
                        message = """
                        **No Profanity Detected**

                        Your audio is clean and free of any profane content.
                        """
                        st.success(message, icon="✔️")

                    else:
                        audio_utils = AudioUtils()
                        path_to_clean_audio_file = audio_utils.text_to_speech(
                            is_profane
                        )

                        message = f"""
                        **Profanity Detected**

                        Your input audio has been flagged for containing profane content.
                        """
                        st.error(message, icon="🚨")

                        with profile_pic_container:
                            st.write("Suggested Non-Profane Version")
                            st.audio(path_to_clean_audio_file)

                            os.remove(f"assets/temp/{input_audio.name}")
                            os.remove(path_to_clean_audio_file)

            elif tab_filter_type == "Video Analysis":
                input_video = st.file_uploader(
                    "Choose a video file", label_visibility="collapsed"
                )

                temp_section_1, temp_section_2, _ = st.columns([1, 0.22, 2.4])

                with temp_section_1:
                    button_analyze_video = st.button(
                        "🛠️ Analyze this Video", use_container_width=True
                    )

                with temp_section_2:
                    st.button("🗑️", use_container_width=True)

                if button_analyze_video:
                    with st.spinner(
                        "Processing your input video for profane content. Please wait a moment..."
                    ):
                        with open(
                            f"assets/temp/{input_video.name}", "wb"
                        ) as video_file:
                            video_file.write(input_video.getbuffer())

                        video_profanity_filter = VideoProfanityFilter()
                        is_profane = video_profanity_filter.detect_toxicity(
                            f"assets/temp/{input_video.name}"
                        )

                    if type(is_profane) is bool:
                        message = """
                        **No Profanity Detected**

                        Your video is clean and free of any profane content.
                        """
                        st.success(message, icon="✔️")

                    else:
                        backend_utils = BackendUtils()
                        profanity_timestamp = backend_utils.extract_list_from_text(
                            is_profane
                        )

                        timestamp = profanity_timestamp[1]

                        profanity_timestamp_end = (
                            datetime.strptime(timestamp, "%M:%S") + timedelta(seconds=2)
                        ).strftime("%M:%S")

                        message = f"""
                        **Profanity Detected**

                        Your input video has been flagged for containing profane content between {profanity_timestamp[0]} to {profanity_timestamp_end}
                        """
                        st.error(message, icon="🚨")

                        with profile_pic_container:
                            try:
                                st.write("Detected Profane Segment")
                                start_minutes, start_seconds = profanity_timestamp[
                                    0
                                ].split(":")
                                (
                                    end_minutes,
                                    end_seconds,
                                ) = profanity_timestamp_end.split(":")

                                st.video(
                                    f"assets/temp/{input_video.name}",
                                    start_time=f"{int(start_minutes)}m{int(start_seconds):02d}s",
                                    end_time=f"{int(end_minutes)}m{int(end_seconds):02d}s",
                                )

                            except Exception as err:
                                pass

                            os.remove(f"assets/temp/{input_video.name}")

        else:
            with st.sidebar:
                with st.container(border=True):
                    input_search_string = st.text_input(
                        "Enter the search term",
                        placeholder="Enter the search term",
                        on_change=set_generate_results_cache_to_false,
                    )
                    input_results_count = st.slider(
                        "Number of Tweets to Analyze",
                        100,
                        5000,
                        2000,
                        100,
                        on_change=set_generate_results_cache_to_false,
                    )

                    button_analyze_tweets = st.button(
                        "Analyze Results", on_click=set_generate_results_cache_to_true
                    )

                st.write(" ")
                st.markdown("<BR><BR><BR>", unsafe_allow_html=True)

                if st.button("Log Out from CleanSpeak", use_container_width=True):
                    st.session_state.logged_in_username = False
                    st.rerun()

            if (
                button_analyze_tweets or st.session_state.generate_results_cache
            ) and len(input_search_string) > 0:
                data = search_tweets(input_search_string, input_results_count)

                st.write(" ")
                colm, coln = st.columns([5.6, 0.4])

                with colm:
                    st.markdown("<H3>Twitter Dashboard</H3>", unsafe_allow_html=True)

                with coln:
                    btn_face = (
                        st.session_state.themes["light"]["button_face"]
                        if st.session_state.themes["current_theme"] == "light"
                        else st.session_state.themes["dark"]["button_face"]
                    )

                    st.button(
                        btn_face,
                        use_container_width=True,
                        type="secondary",
                        on_click=change_streamlit_theme,
                    )

                st.write(" ")

                cola, colb, colc = st.columns(3)

                with cola:
                    with st.container(border=True):
                        colx, coly = st.columns([1, 3])

                        with colx:
                            st.image(
                                Image.open(
                                    "assets/app_graphics/trident_yellow.png"
                                ).resize((50, 50))
                            )

                        with coly:
                            st.markdown(
                                f"Total Tweets<BR><B>{len(data)} Tweets Analyzed</B>",
                                unsafe_allow_html=True,
                            )

                with colb:
                    with st.container(border=True):
                        colx, coly = st.columns([1, 3])

                        with colx:
                            st.image(
                                Image.open(
                                    "assets/app_graphics/trident_green.png"
                                ).resize((50, 50))
                            )

                        with coly:
                            st.markdown(
                                f"Positive Tweets<BR><B>{data[data['sentiment'] == 'positive'].shape[0]} Non Profane Tweets</B>",
                                unsafe_allow_html=True,
                            )

                with colc:
                    with st.container(border=True):
                        colx, coly = st.columns([1, 3])

                        with colx:
                            st.image(
                                Image.open(
                                    "assets/app_graphics/trident_red.png"
                                ).resize((50, 50))
                            )

                        with coly:
                            st.markdown(
                                f"Negative Tweets<BR><B>{data[data['sentiment'] == 'negative'].shape[0]} Profanous Tweets</B>",
                                unsafe_allow_html=True,
                            )

                st.write(" ")
                cola, colb, colc = st.columns(3)

                with cola:
                    with st.container(border=True, height=400):
                        st.map(
                            data,
                            latitude="lat",
                            longitude="long",
                            use_container_width=True,
                        )

                with colb:
                    with st.container(border=True, height=400):
                        negative_df = data[data["sentiment"] == "negative"]
                        word_counts = Counter(
                            " ".join(negative_df["cleaned_tweet"]).split()
                        )

                        common_words = word_counts.most_common(10)
                        common_words_df = pd.DataFrame(
                            common_words, columns=["Word", "Count"]
                        )

                        st.line_chart(common_words_df.set_index("Word"), height=390)

                with colc:
                    with st.container(border=True, height=400):
                        sentiments_counts = data["sentiment"].value_counts()

                        sentiments_data = {
                            "labels": sentiments_counts.index,
                            "values": sentiments_counts.values,
                        }

                        config = {
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": [
                                "zoom2d",
                                "pan2d",
                                "select2d",
                                "lasso2d",
                                "zoomIn2d",
                                "zoomOut2d",
                                "autoScale2d",
                                "resetScale2d",
                                "hoverClosestCartesian",
                                "hoverCompareCartesian",
                                "toggleSpikelines",
                                "toImage",
                                "sendDataToCloud",
                                "zoom3d",
                                "pan3d",
                                "resetCameraDefault3d",
                                "resetCameraLastSave3d",
                                "hoverClosest3d",
                                "orbitRotation",
                                "tableRotation",
                                "toggleHover",
                                "resetGeo",
                                "hoverClosestGeo",
                                "resetViews",
                                "toggleHover",
                                "resetViewMapbox",
                                "zoomInGeo",
                                "zoomOutGeo",
                                "hoverClosestGl2d",
                                "hoverClosestPie",
                                "toggleHover",
                                "resetViews",
                            ],
                        }

                        fig = px.pie(sentiments_data, values="values")
                        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})
                        st.plotly_chart(fig, config=config)

                st.write(" ")
                section_1, section_2 = st.columns([2, 1])

                with section_1:
                    with st.container(border=True, height=430):
                        tab_view_tweets_by_sentiment = sac.tabs(
                            [
                                sac.TabsItem(label="Positive Tweets"),
                                sac.TabsItem(label="Neutral Tweets"),
                                sac.TabsItem(label="Negative Tweets"),
                            ],
                            align="left",
                        )

                        if tab_view_tweets_by_sentiment == "Positive Tweets":
                            positive_df = data[data["sentiment"] == "positive"]

                            sampled_df = positive_df.head()
                            result = sampled_df[
                                ["User", "Date", "Time", "Tweet"]
                            ].to_dict(orient="records")

                            for df in result:
                                cola, colb, colc = st.columns([0.75, 3.6, 1.6])

                                with cola:
                                    st.image(
                                        Image.open(
                                            "assets/app_graphics/trident_green.png"
                                        ).resize((50, 50))
                                    )

                                with colb:
                                    st.markdown(
                                        f"<B>Tweet by @{df['User'].lower()}</B><BR>Published on {df['Date']}</B> at {df['Time'][:-3]} GMT",
                                        unsafe_allow_html=True,
                                    )

                                with colc:
                                    if st.button(
                                        f"View Details",
                                        use_container_width=True,
                                        key=f"_button_tweet_{result.index(df)}_details",
                                    ):
                                        view_tweet_details(
                                            df["User"],
                                            df["Date"],
                                            df["Time"],
                                            df["Tweet"],
                                        )

                        elif tab_view_tweets_by_sentiment == "Neutral Tweets":
                            neutral_df = data[data["sentiment"] == "neutral"]

                            sampled_df = neutral_df.head()
                            result = sampled_df[
                                ["User", "Date", "Time", "Tweet"]
                            ].to_dict(orient="records")

                            for df in result:
                                cola, colb, colc = st.columns([0.75, 3.6, 1.6])

                                with cola:
                                    st.image(
                                        Image.open(
                                            "assets/app_graphics/trident_yellow.png"
                                        ).resize((50, 50))
                                    )

                                with colb:
                                    st.markdown(
                                        f"<B>Tweet by @{df['User'].lower()}</B><BR>Published on {df['Date']}</B> at {df['Time'][:-3]} GMT",
                                        unsafe_allow_html=True,
                                    )

                                with colc:
                                    if st.button(
                                        f"View Details",
                                        use_container_width=True,
                                        key=f"_button_tweet_{result.index(df)}_details",
                                    ):
                                        view_tweet_details(
                                            df["User"],
                                            df["Date"],
                                            df["Time"],
                                            df["Tweet"],
                                        )

                        elif tab_view_tweets_by_sentiment == "Negative Tweets":
                            negative_df = data[data["sentiment"] == "negative"]

                            sampled_df = negative_df.head()
                            result = sampled_df[
                                ["User", "Date", "Time", "Tweet"]
                            ].to_dict(orient="records")

                            for df in result:
                                cola, colb, colc = st.columns([0.75, 3.6, 1.6])

                                with cola:
                                    st.image(
                                        Image.open(
                                            "assets/app_graphics/trident_red.png"
                                        ).resize((50, 50))
                                    )

                                with colb:
                                    st.markdown(
                                        f"<B>Tweet by @{df['User'].lower()}</B><BR>Published on {df['Date']}</B> at {df['Time'][:-3]} GMT",
                                        unsafe_allow_html=True,
                                    )

                                with colc:
                                    if st.button(
                                        f"View Details",
                                        use_container_width=True,
                                        key=f"_button_tweet_{result.index(df)}_details",
                                    ):
                                        view_tweet_details(
                                            df["User"],
                                            df["Date"],
                                            df["Time"],
                                            df["Tweet"],
                                        )

                with section_2:
                    with st.container(border=True, height=252):
                        sentiment_counts = data["sentiment"].value_counts()

                        sentiment_df = pd.DataFrame(
                            {
                                "Sentiment": sentiment_counts.index,
                                "Count": sentiment_counts.values,
                            }
                        )

                        st.bar_chart(sentiment_df.set_index("Sentiment"), height=235)

                    with st.container(border=True):
                        text = " ".join(word for word in data["cleaned_tweet"])

                        wordcloud = WordCloud(random_state=50).generate(text)
                        plt.imshow(wordcloud, interpolation="bilinear")

                        plt.axis("off")
                        st.pyplot(plt)

                st.write(" ")

            else:
                st.write(" ")
                st.markdown("<BR>", unsafe_allow_html=True)

                sac.result(
                    label="CleanSpeak Tweet Analysis",
                    description="Enter a keyword to analyze related tweets",
                    status=404,
                )

    else:
        st.markdown("<BR>", unsafe_allow_html=True)
        cola, colb = st.columns(2)

        with cola:
            st.markdown("<BR><BR><BR><BR>", unsafe_allow_html=True)
            st.image(Image.open("assets/dotwave_intro_img.gif"))

            colx, _ = st.columns([5, 1])

            with colx:
                st.markdown(
                    "<H2>Get Started Here:</H2><P align='justify'>This app detects and filters profanity in text, image, audio, and video, providing better alternatives, while highlighting profane content location</P>",
                    unsafe_allow_html=True,
                )
                st.markdown("", unsafe_allow_html=True)

            colx, coly, _ = st.columns(3)

            with colx:
                if st.button("Login to Cleanspeak", use_container_width=True):
                    user_login()

            with coly:
                if st.button("Create New Account", use_container_width=True):
                    create_account()

            colx, _ = st.columns([5, 1])

        with colb:
            side_bg = "assets/app_graphics/md_wallpaper.png"
            st.image(side_bg, use_column_width=True)
