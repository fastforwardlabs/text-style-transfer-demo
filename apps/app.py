from calendar import c
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from apps.data_utils import DATA_PACKET
from src.style_classification import StyleIntensityClassifier

# SESSION STATE UTILS
if "page_progress" not in st.session_state:
    st.session_state.page_progress = 1


def increment_page_progress():
    st.session_state.page_progress += 1


def reset_page_progress_state():
    st.session_state.page_progress = 1


class DisableableButton:
    """
    Utility class for creating "disable-able" buttons upon click.

    We initialize an empty container, then update that container with buttons
    upon calling `create_enabled_button` and `disable` methods where clicking
    is enabled and then disabled, respectively.

    """

    def __init__(self, button_number, button_text):
        self.button_number = button_number
        self.button_text = button_text

    def _init_placeholder_container(self):
        self.ph = st.empty()

    def create_enabled_button(self):
        self._init_placeholder_container()
        self.ph.button(
            self.button_text,
            on_click=increment_page_progress,
            key=f"ph{self.button_number}_before",
            disabled=False,
        )

    def disable(self):
        self.ph.button(
            self.button_text, key=f"ph{self.button_number}_after", disabled=True
        )


# MODEL CACHING
@st.cache(
    hash_funcs={tokenizers.Tokenizer: lambda _: None},
    allow_output_mutation=True,
    show_spinner=False,
)
def get_cached_style_intensity_classifier():
    return StyleIntensityClassifier(DATA_PACKET[style_attribute].cls_model_path)


@st.cache(
    hash_funcs={tokenizers.Tokenizer: lambda _: None},
    allow_output_mutation=True,
    show_spinner=False,
)
def get_cached_word_attributions(text_sample):
    it = InterpretTransformer(
        cls_model_identifier=DATA_PACKET[style_attribute].cls_model_path
    )
    return it.visualize_feature_attribution_scores(text_sample).data


# PAGE CONFIG
ffl_favicon = Image.open("static/images/cldr-favicon.ico")
st.set_page_config(
    page_title="CFFL: Text Style Transfer",
    page_icon=ffl_favicon,
    layout="centered",
    initial_sidebar_state="expanded",
)

# SIDEBAR
ffl_logo = Image.open("static/images/ffllogo2@1x.png")
st.sidebar.image(ffl_logo)

st.sidebar.write(
    "This prototype accompanies our [Text Style Transfer]() blog series in which we \
    explore the task of automatically neutralizing subjectivity bias in free text."
)

st.sidebar.markdown("## Select a style attribute")
style_attribute = st.sidebar.selectbox(
    "What style would you like to transfer between?",
    options=("subjective-to-neutral", "informal-to-formal"),
    help="ADD HELP MESSAGE HERE!",
)

st.sidebar.markdown("## Text generation parameters")
st.sidebar.write("**max_gen_length**")
max_gen_length = st.sidebar.slider(
    "Whats the maximum generation length desired?", 1, 250, 200, 10
)
st.sidebar.write("**num_beams**")
num_beams = st.sidebar.slider("Whats the maximum generation length desired?", 1, 8, 4)
st.sidebar.write("**temperature**")
num_beams = st.sidebar.slider(
    "What sensitivity value to model next token probabilities?", 0.0, 1.0, 1.0
)

st.sidebar.button("Start over!", on_click=reset_page_progress_state)

# MAIN CONTENT
st.markdown(
    "# :twisted_rightwards_arrows: Text Style Transfer :twisted_rightwards_arrows:"
)
st.write(
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Feugiat vivamus at augue eget. Mattis enim ut tellus elementum sagittis vitae et leo duis. Rhoncus urna neque viverra justo nec ultrices. Lacinia at quis risus sed vulputate odio. Vitae purus faucibus ornare suspendisse. Quam lacus suspendisse faucibus interdum posuere lorem ipsum. Duis ut diam quam nulla porttitor massa id neque. Eget duis at tellus at urna condimentum. Ut faucibus pulvinar elementum integer enim neque volutpat."
)


## 1. INPUT EXAMPLE
if st.session_state.page_progress > 0:
    st.write("### 1. Input some text")

    col1_1, col1_2 = st.columns([1, 3])

    with col1_1:
        input_type = st.radio(
            "Type your own or choose from a preset example",
            ("Choose preset", "Enter text"),
            horizontal=False,
        )
    with col1_2:
        if input_type == "Enter text":
            text_sample = st.text_input(
                f"Enter some text to modify style from {style_attribute}",
                help="You can also select one of our preset examples by toggling the radio button to the left.",
            )
        else:
            text_sample = option = st.selectbox(
                f"Select a preset example to modify style from {style_attribute}",
                SUBJECTIVE_TO_NEUTRAL["examples"],
                help="You can also enter your own text by toggling the radio button to the left.",
            )

        db1 = DisableableButton(1, "Let's go!")
        db1.create_enabled_button()

## 2. CLASSIFY INPUT
if st.session_state.page_progress > 1:
    db1.disable()

    st.write("### 2. Style detection")
    st.write(
        f"""Before we can transfer style, we need to ensure the input text isn't already of the target style! To do so, \
            we classify the sample text with a model that has been fine-tuned to differentiate between \
            **{SUBJECTIVE_TO_NEUTRAL["source_attribute"]}** and **{SUBJECTIVE_TO_NEUTRAL["target_attribute"]}** tones."""
    )
    st.write(f'**Input Text:** *"{text_sample}"*')
    st.write("")

    cls_result = sic.score(text_sample)
    cls_result_df = pd.DataFrame(
        cls_result[0]["distribution"],
        columns=["Score"],
        index=[v for k, v in sic.pipeline.model.config.id2label.items()],
    )

    with st.container():
        col2_1, col2_2, col2_3 = st.columns([4, 1, 4])

        with col2_1:
            st.bar_chart(cls_result_df)

        with col2_2:
            pass

        with col2_3:
            st.dataframe(cls_result_df)

    st.info(
        f"""**We have room for improvement!** \n\n\n The distribution of classification scores suggests that the input text is more \
         **{SUBJECTIVE_TO_NEUTRAL["source_attribute"]}** than **{SUBJECTIVE_TO_NEUTRAL["target_attribute"]}**. Therefore, \
            an automated style transfer may improve our intended tone of voice."""
    )

    db2 = DisableableButton(2, "Next")
    db2.create_enabled_button()

## 3. Here's why
if st.session_state.page_progress > 2:
    db2.disable()
    st.write("### 3. Here's why")
    st.write(
        f"""Before we can transfer style, we need to ensure the input text isn't already of the target style! To do so, \
            we classify the sample text with a model that has been fine-tuned to differentiate between \
            **{SUBJECTIVE_TO_NEUTRAL["source_attribute"]}** and **{SUBJECTIVE_TO_NEUTRAL["target_attribute"]}** tones."""
    )


## 4. SUGGEST GENERATED EDIT

## 5. EVALUATE QUALITY OF GENERATED SUGGESTION
