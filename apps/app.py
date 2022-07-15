import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from apps.data_utils import (
    DATA_PACKET,
    format_classification_results,
)
from apps.app_utils import (
    reset_page_progress_state,
    DisableableButton,
    get_cached_style_intensity_classifier,
    get_cached_word_attributions,
    get_sti_metric,
    get_cps_metric,
    generate_style_transfer,
)
from apps.visualization_utils import build_altair_classification_plot

# SESSION STATE UTILS
if "page_progress" not in st.session_state:
    st.session_state.page_progress = 1

if "st_result" not in st.session_state:
    st.session_state.st_result = False


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
    options=DATA_PACKET.keys(),
    help="ADD HELP MESSAGE HERE!",
)
STYLE_ATTRIBUTE_DATA = DATA_PACKET[style_attribute]

st.sidebar.button("Start over!", on_click=reset_page_progress_state)

# MAIN CONTENT
st.markdown("# Intelligent Writing Assistance with Text Style Transfer")

st.write(
    """The goal of this application is to demonstrate how the NLP task of _text style transfer_ can be applied to enhance the human writing experience. \
    In this sense, we intend to peel back the curtains on how intelligent writing assistants might function — walking through the logical steps needed to \
    automatically re-style a piece of text while building up confidence in the model output. \
    \n\n We emphasize the imperative for a human-in-the-loop user experience when dealing with natural language generation systems. We believe text style \
    transfer has the potential to empower writers to better express themselves, but not by blindly generating text. Rather, generative models, in conjunction with \
    interpretability methods, should be be combined to help writers understand the nuances of linguistic "style" and suggest edits that may improve their writing."""
)

## 1. INPUT EXAMPLE
if st.session_state.page_progress > 0:
    st.write("### 1. Input some text")

    with st.container():

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
                option = st.selectbox(
                    f"Select a preset example to modify style from {style_attribute}",
                    [
                        f"Example {i+1}"
                        for i in range(len(STYLE_ATTRIBUTE_DATA.examples))
                    ],
                    help="You can also enter your own text by toggling the radio button to the left.",
                )

                idx_key = int(option.split(" ")[-1]) - 1
                text_sample = STYLE_ATTRIBUTE_DATA.examples[idx_key]

        st.text_area(
            "Preview Text",
            value=text_sample,
            placeholder="Enter some text above or toggle to choose a preset!",
            disabled=True,
        )

    if text_sample != "":
        db1 = DisableableButton(1, "Let's go!")
        db1.create_enabled_button()

## 2. CLASSIFY INPUT
if st.session_state.page_progress > 1:
    db1.disable()

    st.write("### 2. Detect style")
    st.write(
        f"""
            Before we can transfer style, we need to ensure the input text isn't already of the target style! To do so,
            we classify the sample text with [a model]({STYLE_ATTRIBUTE_DATA.build_model_url("cls")}) that has been fine-tuned to differentiate between
            {STYLE_ATTRIBUTE_DATA.attribute_AND_string} tones. 
            
            In a product setting, you could imagine this style detection process running inside your favorite word processor, prompting you for action when it 
            detects language that is at odds with your desired tone of voice.
            """
    )

    with st.spinner("Detecting style, hang tight!"):

        sic = get_cached_style_intensity_classifier(style_data=STYLE_ATTRIBUTE_DATA)
        cls_result = sic.score(text_sample)

    cls_result_df = pd.DataFrame(
        cls_result[0]["distribution"],
        columns=["Score"],
        index=[v for k, v in sic.pipeline.model.config.id2label.items()],
    )

    with st.container():

        format_cls_result = format_classification_results(
            id2label=sic.pipeline.model.config.id2label, cls_result=cls_result
        )

        st.markdown("##### Distribution Between Style Classes")
        chart = build_altair_classification_plot(format_cls_result)
        st.altair_chart(chart.interactive(), use_container_width=True)

        st.markdown(
            f"""
            - **Input Text:** *"{text_sample}"*
            - **Classification Result:** \t {cls_result[0]["label"].capitalize()} ({round(cls_result[0]["score"]*100, 2)}%)
            """
        )
        st.write(" ")

    if cls_result[0]["label"].lower() != STYLE_ATTRIBUTE_DATA.target_attribute:
        st.info(
            f"""**Looks like we have room for improvement!** \n\n\n The distribution of classification scores suggests that the input text is more \
            {STYLE_ATTRIBUTE_DATA.attribute_THAN_string}. Therefore, \
                an automated style transfer may improve our intended tone of voice."""
        )
        db2 = DisableableButton(2, "Let's see why")
        db2.create_enabled_button()
    else:
        st.success(
            f"""**No work to be done!** \n\n\n The distribution of classification scores suggests that the input text is less \
            {STYLE_ATTRIBUTE_DATA.attribute_THAN_string}. Therefore, there's no need to transfer style. \
            Enter a different text prompt or select one of the preset examples to re-run the analysis with."""
        )

## 3. Here's why
if st.session_state.page_progress > 2:
    db2.disable()
    st.write("### 3. Interpret the classification result")
    st.write(
        f"""Interpreting our model's output is a crucial practice that helps build trust and justify taking real-world action from the \
            model predictions. In this case, we apply a popular model interpretability technique called [Integrated Gradients](https://arxiv.org/pdf/1703.01365.pdf) \
            to the Transformer-based classifier to explain the model's prediction in terms of its features."""
    )

    with st.spinner("Interpreting the prediction, hang tight!"):
        word_attributions_visual = get_cached_word_attributions(
            text_sample=text_sample, style_data=STYLE_ATTRIBUTE_DATA
        )
        components.html(html=word_attributions_visual, height=150, scrolling=True)

    st.write(
        f"""The visual above displays word attributions using the [Transformers Interpret](https://github.com/cdpierse/transformers-interpret) library. \
            Positive attribution values (green) indicate tokens that contribute positively towards the \
            predicted class ("{STYLE_ATTRIBUTE_DATA.source_attribute}"), while negative values (red) indicate tokens that contribute negatively towards the predicted class. \
            \n\n\n Visualizing word attributions is a helpful way to build intuition about what makes the input text "{STYLE_ATTRIBUTE_DATA.source_attribute}"!"""
    )
    db3 = DisableableButton(3, "Next")
    db3.create_enabled_button()


## 4. SUGGEST GENERATED EDIT
if st.session_state.page_progress > 3:
    db3.disable()

    st.write("### 4. Generate a suggestion")
    st.write(
        f"Now that we've verified the input text is in fact *{STYLE_ATTRIBUTE_DATA.source_attribute}* and understand why that's the case, we can utilize a \
            text style transfer model to generate a suggested replacement that retains the same semantic meaning, but achieves the *{STYLE_ATTRIBUTE_DATA.target_attribute}* target style.\
            \n\n Expand the accordian below to toggle generation parameters, then click the button to transfer style!"
    )

    with st.expander("Toggle generation parameters"):

        # st.markdown("##### Text generation parameters")
        st.write("**max_gen_length**")
        max_gen_length = st.slider(
            "Whats the maximum generation length desired?", 1, 250, 200, 10
        )
        st.write("**num_beams**")
        num_beams = st.slider(
            "How many beams to use for beam-search decoding?", 1, 8, 4
        )
        st.write("**temperature**")
        temperature = st.slider(
            "What sensitivity value to model next token probabilities?",
            0.0,
            1.0,
            1.0,
        )

    st.markdown(
        f"""
        **:hugging_face: Model:** [{STYLE_ATTRIBUTE_DATA.seq2seq_model_path}]({STYLE_ATTRIBUTE_DATA.build_model_url("seq2seq")})
        """
    )

    col4_1, col4_2, col4_3 = st.columns([1, 8, 8])
    with col4_2:
        st.markdown(
            f"""
            - **Max Generation Length:** {max_gen_length}
            - **Number of Beams:** {num_beams}
            - **Temperature:** {temperature}
            """
        )
    with col4_3:
        with st.container():

            st.write("")
            st.button(
                "Generate style transfer",
                key="generate_text",
                on_click=generate_style_transfer,
                kwargs={
                    "text_sample": text_sample,
                    "style_data": STYLE_ATTRIBUTE_DATA,
                    "max_gen_length": max_gen_length,
                    "num_beams": num_beams,
                    "temperature": temperature,
                },
            )

    if st.session_state.st_result:
        st.warning(
            f"""**{STYLE_ATTRIBUTE_DATA.source_attribute.capitalize()} Input:** "{text_sample} """
        )
        st.info(
            f"""
            **{STYLE_ATTRIBUTE_DATA.target_attribute.capitalize()} Suggestion:** "{st.session_state.st_result[0]}" """
        )
        db4 = DisableableButton(4, "Next")
        db4.create_enabled_button()

## 5. EVALUATE THE SUGGESTION
if st.session_state.page_progress > 4:
    db4.disable()
    st.write("### 5. Evaluate the suggestion")
    st.markdown(
        """
        Blindly prompting a writer with style suggestions without first checking quality would make for a noisy, error-prone product
        with a poor user experience -- ultimately, we only want to suggest high quality edits. How do we check for quality?

        A comprehensive quality evaluation for text style transfer output should consider three criteria:
        1. **Style strength** - To what degree does the generated text achieve the target style? 
        2. **Content preservation** - To what degree does the generated text retain the semantic meaning of the source text?
        3. **Fluency** - To what degree does the generated text appear as if it were produced naturally by a human?

        Below, we apply automated evaluation metrics -- _Style Transfer Intensity (STI)_ & _Content Preservation Score (CPS)_ -- to
        measure the first two of these criteria by comparing the input text to the generated suggestion.
        """
    )

    with st.spinner("Evaluating text style transfer, hang tight!"):

        sti = get_sti_metric(
            input_text=text_sample,
            output_text=st.session_state.st_result[0],
            style_data=STYLE_ATTRIBUTE_DATA,
        )
        cps = get_cps_metric(
            input_text=text_sample,
            output_text=st.session_state.st_result[0],
            style_data=STYLE_ATTRIBUTE_DATA,
        )

    st.markdown(
        """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True,
    )

    col5_1, col5_2, col5_3 = st.columns([3, 1, 3])

    with col5_1:
        st.metric(
            label="Style Transfer Intensity (STI)",
            value=f"{sti[0]*100}",
        )
        st.caption(
            f"""
                STI compares the style class distributions (determined by the [style classifier]({STYLE_ATTRIBUTE_DATA.build_model_url("cls")}))
                between the input text and generated suggestion using Earth Mover's Distance. Therefore, STI can be thought of as the degree to which style was altered ranging
                between -100 and 100. EXPLAIN MORE
                """
        )

    with col5_3:
        st.metric(
            label="Content Preservation Score (CPS)",
            value=f"{cps[0]*100}%",
        )
        st.caption(
            f"""
                CPS compares the latent space embeedings (determined by [SentenceBERT]({STYLE_ATTRIBUTE_DATA.build_model_url("sbert")}))
                between the input text and generated suggestion using cosine similarity. Therefore, CPS can be thought of as the percentage of content that was perserved
                during the style transfer.
                """
        )
