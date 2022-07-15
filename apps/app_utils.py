from typing import List

import tokenizers
import streamlit as st


from apps.data_utils import string_to_list_string


from apps.data_utils import StyleAttributeData

from src.style_transfer import StyleTransfer
from src.style_classification import StyleIntensityClassifier
from src.content_preservation import ContentPreservationScorer
from src.transformer_interpretability import InterpretTransformer

# CALLBACKS
def increment_page_progress():
    st.session_state.page_progress += 1


def reset_page_progress_state():
    st.session_state.page_progress = 1


# UTILITY CLASSES
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


