from typing import List, Union

import torch
from transformers import pipeline


class StyleTransfer:
    """
    Model wrapper for a Text2TextGeneration pipeline used to transfer a style attribute on a given piece of text.

    Attributes:
        model_identifier (str) - Path to the model that will be used by the pipeline to make predictions
        max_gen_length (int) - Upper limit on number of tokens the model can generate as output

    """

    def __init__(
        self,
        model_identifier: str,
        max_gen_length: int = 200,
        num_beams=4,
        temperature=1,
    ):
        self.model_identifier = model_identifier
        self.max_gen_length = max_gen_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        self._build_pipeline()

    def _build_pipeline(self):

        self.pipeline = pipeline(
            task="text2text-generation",
            model=self.model_identifier,
            device=self.device,
            max_length=self.max_gen_length,
            num_beams=self.num_beams,
            temperature=self.temperature,
        )

    def transfer(self, input_text: Union[str, List[str]]) -> List[str]:
        """
        Transfer the style attribute on a given piece of text using the
        initialized `model_identifier`.

        Args:
            input_text (`str` or `List[str]`) - Input text for style transfer

        Returns:
            generated_text (`List[str]`) - The generated text outputs

        """
        return [item["generated_text"] for item in self.pipeline(input_text)]
