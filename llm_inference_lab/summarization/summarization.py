import os
from dataclasses import dataclass, field
import logging
from typing import Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PegasusForConditionalGeneration,
    PegasusTokenizerFast,
    BartForConditionalGeneration,
    BartTokenizerFast,
    BatchEncoding,
    PretrainedConfig,
)
import torch

from llm_inference_lab.tools.torch.quantization import dynamic_quantization
from llm_inference_lab.utils.models import pegasus_model

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TextSummarizerResult(BaseModel):
    """
    BaseModel for the result of a text summarization containing the summary or an error

    Attributes
        summary (list): Summary of text
        error (list): Error that occurred during summarization. Empty string if no errors occurred
    """

    summary: list


@dataclass
class TextSummarizer:  # pylint:disable=too-many-instance-attributes
    """Text summarization via Pegasus model architecture or DistilBart model architecture.
    Defaults to Pegasus model architecture.

    model_location (str): Path to model location. Options, '../models/pegasus-cnn_dailymail'
        or '../models/distilbart-cnn-12-6'. Defaults to
        '../models/pegasus-cnn_dailymail'
    num_beams (int): Number of beams for beam search
    device (str): Cuda device to use if available. Defaults to 'cuda:0'
    tokenizer (SummarizerTokenizer): Tokenizer to use. Defaults to SummarizerTokenizer()
    use_dynamic_quantization (bool): Whether to use pytorch dynamic quantization. Defaults to True.
    """

    os.environ["MKL_NUM_THREAD"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_location: str = pegasus_model.path
    device: str = "cuda:0"
    use_dynamic_quantization: bool = False
    tokenizer_path: str = pegasus_model.path
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = False
    tokenizer: PegasusTokenizerFast | BartTokenizerFast = field(init=False)
    config: PretrainedConfig = field(init=False)

    model: PegasusForConditionalGeneration | BartForConditionalGeneration = field(
        init=False
    )
    use_gpu: bool = field(init=False, default=False)
    is_initialized: bool = field(init=False, default=False)

    def __post_init__(self):
        """
        Post-instantiation tasks
        """
        # Load model configuration
        self.config = AutoConfig.from_pretrained(self.tokenizer_path)

        # Load model tokenizer
        self.tokenizer = self.load_tokenizer()

        # Initialize and load pretrained model
        if not self.is_initialized:
            logger.info("Checking for gpu")
            self._check_for_gpu()
            logger.info("Loading model")
            self._load_model()

            self.is_initialized = True


    def __call__(self, text: list[str]) -> TextSummarizerResult:
        """Summarizes text

        Args:
            text (str): Text to be summarized

        Returns:
            TextSummarizerResult: The text summary and an empty string indicating that no error
                occurred. There are several ways that summarization can fail and result in an
                error:
                - If text is an empty string, summary will be an emtpy string and an error string
                  will also be included
                - If text does not generate more input_ids than what the model was trained for,
                  then the text is too short to generate a genuine summary. The input text is
                  returned as the summary as well as an error string
                - Any other exception will result in returning an empty string for the summary, and
                  the string-serialized exception as the error string.

        """
        try:

            summaries = self.get_summaries(text)
            return TextSummarizerResult(summary=summaries)

        except Exception as exc:
            logging.exception(exc)
            return TextSummarizerResult(summary=[""])


    def get_token_count(self, input_id) -> int:
        try:
            return len(input_id[0])
        except Exception:
            return 0

    # Helper function for exceptions
    def get_token_count_exc(self, input_id) -> str:
        try:
            str(len(input_id[0]))
            return ""
        except Exception as exc:
            return str(exc)

    # Helper function to get summaries
    def get_summaries(self, text: list[str]) -> list[str]:
        batch_encoding = self.encode(text)
        input_ids = self.get_input_ids(batch_encoding)
        summary_ids = self.generate_summary_ids(input_ids)
        summary = self.decode(summary_ids)
        summary = [summ.replace("<n>", "") for summ in summary]
        return summary

    ###########################
    # MODEL FUNCTIONS         #
    ###########################

    def encode(self, text: list[str]) -> BatchEncoding:
        """Tokenizes text

        Args:
            text (str): Text to be tokenized

        Returns:
            BatchEncoding: Tokenization output
        """
        token_inputs: BatchEncoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if self.use_gpu:
            token_inputs.to(self.device)
        return token_inputs

    def get_input_ids(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Extracts input_ids from a BatchEncoding if it has them.

        Args:
            batch_encoding (BatchEncoding): BatchEncoding

        Raises:
            ValueError: When 'input_ids' key missing in batch_encoding

        Returns:
            Optional[torch.Tensor]: BatchEncoding input ids or None if input_ids is missing
        """
        input_ids = batch_encoding.get("input_ids", None)
        if input_ids is None:
            raise ValueError("input_ids missing from BatchEncoding")
        return input_ids

    def generate_summary_ids(self, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Generates summary ids from selected Model

        Args:
            input_ids (Optional[torch.Tensor]): Input ids

        Raises:
            ValueError: If model doesn't produces a tensor

        Returns:
            torch.Tensor: Summary ids generated by Pegasus Model
        """
        summary_ids_tensor = self.model.generate(
            input_ids
        )

        if not isinstance(summary_ids_tensor, torch.Tensor):
            raise ValueError("Expected self.model.generate to return torch.LongTensor")

        return summary_ids_tensor
    
    def decode(self, summary_ids: torch.Tensor) -> list[str]:
        """Decodes tokens back into a string

        Args:
            summary_ids (torch.Tensor): Tokens to decode

        Returns:
            str: String output
        """
        return self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )

    ###########################
    # MODEL LOAD FUNCTIONS    #
    ###########################

    def _load_model_from_path(self):
        """Loads model from self.model_location

        Raises:
            ValueError: If model type is incorrect
        """
        model = self.load_model()

        if self.use_dynamic_quantization:
            if isinstance(model, PegasusForConditionalGeneration):
                model = dynamic_quantization(model)
        if not isinstance(
            model, (PegasusForConditionalGeneration, BartForConditionalGeneration)
        ):
            raise ValueError(
                "PegasusTextSummarizer can only be used with",
                "PegasusForConditionalGeneration models",
                "and BartForConditionalGeneration models",
            )
        self.model = model
    
    def _check_for_gpu(self):
        """Sets the use_gpu property"""
        if torch.cuda.is_available() and not self.device == "cpu":
            self.use_gpu = True

    def _load_model_gpu(self):
        """Loads the model into GPU"""
        self._check_for_gpu()
        if self.use_gpu:
            log_str = f"Using gpu device {self.device}"
            logger.info(log_str)
            self.model.to(self.device)
    
    def load_tokenizer(self):
        """
        Loads tokenizer from path and checks that type is
        `PegasusTokenizerFast` or `BartTokenizerFast`
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if not isinstance(tokenizer, (PegasusTokenizerFast, BartTokenizerFast)):
            raise ValueError(
                "TextSummarizer can only be used with PegasusTokenizerFast",
                "and BartTokenizerFast tokenizers",
            )
        return tokenizer

    def load_model(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(pegasus_model.path)
        return model