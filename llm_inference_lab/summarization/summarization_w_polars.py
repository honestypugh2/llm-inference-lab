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

import polars as pl

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
    error: list


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
    num_beams: int = 4
    device: str = "cuda:0"
    use_dynamic_quantization: bool = False
    tokenizer_path: str = pegasus_model.path
    max_length: int = 512
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
            batch_df = (
                pl.DataFrame(
                    {
                        "text": text,
                        "index": range(len(text)),  # for preserving input order
                    },
                    schema={"text": str, "index": int},
                )
                .with_columns([pl.col("text").str.strip().alias("cleaned_text")])
                .with_columns([pl.col("text").str.n_chars().alias("char_count")])
                .with_columns(
                    [pl.col("cleaned_text").str.n_chars().alias("char_count_cleaned")]
                )
            )
            empty_string_output_df = self._handle_empty_strings(batch_df)
            initial_encodings_df = self._initial_encoding(batch_df)
            non_summarizable_output_df = self._handle_non_summarizable_strings(
                initial_encodings_df
            )
            summarizable_df = self._summarize_batch(initial_encodings_df)

            # Concatenate all tables together and sort by index
            batch_df = pl.concat(
                [
                    empty_string_output_df,
                    summarizable_df,
                    non_summarizable_output_df,
                ]
            ).sort("index")

            summaries = list(batch_df["summary"])
            errors = list(batch_df["error"])

            return TextSummarizerResult(summary=summaries, error=errors)

        except Exception as exc:
            logging.exception(exc)
            return TextSummarizerResult(summary=[""], error=[str(exc)])

    ###########################################
    # HELPER FUNCTIONS TO PREPARE TEXT       #
    ##########################################
    # Helper function to get empty strings for polars dataframe
    def _handle_empty_strings(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        # Create output dataframe of Empty Strings
        empty_string_output_df = dataframe.filter(pl.col("char_count_cleaned") == 0)
        # Copy the contents of the text colimn and create a new column
        # called summary and a new column called error
        empty_string_output_df = empty_string_output_df.with_columns(
            [
                # Copy contents
                pl.col("text").alias("summary"),
                pl.lit("Empty String").alias("error"),
            ]
        )
        # Drop unnecessary columns
        empty_string_output_df = empty_string_output_df.drop(
            ["cleaned_text", "char_count", "char_count_cleaned"]
        )
        return empty_string_output_df

    def _initial_encoding(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        # Perform initial encoding pass to get token count
        token_df = dataframe.filter(pl.col("char_count_cleaned") > 0)
        tokens_list = [
            self.encode(txt).get("input_ids") for txt in list(token_df["text"])
        ]
        token_count_list = [self.get_token_count(tok) for tok in tokens_list]
        token_count_exc_list = [self.get_token_count_exc(tok) for tok in tokens_list]
        token_df = token_df.with_columns(
            [
                pl.Series("tokens", tokens_list),
                pl.Series("token_count", token_count_list),
                pl.Series("token_count_exception", token_count_exc_list),
            ]
        )
        return token_df

    def _handle_non_summarizable_strings(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        # Create output dataframe for rows that don't create enough tokens
        non_summarizable_output_df = dataframe.filter(
            pl.col("token_count") < self.tokenizer.config.min_length
        )
        tooshort_error_list = [
            "Source text too short to summarize. Using original text." if v == "" else v
            for v in list(non_summarizable_output_df["token_count_exception"])
        ]
        non_summarizable_output_df = non_summarizable_output_df.with_columns(
            [
                pl.col("text").alias("summary"),
                pl.Series("error", tooshort_error_list),
            ]
        )
        non_summarizable_output_df = non_summarizable_output_df.with_columns(
            [
                pl.col("error").cast(pl.Utf8),
            ]
        )
        # Drop unnecessary columns
        non_summarizable_output_df = non_summarizable_output_df.drop(
            [
                "cleaned_text",
                "char_count",
                "char_count_cleaned",
                "tokens",
                "token_count",
                "token_count_exception",
            ]
        )
        return non_summarizable_output_df

    def _summarize_batch(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        # Finally, perform summarization
        # Create summarizable df (non_empty, minimal token length)
        summarizable_df = dataframe.filter(
            pl.col("token_count") >= self.tokenizer.config.min_length
        )
        if len(summarizable_df) > 0:
            summaries = self.get_summaries(list(summarizable_df["text"]))

            summarizable_df = summarizable_df.with_columns(
                [
                    pl.Series("summary", summaries),
                    pl.lit("").alias("error"),
                ]
            )
            # Drop unnecessary columns
            summarizable_df = summarizable_df.drop(
                [
                    "cleaned_text",
                    "char_count",
                    "char_count_cleaned",
                    "tokens",
                    "token_count",
                    "token_count_exception",
                ]
            )
        else:
            summarizable_df = pl.DataFrame(
                schema={"text": str, "index": int, "summary": str, "error": str}
            )

        return summarizable_df

    # Helper function for polars
    def get_token_count(self, input_id) -> int:
        try:
            return len(input_id[0])
        except Exception:
            return 0

    # Helper function for polars exceptions
    def get_token_count_exc(self, input_id) -> str:
        try:
            str(len(input_id[0]))
            return ""
        except Exception as exc:
            return str(exc)

    # Helper function to get summaries for polars dataframe
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
            max_length=self.max_length,
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
            input_ids, num_beams=self.num_beams, early_stopping=True, max_new_tokens=128
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