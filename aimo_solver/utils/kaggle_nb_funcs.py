"""Python functions from Kaggle notebooks that are used in development of AIMO solver."""

# Competition Specific Imports from HF and torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    pipeline,
    set_seed as transformers_set_seed
)
from vllm import LLM, SamplingParams
import torch

# # Patch issue from https://github.com/vllm-project/vllm/issues/1116
# if torch.cuda.device_count()>1:
#     import ray
#     ray.shutdown()
#     ray.init(num_gpus=torch.cuda.device_count())

import pandas as pd
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import sklearn

# Built-In Imports (mostly don't worry about these)
from typing import Iterable, Any, Callable
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import subprocess
import warnings
import requests
import textwrap
import hashlib
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import copy
import math
import time
import gzip
import ast
import sys
import io
import gc
import re
import os

# Visualization Imports (overkill)
from IPython.core.display import HTML, Markdown
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import plotly.express as px
import seaborn as sns
from PIL import Image, ImageEnhance
import matplotlib
import plotly
import PIL

# Set options to help
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
tqdm.pandas()
Image.MAX_IMAGE_PIXELS = 5_000_000_000


def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)


seed_it_all()

print("\n\n... IMPORTS COMPLETE ...\n")


def load_vllm_model_and_tokenizer(
        model_path: str,
        model_dtype: str = "half",
        enforce_eager: bool = True,
        gpu_memory_utilization: float = 0.999,
        swap_space: int = 4,
        max_model_len: int = 1024,
        kv_cache_dtype: str = "fp8_e5m2",
        tensor_parallel_size: int | str = 1,
):
    """Initializes and returns the specified language model and its associated tokenizer.

    This is primarily used in the context of this competition for the DeepSeek Math RL model.

    While the function and descriptions are mine, the underlying code comes from this notebook:
        https://www.kaggle.com/code/bsmit1659/aimo-vllm-accelerated-tot-sc-deepseekmath/notebook

    Args:
        model_path (str):
            The path to the pre-trained model's checkpoint file on the local filesystem.
            This file contains the learned weights and parameters of the language model.
        model_dtype (str, optional):
            The data type to use for the model's computations.
            Defaults to "half" which represents 16-bit half-precision floating-point format.
            This can help reduce memory usage and improve performance on GPUs.
        enforce_eager (bool, optional):
            Whether to enforce eager execution mode for the model.
            Eager execution allows for immediate evaluation of operations without building a computational graph.
        gpu_memory_utilization (float, optional):
            The fraction of available GPU memory to allocate for the model.
            This controls the trade-off between memory usage and performance. Higher values allocate more memory
            to the model, potentially improving performance but limiting the available memory for other tasks.
        swap_space (int, optional):
            The size of the swap space (in GB) to use for model loading.
            Swap space is used when the model's memory requirements exceed the available GPU memory.
            It allows the model to be loaded by swapping data between GPU memory and CPU memory.
        max_model_len (int, optional):
            The maximum sequence length (in tokens) that the model can process.
            This determines the maximum context size the model can handle in a single forward pass.
            Longer sequences will be truncated to fit within this limit.
        kv_cache_dtype (str, optional):
            The data type to use for the key-value cache in the model.
            The key-value cache stores intermediate activations to speed up computation.
            This can help reduce memory usage while maintaining acceptable precision.
            Defaults to "fp8_e5m2" which represents:
                - an 8-bit floating-point format
                - with exponent bias (5)
                - and mantissa size (2)
        tensor_parallel_size (int | str, optional):
            The number of GPU devices to use for tensor parallelism.
            Tensor parallelism splits the model across multiple GPUs to distribute the computation.
            Defaults to 1, which means no tensor parallelism is used. Use 2 for 2xT4.
            If set to "system" than torch.cuda.device_count() will be used.

    Returns:
        tuple:
            A tuple containing the initialized DeepSeek language model (LLM) and its associated tokenizer.
                - llm (LLM): The initialized DeepSeek language model.
                - tokenizer (Tokenizer): The tokenizer associated with the language model.
    """
    _llm = LLM(
        model=model_path,
        dtype=model_dtype,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=swap_space,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tensor_parallel_size if isinstance(tensor_parallel_size,
                                                                int) else torch.cuda.device_count()
    )
    _tokenizer = _llm.get_tokenizer()
    return _llm, _tokenizer


# https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag
def clean_memory() -> None:
    """Function to clean RAM & vRAM"""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


def flatten_l_o_l(nested_list):
    """ Flatten a list of lists into a single list.

    Args:
        nested_list (Iterable):
            – A list of lists (or iterables) to be flattened.

    Returns:
        A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """ Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional):
            – The symbol to use for the horizontal line
        line_len (int, optional):
            – The length of the horizontal line in characters
        newline_before (bool, optional):
            – Whether to print a newline character before the line
        newline_after (bool, optional):
            – Whether to print a newline character after the line

    Returns:
        None; A divider with pre/post new-lines (optional) is printed
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();


def display_hr(newline_before=False, newline_after=False):
    """ Renders a HTML <hr>

    Args:
        newline_before (bool, optional):
            – Whether to print a newline character before the line
        newline_after (bool, optional):
            – Whether to print a newline character after the line

    Returns:
        None; A divider with pre/post new-lines (optional) is printed
    """
    if newline_before: print();
    display(HTML("<hr>"))
    if newline_after: print();


def wrap_text(text, width=88):
    """Wrap text to a specified width.

    Args:
        text (str):
            - The text to wrap.
        width (int):
            - The maximum width of a line. Default is 88.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width)


def wrap_text_by_paragraphs(text, width=88):
    """Wrap text by paragraphs to a specified width.

    Args:
        text (str):
            - The text containing multiple paragraphs to wrap.
        width (int):
            - The maximum width of a line. Default is 88.

    Returns:
        str: The wrapped text with preserved paragraph separation.
    """
    paragraphs = text.split('\n')  # Assuming paragraphs are separated by newlines
    wrapped_paragraphs = [textwrap.fill(paragraph, width) for paragraph in paragraphs]
    return '\n\n'.join(wrapped_paragraphs)


def hide_asy_text(text: str) -> tuple[str, dict[str, str]]:
    """Replaces text within [asy]...[/asy] blocks with unique placeholders.

    Args:
        text (str):
            The original text containing blocks to be hidden.

    Returns:
        tuple[str, dict[str, str]]:
            A tuple containing the modified text with placeholders
            and a dictionary mapping placeholders to the original text blocks.
    """
    pattern = r'\[asy\](.*?)\[/asy\]'
    placeholders = {}

    def _replacer(match: re.Match) -> tuple[str, dict[str, str]]:
        """This function is used to replace the text within [asy]...[/asy] blocks.

        It replaces the text with a unique placeholder and stores the original text.

        Args:
            match (re.Match): The matched object.

        Returns:
            str: The original text corresponding to the placeholder.
        """
        original = match.group(1)
        placeholder = f"UNIQUE_STRING_{len(placeholders)}"
        placeholders[placeholder] = original
        return f"[asy]{placeholder}[/asy]"

    modified_text = re.sub(pattern, _replacer, text)
    return modified_text, placeholders


def unhide_asy_text(text: str, placeholders: dict[str, str]) -> str:
    """Restores the original text blocks within [asy]...[/asy] from the placeholders.

    Args:
        text (str):
            The text with placeholders to be restored.
        placeholders (dict[str, str]):
            A dictionary mapping placeholders back to the original text.

    Returns:
        str: The text with all placeholders restored to their original content.
    """
    pattern = r'\[asy\](UNIQUE_STRING_\d+)\[/asy\]'

    def _replacer(match: re.Match) -> str:
        """This function is used to replace the placeholders with the original text.

        Args:
            match (re.Match): The matched object.

        Returns:
            str: The original text corresponding to the placeholder.
        """
        placeholder = match.group(1)
        return f"[asy]{placeholders.get(placeholder, 'ERROR: Text not found')}[/asy]"

    restored_text = re.sub(pattern, _replacer, text)
    return restored_text


def load_aops_dataset_as_df(
        csv_path: str,
        coerce_answers: bool = True,
        drop_diagram_questions: bool = True,
        remove_asy_blocks_from_solution: bool = True
) -> pd.DataFrame:
    """This will return a dataframe for the Art of Problem Solving Dataset based on various options.

    Options include:
        - Fixing the answer column by coercing values
            - removing lfill 0s
            - replacing periods added to the right side incorrectly
            - removing commas
        - Removing problems with Asymptote diagrams in problem description (as no diagrams are found in test set)
        - Removing parts Asymptote diagrams from solution description (as no diagrams are found in test set)

    Args:
        csv_path (str): The path to the csv file
        coerce_answers (bool): Whether to fix the answer column
        drop_diagram_questions (bool): Whether to drop questions with Asymptote diagrams
        remove_asy_blocks_from_solution (bool): Whether to remove Asymptote blocks from solution

    Returns:
        pd.DataFrame: The loaded dataset
    """
    _df = pd.read_csv(csv_path)

    if coerce_answers:
        _df["answer"] = _df["answer"].apply(lambda x: x[:-1] if str(x)[-1] == "." else x)
        _df["answer"] = _df["answer"].apply(lambda x: x.replace(",", ""))
        _df["answer"] = _df["answer"].apply(lambda x: int(x) if str(x).startswith("0") and "." not in str(x) else x)

    if drop_diagram_questions:
        _df = _df[_df.solution.str.lower().str.contains("[asy]")]

    if remove_asy_blocks_from_solution:
        _df["solution"] = _df["solution"].apply(lambda text: re.sub(r'\[asy\](.*?)\[/asy\]', '', text))

    return _df.reset_index(drop=True)


def get_problem(df: pd.DataFrame, problem_id: str = None, problem_str: str = None,
                problem_link: str = None) -> pd.DataFrame:
    """This function will retrieve a dataframe subset of the aops_df that matches the desired problem.

    If no problem specifier is provided then a random problem will be retrieved.

    Args:
        df (pd.DataFrame): The dataframe containing problem information.
        problem_id (str, optional): The specific problem ID to filter by.
        problem_str (str, optional): A substring of the problem text to filter by.
        problem_link (str, optional): The specific problem link to filter by.

    Raises:
        ValueError: If no criteria are provided and the dataframe is empty.

    Returns:
        pd.DataFrame: A subset of `aops_df` based on the provided criteria or a single random problem if no criteria are provided.
    """
    # Create a copy to avoid modifying the original dataframe
    _df = copy.deepcopy(df)

    # Check if any criteria is provided
    if problem_id is not None:
        _df = _df[_df['problem_id'] == problem_id]
    if problem_str is not None:
        _df = _df[_df['problem'].str.contains(problem_str, case=False, na=False)]
    if problem_link is not None:
        _df = _df[_df['link'] == problem_link]

    # If no criteria is specified, select a random problem
    if problem_id is None and problem_str is None and problem_link is None:
        if _df.empty:
            raise ValueError("The input dataframe is empty. Cannot select a random problem.")
        _df = get_problem(_df, problem_id=_df.problem_id.sample(1).values[0])
    return _df.reset_index(drop=True)


def problem_to_html(
        problem_str: str,
        problem_link: str | None = None,
        problem_id: str | None = None,
        bg_color: str = "#f5f5f5"
) -> HTML:
    """Generates an HTML representation of a problem description, optionally including a link.

    Args:
        problem_str (str): The text describing the problem.
        problem_link (Optional[str]): A URL linking to further details about the problem, defaults to None.
        bg_color (str): The background color for the HTML div element, defaults to "#f5f5f5".

    Returns:
        HTML: An HTML object suitable for display in IPython environments.
    """
    # Remove ASY text as we don't want to adjust it's formatting
    problem_str, placeholders = hide_asy_text(problem_str)

    #  -- Prettify  --
    _html_str = f'<div style="background-color: {bg_color}; border-radius: 10px; padding: 20px; margin: 20px;"> <b>PROBLEM:</b><br><br>{problem_str.replace("...", "...<br><br>").replace(".", ".<br><br>")}</div>'
    if problem_link is not None and not pd.isna(problem_link):
        _html_str = _html_str.replace("PROBLEM:", 'PROBLEM  <a href="' + problem_link + '">[LINK]</a>:')
    if problem_id is not None:
        _html_str = _html_str.replace("PROBLEM", f"PROBLEM ID: {problem_id}")

    # Put the ASY text back in
    _html_str = unhide_asy_text(_html_str, placeholders)

    return HTML(_html_str)


# Basic Hex Colour for Success is #e9fce9
def solution_to_html(solution_str: str, solution_num: int | str | None = None,
                     solution_value: int | float | str | None = None, bg_color: list[str] | str = "pastel"):
    """Generates an HTML representation of a solution with dynamic background colors and optional details.

    Args:
        solution_str (str):
            The text describing the solution.
        solution_num (int | str, optional):
            A number or identifier for the solution, defaults to None.
        solution_value (int | float | str, optional):
            A value associated with the solution.
        bg_color (list[str] | str, optional):
            The background color(s) for the HTML div element.
            This can be a hex color, a list of hex colors, or a seaborn palette name, defaults to "pastel".

    Returns:
        HTML: An HTML object suitable for display in IPython environments.

    """

    def get_colors(color_input: str | list[str]) -> list[str]:
        """Resolves the background color input into a list of hexadecimal color codes.

        Args:
            color_input (str | list[str]):
                A hex color, a list of hex colors, or a seaborn palette name.

        Returns:
            list[str]: A list of hexadecimal color codes.
        """
        if isinstance(color_input, str) and color_input.startswith("#"):
            return [color_input]
        elif isinstance(color_input, list) and all(isinstance(item, str) for item in color_input):
            return color_input
        else:
            return sns.color_palette(color_input).as_hex()

    # Remove ASY text as we don't want to adjust it's formatting
    solution_str, placeholders = hide_asy_text(solution_str)

    # -- Prettify --
    # Resolve background colors using the internal function
    colors = get_colors(bg_color)
    # Generate the main HTML string for the solution
    color_index = solution_num % len(colors) if solution_num is not None else 0
    _html_str = f'<div style="background-color: {colors[color_index]}; border-radius: 10px; padding: 20px; margin: 20px;"> <b>SOLUTION:</b><br><br>{solution_str.replace("...", "...<br><br>").replace(".", ".<br><br>")}</div>'
    # Add solution number if specified
    if solution_num is not None:
        _html_str = _html_str.replace("SOLUTION:", f"SOLUTION #{solution_num}:")
    if solution_value is not None:
        _html_str = _html_str.replace("</div>",
                                      f'<br><br><b>SOLUTION VALUE: <font color="red">{solution_value}</font></b><br><br></div>')

    # Put the ASY text back in
    _html_str = unhide_asy_text(_html_str, placeholders)
    _html_str = _html_str.replace(r"\[", r"<br><br>\[").replace(r"\]", r"\]<br><br>")
    return HTML(_html_str)


def review_problem(df: pd.DataFrame, problem_id: str | None = None, show_all_solutions: bool = False):
    """This function will retrieve a dataframe subset of the aops_df that matches the desired problem.

    It will then iterate over the provided solutions and display the example in an asthetically pleasing way.
    If no problem specifier is provided then a random problem will be retrieved.

    Args:
        _df (pd.DataFrame): The dataframe containing problem information.
        problem_id (str, optional): The specific problem ID to filter by.
        show_all_solutions (bool, optional): Whether to show all or just the first solution

    Raises:
        ValueError: If no criteria are provided and the dataframe is empty.

    Returns:
        pd.DataFrame: A subset of `aops_df` based on the provided criteria or a single random problem if no criteria are provided.
    """
    _df = get_problem(df, problem_id=problem_id)
    _df_link = _df.link[0] if "link" in _df.columns else None
    display(problem_to_html(_df.problem[0], _df_link, _df.problem_id[0]))

    for i, (_, row) in enumerate(_df.iterrows()):
        display(solution_to_html(row.solution, i + 1, row.answer))
        if not show_all_solutions:
            break

    return _df


def extract_and_evaluate_solution(text: str, re_pattern: str | None = r"\\boxed{((?:[^{}]+|{[^{}]*})*)}",
                                  verbose: bool = False) -> int | float | None:
    """Extracts a LaTeX expression from a given text and evaluates it numerically.

    If a regex is provided but no match is found... the parsing will assume full text requires evaluation.

    Args:
        text (str):
            The text containing the LaTeX expression.
        re_pattern (str, optional):
            A regular expression pattern to extract the LaTeX expression enclosed in specific LaTeX commands like \\boxed{}.
            If None, evaluates the entire text as a LaTeX expression.

    Returns:
        int | float | None: The evaluated numerical result as an integer or float, or None if no expression is found or an error occurs in parsing.
    """
    # Use the provided regular expression pattern, or default to the entire text
    latex_expression = text
    if re_pattern:
        match = re.search(re_pattern, text)
        if match:
            latex_expression = match.group(1)
    else:
        latex_expression = text

    try:
        # Convert LaTeX to a sympy expression
        sympy_expression = sp.sympify(parse_latex(latex_expression))

        # Evaluate the expression to a numerical result and determine type
        evaluated_expression = sympy_expression.evalf()
        if evaluated_expression.is_Integer:
            return int(evaluated_expression)
        else:
            return float(evaluated_expression)
    except Exception as e:
        if verbose:
            print(f"Error parsing or evaluating the expression: {e}")
        return -1.0


def remove_multiple_choice_options_in_problem(problem_str):
    return problem_str.rsplit("$\\textbf{(A", 1)[0].rsplit("$\\text{(A", 1)[0].strip()


def remove_asy_block(text):
    if text.count("[asy]") > 0:
        text = text.split("[asy]", 1)[0] + text.split("[/asy]", 1)[-1]
        remove_asy_block(text)
    return text.strip()


def fix_and_filter_external_data(df: pd.DataFrame, drop_non_pos_int_rows: bool = True):
    """"""
    _df = df.copy()

    # Drop rows with infinite values in answer
    _df = _df[~_df["answer"].replace([np.inf, -np.inf], np.nan).isna()]

    # Force answer to be appropriate dtype (int or float)
    _df["answer"] = _df["answer"].apply(lambda x: int(x) if float(int(x)) == float(x) else float(x))

    # Remove multiple choices in problem string
    _df.loc[~pd.isna(_df["letter"]), "problem"] = _df.loc[~pd.isna(_df["letter"]), "problem"].apply(
        remove_multiple_choice_options_in_problem)

    # Replace hashtag solutions with boxed
    _df.loc[_df.solution.str.contains("#### "), "solution"] = _df.loc[
        _df.solution.str.contains("#### "), "solution"].apply(
        lambda x: x.strip().rsplit("####", 1)[0] + "$\\boxed{" + x.strip().rsplit("####", 1)[-1].strip() + "}$"
    )

    # Drop rows where the answer is less than 0 or non-int
    if drop_non_pos_int_rows:
        _df = _df[_df.answer.apply(lambda x: isinstance(x, int) and 0 <= x)]

    # Drop rows where there is

    # Modulo the answer and store the original in a separate column
    _df["original_answer"] = _df["answer"].copy()
    _df["answer"] = _df["answer"].apply(lambda x: x % 1000)

    return _df.reset_index(drop=True)


def filter_by_consensus(df: pd.DataFrame, id_col: str = 'problem_id', answer_col: str = 'answer') -> pd.DataFrame:
    """
    Filters a DataFrame to retain only rows where the answer has the majority consensus
    for each unique problem identifier.

    This function groups the DataFrame by a problem identifier and determines the most frequent
    answer for each group. Only rows where the answer matches the most frequent (mode) answer
    for their corresponding group are retained.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        id_col (str, optional): The column name in the DataFrame that contains the problem identifiers.
                                Defaults to 'problem_id'.
        answer_col (str, optional): The column name in the DataFrame that contains the answers.
                                    Defaults to 'answer'.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows with the majority consensus answer
                      for each unique problem identifier.

    Example:
        >>> data = {'problem_id': ['1', '1', '1', '2', '2', '3'],
        ...         'answer': [10, 10, 4, 150, 150, 3]}
        >>> df = pd.DataFrame(data)
        >>> filtered_df = filter_by_consensus(df)
        >>> print(filtered_df)
    """
    _df = df.copy()

    # Calculate the mode of the answers for each problem_id
    mode_df = _df.groupby(id_col)[answer_col].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    mode_df.rename(columns={answer_col: 'mode_answer'}, inplace=True)

    # Merge this back with the original DataFrame to filter
    merged_df = _df.merge(mode_df, on=id_col)

    # Keep only rows where the answer matches the mode answer
    result_df = merged_df[merged_df[answer_col] == merged_df['mode_answer']]

    # Remove the temporary mode_answer column
    result_df = result_df.drop(columns=['mode_answer'])

    return result_df.reset_index(drop=True)


def set_seed(seed: int = 42) -> None:
    """Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed number. Default is 42.
    """
    transformers_set_seed(seed)


def create_quantization_config(load_in_4bit: bool = True,
                               quant_type: str = "nf4",
                               compute_dtype=torch.bfloat16,
                               use_double_quant: bool = True) -> BitsAndBytesConfig:
    """Creates a configuration for model quantization to optimize model size and inference speed.

    Args:
        load_in_4bit (bool): Whether to load models in 4-bit precision.
        quant_type (str): Type of quantization, 'nf4' for noise-free 4-bit.
        compute_dtype: Data type for computation, typically torch.bfloat16 for mixed precision.
        use_double_quant (bool): Whether to use double quantization.

    Returns:
        BitsAndBytesConfig: A configuration object for BitsAndBytes.
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def load_model_and_tokenizer(
        model_path: str = "/kaggle/input/deepseek-math",
        quantization_config: BitsAndBytesConfig | None = None
) -> tuple:
    """Loads the tokenizer and model with specific quantization configurations.

    Args:
        model_path (str): Path to the model directory.
        quantization_config (BitsAndBytesConfig): Quantization configuration for the model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    config = AutoConfig.from_pretrained(model_path)
    config.gradient_checkpointing = True

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        config=config
    )

    return model, tokenizer


def initialize_pipeline(model, tokenizer) -> pipeline:
    """Initializes a pipeline for text generation using the provided model and tokenizer.

    Args:
        model: The pre-trained model to be used for text generation.
        tokenizer: The tokenizer for text preprocessing.

    Returns:
        pipeline: A configured pipeline for text generation.
    """
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype='auto',
        device_map="auto"
    )


def setup_torch_backend(enable_mem_efficient_sdp: bool = False) -> None:
    """Configures PyTorch backend settings.

    Args:
        enable_mem_efficient_sdp (bool): Flag to enable memory efficient scatter-gather.
                                        Default is False.
    """
    torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient_sdp)


def naive_parse(answer: str) -> str:
    """Extracts the last contiguous sequence of digits from a given string.

    This function is based on the function that is floating around in the top scoring code notebooks.
    I'm not sure who the original author was... but once I know I will attribute accordingly.

    Args:
        answer: A string from which to extract the digit sequence.

    Returns:
        A string containing the last sequence of digits found in the input string.
        Returns an empty string if no digits are found.

    Examples:
        naive_parse("example123test456") returns "456"
        naive_parse("no digits here!") returns ""
    """
    last_digits = ''
    found_digit = False

    for char in reversed(answer):
        if char.isdigit():
            last_digits += char
            found_digit = True
        elif found_digit:
            # Break the loop once the first non-digit is found after finding digits
            break

    # Reverse to correct the order of digits
    return last_digits[::-1]


def postprocess_final_answer(expression: str, modulo: int = 1000) -> int:
    """Postprocesses the final answer by returning the rounded/modulod value.

    Args:
        expression: The mathematical expression to evaluate as a string. (raw final answer)
        modulo: The modulo value to use in the calculation.

    Returns:
        An integer result of the evaluated expression modulo the specified value.
    """
    try:
        result = round(float(eval(expression)))
        return result % modulo
    except Exception as e:
        print(f"Exception occured in `postprocess_final_answer`: {e}")
        return -1


def execute_code(code: str, timeout_seconds: int = 7, filename: str = 'code_to_execute.py', modulo: int = 1000,
                 sympy_star_import: bool = True) -> int:
    """Executes the given Python code snippet and processes the output.

    Args:
        code: The Python code to execute.
        timeout_seconds: Maximum allowed time for code execution in seconds.
        filename: The filename to which the code will be written before execution.
        modulo: The modulo value to use for processing the output.
        sympy_star_import: Whether to always import everything from sympy

    Returns:
        An integer result derived from the execution output or -1 if an error occurs.
    """
    try:
        with open(filename, 'w') as fout:
            fout.write(code if not sympy_star_import else 'from sympy import *\n' + code)

        batcmd = f'timeout {timeout_seconds} {sys.executable} {filename}'
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return postprocess_final_answer(shell_output, modulo)
    except Exception as e:
        print(f"Exception occured in `execute_code`: {e}")
        return -1


def extract_and_process_math(result: str, modulo: int = 1000) -> int:
    """Extracts and evaluates the mathematical expression from the given string.

    Args:
        result: The string containing the expression in a LaTeX-like \\boxed{} format.
        modulo: The modulo value to use for the final result.

    Returns:
        An integer result of the evaluated expression or -1 if an error occurs.
    """
    try:
        result_output = re.findall(r'\\boxed\{(.*)\}', result)
        if result_output:
            expression = result_output[-1]
        else:
            expression = naive_parse(result)

        if expression:
            return postprocess_final_answer(expression, modulo)
        return -1
    except Exception as e:
        print(f"Exception occured in `extract_and_process_math`: {e}")
        return -1


def process_output(output: str, timeout_seconds: int = 7, filename: str = 'code_to_execute.py',
                   modulo: int = 1000) -> tuple:
    """Processes the provided output string to execute contained code and extract mathematical results.

    Args:
        output: The string that may contain Python code in triple backticks and/or a mathematical expression in \\boxed{}.
        timeout_seconds: Maximum allowed time for code execution in seconds.
        filename: The filename for saving and executing the Python code.
        modulo: The modulo value to use for processing the outputs.

    Returns:
        A tuple (result_output, code_output) where each is an integer result of the processing or -1 if an error occurs.
    """
    code_output = -1
    result_output = -1

    # Extract and execute code from output
    code_pattern = re.compile(r'```(?:\S*?\n)?(.*?)```', re.DOTALL)
    code_match = code_pattern.search(output)
    if code_match:
        code = code_match.group(1)
        code_output = execute_code(code, timeout_seconds, filename, modulo)
        # print('CODE RESULTS', code_output)

    # Extract and process mathematical result
    result_output = extract_and_process_math(output, modulo)
    # print('BOXED', result_output)

    return result_output, code_output


def prepare_problem_statement(problem: str, tool_instruction: str | None = None, tokenizer: Any = None,
                              apply_chat_template: bool = True, use_simple: bool = False, ) -> str:
    """Prepares the complete problem statement by appending the tool instruction to the problem text.

    Args:
        problem (str):
            The original problem text.
        tool_instruction (str):
            Additional instructions or information to append to the problem.
        tokenizer ():
            The huggingface tokenizer
        apply_chat_template (bool, optional):
            Whether to apply the HF prompt template (requires )
            If no tokenizer is provided apply_chat_template will not work.
        use_simple (bool, optional):
            Whether to do 0 prompt engineering.



    Returns:
        A complete problem statement ready for processing.
    """
    if not use_simple and tool_instruction is not None:
        prompt_str = tool_instruction + f"\nQUESTION:\n{problem}\n\nYou must write out the logical solution in a step by step fashion before you write any python code to solve the problem.\n\nSOLUTION:\n"
    else:
        prompt_str = problem + "\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."

    if apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_str}],
            tokenize=False
        )
    else:
        return prompt_str


def aggregate_results(code_results: Iterable, boxed_results: Iterable, boxed_copies_over_code_fail: bool = True,
                      use_code_and_boxed: bool = False) -> int:
    """Aggregates the outputs, resolves errors, and determines the most common valid output.

    Args:
        code_results: List of code outputs.
        boxed_results: List of mathematical result outputs.
        boxed_copies_over_code_fail: Whether the non-error boxed results will copy over the failed code results
        use_code_and_boxed: Whether to aggregate results from both code and boxed results initially

    Returns:
        The most common valid output among the provided answers or -1 if none are valid.
    """
    # So we can pop
    code_results, boxed_results = list(code_results), list(boxed_results)
    if all(x < 0 for x in code_results):
        boxed_copies_over_code_fail = True

    # Get the results array
    if boxed_copies_over_code_fail and not use_code_and_boxed:
        results = []
        for i in range(len(code_results) - 1, -1, -1):
            if code_results[i] > 0:
                results.append(code_results[i])
            else:
                code_results.pop(i)
                results.append(boxed_results.pop(i))
        results = results[::-1]
    elif not boxed_copies_over_code_fail and not use_code_and_boxed:
        results = code_results
    else:
        results = code_results + boxed_results
    results = np.array(results)

    # Handle negatives as invalid results and handle negatives in boxed_results if needed
    results = np.where(results < 0, -1, results)
    boxed_results = np.where(np.array(boxed_results) < 0, -1, np.array(boxed_results))

    # Get most common
    most_common_results_w_counts = [x for x in Counter(results).most_common() if x[0] != -1]
    if len(most_common_results_w_counts) == 0:
        return 1
    elif len(most_common_results_w_counts) == 1:
        return int(abs(most_common_results_w_counts[0][0]))
    if most_common_results_w_counts[0][1] == most_common_results_w_counts[1][1] and not use_code_and_boxed:
        most_common_results_w_counts = [x for x in Counter(np.concatenate(
            (results, results, results, results, results, results, results, boxed_results))).most_common() if
                                        x[0] != -1]
    return int(abs(most_common_results_w_counts[0][0]))


def run_pipeline(
        model_pipeline: Callable,
        query_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.85,
        num_repetitions: int = 2,
) -> list:
    """Executes the text-generation pipeline multiple times and collects outputs.

    Args:
        model_pipeline: The initialized text generation pipeline.
        query_prompt: Input text for the pipeline.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Controls randomness in output generation.
        num_repetitions: Number of times to run the pipeline for each input.

    Returns:
        A list of outputs from the pipeline.
    """

    # Initialize the empty results list for this particular query prompt
    results = []

    # For N repetitions we will repeatedly attempt the problem.
    for _ in tqdm(range(num_repetitions)):
        try:
            raw_output = model_pipeline(
                query_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_full_text=False
            )
            results.append(raw_output[0]['generated_text'])
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Exception occured in `model_pipeline`: {e}")
            results.append(None)
    return results


class VLLMPipeline:
    def __init__(
            self,
            model: object | None = None,
            tokenizer: object | None = None,
            stop_words: list[str] | None = None,
            model_sampling_params: dict[str, Any] | None = None,
            **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        # Set stop words and fallback
        self.stop_words = stop_words
        if stop_words is None:
            self.stop_words = stop_words or [
                tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
        self.model_sampling_params = model_sampling_params or {}
        self.model_sampling_params["stop"] = self.stop_words

    def __call__(
            self,
            query_prompt: str | list[str],
            max_new_tokens: int,
            temperature: float,
            do_sample: bool = True,
            return_full_text: bool = False,
            stop_word_overrides: list[str] | None = None,
            n_repeats: int = 1,
            sampling_kwargs: dict[str, Any] | None = None,
            batch_style: str = "multiply",
            do_cleanup: bool = False,
    ):
        # batch style is "multiply" or "sampling_param_n"
        # Coerce into batch format
        if isinstance(query_prompt, str):
            query_prompt = [query_prompt]

        # Validate sampling is allowed and if not adjust temperature
        temperature = 0.0000001 if not do_sample else temperature

        # Get sampling parameters and update with call specific params
        sampling_params_kwargs = {**self.model_sampling_params}
        sampling_params_kwargs.update(sampling_kwargs or {})
        sampling_params_kwargs.update({"temperature": temperature, "max_tokens": max_new_tokens})

        # Parse input batch
        if batch_style == "multiply" and len(query_prompt) == 1:
            query_prompt = query_prompt * n_repeats
        elif batch_style == "sampling_param_n":
            sampling_params_kwargs.update({"n": n_repeats})

        # Finalize sampling params
        _sampling_params = SamplingParams(**sampling_params_kwargs)

        # Do inference
        model_output = self.model.generate(query_prompt, _sampling_params)

        # Parse output
        if batch_style == "multiply":
            model_output = [output.outputs[0].text for output in model_output]
        elif batch_style == "sampling_param_n":
            model_output = [output.text for output in model_output[0].outputs]

        # Cleanup
        if do_cleanup:
            torch.cuda.empty_cache()
            gc.collect()

        return model_output


def initialize_vllm_pipeline(
        model,
        tokenizer,
        stop_words: list[str] | None = None,
        model_sampling_params: dict[str, Any] | None = None,
        **kwargs,
) -> VLLMPipeline:
    """Artificial pipeline construct so we can mimic the transformers workflow.

    Args:
        model: The pre-trained model to be used for text generation.
        tokenizer: The tokenizer for text preprocessing.

    Returns:
        VLLMPipeline: A configured pipeline for text generation.
    """
    return VLLMPipeline(
        model=model,
        tokenizer=tokenizer,
        stop_words=stop_words,
        model_sampling_params=model_sampling_params,
        **kwargs
    )


def run_vllm_pipeline(
        model_pipeline: VLLMPipeline,
        query_prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.85,
        num_repetitions: int = 2,
        sampling_kwargs: dict[str, Any] | None = None
) -> list:
    """Executes the text-generation pipeline multiple times and collects outputs.

    Args:
        model_pipeline: The initialized text generation pipeline.
        query_prompt: Input text for the pipeline.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Controls randomness in output generation.
        num_repetitions: Number of times to run the pipeline for each input.

    Returns:
        A list of outputs from the pipeline.
    """
    try:
        model_results = model_pipeline(
            query_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_repeats=num_repetitions,
            sampling_kwargs=sampling_kwargs
        )
    except Exception as e:
        print(f"Exception occured in `model_pipeline`: {e}")
        model_results = ["", ] * num_repetitions

    return model_results


def get_aimo_examples(df, num_of_examples: int = 1, source="AOPS", idx: int = None):
    if idx is None:
        return df[df.source==source].sample(num_of_examples)
    else:
        return df[df.source==source].reset_index().iloc[idx:idx+num_of_examples]
