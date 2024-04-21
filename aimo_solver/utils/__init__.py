"""The main utilities package for the aimo_solver package."""

from .kaggle_nb_funcs import (
    seed_it_all,
    load_vllm_model_and_tokenizer,
    clean_memory,
    flatten_l_o_l,
    print_ln,
    display_hr,
    wrap_text,
    wrap_text_by_paragraphs,
    hide_asy_text,
    unhide_asy_text,
    load_aops_dataset_as_df,
    get_problem,
    problem_to_html,
    solution_to_html,
    review_problem,
    extract_and_evaluate_solution,
    remove_multiple_choice_options_in_problem,
    remove_asy_block,
    fix_and_filter_external_data,
    filter_by_consensus,
    set_seed,
    create_quantization_config,
    load_model_and_tokenizer,
    initialize_pipeline,
    setup_torch_backend,
    naive_parse,
    postprocess_final_answer,
    execute_code,
    extract_and_process_math,
    process_output,
    prepare_problem_statement,
    aggregate_results,
    run_pipeline,
    initialize_vllm_pipeline,
    run_vllm_pipeline,
    get_aimo_examples
)

__all__ = [
    "seed_it_all",
    "load_vllm_model_and_tokenizer",
    "clean_memory",
    "flatten_l_o_l",
    "print_ln",
    "display_hr",
    "wrap_text",
    "wrap_text_by_paragraphs",
    "hide_asy_text",
    "unhide_asy_text",
    "load_aops_dataset_as_df",
    "get_problem",
    "problem_to_html",
    "solution_to_html",
    "review_problem",
    "extract_and_evaluate_solution",
    "remove_multiple_choice_options_in_problem",
    "remove_asy_block",
    "fix_and_filter_external_data",
    "filter_by_consensus",
    "set_seed",
    "create_quantization_config",
    "load_model_and_tokenizer",
    "initialize_pipeline",
    "setup_torch_backend",
    "naive_parse",
    "postprocess_final_answer",
    "execute_code",
    "extract_and_process_math",
    "process_output",
    "prepare_problem_statement",
    "aggregate_results",
    "run_pipeline",
    "initialize_vllm_pipeline",
    "run_vllm_pipeline",
    "get_aimo_examples"
]
