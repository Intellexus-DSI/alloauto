import logging
import os
import json
import pandas as pd
from pathlib import Path
import traceback
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = utils.set_logger(logger, full_msg=False)

messages = []
def _append_msg(msg: str) -> None:
    if not msg:
        return
    messages.append(msg)

def main():

    # Load API keys from keys.yaml
    logger.info("Loading keys from keys.yaml")
    _append_msg("Loading keys from keys.yaml")
    try:
        utils.set_env_vars("keys.yaml")
    except Exception as e:
        logger.error(f"Error loading keys from keys.yaml: {e}")
        _append_msg(f"Error loading keys from keys.yaml: {e}")
        return
    logger.info("Keys loaded from keys.yaml")
    _append_msg("Keys loaded from keys.yaml")
    
    #Load configuration from config.yaml
    logger.info("Loading config from config.yaml")
    _append_msg("Loading config from config.yaml")
    try:
        config = utils.load_config("config.yaml")
    except Exception as e:
        logger.error(f"Error loading config from config.yaml: {e}")
        _append_msg(f"Error loading config from config.yaml: {e}")
        return
    
    # Load data
    DATA_DIR = config.get('data_dir', '')
    DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR)
    if not DATA_DIR or not Path(DATA_DIR).exists():
        logger.error(f"❌ Error: Data dir not found: {DATA_DIR}")
        return 

    DATA_FILE_NAME = config.get('data_file_name', '')
    logger.info(f"🔄 Loading data from {os.path.join(DATA_DIR, DATA_FILE_NAME)}")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE_NAME))
        _append_msg(f"Loaded {len(df)} samples from {os.path.join(DATA_DIR, DATA_FILE_NAME)}")
        logger.info(f"Loaded {len(df)} samples from {os.path.join(DATA_DIR, DATA_FILE_NAME)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Load results 
    RESULTS_FILE = config.get('results_model_name', 'gemini-2.5-flash')
    RESULTS_FILE = f"./results/results_{RESULTS_FILE}.jsonl"
    RESULTS_FILE = os.path.join(os.path.dirname(__file__), RESULTS_FILE)
    logger.info(f"Loading results from {RESULTS_FILE}")
    _append_msg(f"Loading results from {RESULTS_FILE}")
    if not RESULTS_FILE or not Path(RESULTS_FILE).exists():
        logger.error(f"❌ Error: Results file not found: {RESULTS_FILE}")
        return
    results = list(utils.load_results_json(RESULTS_FILE))
    _append_msg(f"Loaded {len(results)} results from {RESULTS_FILE}")
    logger.info(f"Loaded {len(results)} results from {RESULTS_FILE}")

    # Preapare entries for prompt
    logger.info(f"Preparing entries for reasoning profile")
    _append_msg(f"Preparing entries for reasoning profile")
    debug_samples = config.get('debug_samples', -1)
    n_samples = debug_samples if debug_samples > 0 else len(results)

    if debug_samples > 0:
        logger.info(f"Debug mode: processing only {debug_samples} samples")
        _append_msg(f"Debug mode: processing only {debug_samples} samples")
    
    entries = []
    for idx, result in enumerate(results):
        pred_first_segment = result['first_segment']
        pred_predictions = result['predictions']
        reasoning = result['reasoning']
        text = df.loc[idx, 'tokens']
        true_predictions = utils.get_true_predictions(df.loc[idx, 'labels'])
       
        block = f"""
        Example {idx+1}:
        Text: {text}
        Model first segment: {pred_first_segment}
        Model prediction: {pred_predictions}
        Model reasoning: {reasoning}
        True first segment: {true_predictions["real_first_segment"]}
        True switches: {true_predictions["real_predictions"]}
        """

        entries.append(block)
        if idx == debug_samples:
            break
    entries = "\n".join(entries)
    logger.info(f"Entries prepared for reasoning profile")
    _append_msg(f"Entries prepared for reasoning profile")

    # Initialize LLM
    logger.info("🔄 Initializing LLM")
    try:
        llm = utils.get_llm(
            temperature=config.get('temperature', 0.3),
            model_name=config.get('model_name', "gemini-2.5-flash"),
            max_output_tokens=config.get('max_tokens', None)
        )
        _append_msg(f"✅ LLM initialized: {config.get('model_name', 'gemini-2.5-flash')}")
        logger.info(f"✅ LLM initialized: {config.get('model_name', 'gemini-2.5-flash')}")
    except Exception as e:
        logger.error(f"❌ Error initializing LLM: {e}")
        return

    # Get prompt
    prompt = utils.get_reasoning_prompt()
    logger.info(f"Prompt loaded")
    _append_msg(f"Prompt loaded {prompt}")

    # Create chain
    chain = prompt | llm
    
    # Create profile
    response = chain.invoke({"n_samples": n_samples, "entries": entries})
    response_content = response.content if hasattr(response, 'content') else ""
    cleaned_response = response_content.strip('`').replace('json\n', '')
    try:
        result = json.loads(cleaned_response)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON response: {cleaned_response}")
    
    result_string = json.dumps(result, indent=4)
    print("=" * 50)
    print("=" * 18, "LLM RESULTS:", "=" * 18)
    print("=" * 50)
    print(result_string)

    _append_msg("=" * 50)
    _append_msg("=" * 18 +"LLM RESULTS:"+"=" * 18)
    _append_msg("=" * 50)
    _append_msg(result_string)
    
    print("=" * 50)
    _append_msg("=" * 50)
    # Save results
    model_name = config.get('model_name', 'gemini-2.5-flash')
    result_model_name = config.get('results_model_name', 'gemini-2.5-flash')
    try:
        results_dir = os.path.join(os.path.dirname(__file__), "reasoning_profile")
        os.makedirs(results_dir, exist_ok=True)
        if debug_samples > 0:
            output_filename = f"reasoning_profile_{debug_samples}_samples_from_{result_model_name}_into_{model_name}.jsonl"
        else:   
            output_filename = f"reasoning_profile_from_{result_model_name}_into_{model_name}.jsonl"
        output_file = os.path.join(results_dir, output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_string)
        _append_msg(f"💾 Results saved to: {output_file}")
        logger.info(f"💾 Results saved to: {output_file}")
    except Exception as e:
        _append_msg(f"❌ Error saving results: {e}")
        logger.error(f"❌ Error saving results: {e}")
        logger.error(traceback.format_exc())
    
    print("=" * 50)
    logger.info("📈 Writing report")
    report_dir = os.path.join(os.path.dirname(__file__), "reasoning_profile")
    os.makedirs(report_dir, exist_ok=True)
    if debug_samples > 0:
        report_filename = f"REPORT_{debug_samples}_samples_from_{result_model_name}_into_{model_name}.log"
    else:   
        report_filename = f"REPORT_{result_model_name}_into_{model_name}.log"
    report_file = os.path.join(report_dir, report_filename)
    utils.write_report(messages, report_file)
    logger.info(f"✅ Report written to: {report_file}")
if __name__ == "__main__":
    main()

    