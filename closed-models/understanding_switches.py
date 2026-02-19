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
    

    # Preapare entries for prompt
    logger.info(f"Preparing entries for reasoning profile")
    _append_msg(f"Preparing entries for reasoning profile")
    debug_samples = config.get('debug_samples', -1)
    n_samples = debug_samples if debug_samples > 0 else len(df)

    if debug_samples > 0:
        logger.info(f"Debug mode: processing only {debug_samples} samples")
        _append_msg(f"Debug mode: processing only {debug_samples} samples")
    
    entries = []
    for idx, row in df.iterrows():
        text = row['tokens']
        true_predictions = utils.get_true_predictions(row['labels'])

        block = f"""
        Example {idx+1}:
        Text: {text}
        First segment type: {true_predictions["real_first_segment"]}
        Switches between AUTO and ALLO: {true_predictions["real_predictions"]}
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
    prompt = utils.get_understanding_switches_prompt()
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
    try:
        results_dir = os.path.join(os.path.dirname(__file__), "understanding_switches")
        os.makedirs(results_dir, exist_ok=True)
        if debug_samples > 0:
            output_filename = f"{model_name}_{debug_samples}_samples_explain_switches.jsonl"
        else:   
            output_filename = f"{model_name}_explain_switches.jsonl"
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
    report_dir = os.path.join(os.path.dirname(__file__), "understanding_switches")
    os.makedirs(report_dir, exist_ok=True)
    if debug_samples > 0:
        report_filename = f"REPORT_{debug_samples}_{model_name}_explain_switches.log"
    else:   
        report_filename = f"REPORT_{model_name}_explain_switches.log"
    report_file = os.path.join(report_dir, report_filename)
    utils.write_report(messages, report_file)
    logger.info(f"✅ Report written to: {report_file}")
if __name__ == "__main__":
    main()

    