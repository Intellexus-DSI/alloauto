from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import os
import json
import yaml
import json
import logging
from typing import List, Generator
import numpy as np

def get_llm(temperature=0.3, model_name="gemini-2.5-flash", max_output_tokens=None):
    """Initialize LLM with specified parameters"""
    if model_name == "gemini-2.5-flash" or model_name == "gemini-2.5-pro":
        return ChatGoogleGenerativeAI(
            model=model_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
    elif model_name == "gpt-4o":
        return ChatOpenAI(
            model=model_name,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
    elif model_name == "meta-llama/Llama-4-Scout-17B-16E-Instruct" or model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
        return ChatTogether(
            model=model_name,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
    elif model_name == "claude-sonnet-4-5-20250929":
        return ChatAnthropic(
            model=model_name,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

# region Old prompts
#=======LONG PROMPT=======#
# PROMPT = {
#     "system": """You are a Tibetan Buddhist philology expert and computational linguist. Read Tibetan text and segment it into contiguous spans of two types:
#     AUTO (autochthonous Tibetan): passages originally composed in Tibetan.
#     ALLO (allochthonous Tibetan): passages translated into Tibetan (typically from Sanskrit or related Indic sources).
#     Output: Return only JSON of the form: "prediction": [i1, i2, ...]
#     Each integer is a 0‑based word index that marks the start of a new segment after a label change.
#     You must output at least one index (best-guess boundary). Never return an empty list.
#     Word indexing (basis for indices):
#     - Normalize consecutive whitespace to single spaces for counting.
#     - The first word has index 0.
#     - Do not include index 0 in prediction.
#     - Indices must be integers, unique, ascending, each within [1, W-1] for total words W.
#     Minimal segmentation guidance (to avoid empty outputs):
#     - Choose a starting label for the first span (default AUTO) and scan left→right.
#     - Switch to ALLO when you encounter sustained translationese cues, or a strong hard marker. Switch back to AUTO when those cues subside for a similar stretch.
#     - If uncertain in a long passage (≥150 words), make at least one best‑guess switch at the most probable boundary rather than returning an empty list.
#     Common ALLO cues (examples—not exhaustive):
#     - Dhāraṇī/mantra blocks or seed syllables: ཨོཾ, ཧཱུྃ, ཙྪཿ, repeated invocatory formulae; long phonetic strings with minimal Tibetan particles.
#     - Dense Indic proper names/transliterations and non‑Tibetan orthography bursts (e.g., stacked consonant transliterations, visarga‑like signs, siddham marks).
#     - Literal scholastic calque feel: extended chains mirroring Sanskrit compounds and rigid enumerations; repetitive frames akin to namo, iti, evaṃ mayā śrutam equivalents.
#     - Function word profile shift: unusually sparse use of idiomatic Tibetan connective particles relative to long technical noun compounds.
#     Common AUTO cues:
#     - Idiomatic expository/narrative flow, local examples, freer paraphrase, pragmatic asides.
#     - Tibetan rhetorical turns and connective particles used in natural distribution.
#     Hard markers that justify an immediate switch (even within <12 words):
#     - Start/end of mantra/dhāraṇī section.
#     - A contiguous run of transliteration‑like syllables or non‑Tibetan letter clusters.
#     Tie‑breaking & stability:
#     - When signals are mixed, maintain the current label until evidence accumulates; but avoid returning an empty list for long mixed passages—choose the most plausible single boundary.
#     Formatting constraints:
#     - Output only the JSON object.
#     - Do not include any explanation, counts, or labels.
# """,
#     "user": "Text: {text}\n"
# }
#=======LONG PROMPT=======#

#=======SHORT PROMPT=======#
# PROMPT_OLD = {
#     "system": """
#     You are a Tibetan Buddhist philology expert and computational linguist. Your task is to read Tibetan text and segment it into contiguous spans of two types:
#     AUTO (autochthonous Tibetan): passages originally composed in Tibetan. ALLO (allochthonous Tibetan): passages translated into Tibetan (typically from Sanskrit or related Indic sources).
#     Return only JSON of the form: ("first_segment": "allo" | "auto", "prediction": [i1, i2, ...] Each integer is a 0‑based word index that marks the start of a new segment after a label change. If there is no switch, return "prediction": [].
#     Split the given text into words as follows:
#     - Normalize consecutive whitespace to single spaces for counting.
#     - first_segment is the first segment of the text, either "allo" or "auto".
#     - The first word has index 0.
#     - Do not include index 0 in prediction.
#     - Indices must be integers, unique, sorted ascending, each within [1, {total_tokens}].
#     - An index must be less than {total_tokens}.
#     Output only the JSON object; no explanations or extra text.
#     """,
#     "human": "{text}\n"
# }
#endregion

PROMPT = {
    "system": """
    You are a Tibetan Buddhist philology expert and computational linguist. 
    Your task is to read Tibetan text and segment it into contiguous spans of two types:
    - AUTO (autochthonous Tibetan): passages originally composed in Tibetan.
    - ALLO (allochthonous Tibetan): passages translated into Tibetan (typically from Sanskrit or related Indic sources).

    We are building a profile for Tibetan text segmentation.
    After segmenting, you must output detailed reasoning *inside the JSON object* under the key "reasoning".
    Do NOT include any explanations or commentary outside the JSON.
    Return ONLY a JSON object of the form:
    {{
    "reasoning": "<short explanation of why and where the text switches>",
    "first_segment": "auto" | "allo",
    "prediction": [i1, i2, ...]
    }}
    Definitions:
    - "reasoning" briefly explains in 2-3 sentences *why* you placed those boundaries (e.g., indicators of translationese, mantra markers, syntax shifts, stylistic transitions, etc.).
    - "first_segment" is the label of the first span in the text: "auto" or "allo".
    - "prediction" is a list of 0-based WORD indices that mark the start of a new segment after a label change.
    - If there is no switch, output: {{"reasoning": "No clear switch detected.", "first_segment": "auto"|"allo", "prediction": []}}
    Word indexing rules:
    - Normalize consecutive whitespace to single spaces for counting.
    - The first word has index 0.
    - Do not include index 0 in prediction.
    - Indices must be integers, unique, strictly ascending, each within [1, {total_tokens}].

    Strict formatting:
    - Output only valid JSON with all three keys: "reasoning", "first_segment", "prediction".
    - Do not include any text or commentary outside the JSON.
    """,
    "human": "{text}\n"
}

COT_PROMPT = {
    "system": """
    You are a Tibetan Buddhist philology expert and computational linguist.
    Your task is to read Tibetan text and segment it into contiguous spans of two types:
    - AUTO (autochthonous Tibetan): passages originally composed in Tibetan.
    - ALLO (allochthonous Tibetan): passages translated into Tibetan (typically from Sanskrit or related Indic sources).

    You must output detailed reasoning for your segmentation, but only *inside the JSON object* under the key "reasoning".
    Do NOT include explanations or commentary outside the JSON.

    Return ONLY a JSON object of the form:
    {{
    "first_segment": "auto" | "allo",
    "prediction": [i1, i2, ...],
    "reasoning": "<short explanation of why and where the text switches>"
    }}

    Definitions:
    - "first_segment" is the label of the first span in the text: "auto" or "allo".
    - "prediction" is a list of 0-based WORD indices that mark the start of a new segment after a label change.
    - "reasoning" is describing *why* you placed those boundaries (e.g., indicators of translationese, mantra markers, syntax shifts, etc.).
    - If there is no switch, output: {{"first_segment": "auto"|"allo", "prediction": [], "reasoning": "No clear switch detected."}}

    Word indexing rules:
    - Normalize consecutive whitespace to single spaces for counting.
    - The first word has index 0.
    - Do not include index 0 in prediction.
    - Indices must be integers, unique, strictly ascending, each within [1, {total_tokens}].
    - Every index must be < {total_tokens}.

    Strict formatting:
    - Output only valid JSON with all three keys: "first_segment", "prediction", "reasoning".
    - Do not include any text or commentary outside the JSON.
    """,
    "human": "Text: {text}\n"
}

REASONING_PROFILE_PROMPT = {
    "system": """
    You are a Tibetan Buddhist philology researcher and computational linguist.
    Your task is to analyze a batch of Tibetan text segmentation examples and produce a single *global segmentation profile*.

    Each example provides:
    - The original Tibetan text.
    - The model's predicted switch indices and reasoning.
    - The true switch indices.

    Your goal:
    1. Identify **recurring linguistic or stylistic features** that distinguish AUTO and ALLO passages.
    2. Note **patterns in reasoning** that match or mismatch the true switch indices.
    3. Detect **common boundary cues** where segment transitions usually occur.
    4. Summarize all of this as a unified profile.

    Return **only valid JSON** with the structure:
    {{
    "key_observations": "a paragraph summarizing key observations",
    "summary": "a paragraph summarizing global segmentation behavior"
    }}

    Formatting rules:
    - Use double quotes, no trailing commas.
    - Every key must appear.
    - Write concise but analytical descriptions.
    """,
    "human": """
    Analyze the following {n_samples} dataset entries:

    {entries}

    Derive a single comprehensive segmentation profile capturing shared AUTO vs ALLO features,
    recurrent cues for switches, and model behavior relative to gold labels.
    Return only the JSON profile object.
    """
}

PROMT_UNDERSTANDING_LABELS = {
     "system": """
    You are a Tibetan Buddhist philology expert and computational linguist.
    You will receive a collection of Tibetan texts, each with its *true switch indices* that mark 
    boundaries between AUTO (autochthonous Tibetan) and ALLO (allochthonous Tibetan).

    Your task is to analyze the entire dataset as a whole and produce a concise analytical summary 
    that describes the recurring linguistic, stylistic, and structural patterns that typically define 
    AUTO and ALLO segments, as well as the typical cues or transitions observed at switch points.

    Focus on patterns — not evaluating or predicting. Think like a researcher explaining what these 
    switches reveal about the nature of Tibetan composition and translation.

    Return only valid JSON with this structure:
    {{
    "key_observations": "Describe the most consistent patterns observed across the dataset — how AUTO and ALLO differ linguistically, what triggers switches, and any broader stylistic regularities.",
    "summary": "Provide a concise research-style paragraph summarizing what this collection of gold switches reveals about Tibetan segmentation as a phenomenon (e.g., stylistic shifts, mantra inclusions, translation markers, etc.)."
    }}

    Formatting rules:
    - Use double quotes and no trailing commas.
    - Both keys must appear.
    - Be analytical and textual; do not output examples or indices.
    - Focus on aggregated insights from all texts.
    """,
    "human": """
    You are given {n_samples} Tibetan texts, each with its gold switch indices.

    {entries}

    Analyze the full collection to discover overall segmentation patterns.
    Explain what these true switches suggest about Tibetan AUTO vs ALLO boundaries.
    Return only the JSON object.
    """
}

PROMPT_REASONING_ANALYSIS = {
    "system": """
    You are a Tibetan Buddhist philology researcher and computational linguist.
    Your task is to analyze a batch of reasoning traces from a Tibetan segmentation model.
    Each example includes:
    - The original Tibetan text.
    - The model’s reasoning for its segmentation decisions.
    - The model’s predicted switch indices.
    - The true (gold) switch indices.

    You are NOT evaluating linguistic content or segmentation quality itself.
    Instead, focus on the *meta-logic* behind the reasoning: how the model explains its decisions.

    Your goals:
    1. Observe **recurring reasoning patterns or motifs** that appear across examples.
    2. Identify how these patterns relate to prediction correctness (consistent logic vs. mismatches).
    3. Assess the **quality and coherence** of reasoning: whether it is evidence-based, repetitive, speculative, or inconsistent.
    
    Return **only valid JSON** with the following structure:
    {{
    "key_observations": "a short paragraph describing recurring reasoning patterns, logic structures, and consistency across examples. Highlight when reasoning repeats, shifts, or contradicts itself. use maximum 10 examples.",
    "summary": "Provide a concise global evaluation of reasoning quality across all samples — how systematic, evidence-based, or error-prone it is overall."
    }}

    Formatting rules:
    - Use double quotes, no trailing commas.
    - Every key must appear.
    - Focus on reasoning patterns, not segmentation details.
    """,
    "human": """
    Analyze the following {n_samples} examples:

    {entries}

    Each example contains: Text, Model Reasoning, Model Prediction, and Gold Indices.
    Study how the reasoning logic behaves across examples, find repeated patterns,
    and summarize which reasoning styles lead to correct vs incorrect segmentation.
    Return only the JSON profile object.
    """
}

PROMPT_ZERO_SHOT =  ChatPromptTemplate.from_messages([
    ("system", PROMPT["system"]),
    ("human", PROMPT["human"])
])

PROMPT_ZERO_SHOT_COT =  ChatPromptTemplate.from_messages([
    ("system", COT_PROMPT["system"]),
    ("human", COT_PROMPT["human"])
])

# === FEW-SHOT PROMPT ===
# Define your Sanskrit examples with CORRECTED output format
examples = [
    # Example 1: 
    {
        "input":  "grims lhod ran pa'i mtshams ni rang gis brtags pa na rig pa 'phang 'di tsam cig bstod na rgod pa skye nges par 'dug snyam pa'i tshad de las ni lhod la / 'di tsam cig la bzhag na bying pa skye sla bar 'dug snyam pa'i tshad de las kyang 'phang mtho ba'i 'jog tshul la bya'o / lnga ba de'i shes byed 'god pa ni / 'phags pa thogs med kyis kyang / de la 'jog par byed pa dang yang dag par 'jog par byed pa la ni bsgrims te 'jug pa'i yid la byed pa yod do / zhes sems dang po gnyis kyi skabs su gsungs shing / sgom rim dang po las kyang bying ba bsal la dmigs pa de nyid dam du gzung ngo / zhes bshad do / drug pa dran pa bsten tshul ma shes pa'i skyon ni /",
        "output": '{"first_segment": "auto", "prediction": [76, 103, 120, 133]}'
    },
    # Example 2: 
    {
        "input": "byams pa chen po rgyun chad pa nyan thos pa la dgag bya'i gtso bo ma yin pa'i phyir ro / mu gsum yongs su dag pa 'di ni ngas rim gyis bslab pa'i gzhi bsngam pa'i phyir zhes sogs kyis kyang / nyan thos pas sha za ba bkag pa ma yin te / dper na / bstan pa la rim gyis gzhug pa'i phyir du / theg pa gsum gsungs pas nyan thos pa la rang don don gnyer gyi bsam pa ma bkag pa bzhin no / des na mdo de dag gis byang sems las dang po pa sha la sred pa'i @# / dbang gis byams pa chen po rgyun chad par 'gyur ba la sha sred pas za ba bkag pa yin te / sha za ba ni byams pa chen po chad par 'gyur ro / zhes dang / ngas ni lus g. yog pa'i phyir gos sna tshogs kyang kha dog ngan par bsgyur bar bya'o zhes bstan na / sha za ba'i ro la chags pa lta ci smos / zhes sogs dang /",
        "output": '{"first_segment": "auto", "prediction": [130, 143, 146, 179]}'
    },
    # Example 3: 
    {
        "input": "yod des rnal bzhin gnyid log na / de las ma rungs gzhan ci yod / ces gsungs pa dang / spyod 'jug las kyang / thams cad bor te cha dgos par / bdag gis de ltar ma shes nas / mdza' dang mi mdza'i don gyi phyir / sdig pa rnam pa sna tshogs byas / zhes gsungs so /",
        "output": '{"first_segment": "auto", "prediction": [26, 58]}'
    }
]

# Define how each example should be formatted
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Create the few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# region few-shot system prompts
# few_shot_system = ChatPromptTemplate.from_messages([
#     ('system', """The following Human–AI pairs are EXAMPLES."""),
#     ('system', """END OF EXAMPLES \n """)
# ])
#endregion

# Reuse the zero-shot prompt and add few-shot examples
PROMPT_FEW_SHOT = ChatPromptTemplate.from_messages([
    *PROMPT_ZERO_SHOT.messages[:-1],  # All messages except the last user message
    #few_shot_system.messages[0],
    few_shot_prompt,
   #few_shot_system.messages[1],
    PROMPT_ZERO_SHOT.messages[-1]     # The original user message template
])

# === CONFIGURATION-BASED PROMPT SELECTION ===

def get_prompt(use_few_shot=False, cot=False):
    """
    Get the appropriate prompt template based on configuration.
    Args:
        use_few_shot (bool): If True, use few-shot prompting. If False, use simple prompt.   
    Returns:
        ChatPromptTemplate: The selected prompt template
    """
    if use_few_shot and cot:
        raise ValueError("COT and Few-Shot prompting cannot be used together")
    if use_few_shot:
        print("Using Few-Shot Prompting approach")
        return PROMPT_FEW_SHOT
    elif cot:
        print("Using COT zero-shot Prompting approach")
        return PROMPT_ZERO_SHOT_COT
    else:
        print("Using zero-shot Prompting approach")
        return PROMPT_ZERO_SHOT

def set_env_vars(config_path="keys.yaml"):
    """Set OS env variables from a YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
    with open(config_path, "r") as f:
        keys = yaml.safe_load(f)
        for key, value in keys.items():
            os.environ[key] = str(value)

def save_results(results, output_file):
    """Save results to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}
    
        
def set_logger(logger: logging.Logger, full_msg: bool = False) -> logging.Logger:
    """Set logger to report file"""
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    if full_msg:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    else:
        fmt = logging.Formatter("%(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger

def write_report(messages:List[str], report_file: str) -> None:
    """Write messages to report file"""
    with open(report_file, "w", encoding="utf-8") as f:
        for message in messages:
            f.write(message + "\n")

def load_results_json(file_path: str) -> Generator[dict, None, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            yield {
                'reasoning': json_line['reasoning'],
                'first_segment': json_line['first_segment'],
                'predictions': json_line['predictions'],
                'total_tokens': json_line['total_tokens'],
                'usage_metadata': json_line['usage_metadata'],
                'labeled_array': json_line['labeled_array'],
                'sample_id': json_line['sample_id'],
            }
            
def fill_class_segments(results: List[dict]) -> List[List[int]]:
    """Fill class segments in results"""
    full_predictions = []
    for result in results:
        first_segment = result['first_segment']
        segment_breaks = list(result['predictions'])
        total_tokens = result['total_tokens']

        if first_segment == 'auto':
            curr_class, curr_switch_class = 0, 3
        elif first_segment == 'allo':
            curr_class, curr_switch_class = 1, 2
        else:
            raise ValueError(f"Invalid first segment: {first_segment}")

        if segment_breaks == []:
            full_predictions.append([curr_class] * total_tokens)
            continue

        prev_break_idx = 0
        curr_labels = [None] * total_tokens
        for break_idx in segment_breaks:
            for i in range(prev_break_idx, break_idx):
                curr_labels[i] = curr_class
            curr_labels[break_idx] = curr_switch_class

            curr_class = 0 if curr_class == 1 else 1
            curr_switch_class = 2 if curr_switch_class == 3 else 3
            prev_break_idx = break_idx + 1

        for i in range(prev_break_idx, total_tokens):
            curr_labels[i] = curr_class

        full_predictions.append(curr_labels)
    return full_predictions
    
def convert_to_labeled_array(prediction: List[int], first_segment: str, total_tokens: int) -> List[int]:
    """
    Convert single prediction to array of labeled tokens
    Args:
        prediction: List[int] - List of indices where the segment changes
        first_segment: str - The first segment of the text
        total_tokens: int - The total number of tokens in the text
    Returns:
        List[int] - List of labeled tokens
    """
    if first_segment == 'auto':
        curr_class, curr_switch_class = 0, 3
    elif first_segment == 'allo':
        curr_class, curr_switch_class = 1, 2
    else:
        raise ValueError(f"Invalid first segment: {first_segment}")

    if prediction == []:
        return [curr_class] * total_tokens

    prev_break_idx = 0
    labels = [None] * total_tokens
    for break_idx in prediction:
        if break_idx == total_tokens:
            return labels
        for i in range(prev_break_idx, break_idx):
            labels[i] = curr_class
        labels[break_idx] = curr_switch_class

        curr_class = 0 if curr_class == 1 else 1
        curr_switch_class = 2 if curr_switch_class == 3 else 3
        prev_break_idx = break_idx + 1

    for i in range(prev_break_idx, total_tokens):
        labels[i] = curr_class

    return labels

def get_closed_models_predictions(file_path: str) -> List[int]:
    """Get metrices for closed models predictions"""
    results = load_results_json(file_path)
    predictions = []
    for result in results:
        predictions.extend(result['labeled_array'])
    return predictions

def get_reasoning_prompt()-> ChatPromptTemplate:
    """Get reasoning prompt"""
    return ChatPromptTemplate.from_messages([
        ("system", PROMPT_REASONING_ANALYSIS["system"]),
        ("human", PROMPT_REASONING_ANALYSIS["human"])
    ])

def get_true_predictions(labels_array: str) -> dict:
    """Get true predictions from string array"""
    pred = np.array([int(x) for x in labels_array.split(",")])
    first_segment = 'auto' if pred[0] == 0 else 'allo'
    return {"real_first_segment": first_segment, "real_predictions": np.where((pred == 2) | (pred == 3))[0].tolist()}

def get_understanding_switches_prompt()-> ChatPromptTemplate:
    """Get understanding predictions prompt"""
    return ChatPromptTemplate.from_messages([
        ("system", PROMT_UNDERSTANDING_LABELS["system"]),
        ("human", PROMT_UNDERSTANDING_LABELS["human"])
    ])