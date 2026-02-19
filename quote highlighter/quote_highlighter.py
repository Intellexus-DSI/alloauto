#!/usr/bin/env python3
import re
import html

INPUT_PATH = "./D2672.txt"  # Change to your input file path
OUTPUT_PATH = "./D2672_quotes.html"  # Change to your output file path
MAX_QUOTE_LENGTH_CHARS = 500

potential_openers = [
    "las kyang ji skad du", "las kyang", "las", "ji skad du", "las 'byung ba", "las ji skad du",
    "de bzhin du", "nas", "de'i phyir"
]
# sort them by decreasing length
potential_openers.sort(key=len, reverse=True)
potential_closers = [
    "zhes gsungs so", "zhes ji skad gsungs pa lta bu'o", "zhes 'byung ngo", "zhes gsungs pa'i phyir ro",
    "ces gsungs so", "zhes gsungs te", "zhes gsungs pa", "zhes gsungs pa yin no",
    "zhes gsungs pa lta bu'o", "gsungs so", "zhes bya ba dang", "ces gsungs te",
    "ces gsungs pa'i phyir ro", "zhes bya ba gsungs so", "zhes bshad do", "ces gsungs pa lta bu'o",
    "zhes pa", "zhes gsungs pa nyid blta bar bya'o", "zhes gsungs pa'o", "zhes bshad pa'i phyir ro",
    "ces gsungs pa", "zhes bshad pa lta bu'o", "ces bya ba gsungs te", "zhes gsungs",
    "zhe'o", "zhes rgya cher gsungs so", "gsungs pa lta bu'o", "zhes rgya cher gsungs pa lta bu'o"
]
# sort them by decreasing length
potential_closers.sort(key=len, reverse=True)
# add "zhes(?! bya ba)" to the end of the closers
# we do this so the zhes with negative lookahead matches last
potential_closers.append("zhes(?! bya ba)")

opener_regex ="(" + "|".join(re.escape(opener) for opener in potential_openers) + ")"
closer_regex = "(" + "|".join(re.escape(closer) for closer in potential_closers) + ")"
print("Opener regex is :", opener_regex.replace("\\", ""))
print("Closer regex is :", closer_regex.replace("\\", ""))

def split_clauses(text):
    return [clause.strip() for clause in re.split(r"\s*/{1,2}\s*", text) if clause.strip()]

def detect_quotes(clauses):
    labels = ["none"] * len(clauses)
    i = 0
    while i < len(clauses):
        clause = clauses[i]
        # Check if the clause starts with a potential opener
        match_opener = re.search(opener_regex, clause)
        if match_opener:
            for j in range(i + 1, len(clauses)):
                match_closer = re.search(closer_regex, clauses[j])
                if match_closer:
                    #print("\nmatch opener group 0 is :", match_opener.group(0))
                    #print("match closer group 0 is :", match_closer.group(0))
                    # if match_opener only matched "las" or "nas", and match_closer
                    # only matched "zhes", skip this one
                    if (match_opener.group(0) in ["las", "nas"] and match_closer.group(0) == "zhes"):
                        print("Skipping ambiguous syllable case...")
                        continue

                    quote_content = " ".join(clauses[i+1:j])
                    if len(quote_content) <= MAX_QUOTE_LENGTH_CHARS:
                        labels[i] = "opener"
                        for k in range(i + 1, j):
                            labels[k] = "inquote"
                        labels[j] = "closer"
                        i = j  # jump forward
                    break
        i += 1
    return labels

def render_html(clauses, labels):
    color_map = {
        "opener": "#d0f0fd",
        "inquote": "#fef7cb",
        "closer": "#ffd0d0",
        "none": "white"
    }
    html_lines = []
    html_lines.append("<html><head><meta charset='utf-8'><style>")
    html_lines.append("body { font-family: monospace; }")
    html_lines.append(".heatmap { display: flex; flex-wrap: wrap; margin-bottom: 1em; }")
    html_lines.append(".heatbox { width: 10px; height: 10px; margin: 1px; border: 1px solid #ccc; }")
    for label, color in color_map.items():
        html_lines.append(f".{label} {{ background-color: {color}; }}")
    html_lines.append("span.opener, span.inquote, span.closer { padding: 0 2px; border-radius: 3px; }")
    html_lines.append("</style></head><body>")
    html_lines.append("<h2>Quote Detection Heatmap</h2><div class='heatmap'>")

    # Heatmap
    for idx, lbl in enumerate(labels):
        html_lines.append(f"<a href='#line-{idx:04d}'><div class='heatbox {lbl}'></div></a>")
    html_lines.append("</div><hr>")

    # Full text with inline highlights
    for i, (clause, label) in enumerate(zip(clauses, labels)):
        clause_html = html.escape(clause)

        # highlight substrings inside clause
        highlighted = clause_html
        if label == "opener":
            match = re.search(opener_regex, clause)
            if match:
                start, end = match.span()
                escaped = html.escape(clause)
                highlighted = (
                    html.escape(clause[:start]) +
                    f"<span class='opener'>{html.escape(clause[start:end])}</span>" +
                    html.escape(clause[end:])
                )
        elif label == "closer":
            match = re.search(closer_regex, clause)
            if match:
                start, end = match.span()
                highlighted = (
                    html.escape(clause[:start]) +
                    f"<span class='closer'>{html.escape(clause[start:end])}</span>" +
                    html.escape(clause[end:])
                )
        elif label == "inquote":
            highlighted = f"<span class='inquote'>{clause_html}</span>"

        html_lines.append(f"<div id='line-{i:04d}'><strong>{i:04d}:</strong> {highlighted}</div>")


    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        full_text = f.read()
    full_text = full_text.replace("\n", " ").replace("\r", " ")

    clauses = split_clauses(full_text)
    labels = detect_quotes(clauses)
    html_output = render_html(clauses, labels)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.write(html_output)

    print(f"✅ Done. Output written to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
