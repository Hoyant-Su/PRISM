import math
from scipy.stats import chi2
import fitz
import re

def fishers_method(p_values):
    X = -2 * sum([math.log(p) for p in p_values])
    df = 2 * len(p_values)
    p_combined = 1 - chi2.cdf(X, df)
    return p_combined


def extract_p_value_from_pdf(pdf_path):
    pattern = re.compile(r'P-value:\s*([-+]?\d*\.\d+|\d+)([eE][-+]?\d+)?')
    p_values = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        matches = pattern.findall(text)

        for match in matches:
            number_str = match[0] + (match[1] if match[1] else '')
            try:
                p_val = float(number_str)
                p_values.append(p_val)
            except ValueError:
                pass

    return p_values

import re
def parse_log_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pattern_max_c = re.compile(r'Max C-index & AUC: \(([\d\.]+),.*\)')
    max_c_index = None
    for line in reversed(lines):
        m = pattern_max_c.search(line)
        if m:
            max_c_index = float(m.group(1))
            break
    if max_c_index is None:
        print(txt_path)
        print("Max C-index not found")
        return None

    separator_line_pattern = re.compile(r'- ={10,}')
    separator_indices = [i for i, l in enumerate(lines) if separator_line_pattern.search(l)]

    blocks = []
    for i in range(len(separator_indices) - 1):
        start = separator_indices[i] + 1
        end = separator_indices[i + 1]
        block_lines = lines[start:end]
        blocks.append(block_lines)

    pattern_threshold = re.compile(r'Threshold for multicollinearity:\s*([0]*\.?[0-9]+)')

    pattern_penalizer = re.compile(r'Penalizer:([\d\.]+)')
    pattern_test_c = re.compile(r'Test C-Index: \(([\d\.]+),')

    results = []

    for block in blocks:
        threshold = None
        penalizer = None
        test_c = None

        for line in block:
            if threshold is None:
                m = pattern_threshold.search(line)
                if m:
                    threshold = float(m.group(1))
            if penalizer is None:
                m = pattern_penalizer.search(line)
                if m:
                    penalizer = float(m.group(1))
            if test_c is None:
                m = pattern_test_c.search(line)
                if m:
                    test_c = float(m.group(1))
            if threshold is not None and penalizer is not None and test_c is not None:
                break

        if threshold is None or penalizer is None or test_c is None:
            print(threshold)
            print(penalizer)
            print(test_c)
            print('--------')
            continue

        results.append({
            "threshold": threshold,
            "penalizer": penalizer,
            "test_c_index": test_c,
            "block": block,
        })

    tolerance = 1e-6
    matched = None
    for r in results:
        if abs(r["test_c_index"] - max_c_index) < tolerance:
            matched = r
            break

    if matched is None:
        print("No Test C-Index matching the Max C-index was found")
