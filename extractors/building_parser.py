import pdfplumber
from collections import defaultdict


def extract_table_like_data(pdf_path, x_threshold=150):
    result = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1, y_tolerance=2)
            lines_by_y = defaultdict(list)

            for word in words:
                y_center = round((word['top'] + word['bottom']) / 2, 1)
                lines_by_y[y_center].append(word)

            for line in lines_by_y.values():
                line = sorted(line, key=lambda w: w['x0'])
                key_words = [w['text'] for w in line if w['x1'] < x_threshold]
                value_words = [w['text']
                               for w in line if w['x0'] >= x_threshold]
                if key_words and value_words:
                    key = " ".join(key_words)
                    value = " ".join(value_words)
                    result[key] = value
    return result
