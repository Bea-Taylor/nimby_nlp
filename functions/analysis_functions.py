import regex

class AnalysisFunctions:

    def __init__(self):
        pass

    def keyword_match(df, column, keyword):
        pattern = rf'\b{regex.escape(keyword)}s?\b'
        matched_comments = df[df[column].str.contains(pattern, flags=regex.IGNORECASE, regex=True, na=False)]
        n = len(matched_comments)

        print(f"Number of comments mentioning '{keyword}': {n}")
        print("Percent of comments mentioning '{keyword}': {:.2f}%".format((n / len(df)) * 100))

        return matched_comments