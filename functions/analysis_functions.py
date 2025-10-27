import regex

def keyword_match(df, column, keyword):
    pattern = rf'\b{regex.escape(keyword)}s?\b'
    matched_comments = df[df[column].str.contains(pattern, flags=regex.IGNORECASE, regex=True, na=False)]
    n = len(matched_comments)
    print(f"Number of comments mentioning '{keyword}': {n}")
    return matched_comments