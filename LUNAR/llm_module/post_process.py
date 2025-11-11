import re
import string
import pandas as pd
import regex as re

param_regex = [r'{([ :_#.\-\w\d]+)}', r'{}']


def process_string(s):
    # Tokenize by whitespace
    tokens = s.split()
    
    # Process each token
    processed_tokens = []
    for token in tokens:
        # Keep track of where we've processed up to
        search_start = 0
        
        # Process each '<*>' sequentially
        while True:
            # Find next '<*>' starting from search_start
            marker = '<*>'
            pos = token.find(marker, search_start)
            
            if pos == -1:
                # No more '<*>' found
                break
            
            # Get the part before '<*>'
            before = token[:pos]
            
            # Get the part after '<*>'
            after = token[pos + 3:]  # Skip '<*>' (3 characters)
            
            end_chars = ('(', ')', '[', ']', ',', '.', ':', '<', '>', '#', '$', '/')
            first_end_pos = -1
            
            for i, char in enumerate(after):
                if char in end_chars:
                    first_end_pos = i
                    break
            
            # Build the processed token
            if first_end_pos >= 0:
                # Found an end character, keep it and everything after
                token = before + marker + after[first_end_pos:]
            else:
                # No end character found, remove everything after '<*>'
                token = before + marker
            
            # Move search_start past the current '<*>' we just processed
            search_start = pos + 3
        
        processed_tokens.append(token)
    
    # Concatenate tokens back with spaces
    return ' '.join(processed_tokens)
    ## Tokenize by whitespace
    #tokens = s.split()

    ## Process each token
    #processed_tokens = []
    #for token in tokens:
    #    if len(re.findall(r"<\*>", token)) == 1:
    #    #if '<*>' in token:
    #        # Find position of '<*>'
    #        idx = token.index('<*>')

    #        # Get the part before '<*>'
    #        before = token[:idx]

    #        # Get the part after '<*>'
    #        after = token[idx + 3:]  # Skip '<*>' (3 characters)

    #        # Find the first occurrence of ), ], , or .
    #        end_chars = ('(', ')', '[', ']', ',', '.', ':', '<', '>', '#', '$', '/')
    #        first_end_pos = -1

    #        for i, char in enumerate(after):
    #            if char in end_chars:
    #                first_end_pos = i
    #                break

    #        # Build the processed token
    #        if first_end_pos >= 0:
    #            # Found an end character, keep it and everything after
    #            processed_token = before + '<*>' + after[first_end_pos:]
    #        else:
    #            # No end character found, remove everything after '<*>'
    #            processed_token = before + '<*>'

    #        processed_tokens.append(processed_token)
    #    else:
    #        # Token doesn't contain '<*>', keep as is
    #        processed_tokens.append(token)

    ## Concatenate tokens back with spaces
    #return ' '.join(processed_tokens)


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean) # we don't use this
    US (User String) # we don't use this
    DG (Digit)
    HEX (Hex Variables)
    PS (Path-like String) # we don't use this
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    # boolean = {}
    # default_strings = {}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:', r'\=', r'\|', r'\"', r'\'', r'\[',
        r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    # if user_strings:
    # default_strings = default_strings.union(user_strings)

    # apply DS
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # apply PS
    # p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    # new_p_tokens = []
    # for p_token in p_tokens:
    # if re.match(r'^(\/[^\/]+)+$', p_token):
    # p_token = '<*>'
    # new_p_tokens.append(p_token)
    # template = ''.join(new_p_tokens)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')',
                      template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        # for to_replace in boolean.union(default_strings):
        # if token.lower() == to_replace.lower():
        # token = '<*>'

        # apply DG
        if re.match(r'^\d+$', token):
            token = '<*>'

        # newly added by me
        # apply Hex
        if re.match(r'0x[0-9a-fA-F]+', token):
            token = '<*>'
        while "0x<*>" in token:
            token = token.replace("0x<*>", "<*>")

        # apply WV
        if re.match(r'^[^\s\/]+<\*>[^\s\/]+$', token):
            if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                token = '<*>'

        # collect the result
        new_tokens.append(token)
    # make the template using new_tokens
    template = ''.join(new_tokens)
    # debug - lezhang.thu - start
    template = process_string(template)
    # debug - lezhang.thu - end

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    #print("CV: ", template)
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break
    #print("CV: ", template)

    while " #<*># " in template:
        template = template.replace(" #<*># ", " <*> ")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')

    while " '<*>' " in template:
        template = template.replace(" '<*>' ", " <*> ")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    # newly added by me
    # debug
    if False:
        while " <*>. " in template:
            template = template.replace(" <*>. ", " <*> ")
        while " <*>, " in template:
            template = template.replace(" <*>, ", " <*> ")
    # debug
    # lezhang.thu
    while "( <*>, <*>)" in template:
        template = template.replace("( <*>, <*>)", "(<*>, <*>)")

    while "<*>+<*>" in template:
        template = template.replace("<*>+<*>", "<*>")
    while "<*>##<*>" in template:
        template = template.replace("<*>##<*>", "<*>")
    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")
    while "<*>-<*>" in template:
        template = template.replace("<*>-<*>", "<*>")
    while " <*> <*> " in template:
        template = template.replace(" <*> <*> ", " <*> ")
    while template.endswith(" <*> <*>"):
        template = template[:-8] + " <*>"
    while template.startswith("<*> <*> "):
        template = "<*> " + template[8:]

    # newly added by me
    while "<*>,<*>" in template:
        template = template.replace("<*>,<*>", "<*>")
    while "(<*> <*>)" in template:
        template = template.replace("(<*> <*>)", "(<*>)")
    while " /<*> " in template:
        template = template.replace(" /<*> ", " <*> ")
    if template.endswith(" /<*>"):
        template = template[:-5] + " <*>"

    # Attribute key-value pair
    if template.count("=<*>") >= 3:
        template = template.replace("= ", "=<*> ")
    return template


def post_process_template(template, regs_common):
    if False:
        # template = re.sub(r'\{(\w+)\}', "<*>", template)
        # print("Be:", template)
        template = re.sub(r'\{\{(.+?)\}\}', "<*>", template)
        # print("0:", template)
        template = re.sub(r'\{(.*?)\}', "<*>", template)
        # print("1:", template)
        for reg in regs_common:
            template = reg.sub("<*>", template)
    else:
        template = re.sub(r'\{[A-Za-z0-9_-]+\}', '<*>', template)
    # print("2:", template)
    # debug
    template = correct_single_template(template)
    # print("3:", template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            print(f"\tPost Template: `{template}`")
            return template, True
    print("Get a too general template. Error.")
    return "", False


def replace_hex_with_placeholder(string):
    hex_pattern = r'0x[0-9a-fA-F]+'
    replaced_string = re.sub(hex_pattern, '<*>', string)
    return replaced_string


def extract_markdown_tables(markdown_text):
    """
    Regex pattern for matching markdown tables
    """
    # This pattern looks for lines starting and ending with a pipe symbol, and containing at least one dash-separated line
    table_pattern = r"(\|.*\|\s*\n\|[-| :]+\|.*\n(?:\|.*\|\n?)*)"

    # Find all matches in the markdown text
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)

    return tables


def markdown_table_to_dataframe(markdown_table):
    """
    Convert a markdown table to a pandas DataFrame

    Parameters
    ----------
    markdown_table : str
        Markdown table (with irrelevant content removed) to convert

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the data from the markdown table
    """
    # Split the table into lines
    lines = markdown_table.strip().split("\n")

    # Extract headers
    headers = lines[0].split("|")[1:-1]  # Remove the outer empty strings
    headers = [header.strip() for header in headers]

    # Extract rows
    rows = []
    for line in lines[2:]:  # Skip the first two lines (headers and dashes)
        row = line.split("|")[1:-1]  # Remove the outer empty strings
        row = [cell.strip() for cell in row]
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df
