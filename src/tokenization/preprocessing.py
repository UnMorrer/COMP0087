# Previous work: ner_experimentation.py
# Script that pre-processes AI and human-generated text in the same way

import re


def human_text_preprocessing(text):
    """
    Function to pre-process human essays. This re-codes a number
    of masks used in the original dataset, as outlined below:
    # @MONTH + @DATE + @TIME -> @TIME
    # @CAPS + @PERSON + @DR -> @PER (match NER label)
    # @CITY + @STATE + @LOCATION -> @LOC (match NER label)
    # @ORGANIZATION -> @ORG (match NER label)
    # @MONEY + @PERCENT + @NUM -> @NUM
    # @EMAIL -> @EMAIL

    Unique masks in human-written essays:
    '@MONTH', '@CAPS', '@MONEY', '@PERSON', '@PERCENT', '@STATE', '@LOCATION',
    '@NUM', '@DATE', '@EMAIL', '@CITY', '@DR', '@ORGANIZATION', '@TIME'

    Inputs:
    Text - str: The human-written essay text

    Returns:
    Cleaned_Text - str: The human-written essay text
    (after applying pre-processing outlined above)
    """
    # Applying above rules, one-by-one
    cleaned_text = re.sub(r"/b@MONTH/d*/b", "@TIME", text)
    cleaned_text = re.sub(r"/b@DATE/d*/b", "@TIME", cleaned_text)
    cleaned_text = re.sub(r"/b@TIME/d*/b", "@TIME", cleaned_text)
    cleaned_text = re.sub(r"/b@CAPS/d*/b", "@PER", cleaned_text)
    cleaned_text = re.sub(r"/b@PERSON/d*/b", "@PER", cleaned_text)
    cleaned_text = re.sub(r"/b@DR/d*/b", "@PER", cleaned_text)
    cleaned_text = re.sub(r"/b@CITY/d*/b", "@LOC", cleaned_text)
    cleaned_text = re.sub(r"/b@STATE/d*/b", "@LOC", cleaned_text)
    cleaned_text = re.sub(r"/b@LOCATION/d*/b", "@LOC", cleaned_text)
    cleaned_text = re.sub(r"/b@ORGANIZATION/d*/b", "@ORG", cleaned_text)
    cleaned_text = re.sub(r"/b@MONEY/d*/b", "@NUM", cleaned_text)
    cleaned_text = re.sub(r"/b@PERCENT/d*/b", "@NUM", cleaned_text)
    cleaned_text = re.sub(r"/b@NUM/d*/b", "@NUM", cleaned_text)
    cleaned_text = re.sub(r"/b@EMAIL/d*/b", "@EMAIL", cleaned_text)
    
    return cleaned_text


# AI-generated essays
# This extracts: LOCation, MISCellanous, ORGanization, PERson
# We are missing: Time/Month, Money, Number, Dr, Percent
# Above: Dr. XY is identified as PERSON

# Plan: 
# Month/Time/Date -> unique format, can be found
# Money: Currency (dollar/$...) -> can be found
# Number and percent (digits%) -> unique format, can be found
# Email address (..@.. .com) -> unique format, can be found
# +1 strip newline (\n) characters - NOT present in human essays
def generated_text_preprocessing(text):
    pass