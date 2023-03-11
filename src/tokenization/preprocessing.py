# Previous work: ner_experimentation.py
# Script that pre-processes AI and human-generated text in the same way

# Common libraries
import re
import torch
import dateutil.parser as date_parser # https://github.com/dateutil/dateutil
import transformers

# Custom packages

def human_text_preprocessor(text):
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
    cleaned_text = re.sub(r"@MONTH\d*", "@TIME", text)
    cleaned_text = re.sub(r"@DATE\d*", "@TIME", cleaned_text)
    cleaned_text = re.sub(r"@TIME\d*", "@TIME", cleaned_text)
    cleaned_text = re.sub(r"@CAPS\d*", "@PER", cleaned_text)
    cleaned_text = re.sub(r"@PERSON\d*", "@PER", cleaned_text)
    cleaned_text = re.sub(r"@DR\d*", "@PER", cleaned_text)
    cleaned_text = re.sub(r"@CITY\d*", "@LOC", cleaned_text)
    cleaned_text = re.sub(r"@STATE\d*", "@LOC", cleaned_text)
    cleaned_text = re.sub(r"@LOCATION\d*", "@LOC", cleaned_text)
    cleaned_text = re.sub(r"@ORGANIZATION\d*", "@ORG", cleaned_text)
    cleaned_text = re.sub(r"@MONEY\d*", "@NUM", cleaned_text)
    cleaned_text = re.sub(r"@PERCENT\d*", "@NUM", cleaned_text)
    cleaned_text = re.sub(r"@NUM\d*", "@NUM", cleaned_text)
    cleaned_text = re.sub(r"@EMAIL\d*", "@EMAIL", cleaned_text)
    
    return cleaned_text


def ai_text_preprocessor(text, tokenizer, model, conversion):
    """
    Function to pre-process Ai-written essays. This extracts
    a number of useful masks, based on Named Entity Recognition (NER)
    models:
    LOCation, MISCellanous, ORGanization, PERson
    HuggingFace Model card: https://huggingface.co/dslim/bert-base-NER
    
    NOTE - compared to human-written text, we are missing:
    Time/Month, Money, Number, Dr, Percent masks

    Month/Time/Date -> Saw some manually, but no models to tag this

    Money/Number/Percent -> NO numbers in AI-generated training data!
    numbers = re.findall(r"/d*", text)
    Also, no percentage (%) sign in generated data

    Email address (..@.. .com) -> unique format, can be found
    emails = re.findall(r"\b\w+@\w+\.com\b", text)
    -> NO email addresses in generated data!

    Remove newline (\n) characters - NOT present in human essays

    Replace [Your Name] etc. prompts with proper masking
    List of prompts:
    [Name of Newspaper], [Your Name], [Name], [NAME], [name],
    [YOUR NAME], [your name], [Insert Newspaper Name], [Editor]

    Inputs:
    Text - str: The AI-written essay text
    tokenizer - transformers tokenizer
    model - transformers model
    -> Above 2 used to derive NER predictions
    conversion - dict: Entity token conversions from model

    Returns:
    Cleaned_Text - str: The AI-written essay text
    (after applying pre-processing outlined above)
    """

    cleaned_text = text.replace("\n", " ")
    # replace multiple spaces with a single space
    cleaned_text = re.sub(" +", " ", cleaned_text)

    # Find named entities using ML
    ml = transformers.pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=0 if torch.cuda.is_available() else -1 #Enable GPU support
    )
    results = ml(cleaned_text)

    # Replace entities found in reverse order
    for ne in reversed(results):
        start_char = ne["start"]
        end_char = ne["end"]
        entity_token = conversion[ne["entity"]]
        cleaned_text = cleaned_text[:start_char] + entity_token + cleaned_text[end_char:]

    # Replace [Your name] and other prompts with mask
    newspaper_regex = re.compile(r"\[.*newspaper.*\]", re.IGNORECASE)
    name_regex = re.compile(r"\[.*name\]", re.IGNORECASE)
    editor_regex = re.compile(r"\[editor\]", re.IGNORECASE)
    
    cleaned_text = re.sub(newspaper_regex, "@ORG", cleaned_text)
    cleaned_text = re.sub(name_regex, "@PER", cleaned_text)
    cleaned_text = re.sub(editor_regex, "@PER", cleaned_text)

    return cleaned_text