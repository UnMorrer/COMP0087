# Previous work: ner_experimentation.py
# Script that masks named entities in generated text

import transformers as trfs

# Example configuration
# ner = trfs.pipeline("ner", grouped_entities=True) # TODO: Specify model revision + framework
# resp = ner("This is a test text to see how much this encoding can handle. I started working for Dr. Hammerschmidt in February 2023 for a salary of £60,000 ($75,000). My e-mail address is hammerschmidt@gmail.com")

# This extracts: LOCation, MISCellanous, ORGanization, PERson
# We are missing: Time/Month, Money, Number, Dr, Percent
# Above: Dr. XY is identified as PERSON

# Plan: Month/Time -> unique format, can be found
# Money: Currency (dollar/$) -> can be found
# Number and percent (digits%) -> unique format, can be found
# Email address (..@.. .com) -> unique format, can be found

# Re-code human essays:
# @MONTH + @DATE + @TIME -> @TIME (or @MISC)
# @CAPS + @PERSON + @DR -> @PER (match NER label)
# @CITY + @STATE + @LOCATION -> @LOC (match NER label)
# @ORGANIZATION -> @ORG (match NER label)
# @MONEY + @PERCENT + @NUM -> @NUM
# @EMAIL -> @EMAIL
# Check: @MISC out of NER
# '@MONTH', '@CAPS', '@MONEY', '@PERSON', '@PERCENT', '@STATE', '@LOCATION', '@NUM', '@DATE', '@EMAIL', '@CITY', '@DR', '@ORGANIZATION', '@TIME'
