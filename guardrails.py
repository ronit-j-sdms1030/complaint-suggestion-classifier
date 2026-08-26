"""
Pre-model guardrail for the PMC complaint/suggestion classifier.

The fine-tuned MuRIL model is binary (complaint vs. suggestion) -- it was never trained
with a third option, so it will happily force a confident label onto input that isn't a
civic complaint or suggestion at all ("who are you?", "hi", "test", random gibberish).
This module catches that class of input *before* it reaches the model and labels it
"junk" instead of guessing.

Usage in a production inference pipeline:
    from guardrails import is_offtopic
    if is_offtopic(text):
        final_label = "junk"
    else:
        final_label = <run the real classifier>

Design: any recognizable civic-topic vocabulary in the text immediately exempts it from
being flagged junk, regardless of length or accompanying small talk ("thanks, please also
fix the water pipe" stays real) -- civic keyword presence is checked first and short-
circuits to False. Only text with NO civic vocabulary at all is evaluated against the
junk signals (exact greeting/meta-chat phrases, no alphabetic characters, or too few words
to plausibly be a real complaint/suggestion).

Known limitation: this is a rule-based prefilter, not a learned classifier -- longer
gibberish with no civic keywords and no greeting-phrase match (e.g. random word salad)
will slip through and get forced into complaint/suggestion by the real model, same as
before. Catching that reliably would need a trained 3-class model, not a keyword filter.
"""

MIN_WORDS = 3

# Common greetings / meta-conversation / chit-chat that is not a civic complaint or
# suggestion at all, across the languages this classifier serves. Matched as substrings,
# but only ever consulted for text that has ALREADY failed the civic-keyword check below,
# so short real complaints with an incidental "thanks" or "ok" in them are never at risk.
OFFTOPIC_PHRASES = [
    # English
    "who are you", "what is this", "hello", "hi there", "hey there", "good morning",
    "good afternoon", "good evening", "how are you", "test", "testing", "ok", "okay",
    "thanks", "thank you", "are you a bot", "is this a bot", "what can you do",
    "just checking", "nothing", "no message",
    # Hindi (Devanagari)
    "आप कौन हैं", "तुम कौन हो", "यह क्या है", "नमस्ते", "कैसे हो", "क्या हाल है",
    "धन्यवाद", "टेस्ट",
    # Marathi (Devanagari)
    "तू कोण आहेस", "तुम्ही कोण आहात", "हे काय आहे", "नमस्कार", "कसे आहात",
    "धन्यवाद", "टेस्ट",
    # Romanized Hindi/Marathi
    "kaun ho tum", "tum kaun ho", "aap kaun ho", "tu kon ahes", "tumhi kon ahat",
    "kay ahe he", "kasa ahes", "kase ahat", "kay chalay", "test", "testing",
    "thank you", "thanks", "dhanyawad", "namaste", "namaskar",
]

# Broad civic-topic vocabulary -- presence of ANY of these exempts the text from being
# flagged junk. Deliberately broad (topics + action/complaint words + PMC references),
# since a false "junk" flag on a real complaint is a worse error than letting a genuine
# edge case through to the real classifier.
CIVIC_KEYWORDS = [
    # English
    "road", "water", "light", "garbage", "drainage", "chamber", "tree", "park",
    "toilet", "parking", "footpath", "school", "hospital", "complaint", "request",
    "please", "repair", "install", "provide", "problem", "issue", "pmc", "municipal",
    "corporation", "pothole", "sewage", "drain", "streetlight", "divider", "breaker",
    "encroach", "bin", "dustbin", "cctv", "bench", "camera", "signal", "footbridge",
    "sidewalk", "sanitation", "electricity", "pipeline", "sewer", "footpath",
    # Hindi/Marathi (Devanagari) -- broad civic nouns
    "रस्ता", "पाणी", "पानी", "लाईट", "लाइट", "कचरा", "गटार", "चेंबर", "झाड", "उद्यान",
    "शौचालय", "पार्किंग", "फुटपाथ", "शाळा", "स्कूल", "रुग्णालय", "अस्पताल", "तक्रार",
    "विनंती", "अनुरोध", "समस्या", "दुरुस्ती", "गतिरोधक", "पालिका", "महानगरपालिका",
    "मनपा", "स्ट्रीट", "पथदिवा", "पथदिवे", "नाला", "नाली", "सांडपाणी", "बगीचा",
    # Romanized
    "rasta", "raste", "pani", "kachra", "jhad", "zad", "shala", "school",
    "tras", "trass", "vinanti", "vinanti", "samasya", "durusti", "gatirodhak",
    "pathdiva", "pathdive", "pmc", "mnapa", "mahapalika", "nala", "gutar",
]


def _word_count(text):
    return len(str(text).strip().split())


def _contains_any(text, phrases):
    tl = str(text).lower()
    return any(p.lower() in tl for p in phrases)


def _has_alpha(text):
    return any(ch.isalpha() for ch in str(text))


def is_offtopic(text):
    """
    Returns True if this text should be labeled "junk" instead of being forced through
    the binary complaint/suggestion classifier.
    """
    text = str(text).strip()

    if not text:
        return True

    if not _has_alpha(text):
        return True

    if _contains_any(text, CIVIC_KEYWORDS):
        return False

    if _contains_any(text, OFFTOPIC_PHRASES):
        return True

    if _word_count(text) < MIN_WORDS:
        return True

    return False


if __name__ == "__main__":
    cases = [
        # Should be flagged junk
        ("who are you?", True),
        ("Who are you", True),
        ("hi", True),
        ("hello good morning", True),
        ("test", True),
        ("thanks", True),
        ("", True),
        ("12345", True),
        ("???", True),
        ("asdkfj", True),
        ("aap kaun ho", True),
        ("tumhi kon ahat", True),
        ("नमस्कार", True),
        # Should NOT be flagged junk (real, if terse, complaints/suggestions)
        ("no water", False),
        ("पथदिवा बंद", False),
        ("road is bad", False),
        ("ok please fix the water problem in our society", False),
        ("thanks for fixing the road but the streetlight is still off", False),
        ("please install a cctv camera near the market", False),
        ("गटार तुंबले आहे", False),
    ]
    for text, expected in cases:
        result = is_offtopic(text)
        status = "OK" if result == expected else "MISMATCH"
        print(f"[{status}] expected={expected} got={result} | {text!r}")
