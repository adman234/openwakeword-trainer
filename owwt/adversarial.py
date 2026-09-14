"""Phrases that sound like the wake word but are not it, used as hard negative examples.

Similar words come from the CMU pronouncing dictionary: words whose phoneme sequence is within one or
two edits of each word in the wake phrase (exact homophones are skipped, since those really would
sound like the wake word).
"""
import difflib
import functools
import random
import re
from collections import defaultdict

GENERIC_PHRASES = [
    "what time is it", "turn on the lights", "can you hear me", "i will be there in a minute",
    "did you feed the cat", "where are my keys", "let's watch a movie tonight", "that sounds good to me",
    "please close the door", "how was your day", "i'm going to the store", "set a timer for ten minutes",
    "it's really cold outside", "who wants pizza", "turn the music down", "call me when you get home",
    "have you seen my phone", "the weather looks nice today", "i need a cup of coffee", "good morning everyone",
    "see you later", "thank you very much", "what are we having for dinner", "can you help me with this",
    "the kids are asleep", "don't forget your jacket", "is the oven still on", "let me think about it",
    "we should go for a walk", "turn off the tv", "open the window please", "hello how are you",
    "hey there", "okay sounds great", "just a second", "are you ready to go", "i'm not sure about that",
    "play some music", "what's the news today", "the package arrived", "nice to meet you",
    "it's time for bed", "wash your hands", "i love this song", "how much does it cost",
    "where did you put it", "that was delicious", "can we talk later", "my battery is low",
    "the meeting starts at nine", "turn up the heat", "did you lock the car", "happy birthday",
    "i'll call you back", "what's on the calendar", "the dog needs to go out", "hey what's up",
    "no thanks i'm good", "check the mail", "who is at the door",
]


# The pronouncing dictionary includes words nobody wants to see suggested as a recording prompt
HIDDEN_WORDS = {"gay", "gays", "fag", "fags", "negro", "nigga", "nigger", "slut", "whore", "cunt", "fuck", "shit",
                "bitch", "dick", "cock", "piss", "tits", "rape", "retard", "spic", "kike", "chink", "dyke", "homo"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", text.lower())).strip()


@functools.lru_cache(maxsize=1)
def _dictionary():
    import pronouncing

    pronouncing.init_cmu()
    by_length = defaultdict(list)
    phones_by_word = {}
    for word, phones in pronouncing.pronunciations:
        if len(word) < 2 or not re.fullmatch(r"[a-z][a-z']*", word):
            continue
        stripped = tuple(re.sub(r"\d", "", p) for p in phones.split())
        if word not in phones_by_word:
            phones_by_word[word] = stripped
            by_length[len(stripped)].append((word, stripped))
    return by_length, phones_by_word


def _distance(a: tuple, b: tuple, limit: int) -> int:
    """Levenshtein distance that gives up early once it exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = i
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            best = min(best, cur[j])
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


@functools.lru_cache(maxsize=512)
def similar_words(word: str, limit: int = 60) -> tuple:
    by_length, phones_by_word = _dictionary()
    word = word.lower()
    target = phones_by_word.get(word)
    spelled_alike = []
    if target is None:
        # Not in the dictionary (names, made-up words): borrow the pronunciation of the closest spelling
        spelled_alike = difflib.get_close_matches(word, list(phones_by_word), n=12, cutoff=0.6)
        if not spelled_alike:
            return ()
        target = phones_by_word[spelled_alike[0]]

    max_edits = 1 if len(target) <= 3 else 2
    found = []
    for n in range(len(target) - max_edits, len(target) + max_edits + 1):
        for candidate, phones in by_length.get(n, ()):
            if candidate == word or phones == target or candidate in HIDDEN_WORDS:
                continue
            d = _distance(target, phones, max_edits)
            if d <= max_edits:
                found.append((d, abs(len(candidate) - len(word)), candidate))
    found.sort()
    words = [w for w in spelled_alike if w != word] + [w for _, _, w in found]
    return tuple(dict.fromkeys(words))[:limit]


def negative_phrases(phrase: str, n: int, custom=(), seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    words = normalize(phrase).split()
    target = " ".join(words)
    similar = [list(similar_words(w)[:40]) for w in words]
    partials = [" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, len(words) + 1)
                if j - i < len(words)]
    custom = [c.strip() for c in custom if c and c.strip()]

    out = []
    attempts = 0
    while len(out) < n and attempts < n * 5:
        attempts += 1
        r = rng.random()
        if partials and r < 0.15:
            text = rng.choice(partials)
        elif custom and r < 0.30:
            text = rng.choice(custom)
        elif r < 0.40 or not any(similar):
            text = rng.choice(GENERIC_PHRASES)
        else:
            replace = set(rng.sample(range(len(words)), rng.randint(1, len(words))))
            parts = [rng.choice(similar[i]) if i in replace and similar[i] else w for i, w in enumerate(words)]
            if len(parts) > 1 and rng.random() < 0.2:
                i = rng.randrange(len(parts))
                parts = parts[i:rng.randrange(i + 1, len(parts) + 1)]
            text = " ".join(parts)
        if normalize(text) != target:
            out.append(text)
    return out


def suggestions(phrase: str, custom=(), k: int = 24) -> list[str]:
    """Readable similar-sounding phrases to show as recording prompts: one word swapped at a time."""
    rng = random.Random(1)
    words = normalize(phrase).split()
    picks = [c.strip() for c in custom if c and c.strip()]
    if len(words) > 1:
        picks += words
    options = []
    for i, word in enumerate(words):
        for alt in similar_words(word)[:12]:
            options.append(" ".join(words[:i] + [alt] + words[i + 1:]))
    rng.shuffle(options)
    picks += options
    target = " ".join(words)
    return [p for p in dict.fromkeys(picks) if normalize(p) != target][:k]
