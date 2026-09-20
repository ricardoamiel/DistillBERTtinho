"""Vocabulary used by the interactive latent space demo.

The words are grouped by topic only so that the scatter plot can be coloured.
The grouping is never given to the models: it is the ground truth we check the
unsupervised geometry against.
"""

WORD_GROUPS = {
    "animals": ["dog", "cat", "horse", "elephant", "tiger", "lion", "wolf", "bear", "rabbit",
                "mouse", "eagle", "shark", "whale", "dolphin", "snake", "frog", "spider",
                "bee", "sheep", "cow", "pig", "chicken", "monkey", "fox", "deer"],
    "countries": ["france", "germany", "spain", "italy", "japan", "china", "india", "brazil",
                  "canada", "mexico", "russia", "egypt", "kenya", "peru", "chile", "argentina",
                  "norway", "sweden", "poland", "greece", "turkey", "vietnam", "thailand",
                  "australia", "portugal"],
    "food": ["bread", "cheese", "rice", "pasta", "soup", "salad", "chicken", "beef", "fish",
             "apple", "banana", "orange", "grape", "potato", "tomato", "onion", "garlic",
             "sugar", "salt", "pepper", "coffee", "tea", "milk", "butter", "chocolate"],
    "sports": ["football", "basketball", "tennis", "baseball", "soccer", "hockey", "golf",
               "boxing", "swimming", "running", "cycling", "skiing", "surfing", "rugby",
               "cricket", "volleyball", "wrestling", "marathon", "olympics", "championship",
               "tournament", "stadium", "coach", "referee", "athlete"],
    "technology": ["computer", "software", "hardware", "internet", "network", "server",
                   "database", "algorithm", "processor", "memory", "keyboard", "screen",
                   "laptop", "smartphone", "browser", "encryption", "robot", "satellite",
                   "semiconductor", "bandwidth", "compiler", "protocol", "firmware",
                   "cloud", "chip"],
    "finance": ["bank", "money", "profit", "loss", "revenue", "investment", "stock", "bond",
                "market", "trader", "economy", "inflation", "interest", "loan", "credit",
                "debt", "budget", "tax", "merger", "shareholder", "dividend", "currency",
                "recession", "startup", "auction"],
    "science": ["physics", "chemistry", "biology", "astronomy", "geology", "experiment",
                "theory", "hypothesis", "molecule", "atom", "gene", "cell", "protein",
                "gravity", "energy", "particle", "telescope", "microscope", "laboratory",
                "research", "evolution", "quantum", "enzyme", "vaccine", "orbit"],
    "emotions": ["happy", "sad", "angry", "afraid", "excited", "bored", "calm", "nervous",
                 "proud", "ashamed", "jealous", "grateful", "lonely", "hopeful", "anxious",
                 "joyful", "miserable", "delighted", "furious", "terrified", "content",
                 "frustrated", "relieved", "disappointed", "surprised"],
    "family": ["mother", "father", "sister", "brother", "daughter", "son", "grandmother",
               "grandfather", "aunt", "uncle", "cousin", "nephew", "niece", "wife", "husband",
               "parent", "child", "baby", "twin", "family", "marriage", "wedding", "divorce",
               "birth", "relative"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black",
               "white", "grey", "violet", "crimson", "scarlet", "turquoise", "indigo",
               "beige", "golden", "silver", "bronze", "emerald", "navy", "maroon", "amber",
               "ivory"],
    "vehicles": ["car", "truck", "bus", "train", "airplane", "helicopter", "boat", "ship",
                 "bicycle", "motorcycle", "tractor", "submarine", "rocket", "taxi", "ferry",
                 "ambulance", "van", "scooter", "yacht", "jet", "engine", "wheel", "driver",
                 "pilot", "garage"],
    "music": ["guitar", "piano", "violin", "drum", "flute", "trumpet", "saxophone", "cello",
              "orchestra", "concert", "melody", "rhythm", "harmony", "singer", "band",
              "album", "song", "jazz", "rock", "classical", "opera", "chorus", "composer",
              "studio", "tune"],
    "medicine": ["doctor", "nurse", "hospital", "patient", "surgery", "diagnosis", "treatment",
                 "medicine", "disease", "infection", "virus", "bacteria", "antibiotic",
                 "therapy", "clinic", "symptom", "fever", "injury", "cancer", "diabetes",
                 "blood", "heart", "lung", "brain", "bone"],
    "weather": ["rain", "snow", "storm", "wind", "cloud", "sunshine", "thunder", "lightning",
                "fog", "hail", "hurricane", "tornado", "drought", "flood", "temperature",
                "humidity", "forecast", "climate", "season", "winter", "summer", "spring",
                "autumn", "frost", "breeze"],
    "clothing": ["shirt", "trousers", "dress", "skirt", "jacket", "coat", "sweater", "shoes",
                 "boots", "hat", "scarf", "gloves", "socks", "belt", "tie", "suit", "jeans",
                 "uniform", "fabric", "cotton", "wool", "silk", "leather", "button", "zipper"],
    "jobs": ["teacher", "engineer", "lawyer", "farmer", "chef", "writer", "artist", "scientist",
             "manager", "accountant", "architect", "journalist", "soldier", "police",
             "firefighter", "carpenter", "plumber", "electrician", "designer", "programmer",
             "salesman", "cashier", "waiter", "librarian", "translator"],
    "politics": ["government", "president", "minister", "parliament", "election", "vote",
                 "democracy", "republic", "senator", "congress", "policy", "law", "treaty",
                 "diplomat", "embassy", "sanction", "protest", "campaign", "candidate",
                 "party", "coalition", "referendum", "constitution", "border", "war"],
    "education": ["school", "university", "student", "teacher", "lecture", "exam", "homework",
                  "degree", "diploma", "classroom", "library", "textbook", "curriculum",
                  "semester", "scholarship", "graduate", "professor", "thesis", "seminar",
                  "tutor", "grade", "course", "campus", "faculty", "dormitory"],
    "motion_verbs": ["run", "walk", "jump", "swim", "fly", "climb", "crawl", "drive", "ride",
                     "sail", "march", "sprint", "dance", "slide", "roll", "push", "pull",
                     "throw", "catch", "carry", "lift", "drop", "chase", "escape", "arrive"],
    "time": ["morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
             "week", "month", "year", "decade", "century", "hour", "minute", "second",
             "monday", "friday", "january", "december", "weekend", "holiday", "birthday",
             "anniversary", "deadline", "schedule"],
}


def flat_words() -> list[tuple[str, str]]:
    """Return (word, group) pairs with duplicates removed, first group wins."""
    seen, out = set(), []
    for group, words in WORD_GROUPS.items():
        for word in words:
            if word not in seen:
                seen.add(word)
                out.append((word, group))
    return out
