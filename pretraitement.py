import re


def nettoyer_texte(texte):
    """Nettoie des paroles : minuscules, sans balises, URL, pseudos, chiffres ni ponctuation."""
    texte = str(texte).lower()
    texte = texte.replace("’", "'")
    texte = re.sub(r"\[.*?\]", " ", texte)               # balises [Chorus], [Verse 1]...
    texte = re.sub(r"\(x\d+\)|\bx\d+\b", " ", texte)    # répétitions (x3), x2
    texte = re.sub(r"http\S+|www\.\S+", " ", texte)      # URL
    texte = re.sub(r"@\w+", " ", texte)                  # pseudos @...
    texte = re.sub(r"[^\w\s']", " ", texte)              # ponctuation (on garde les lettres accentuées)
    texte = re.sub(r"[\d_]", " ", texte)                 # chiffres
    texte = re.sub(r"\s'|'\s", " ", texte)               # apostrophes seules
    texte = re.sub(r"\s+", " ", texte)                   # espaces multiples
    return texte.strip()
