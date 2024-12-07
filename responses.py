responses = {
    "an anti meme that parodies meme culture": "Ez egy anti meme, ami parodizálja a mém kultúrát.",
    "a positive meme that spreads happiness and optimism": "Ez egy pozitív meme, ami boldogságot és optimizmust terjeszt.",
    "a dark meme that contains edgy or controversial humor": "Ez egy dark meme, tele vitatható vagy éles humorral."
}

def get_response(category):
    return responses.get(category, "Ezt meg te sem érted.")
