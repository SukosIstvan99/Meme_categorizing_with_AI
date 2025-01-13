responses = {
     "Positive meme,",
    "Dark meme,"
}

def get_response(category):
    return responses.get(category, "Ezt meg te sem érted.")
