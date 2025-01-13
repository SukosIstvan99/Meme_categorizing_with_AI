import discord
from discord.ext import commands
import clip
import torch
from PIL import Image
from responses import get_response
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

categories = [
    "Positive meme",
    "Dark meme, 911, dead, blood, gore, horror, scary, spooky, dark, evil, satan, devil, demon, hell"
]
text_tokens = clip.tokenize(categories).to(device)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Bot is ready as {bot.user}")

@bot.event
async def on_message(message):
    if message.attachments:
        for attachment in message.attachments:
            await attachment.save("temp.jpg")
            image = preprocess(Image.open("temp.jpg")).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image).float()
                image_features = F.normalize(image_features, p=2, dim=-1)

                text_features = model.encode_text(text_tokens).float()
                text_features = F.normalize(text_features, p=2, dim=-1)

                image_similarity = image_features @ text_features.T
                text_similarity = text_features @ image_features.T

                combined_similarity = (image_similarity + text_similarity) / 2

                print(f"Nyers hasonlóságok: {combined_similarity.cpu().numpy()}")

                probs = combined_similarity.softmax(dim=-1).cpu().numpy()

            print(f"Softmax utáni hasonlóságok: {probs}")
            max_prob = probs.max()
            if max_prob < 0.75:  
                response = "Nem tudom pontosan megmondani, hogy mi az."
            else:
                category = categories[probs.argmax()]
                response = get_response(category)

            await message.channel.send(response)

bot.run("MTMxNDk3Nzk4NzE3ODkyMjAwNA.GYItcE.RS6wb-S5iRIlOZwsG_oYSZJgUiuZzaGygaynak")
