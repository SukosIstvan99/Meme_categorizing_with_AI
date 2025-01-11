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
    "an anti meme that parodies meme culture",
    "a positive meme that spreads happiness and optimism",
    "a dark meme that contains edgy or controversial humor"
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


                similarity = image_features @ text_features.T
                print(f"Nyers hasonlóságok: {similarity.cpu().numpy()}")


                probs = similarity.softmax(dim=-1).cpu().numpy()


            print(f"Softmax utáni hasonlóságok: {probs}")
            category = categories[probs.argmax()]
            response = get_response(category)
            await message.channel.send(response)


bot.run("MTMxNDk3Nzk4NzE3ODkyMjAwNA.Gt518x.jMuJK-e8A7jbQxzVD645ThMCe13wLQ4mxCE7cY")

#