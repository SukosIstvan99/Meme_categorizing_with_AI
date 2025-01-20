import discord
from discord.ext import commands
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import random

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

responses = {
    "positive meme": [
        "You're amazing! Keep shining!",
        "Oh, I love this! You're absolutely great! Stay awesome!",
        "You make the world a better place just by being you!",
        "This just made my day! You're simply fantastic!"
    ],
    "negative meme": [
        "Oh great, just what we all needed today... not!",
        "Perfect, just what my mood needed – not!",
        "Well, that's just wonderful... said no one ever.",
        "Just the kind of positivity I needed today... not."
    ],
    "sarcasm meme": [
        "I can totally see why you posted this. It's really inspiring.",
        "Oh, definitely, this is life-changing stuff here...",
        "Wow, really makes you think... or maybe not.",
        "A true masterpiece of sarcasm, truly moving."
    ]
}

def classify_meme(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=["positive meme", "negative meme", "sarcasm meme"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    return {
        "positive": probs[0][0].item(),
        "negative": probs[0][1].item(),
        "sarcasm": probs[0][2].item()
    }

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

            temp_path = "temp.jpg"
            await attachment.save(temp_path)

            try:
                result = classify_meme(temp_path)

                positive_prob = result["positive"]
                negative_prob = result["negative"]
                sarcasm_prob = result["sarcasm"]

                positive_percent = positive_prob * 100
                negative_percent = negative_prob * 100
                sarcasm_percent = sarcasm_prob * 100

                if positive_prob > max(negative_prob, sarcasm_prob):
                    category = "positive meme"
                elif negative_prob > max(positive_prob, sarcasm_prob):
                    category = "negative meme"
                else:
                    category = "sarcasm meme"

                ai_response = random.choice(responses[category])

                response = (
                    f"**Category**: {category}\n"
                    f"Positive Probability: {positive_percent:.2f}%\n"
                    f"Negative Probability: {negative_percent:.2f}%\n"
                    f"Sarcasm Probability: {sarcasm_percent:.2f}%\n\n"
                    f"**AI Response**: {ai_response}"
                )

                await message.channel.send(response)
            except Exception as e:
                await message.channel.send("An error occurred during meme classification.")
                print(f"Error: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)


bot.run("MTMxNDk3Nzk4NzE3ODkyMjAwNA.GYItcE.RS6wb-S5iRIlOZwsG_oYSZJgUiuZzaGygaynak")