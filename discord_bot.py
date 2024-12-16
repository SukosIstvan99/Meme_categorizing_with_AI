import discord
from discord.ext import commands
import transformers
import torch
import requests
from PIL import Image
from io import BytesIO

intents = discord.Intents.all()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix="!", intents=intents)

def load_model():
    print("Loading model, please wait...")

    try:
        # Modell betöltése
        model = transformers.AutoModelForCausalLM.from_pretrained(
            'mosaicml/mpt-1b-redpajama-200b',
            trust_remote_code=True,
            attn_impl='triton'
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "anas-awadalla/mpt-1b-redpajama-200b",
            trust_remote_code=True
        )

        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Hiba történt a modell betöltése közben: {str(e)}")
        return None, None

# Event: when the bot is online
@bot.event
async def on_ready():
    print(f'Bot is online as {bot.user}')
    await bot.change_presence(activity=discord.Game(name="AI Predikciók"))

    # Várakozás, hogy a bot teljesen betöltse az információkat
    await bot.wait_until_ready()
    
    # Ellenőrizzük, hogy a bot az első szerveren van
    guild = bot.guilds[0]
    channel = guild.get_channel(1314982323271499776)  # Csatorna ID, amit megadtál

    if channel:
        await channel.send("Szia! Én egy AI alapú Discord bot vagyok. 😊")
    else:
        print("Nem található a csatorna.")
# Parancs: !hello
@bot.command()
async def hello(ctx):
    await ctx.send("Szia! Én egy AI alapú Discord bot vagyok. 😊")

# Parancs: !predict [kép link]
@bot.command()
async def predict(ctx, image_url: str):
    await ctx.send("Feldolgozom a képet... ⏳")

    # Modell és tokenizer betöltése egyszer, ha még nem történt meg
    if not hasattr(bot, "model"):
        bot.model, bot.tokenizer = load_model()

    if bot.model is None:
        await ctx.send("Hiba történt a modell betöltése közben. Kérlek próbáld újra később. ❌")
        return

    try:
        # Kép letöltése az URL-ről
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))

            # A képfeldolgozás ide jöhet
            await ctx.send("A képet sikeresen betöltöttem, és a predikció folyamatban van! 🚀")
        else:
            await ctx.send("Nem sikerült letölteni a képet. Kérlek, próbálj egy másik URL-t! ❌")
    except Exception as e:
        await ctx.send(f"Hiba történt a kép feldolgozása közben: {str(e)}")


# Bot indítása
if __name__ == '__main__':
    # A saját Discord bot tokened
    TOKEN = "MTMxNDk3Nzk4NzE3ODkyMjAwNA.GWwtot.GaVzU583834UcC1EXnSG6Mt-_9-qSVPnXz675g"

    try:
        # Modell előtöltése
        bot.model, bot.tokenizer = load_model()
    except Exception as e:
        print(f"Model betöltési hiba: {e}")
        exit(1)

    # Bot futtatása
    bot.run(TOKEN)
