import discord
from discord.ext import commands
from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import torch
import os

# Modell és processor betöltése
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# GPT-2 modell betöltése válasz generálásához
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Kategóriák
categories = ["positive meme", "negative meme", "sarcasm meme"]

def classify_meme(image_path):
   
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
    
    outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1)  
    
    return {
        "positive": probs[0][0].item(),
        "negative": probs[0][1].item(),
        "sarcasm": probs[0][2].item()
    }

def generate_response(category):
   
    prompt = ""
    if category == "positive meme":
        prompt = "Generate a light-hearted, positive, and humorous response to a meme, like something an upbeat person would say."
    elif category == "negative meme":
        prompt = "Generate a funny but sarcastic response to a meme, like something someone with a dark sense of humor would say."
    elif category == "sarcasm meme":
        prompt = "Generate a response that captures the sarcastic nature of the meme, with a witty remark as if a friend was commenting on it."

    # Tokenizálás
    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt")

    # GPT-2 generálás
    outputs = gpt_model.generate(inputs, 
                                  max_length=150, 
                                  num_return_sequences=1, 
                                  no_repeat_ngram_size=2, 
                                  temperature=0.7, 
                                  top_k=50, 
                                  do_sample=True,       
                                  pad_token_id=gpt_tokenizer.eos_token_id,  
                                  attention_mask=torch.ones(inputs.shape, dtype=torch.long)) 

    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response.replace(prompt, "").strip()
    
    return response


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
                
                ai_response = generate_response(category)
                
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













