import os, asyncio, aiohttp, torch, discord
import numpy as np
import torch.nn as nn
from bs4 import BeautifulSoup
from discord.ext import commands

# 🔱 ここをお前のトークンに書き換える！
DISCORD_TOKEN = "MTQ2OTUzMTA4NjQ1ODcxNjI3MQ.GNBExr.rxjXGP46FOMt_pgNWtbSQRTaUzWMyGE2ZgJoVk"
MODEL_PATH = "zeus_v200_brain.pth"

class Zeus(nn.Module):
    def __init__(self):
        super().__init__()
        self.j_emb = nn.Embedding(30000, 8)
        self.fc = nn.Linear(12, 64)
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, batch_first=True), 2)
        self.out = nn.Linear(64, 1)
    def forward(self, x, j):
        e = self.j_emb(j).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.out(self.tr(self.fc(torch.cat([x, e], dim=2)))[:, -1, :])

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.command()
async def z(ctx, rid: str):
    await ctx.send(f"🔱 ZEUS執行中... レースID: {rid}")
    await ctx.send(f"✅ 脳と同期完了。ターゲットを捕捉したぜ。")

bot.run(DISCORD_TOKEN)
