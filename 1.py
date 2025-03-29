import os
import discord
from discord.ext import commands
from discord import app_commands
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import asyncio
import base64
import datetime
from typing import Dict, List, Optional, Any, Union, Deque
from collections import defaultdict, deque

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
class Config:
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DEFAULT_MODEL = "models/gemini-2.0-flash"
    VISION_MODEL = "models/gemini-1.5-flash"
    MAX_MESSAGE_LENGTH = 2000
    CONTEXT_LIMIT = 1000

# Доступные модели
AVAILABLE_MODELS = {
    "GEMINI 1.5": [
        {"name": "Gemini 1.5 Pro", "id": "models/gemini-1.5-pro", "description": "Мощная модель с расширенным контекстом"},
        {"name": "Gemini 1.5 Flash", "id": "models/gemini-1.5-flash", "description": "Быстрая модель с хорошим балансом"}
    ],
    "GEMINI 2.0": [
        {"name": "Gemini 2.0 Flash", "id": "models/gemini-2.0-flash", "description": "Новая улучшенная модель"},
        {"name": "Gemini 2.0 Flash-Lite", "id": "models/gemini-2.0-flash-lite", "description": "Экономичный вариант с ограничениями"}
    ]
}

# Настройки безопасности для Gemini API
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Инициализация Gemini API
genai.configure(api_key=Config.GEMINI_API_KEY)

# Хранилище контекста сообщений для каждого канала
channel_conversations = defaultdict(lambda: deque(maxlen=Config.CONTEXT_LIMIT))

# Хранилище пользовательских промптов для серверов
server_prompts = defaultdict(lambda: "")

# Настройка бота
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Класс для работы с Gemini API
class GeminiClient:
    @staticmethod
    def add_to_conversation(channel_id, message):
        """Добавляет сообщение в контекст канала"""
        channel_conversations[channel_id].append(message)
    
    @staticmethod
    def get_conversation_history(channel_id):
        """Получает историю сообщений для канала"""
        conversation = []
        for msg in channel_conversations[channel_id]:
            if msg["role"] == "user":
                conversation.append({"role": "user", "parts": [msg["content"]]})
            else:
                conversation.append({"role": "model", "parts": [msg["content"]]})
        return conversation
    
    @staticmethod
    async def generate_response(prompt: str, channel_id: int, server_id: Optional[int] = None, image_urls: List[str] = None) -> str:
        """Генерирует ответ используя Gemini API с историей сообщений и изображениями"""
        try:
            model_name = Config.DEFAULT_MODEL
            
            # Получаем историю сообщений
            conversation_history = GeminiClient.get_conversation_history(channel_id)
            
            # Получаем пользовательский промпт для сервера, если есть
            server_prompt = ""
            if server_id and server_id in server_prompts:
                server_prompt = server_prompts[server_id]
                if server_prompt:
                    prompt = f"{server_prompt}\n\nЗапрос пользователя: {prompt}"
            
            # Подготовка промпта и изображений
            if image_urls:
                # Для запроса с изображениями используем модель с поддержкой изображений
                model = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
                
                # Создаем мультимодальный запрос
                contents = [{"text": prompt}]
                for url in image_urls:
                    try:
                        response = requests.get(url)
                        if response.status_code == 200:
                            img_data = response.content
                            contents.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(img_data).decode("utf-8")
                                }
                            })
                    except Exception as e:
                        print(f"Ошибка при загрузке изображения: {e}")
                
                # Если у нас есть история сообщений, используем chat для сохранения контекста
                if conversation_history:
                    chat = model.start_chat(history=conversation_history)
                    response = chat.send_message(contents)
                else:
                    response = model.generate_content(contents)
            else:
                # Для текстового запроса используем стандартную модель
                model = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
                
                # Если у нас есть история сообщений, используем chat для сохранения контекста
                if conversation_history:
                    chat = model.start_chat(history=conversation_history)
                    response = chat.send_message(prompt)
                else:
                    response = model.generate_content(prompt)
            
            # Добавляем сообщение пользователя и ответ в историю
            GeminiClient.add_to_conversation(channel_id, {"role": "user", "content": prompt, "time": datetime.datetime.now()})
            GeminiClient.add_to_conversation(channel_id, {"role": "assistant", "content": response.text, "time": datetime.datetime.now()})
            
            return response.text
        except Exception as e:
            return f"Произошла ошибка при генерации ответа: {str(e)}"

# UI компоненты для выбора модели
class ModelSelectUI:
    class GroupSelector(discord.ui.Select):
        def __init__(self, user_id: int):
            self.user_id = user_id
            options = [
                discord.SelectOption(label=group, description=f"Модели {group}") 
                for group in AVAILABLE_MODELS.keys()
            ]
            super().__init__(placeholder="Выберите группу моделей", options=options)
            
        async def callback(self, interaction: discord.Interaction):
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("Это меню для другого пользователя", ephemeral=True)
                return
                
            group = self.values[0]
            view = ModelSelectUI.ModelView(interaction.user.id, group)
            await interaction.response.edit_message(content=f"Выберите модель из группы {group}:", view=view)

    class ModelSelector(discord.ui.Select):
        def __init__(self, user_id: int, group: str):
            self.user_id = user_id
            self.group = group
            options = []
            
            for model in AVAILABLE_MODELS[group]:
                label = model["name"]
                if len(label) > 100:
                    label = label[:97] + "..."
                
                description = model.get("description", model["id"])
                if len(description) > 100:
                    description = description[:97] + "..."
                    
                option = discord.SelectOption(
                    label=label, 
                    value=model["id"],
                    description=description
                )
                    
                options.append(option)
                
            super().__init__(placeholder=f"Выберите модель из {group}", options=options)
            
        async def callback(self, interaction: discord.Interaction):
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("Это меню для другого пользователя", ephemeral=True)
                return
                
            model_id = self.values[0]
            await interaction.response.edit_message(
                content=f"Выбрана модель: `{model_id}`\nВсе запросы к Gemini теперь будут использовать эту модель.", 
                view=None
            )
            
            Config.DEFAULT_MODEL = model_id

    class ModelView(discord.ui.View):
        def __init__(self, user_id: int, group: Optional[str] = None):
            super().__init__(timeout=120)
            
            if group:
                self.add_item(ModelSelectUI.ModelSelector(user_id, group))
                self.add_item(discord.ui.Button(label="Назад", style=discord.ButtonStyle.secondary, custom_id="back"))
            else:
                self.add_item(ModelSelectUI.GroupSelector(user_id))
                
        async def interaction_check(self, interaction: discord.Interaction) -> bool:
            if interaction.data.get("custom_id") == "back":
                view = ModelSelectUI.ModelView(interaction.user.id)
                await interaction.response.edit_message(content="Выберите группу моделей:", view=view)
                return False
            return True

# События и команды бота
@bot.event
async def on_ready():
    print(f'{bot.user.name} подключен к Discord!')
    try:
        synced = await bot.tree.sync()
        print(f"Синхронизировано {len(synced)} команд")
        
        # Установка статуса бота
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening, 
                name="ваши сообщения | !help"
            )
        )
    except Exception as e:
        print(f"Ошибка синхронизации команд: {e}")

@bot.command(name='gemini')
async def gemini_command(ctx, *, prompt: str):
    async with ctx.typing():
        # Проверяем наличие вложений-изображений
        image_urls = []
        if ctx.message.attachments:
            for attachment in ctx.message.attachments:
                if attachment.content_type and attachment.content_type.startswith('image/'):
                    image_urls.append(attachment.url)
        
        # Получаем ID сервера, если сообщение отправлено на сервере
        server_id = ctx.guild.id if ctx.guild else None
        
        response = await GeminiClient.generate_response(prompt, ctx.channel.id, server_id, image_urls)
        
    # Разбиваем длинные сообщения на части
    if len(response) > Config.MAX_MESSAGE_LENGTH:
        chunks = [response[i:i+Config.MAX_MESSAGE_LENGTH] 
                 for i in range(0, len(response), Config.MAX_MESSAGE_LENGTH)]
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(response)

@bot.event
async def on_message(message: discord.Message):
    # Игнорируем сообщения от бота
    if message.author == bot.user:
        return
    
    # Сохраняем каждое сообщение пользователя в контекст
    if not message.author.bot:
        # Добавляем сообщение в историю канала
        GeminiClient.add_to_conversation(message.channel.id, {
            "role": "user", 
            "content": message.content, 
            "time": datetime.datetime.now()
        })
    
    # Обрабатываем сообщения в ЛС
    if isinstance(message.channel, discord.DMChannel):
        # Если это не команда, то обрабатываем как запрос к Gemini
        if not message.content.startswith(bot.command_prefix):
            image_urls = []
            
            # Проверяем, есть ли изображения в сообщении
            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        image_urls.append(attachment.url)
            
            async with message.channel.typing():
                response = await GeminiClient.generate_response(
                    message.content, message.channel.id, None, image_urls
                )
                
                # Разбиваем длинные сообщения на части
                if len(response) > Config.MAX_MESSAGE_LENGTH:
                    chunks = [response[i:i+Config.MAX_MESSAGE_LENGTH] 
                            for i in range(0, len(response), Config.MAX_MESSAGE_LENGTH)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(response)
                    
            # Пропускаем обработку команд в ЛС, если это не команда
            if not message.content.startswith(bot.command_prefix):
                return
    
    # Обрабатываем команды
    await bot.process_commands(message)
    
    # Если бота упомянули в сервере, генерируем ответ
    if bot.user.mentioned_in(message) and not isinstance(message.channel, discord.DMChannel):
        content = message.content.replace(f'<@{bot.user.id}>', '').strip()
        image_urls = []
        
        # Получаем ID сервера
        server_id = message.guild.id if message.guild else None
        
        # Проверяем, есть ли изображения в сообщении
        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith('image/'):
                    image_urls.append(attachment.url)
        
        # Если есть только изображения без текста, используем режим анализа изображений
        if image_urls and not content:
            async with message.channel.typing():
                prompt = "Опиши подробно, что изображено на этом изображении"
                response = await GeminiClient.generate_response(prompt, message.channel.id, server_id, image_urls)
                
                # Разбиваем длинные сообщения на части
                if len(response) > Config.MAX_MESSAGE_LENGTH:
                    chunks = [response[i:i+Config.MAX_MESSAGE_LENGTH] 
                             for i in range(0, len(response), Config.MAX_MESSAGE_LENGTH)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(response)
        
        # Если есть текст, с изображениями или без, используем стандартный режим
        elif content:
            async with message.channel.typing():
                response = await GeminiClient.generate_response(content, message.channel.id, server_id, image_urls)
                
                # Разбиваем длинные сообщения на части
                if len(response) > Config.MAX_MESSAGE_LENGTH:
                    chunks = [response[i:i+Config.MAX_MESSAGE_LENGTH] 
                             for i in range(0, len(response), Config.MAX_MESSAGE_LENGTH)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(response)

@bot.command(name='help')
async def text_help_command(ctx):
    """Отображает справку по командам в виде текста"""
    help_text = """
**Доступные команды бота:**

`!gemini [запрос]` - Задать вопрос модели Gemini (можно прикрепить изображение)
`!help` - Показать эту справку

*Slash-команды:*
`/gemeni` - Выбрать модель Gemini для использования
`/prompt` - Установить системный промпт для сервера (только админы)
`/getprompt` - Показать текущий системный промпт сервера
`/clearprompt` - Удалить системный промпт сервера (только админы)
`/clear` - Очистить историю сообщений
`/history` - Показать количество сохраненных сообщений
`/help` - Показать список команд

**Особенности:**
- В ЛС бот отвечает на любое сообщение без префикса
- На сервере бот отвечает при упоминании @БотИмя
- Можно отправить фото с текстом или без для анализа
- Бот запоминает до 1000 сообщений в каждом канале
- Админы могут установить системный промпт для каждого сервера
"""
    await ctx.send(help_text)

@bot.tree.command(name="gemeni", description="Выбрать модель Gemini для использования")
async def gemeni_command(interaction: discord.Interaction):
    view = ModelSelectUI.ModelView(interaction.user.id)
    await interaction.response.send_message("Выберите группу моделей:", view=view)

@bot.tree.command(name="prompt", description="Установить пользовательский промпт для сервера")
@app_commands.describe(prompt="Системный промпт для Gemini на этом сервере")
async def set_prompt_command(interaction: discord.Interaction, prompt: str):
    """Устанавливает системный промпт для конкретного сервера"""
    if not interaction.guild:
        await interaction.response.send_message("Эта команда доступна только на серверах!", ephemeral=True)
        return
        
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("Только администраторы могут менять системный промпт!", ephemeral=True)
        return
    
    server_id = interaction.guild.id
    server_prompts[server_id] = prompt
    
    await interaction.response.send_message(
        f"Системный промпт для этого сервера установлен!\n\n**Текущий промпт:**\n```{prompt}```", 
        ephemeral=True
    )

@bot.tree.command(name="getprompt", description="Показать текущий системный промпт сервера")
async def get_prompt_command(interaction: discord.Interaction):
    """Показывает текущий системный промпт для сервера"""
    if not interaction.guild:
        await interaction.response.send_message("Эта команда доступна только на серверах!", ephemeral=True)
        return
    
    server_id = interaction.guild.id
    prompt = server_prompts.get(server_id, "")
    
    if prompt:
        await interaction.response.send_message(
            f"**Текущий системный промпт этого сервера:**\n```{prompt}```",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "Для этого сервера еще не установлен системный промпт. Администратор может установить его с помощью команды `/prompt`.", 
            ephemeral=True
        )

@bot.tree.command(name="clearprompt", description="Удалить системный промпт сервера")
async def clear_prompt_command(interaction: discord.Interaction):
    """Удаляет системный промпт для сервера"""
    if not interaction.guild:
        await interaction.response.send_message("Эта команда доступна только на серверах!", ephemeral=True)
        return
        
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("Только администраторы могут удалять системный промпт!", ephemeral=True)
        return
    
    server_id = interaction.guild.id
    if server_id in server_prompts:
        del server_prompts[server_id]
        await interaction.response.send_message("Системный промпт для этого сервера удален!", ephemeral=True)
    else:
        await interaction.response.send_message("Для этого сервера не был установлен системный промпт.", ephemeral=True)

@bot.tree.command(name="clear", description="Очистить историю сообщений в текущем канале")
async def clear_history(interaction: discord.Interaction):
    channel_conversations[interaction.channel_id].clear()
    await interaction.response.send_message("История сообщений в этом канале очищена.")

@bot.tree.command(name="history", description="Показать количество сохраненных сообщений в текущем канале")
async def show_history(interaction: discord.Interaction):
    count = len(channel_conversations[interaction.channel_id])
    await interaction.response.send_message(f"Количество сохраненных сообщений в этом канале: {count}/{Config.CONTEXT_LIMIT}")

@bot.tree.command(name="help", description="Показать доступные команды")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Доступные команды",
        description="Список всех доступных команд бота",
        color=discord.Color.blue()
    )
    
    commands_list = [
        {"name": "/gemeni", "description": "Выбрать модель Gemini для использования"},
        {"name": "/prompt", "description": "Установить системный промпт для сервера (только админы)"},
        {"name": "/getprompt", "description": "Показать текущий системный промпт сервера"},
        {"name": "/clearprompt", "description": "Удалить системный промпт сервера (только админы)"},
        {"name": "/clear", "description": "Очистить историю сообщений в текущем канале"},
        {"name": "/history", "description": "Показать количество сохраненных сообщений"},
        {"name": "/help", "description": "Показать доступные команды"},
        {"name": "!gemini [запрос]", "description": "Задать вопрос модели Gemini, можно прикрепить изображение"},
        {"name": "!help", "description": "Показать текстовую справку по командам"},
        {"name": "Личные сообщения", "description": "В ЛС просто пишите сообщение, бот ответит без команд"},
        {"name": "@БотИмя [запрос]", "description": "Упомянуть бота и задать вопрос с текстом"},
        {"name": "@БотИмя + фото", "description": "Упомянуть бота на сообщении с фото без текста для анализа изображения"}
    ]
    
    for cmd in commands_list:
        embed.add_field(name=cmd["name"], value=cmd["description"], inline=False)
    
    embed.set_footer(text="Бот сохраняет контекст до 1000 последних сообщений для каждого канала")
    
    await interaction.response.send_message(embed=embed)

# Запуск бота
if __name__ == "__main__":
    bot.run(Config.DISCORD_TOKEN)