import discord
from discord.ext import commands
import aiohttp
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from camel.agents import ChatAgent
from camel.configs import SambaCloudAPIConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class FirecrawlScraper:
    """Handles web scraping of documentation"""

    def __init__(self):
        self.session = None

    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        if self.session:
            await self.session.close()

    async def scrape(self, url: str) -> str:
        """Scrape content from a URL"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    return soup.get_text(separator='\n', strip=True)
                else:
                    logger.error(f"Error scraping {url}: Status {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""


class CustomerServiceBot:
    def __init__(self, discord_token: str, samba_api_key: str, qdrant_url: str):
        """Initialize the customer service bot"""
        # Initialize Discord bot with required intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.discord_token = discord_token

        # Initialize components
        self.scraper = FirecrawlScraper()
        self.setup_qdrant(qdrant_url)
        self.setup_camel_agent(samba_api_key)

        # Register event handlers
        self.setup_events()

    def setup_qdrant(self, qdrant_url: str):
        """Initialize Qdrant vector database"""
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection_name = "support_docs"
        try:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except Exception as e:
            logger.info(f"Collection might already exist: {str(e)}")

    def setup_camel_agent(self, samba_api_key: str):
        """Initialize CAMEL agent with Samba model"""
        model = ModelFactory.create(
            model_platform=ModelPlatformType.SAMBA,
            model_type="Meta-Llama-3.1-405B-Instruct",
            model_config_dict=SambaCloudAPIConfig(max_tokens=800).as_dict(),
            api_key=samba_api_key,
            url="https://api.sambanova.com/v1"
        )

        self.agent = ChatAgent(
            system_message="You are a helpful customer support assistant. Answer questions based on the provided documentation.",
            model=model
        )

    def setup_events(self):
        @self.bot.event
        async def on_ready():
            await self.scraper.setup()
            logger.info(f"{self.bot.user} is now running!")

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return
            await self.bot.process_commands(message)

        @self.bot.command(name="help")
        async def help_command(ctx):
            help_text = """
            Available commands:
            !docs [url] - Add documentation from a URL
            !ask [question] - Ask a question about the documentation
            !help - Show this help message
            """
            await ctx.send(help_text)

        @self.bot.command(name="docs")
        async def add_docs(ctx, url: str):
            await ctx.send("Processing documentation...")
            await self.process_documentation(ctx, url)

        @self.bot.command(name="ask")
        async def ask_question(ctx, *, question: str):
            await self.answer_question(ctx, question)

    async def process_documentation(self, ctx, url: str):
        """Process and store documentation from URL"""
        try:
            content = await self.scraper.scrape(url)
            if not content:
                await ctx.send("Failed to scrape the documentation.")
                return

            response = self.agent.step(
                f"Process and summarize this documentation: {content}"
            )
            processed_content = response.msgs[0].content

            embedding_response = self.agent.step(
                "Generate embeddings for semantic search"
            )
            embeddings = embedding_response.msgs[0].content

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=hash(url),
                        vector=embeddings,
                        payload={
                            "url": url,
                            "content": processed_content
                        }
                    )
                ]
            )

            await ctx.send("Documentation processed and stored successfully!")

        except Exception as e:
            logger.error(f"Error processing documentation: {str(e)}")
            await ctx.send(f"Error processing documentation: {str(e)}")

    async def answer_question(self, ctx, question: str):
        """Answer user questions using stored documentation"""
        try:
            embed_response = self.agent.step(
                f"Generate embedding for question: {question}"
            )
            question_embedding = embed_response.msgs[0].content

            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=question_embedding,
                limit=3
            )

            contexts = [result.payload["content"] for result in search_results]

            response = self.agent.step(
                f"Context: {' '.join(contexts)}\nQuestion: {question}"
            )

            await ctx.send(response.msgs[0].content)

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            await ctx.send(f"Error answering question: {str(e)}")

    async def run(self):
        """Run the Discord bot"""
        try:
            await self.bot.start(self.discord_token)
        finally:
            await self.scraper.cleanup()