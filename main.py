import os
import asyncio
import logging
from dotenv import load_dotenv
from bot import CustomerServiceBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()

    required_vars = {
        'DISCORD_TOKEN': os.getenv('DISCORD_TOKEN'),
        'SAMBA_API_KEY': os.getenv('SAMBA_API_KEY'),
        'QDRANT_URL': os.getenv('QDRANT_URL', 'http://localhost:6333')
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file or environment settings."
        )

    return required_vars


async def main():
    """Main function to run the bot"""
    try:
        logger.info("Loading environment variables...")
        env_vars = load_environment_variables()

        logger.info("Initializing Customer Service Bot...")
        bot = CustomerServiceBot(
            discord_token=env_vars['DISCORD_TOKEN'],
            samba_api_key=env_vars['SAMBA_API_KEY'],
            qdrant_url=env_vars['QDRANT_URL']
        )

        logger.info("Starting the bot...")
        await bot.run()

    except Exception as e:
        logger.error(f"Error running the bot: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutdown initiated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")