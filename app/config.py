import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
class Settings:

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"
    
    QDRANT_URL= os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # GROQ is deprecated in favor of Gemini
    GROQ_API_KEY= os.getenv("GROQ_API_KEY")
    
    # Portkey Gateway Settings
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    # A Portkey config must contain the provider/virtual-key configuration for
    # the aliases below.  A GROQ_API_KEY does not create these aliases.
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID") # e.g. pc-xxx
    GROQ_SLUG = os.getenv("GROQ_SLUG", "groq")
    GROQ_SLUG_2 = os.getenv("GROQ_SLUG_2", "groq-fallback")

    # Keep the gateway opt-in.  This prevents a stray PORTKEY_API_KEY from
    # routing requests to an alias that has not been configured in Portkey.
    PORTKEY_ENABLED = bool(PORTKEY_API_KEY and PORTKEY_CONFIG_ID)

settings=Settings()
