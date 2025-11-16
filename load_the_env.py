import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# OPTION 1: Set API keys directly (Quick Start)
# ============================================
# Uncomment and set your keys here:
# Recommended: Holistic AI Bedrock
# os.environ["HOLISTIC_AI_TEAM_ID"] = "tutorials_api"
# os.environ["HOLISTIC_AI_API_TOKEN"] = "your-token-here"
# Alternative: OpenAI
# os.environ["OPENAI_API_KEY"] = "your-openai-key-here"
# Optional: Valyu
# os.environ["VALYU_API_KEY"] = "your-valyu-key-here"


# ============================================
# OPTION 2: Load from .env file (Recommended)
# ============================================
# Try to load from .env file in parent directory
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print("📄 Loaded configuration from .env file")
else:
    print("⚠️  No .env file found - using environment variables or hardcoded keys")


# ============================================
# Verify API keys are set
# ============================================
print("\n🔑 API Key Status:")
if os.getenv('HOLISTIC_AI_TEAM_ID') and os.getenv('HOLISTIC_AI_API_TOKEN'):
    print("  ✅ Holistic AI Bedrock credentials loaded (will use Bedrock)")
elif os.getenv('OPENAI_API_KEY'):
    print("  ⚠️  OpenAI API key loaded (Bedrock credentials not set)")
    print("     💡 Tip: Set HOLISTIC_AI_TEAM_ID and HOLISTIC_AI_API_TOKEN to use Bedrock (recommended)")
else:
    print("  ⚠️  No API keys found")
    print("     Set Holistic AI Bedrock credentials (recommended) or OpenAI key")

if os.getenv('VALYU_API_KEY'):
    key_preview = os.getenv('VALYU_API_KEY')[:10] + "..."
    print(f"  ✅ Valyu API key loaded: {key_preview}")
else:
    print("  ⚠️  Valyu API key not found - search tool will not work")

print("\n📁 Working directory:", Path.cwd())


# ============================================
# Import Holistic AI Bedrock helper function
# ============================================
# Import from core module (recommended)
import sys
try:
    # Import from same directory
    from holistic_ai_bedrock import HolisticAIBedrockChat, get_chat_model
    print("\n✅ Holistic AI Bedrock helper function loaded")
except ImportError:
    print("\n⚠️  Could not import holistic_ai_bedrock - will use OpenAI only")
    print("   Make sure holistic_ai_bedrock.py exists in tutorials directory")


# Import official packages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

print("\n✅ All imports successful!")