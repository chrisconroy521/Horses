#!/usr/bin/env python3
"""
Environment Setup Script for Ragozin Sheets Parser
This script helps users set up their .env file with the required configuration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_example.exists():
        print("❌ .env.example file not found!")
        print("Please make sure you have the .env.example template file.")
        return False
    
    if env_file.exists():
        print("✅ .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    # Copy .env.example to .env
    with open(env_example, 'r') as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ .env file created from template!")
    return True

def setup_api_key():
    """Set up OpenAI API key"""
    print("\n🔑 OpenAI API Key Setup")
    print("=" * 40)
    
    # Check if API key is already set
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY=your_openai_api_key_here' not in content:
                print("✅ API key appears to be already configured!")
                return True
    
    print("To use this application, you need an OpenAI API key.")
    print("Get your API key from: https://platform.openai.com/api-keys")
    print()
    
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key doesn't start with 'sk-'. Please verify your key.")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            return False
    
    # Update .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            content = f.read()
        
        # Replace the placeholder with actual API key
        content = content.replace('OPENAI_API_KEY=your_openai_api_key_here', f'OPENAI_API_KEY={api_key}')
        
        with open('.env', 'w') as f:
            f.write(content)
        
        print("✅ API key saved to .env file!")
        return True
    else:
        print("❌ .env file not found. Please run setup_env.py first.")
        return False

def test_api_key():
    """Test the API key"""
    print("\n🧪 Testing API Key")
    print("=" * 40)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            print("❌ API key not found or not set properly.")
            return False
        
        # Simple test - try to import the parser
        try:
            from gpt_parser_alternative import GPTRagozinParserAlternative
            parser = GPTRagozinParserAlternative()
            print("✅ API key loaded successfully!")
            print("✅ Parser initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing parser: {e}")
            return False
            
    except ImportError:
        print("❌ python-dotenv not installed. Run: pip install python-dotenv")
        return False

def main():
    """Main setup function"""
    print("🐎 Ragozin Sheets Parser - Environment Setup")
    print("=" * 50)
    
    # Step 1: Create .env file
    if not create_env_file():
        return
    
    # Step 2: Setup API key
    if not setup_api_key():
        return
    
    # Step 3: Test setup
    if test_api_key():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the application: python start.py")
        print("2. Or start individually:")
        print("   - Backend: uvicorn api:app --reload --host 0.0.0.0 --port 8000")
        print("   - Frontend: streamlit run streamlit_app.py")
        print("\nHappy horse racing analysis! 🏇")
    else:
        print("\n❌ Setup failed. Please check your configuration.")

if __name__ == "__main__":
    main() 