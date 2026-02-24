#!/usr/bin/env python3
"""
Server Launcher for Ragozin Sheets Parser
Starts both FastAPI backend and Streamlit frontend simultaneously
"""

import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path
import dotenv
dotenv.load_dotenv()

class ServerLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = False
        
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        try:
            import fastapi
            import uvicorn
            import streamlit
            import requests
            print("✅ All required packages are installed")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Please install dependencies with: pip install -r requirements.txt")
            return False
    
    def check_api_key(self):
        """Check if OpenAI API key is set"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OpenAI API key not set")
            print("   The GPT parser will not be available")
            print("   Run 'python set_api_key.py' to set your API key")
            return False
        else:
            print("✅ OpenAI API key is configured")
            return True
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting FastAPI backend server...")
        
        try:
            # Start the backend server
            self.backend_process = subprocess.Popen(
                [sys.executable, "api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Check if server started successfully
            if self.backend_process.poll() is None:
                print("✅ Backend server started successfully")
                print("   API available at: http://localhost:8000")
                print("   API docs at: http://localhost:8000/docs")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                print(f"❌ Backend server failed to start:")
                print(f"   Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend server: {e}")
            return False
    
    def start_frontend(self):
        """Start the Streamlit frontend server"""
        print("🚀 Starting Streamlit frontend server...")
        
        try:
            # Start the frontend server
            self.frontend_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8502"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment for server to start
            time.sleep(5)
            
            # Check if server started successfully
            if self.frontend_process.poll() is None:
                print("✅ Frontend server started successfully")
                print("   Web interface available at: http://localhost:8502")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                print(f"❌ Frontend server failed to start:")
                print(f"   Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend server: {e}")
            return False
    
    def monitor_servers(self):
        """Monitor server processes and handle shutdown"""
        try:
            while self.running:
                # Check if backend is still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print("❌ Backend server stopped unexpectedly")
                    self.running = False
                    break
                
                # Check if frontend is still running
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print("❌ Frontend server stopped unexpectedly")
                    self.running = False
                    break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown both servers gracefully"""
        self.running = False
        
        if self.backend_process:
            print("🛑 Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            print("🛑 Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        print("✅ Servers stopped")
    
    def run(self):
        """Main method to start both servers"""
        print("🐎 Ragozin Sheets Parser - Server Launcher")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Check API key
        self.check_api_key()
        
        print("\n🚀 Starting servers...")
        
        # Start backend server
        if not self.start_backend():
            return False
        
        # Start frontend server
        if not self.start_frontend():
            self.shutdown()
            return False
        
        print("\n" + "=" * 50)
        print("🎯 Both servers are running!")
        print("\n📱 Access your application:")
        print("   Frontend: http://localhost:8502")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("\n🛑 Press Ctrl+C to stop both servers")
        print("=" * 50)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda sig, frame: self.shutdown())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.shutdown())
        
        # Start monitoring
        self.running = True
        self.monitor_servers()
        
        return True

def main():
    """Main function"""
    launcher = ServerLauncher()
    try:
        launcher.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        launcher.shutdown()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        launcher.shutdown()

if __name__ == "__main__":
    main() 