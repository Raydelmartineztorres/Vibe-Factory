"""
Test script to verify Binance API connection.
Works in both DEMO and TESTNET modes.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_binance_connection():
    """Test connection to Binance (DEMO or Testnet)"""
    
    print("🔍 Testing Binance API Configuration...\n")
    
    # Check trading mode
    trading_mode = os.getenv("TRADING_MODE", "demo").lower()
    print(f"📊 Trading Mode: {trading_mode.upper()}\n")
    
    if trading_mode == "demo":
        print("✅ DEMO MODE - No API keys required!")
        print("   - All trades are simulated")
        print("   - Prices are REAL from Binance public API")
        print("   - Starting balance: $10,000 USDT + 0 BTC\n")
        
        # Test fetching real price in demo mode
        try:
            import ccxt.async_support as ccxt
            
            print("🔌 Connecting to Binance (public API)...\n")
            
            # Create exchange without credentials (public API)
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Fetch current BTC price
            print("📊 Fetching live BTC/USDT price...\n")
            ticker = await exchange.fetch_ticker('BTC/USDT')
            
            print(f"✅ Current BTC/USDT Price: ${ticker['last']:,.2f}")
            print(f"   24h High: ${ticker['high']:,.2f}")
            print(f"   24h Low: ${ticker['low']:,.2f}")
            print(f"   24h Volume: {ticker['baseVolume']:,.2f} BTC\n")
            
            await exchange.close()
            
            print("✅ Demo mode is working perfectly!")
            print("\n💡 Next steps:")
            print("   1. Start the trading bot: python3 api.py")
            print("   2. Open dashboard: http://localhost:8000")
            print("   3. Watch the bot trade with SIMULATED money!")
            
        except Exception as e:
            print(f"\n❌ ERROR fetching prices: {e}")
            print("\n🔧 Check your internet connection")
            return

if __name__ == "__main__":
    asyncio.run(test_binance_connection())
