import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from risk_strategy import RiskStrategy

async def test_multi_asset():
    print("🚀 Starting Multi-Asset Verification with Scanner...")
    
    strategy = RiskStrategy()
    print("✅ Strategy initialized")
    
    # Mock scanner behavior manually since we can't easily wait for the loop in this script
    print("\n🔍 Testing Active Symbol Filtering:")
    strategy.active_symbols = ["BTC/USDT"] # Only BTC is active
    print(f"  Active Symbols: {strategy.active_symbols}")
    
    # Simulate ticks for BTC (Active)
    print("\n📉 Simulating BTC/USDT ticks (Active)...")
    await strategy.on_tick({"symbol": "BTC/USDT", "price": 90000, "timestamp": 1700000000})
    
    # Simulate ticks for ETH (Inactive)
    print("\n📉 Simulating ETH/USDT ticks (Inactive)...")
    await strategy.on_tick({"symbol": "ETH/USDT", "price": 3000, "timestamp": 1700000000})
    
    # Check internal state
    print("\n🔍 Checking Internal State:")
    btc_hist = len(strategy.price_history.get('BTC/USDT', []))
    eth_hist = len(strategy.price_history.get('ETH/USDT', []))
    print(f"  BTC Price History: {btc_hist} entries (Expected > 0)")
    print(f"  ETH Price History: {eth_hist} entries (Expected 0)")
    
    if btc_hist > 0 and eth_hist == 0:
        print("✅ Filtering Logic Confirmed: Only active symbols processed.")
    else:
        print("❌ Filtering Logic FAILED.")

    # Test Scanner Integration (Mock)
    print("\n🔍 Testing Scanner Integration (Mock):")
    # Simulate scanner finding ETH
    strategy.active_symbols.append("ETH/USDT")
    print(f"  Scanner adds ETH. Active: {strategy.active_symbols}")
    
    print("\n📉 Simulating ETH/USDT ticks (Now Active)...")
    await strategy.on_tick({"symbol": "ETH/USDT", "price": 3000, "timestamp": 1700000000})
    
    eth_hist_new = len(strategy.price_history.get('ETH/USDT', []))
    print(f"  ETH Price History: {eth_hist_new} entries (Expected > 0)")
    
    if eth_hist_new > 0:
        print("✅ Dynamic Activation Confirmed!")
    else:
        print("❌ Dynamic Activation FAILED.")

if __name__ == "__main__":
    asyncio.run(test_multi_asset())
