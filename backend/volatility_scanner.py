"""
Volatility Scanner - Selects the most volatile assets for trading
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
import statistics

class VolatilityScanner:
    """Scans multiple assets and ranks by volatility"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or [
            "BTC/USDT",
            "ETH/USDT", 
            "SOL/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "AVAX/USDT",
            "MATIC/USDT"
        ]
        self.volatility_scores = {}
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.last_scan = None
        
    async def scan_volatility(self) -> List[Dict]:
        """
        Scan all symbols and calculate volatility.
        Returns list of symbols sorted by volatility (highest first).
        """
        from broker_api_handler import get_current_price
        
        results = []
        
        for symbol in self.symbols:
            try:
                # Get current price
                price = await get_current_price(symbol, mode="demo")
                
                if price <= 0:
                    continue
                    
                # Store price in history
                self.price_history[symbol].append({
                    'price': price,
                    'time': datetime.now().timestamp()
                })
                
                # Keep only last 100 prices
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)
                
                # Calculate volatility (need at least 20 prices)
                if len(self.price_history[symbol]) >= 20:
                    volatility = self._calculate_volatility(symbol)
                    
                    results.append({
                        'symbol': symbol,
                        'price': price,
                        'volatility': volatility,
                        'volatility_pct': volatility / price * 100  # % volatility
                    })
                    
            except Exception as e:
                print(f"[VOLATILITY] Error scanning {symbol}: {e}")
                continue
        
        # Sort by volatility% (highest first)
        results.sort(key=lambda x: x['volatility_pct'], reverse=True)
        
        self.volatility_scores = {r['symbol']: r['volatility'] for r in results}
        self.last_scan = datetime.now()
        
        return results
    
    def _calculate_volatility(self, symbol: str) -> float:
        """
        Calculate ATR-based volatility for a symbol.
        Higher value = more volatile.
        """
        prices = [p['price'] for p in self.price_history[symbol][-20:]]
        
        # Calculate Average True Range (ATR) as volatility measure
        ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            ranges.append(high_low)
        
        if not ranges:
            return 0.0
            
        atr = statistics.mean(ranges)
        return atr
    
    def get_top_volatile(self, n: int = 3) -> List[str]:
        """
        Returns the top N most volatile symbols.
        Default: top 3.
        """
        if not self.volatility_scores:
            return self.symbols[:n]  # Return first N if no scan yet
        
        # Sort by volatility
        sorted_symbols = sorted(
            self.volatility_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top N symbols
        top_symbols = [symbol for symbol, _ in sorted_symbols[:n]]
        
        print(f"[VOLATILITY] 🔥 Top {n} most volatile: {', '.join(top_symbols)}")
        
        return top_symbols
    
    def should_rescan(self) -> bool:
        """Check if we should rescan (every 5 minutes)"""
        if not self.last_scan:
            return True
        
        time_since_scan = datetime.now() - self.last_scan
        return time_since_scan > timedelta(minutes=5)


# Global scanner instance
_scanner = None

def get_volatility_scanner() -> VolatilityScanner:
    """Get global volatility scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = VolatilityScanner()
    return _scanner
