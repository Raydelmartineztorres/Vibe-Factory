"""
Decision Protocol - Intelligent decision making for multi-asset trading
"""
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TradingDecision:
    """Represents a trading decision"""
    action: str  # "OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD"
    symbol: str
    confidence: float  # 0.0 to 1.0
    size: float
    reason: str
    
class DecisionProtocol:
    """
    Intelligent decision making system for multi-asset trading.
    Analyzes opportunities across multiple assets and makes optimal decisions.
    """
    
    def __init__(self):
        self.max_concurrent_positions = 3
        self.min_confidence = 0.65  # Minimum confidence to trade
        
    def analyze_opportunities(
        self, 
        symbol: str,
        signals: Dict,
        current_position: float,
        portfolio_exposure: float
    ) -> TradingDecision:
        """
        Analyze a trading opportunity and make a decision.
        
        Args:
            symbol: Trading symbol
            signals: Dict with technical signals (rsi, macd, ema, etc)
            current_position: Current position size for this symbol
            portfolio_exposure: Total portfolio exposure (0-1)
            
        Returns:
            TradingDecision object
        """
        
        # 🚫 RULE 1: Don't overtrade - max 3 positions
        if current_position == 0 and portfolio_exposure >= 0.8:
            return TradingDecision(
                action="HOLD",
                symbol=symbol,
                confidence=0.0,
                size=0.0,
                reason="Portfolio exposure too high (>80%)"
            )
        
        # 🚫 RULE 2: Don't open new position if one exists
        if current_position != 0:
            return TradingDecision(
                action="HOLD",
                symbol=symbol,
                confidence=0.0,
                size=0.0,
                reason=f"Already have {'LONG' if current_position > 0 else 'SHORT'} position"
            )
        
        # 📊 ANALYZE SIGNALS
        ema_bullish = signals.get('ema_bullish', False)
        ema_bearish = signals.get('ema_bearish', False)
        macd_bullish = signals.get('macd_bullish', False)
        macd_bearish = signals.get('macd_bearish', False)
        rsi = signals.get('rsi', 50.0)
        volume_ok = signals.get('volume_ok', False)
        ml_signal = signals.get('ml_signal', 'NEUTRAL')
        bullish_pattern = signals.get('bullish_pattern', False)
        bearish_pattern = signals.get('bearish_pattern', False)
        
        # 🎯 CALCULATE CONFIDENCE
        confidence = 0.0
        reasons = []
        
        # BULLISH SCENARIO
        if ema_bullish and macd_bullish and rsi < 70:
            confidence = 0.5
            reasons.append("EMA+MACD bullish")
            
            if volume_ok:
                confidence += 0.1
                reasons.append("volume confirmed")
            
            if ml_signal == "BUY":
                confidence += 0.15
                reasons.append("ML predicts up")
            
            if bullish_pattern:
                confidence += 0.1
                reasons.append("bullish candlestick")
            
            if rsi < 40:
                confidence += 0.05
                reasons.append("oversold RSI")
            
            # Cap at 95%
            confidence = min(0.95, confidence)
            
            if confidence >= self.min_confidence:
                return TradingDecision(
                    action="OPEN_LONG",
                    symbol=symbol,
                    confidence=confidence,
                    size=self._calculate_position_size(confidence),
                    reason=f"LONG: {', '.join(reasons)} ({confidence:.0%})"
                )
        
        # BEARISH SCENARIO
        elif ema_bearish and macd_bearish and rsi > 30:
            confidence = 0.5
            reasons.append("EMA+MACD bearish")
            
            if volume_ok:
                confidence += 0.1
                reasons.append("volume confirmed")
            
            if ml_signal == "SELL":
                confidence += 0.15
                reasons.append("ML predicts down")
            
            if bearish_pattern:
                confidence += 0.1
                reasons.append("bearish candlestick")
            
            if rsi > 60:
                confidence += 0.05
                reasons.append("overbought RSI")
            
            confidence = min(0.95, confidence)
            
            if confidence >= self.min_confidence:
                return TradingDecision(
                    action="OPEN_SHORT",
                    symbol=symbol,
                    confidence=confidence,
                    size=self._calculate_position_size(confidence),
                    reason=f"SHORT: {', '.join(reasons)} ({confidence:.0%})"
                )
        
        # NO CLEAR SIGNAL
        return TradingDecision(
            action="HOLD",
            symbol=symbol,
            confidence=confidence,
            size=0.0,
            reason=f"Confidence too low ({confidence:.0%} < {self.min_confidence:.0%})"
        )
    
    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence.
        Higher confidence = larger position.
        """
        # Base size: 0.001 BTC (or equivalent)
        base_size = 0.001
        
        # Scale by confidence: 65% conf = 1x, 95% conf = 2x
        multiplier = 1.0 + (confidence - 0.65) / 0.3
        
        return base_size * multiplier
    
    def should_close_position(
        self,
        symbol: str,
        position: float,
        entry_price: float,
        current_price: float,
        signals: Dict
    ) -> bool:
        """
        Decide if an open position should be closed.
        
        Returns True if position should close.
        """
        if position == 0:
            return False
        
        # Calculate current PnL %
        if position > 0:  # LONG
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
        
        # 🎯 TAKE PROFIT: Close if profit > 8%
        if pnl_pct > 0.08:
            return True
        
        # 🛑 STOP LOSS: Close if loss > 4%
        if pnl_pct < -0.04:
            return True
        
        # 🔄 SIGNAL REVERSAL: Close if signals flip
        if position > 0:  # We're LONG
            # Close if bearish signals appear
            if signals.get('ema_bearish') and signals.get('macd_bearish'):
                return True
        else:  # We're SHORT
            # Close if bullish signals appear
            if signals.get('ema_bullish') and signals.get('macd_bullish'):
                return True
        
        return False


# Global instance
_decision_protocol = None

def get_decision_protocol() -> DecisionProtocol:
    """Get global decision protocol instance"""
    global _decision_protocol
    if _decision_protocol is None:
        _decision_protocol = DecisionProtocol()
    return _decision_protocol
