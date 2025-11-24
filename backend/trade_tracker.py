"""
Sistema de Tracking de Trades Individuales.

Permite gestionar múltiples trades por separado con PnL individual.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class TradeTracker:
    """Gestiona trades individuales de forma separada."""
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.file = self.cache_dir / "active_trades.json"
        self.trades = self._load()
    
    def _load(self) -> List[Dict]:
        """Carga trades del archivo."""
        if self.file.exists():
            try:
                return json.loads(self.file.read_text())
            except:
                return []
        return []
    
    def _save(self):
        """Guarda trades al archivo."""
        self.file.write_text(json.dumps(self.trades, indent=2))
    
    def _generate_id(self) -> int:
        """Genera un ID único para el trade."""
        if not self.trades:
            return 1
        return max(t.get("id", 0) for t in self.trades) + 1
    
    def open_trade(self, size: float, entry_price: float, side: str = "LONG") -> Dict:
        """
        Abre un nuevo trade.
        
        Args:
            size: Cantidad de BTC
            entry_price: Precio de entrada
            side: LONG o SHORT
            
        Returns:
            Diccionario del trade creado
        """
        trade = {
            "id": self._generate_id(),
            "size": size,
            "entry_price": entry_price,
            "side": side,
            "opened_at": datetime.now().isoformat(),
            "status": "OPEN"
        }
        self.trades.append(trade)
        self._save()
        print(f"[TRACKER] ✅ Trade #{trade['id']} abierto: {size} BTC @ ${entry_price:,.0f} ({side})")
        return trade
    
    def close_trade(self, trade_id: int, exit_price: float) -> Optional[Dict]:
        """
        Cierra un trade específico.
        
        Args:
            trade_id: ID del trade a cerrar
            exit_price: Precio de salida
            
        Returns:
            Trade cerrado o None si no existe
        """
        for trade in self.trades:
            if trade["id"] == trade_id and trade["status"] == "OPEN":
                trade["status"] = "CLOSED"
                trade["exit_price"] = exit_price
                trade["closed_at"] = datetime.now().isoformat()
                
                # Calcular PnL
                if trade["side"] == "LONG":
                    trade["pnl"] = (exit_price - trade["entry_price"]) * trade["size"]
                else:  # SHORT
                    trade["pnl"] = (trade["entry_price"] - exit_price) * trade["size"]
                
                self._save()
                print(f"[TRACKER] 🔒 Trade #{trade_id} cerrado @ ${exit_price:,.0f} | PnL: ${trade['pnl']:,.2f}")
                return trade
        
        return None
    
    def reverse_trade(self, trade_id: int) -> Optional[Dict]:
        """
        Invierte la dirección de un trade (LONG→SHORT o SHORT→LONG).
        
        Args:
            trade_id: ID del trade a invertir
            
        Returns:
            Trade modificado o None si no existe
        """
        for trade in self.trades:
            if trade["id"] == trade_id and trade["status"] == "OPEN":
                old_side = trade["side"]
                trade["side"] = "SHORT" if old_side == "LONG" else "LONG"
                self._save()
                print(f"[TRACKER] 🔄 Trade #{trade_id} invertido: {old_side} → {trade['side']}")
                return trade
        
        return None
    
    def get_active_trades(self) -> List[Dict]:
        """Obtiene todos los trades abiertos."""
        return [t for t in self.trades if t["status"] == "OPEN"]
    
    def get_all_trades(self) -> List[Dict]:
        """Obtiene todos los trades (abiertos y cerrados)."""
        return self.trades.copy()
    
    def calculate_live_pnl(self, trade: Dict, current_price: float) -> Dict:
        """
        Calcula el PnL no realizado de un trade.
        
        Args:
            trade: Diccionario del trade
            current_price: Precio actual del mercado
            
        Returns:
            Trade con campos 'current_price', 'unrealized_pnl_usd', 'unrealized_pnl_pct'
        """
        trade = trade.copy()
        trade["current_price"] = current_price
        
        if trade["side"] == "LONG":
            pnl_usd = (current_price - trade["entry_price"]) * trade["size"]
        else:  # SHORT
            pnl_usd = (trade["entry_price"] - current_price) * trade["size"]
        
        entry_value = trade["entry_price"] * trade["size"]
        pnl_pct = (pnl_usd / entry_value * 100) if entry_value > 0 else 0.0
        
        trade["unrealized_pnl_usd"] = pnl_usd
        trade["unrealized_pnl_pct"] = pnl_pct
        
        return trade


# Instancia global
_tracker = TradeTracker()

def get_tracker() -> TradeTracker:
    """Obtiene la instancia global del tracker."""
    return _tracker
