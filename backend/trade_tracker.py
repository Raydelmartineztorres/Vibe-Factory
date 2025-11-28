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
    
    def open_trade(self, size: float, entry_price: float, side: str = "LONG", symbol: str = "BTC/USDT", trade_id: str | int | None = None, source: str = "manual", fee_rate: float = 0.001) -> Dict:
        """
        Abre un nuevo trade.
        
        Args:
            size: Cantidad de BTC
            entry_price: Precio de entrada
            side: LONG o SHORT
            symbol: Símbolo del activo (ej: BTC/USDT)
            trade_id: ID opcional para el trade
            source: "manual" (👤 usuario) o "auto" (🐱 EL GATO)
            fee_rate: Tasa de comisión (default 0.1%)
            
        Returns:
            Diccionario del trade creado
        """
        # Calcular fee de entrada (simulado)
        entry_value = size * entry_price
        entry_fee = entry_value * fee_rate

        trade = {
            "id": trade_id if trade_id is not None else self._generate_id(),
            "symbol": symbol,
            "size": size,
            "entry_price": entry_price,
            "side": side,
            "opened_at": datetime.now().isoformat(),
            "status": "OPEN",
            "source": source,  # 👤 manual or 🐱 auto
            # 🆕 Fee tracking
            "fee_rate": fee_rate,
            "entry_fee": entry_fee,
            "exit_fee": 0.0, # Se calcula al cerrar
            "total_fees": entry_fee,
            "net_pnl": -entry_fee # Empieza negativo por el fee
        }
        self.trades.append(trade)
        self._save()
        icon = "🐱" if source == "auto" else "👤"
        print(f"[TRACKER] ✅ {icon} Trade #{trade['id']} abierto: {size} {symbol} @ ${entry_price:,.2f} ({side}) | Fee: ${entry_fee:.2f}")
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
                
                # Calcular Fee de Salida
                fee_rate = trade.get("fee_rate", 0.001)
                exit_value = trade["size"] * exit_price
                exit_fee = exit_value * fee_rate
                
                trade["exit_fee"] = exit_fee
                trade["total_fees"] = trade["entry_fee"] + exit_fee

                # Calcular Gross PnL (sin fees)
                if trade["side"] == "LONG":
                    gross_pnl = (exit_price - trade["entry_price"]) * trade["size"]
                else:  # SHORT
                    gross_pnl = (trade["entry_price"] - exit_price) * trade["size"]
                
                # Calcular Net PnL (restando fees)
                trade["pnl"] = gross_pnl - trade["total_fees"]
                trade["net_pnl"] = trade["pnl"]
                
                self._save()
                print(f"[TRACKER] 🔒 Trade #{trade_id} cerrado @ ${exit_price:,.0f} | Gross: ${gross_pnl:,.2f} | Fees: ${trade['total_fees']:.2f} | Net PnL: ${trade['pnl']:,.2f}")
                return trade
        
        return None

    def cleanup_duplicates(self):
        """
        Detecta y cierra trades duplicados para el mismo símbolo.
        Mantiene solo el trade más reciente por símbolo.
        """
        open_trades = {}  # {symbol: [trade1, trade2]}
        
        # Agrupar trades abiertos
        for trade in self.trades:
            if trade["status"] == "OPEN":
                symbol = trade["symbol"]
                if symbol not in open_trades:
                    open_trades[symbol] = []
                open_trades[symbol].append(trade)
        
        # Verificar duplicados
        cleaned_count = 0
        for symbol, trades in open_trades.items():
            if len(trades) > 1:
                # Ordenar por fecha (más reciente al final)
                # Asumimos que el orden en lista es cronológico, pero por si acaso
                # trade['opened_at'] es ISO string, se ordena bien lexicográficamente
                trades.sort(key=lambda x: x["opened_at"])
                
                # Mantener el último, cerrar los anteriores
                to_close = trades[:-1]
                keep = trades[-1]
                
                print(f"[TRACKER] ⚠️ Detectados {len(trades)} trades abiertos para {symbol}. Manteniendo #{keep['id']}.")
                
                for t in to_close:
                    print(f"[TRACKER] 🧹 Limpiando trade duplicado #{t['id']}")
                    t["status"] = "CLOSED"
                    t["exit_price"] = t["entry_price"]  # Break even
                    t["closed_at"] = datetime.now().isoformat()
                    t["pnl"] = 0.0
                    t["notes"] = "Auto-closed duplicate"
                    cleaned_count += 1
        
        if cleaned_count > 0:
            self._save()
            print(f"[TRACKER] ✅ Limpieza completada: {cleaned_count} trades duplicados cerrados.")
    
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
        Calcula el PnL no realizado de un trade, INCLUYENDO FEES estimados.
        
        Args:
            trade: Diccionario del trade
            current_price: Precio actual del mercado
            
        Returns:
            Trade con campos 'current_price', 'unrealized_pnl_usd', 'unrealized_pnl_pct'
        """
        trade = trade.copy()
        trade["current_price"] = current_price
        
        # 1. Calcular Gross PnL (Cambio de precio puro)
        if trade["side"] == "LONG":
            gross_pnl = (current_price - trade["entry_price"]) * trade["size"]
        else:  # SHORT
            gross_pnl = (trade["entry_price"] - current_price) * trade["size"]
        
        # 2. Calcular Fees Estimados (Entrada ya pagada + Salida estimada)
        fee_rate = trade.get("fee_rate", 0.001) # Default 0.1%
        entry_fee = trade.get("entry_fee", trade["entry_price"] * trade["size"] * fee_rate)
        estimated_exit_fee = current_price * trade["size"] * fee_rate
        total_estimated_fees = entry_fee + estimated_exit_fee

        # 3. Calcular Net PnL (Realista)
        net_pnl = gross_pnl - total_estimated_fees
        
        entry_value = trade["entry_price"] * trade["size"]
        pnl_pct = (net_pnl / entry_value * 100) if entry_value > 0 else 0.0
        
        trade["unrealized_pnl_usd"] = net_pnl
        trade["unrealized_pnl_pct"] = pnl_pct
        
        return trade


# Instancia global
_tracker = TradeTracker()

def get_tracker() -> TradeTracker:
    """Obtiene la instancia global del tracker."""
    return _tracker
