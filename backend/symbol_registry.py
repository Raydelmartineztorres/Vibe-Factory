"""
Symbol Registry - Centralized management of tradeable instruments.

Supports: Crypto, Stocks (US/EU), Forex
"""

from typing import TypedDict, Literal
from dataclasses import dataclass


class SymbolMetadata(TypedDict):
    """Metadata for a tradeable symbol."""
    name: str
    exchange: str
    asset_class: Literal["crypto", "stock_us", "stock_eu", "forex"]
    base_currency: str
    quote_currency: str
    min_order_size: float
    tick_size: float
    display_name: str


@dataclass
class SymbolRegistry:
    """Registry of all supported trading instruments."""
    
    # Cryptocurrencies (Top 20 by market cap)
    CRYPTO = {
        "BTC/USDT": SymbolMetadata(
            name="BTC/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="BTC",
            quote_currency="USDT",
            min_order_size=0.0001,
            tick_size=0.01,
            display_name="Bitcoin"
        ),
        "ETH/USDT": SymbolMetadata(
            name="ETH/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="ETH",
            quote_currency="USDT",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="Ethereum"
        ),
        "BNB/USDT": SymbolMetadata(
            name="BNB/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="BNB",
            quote_currency="USDT",
            min_order_size=0.01,
            tick_size=0.01,
            display_name="Binance Coin"
        ),
        "SOL/USDT": SymbolMetadata(
            name="SOL/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="SOL",
            quote_currency="USDT",
            min_order_size=0.01,
            tick_size=0.01,
            display_name="Solana"
        ),
        "XRP/USDT": SymbolMetadata(
            name="XRP/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="XRP",
            quote_currency="USDT",
            min_order_size=1.0,
            tick_size=0.0001,
            display_name="Ripple"
        ),
        "ADA/USDT": SymbolMetadata(
            name="ADA/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="ADA",
            quote_currency="USDT",
            min_order_size=1.0,
            tick_size=0.0001,
            display_name="Cardano"
        ),
        "AVAX/USDT": SymbolMetadata(
            name="AVAX/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="AVAX",
            quote_currency="USDT",
            min_order_size=0.01,
            tick_size=0.01,
            display_name="Avalanche"
        ),
        "DOGE/USDT": SymbolMetadata(
            name="DOGE/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="DOGE",
            quote_currency="USDT",
            min_order_size=1.0,
            tick_size=0.00001,
            display_name="Dogecoin"
        ),
        "DOT/USDT": SymbolMetadata(
            name="DOT/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="DOT",
            quote_currency="USDT",
            min_order_size=0.1,
            tick_size=0.001,
            display_name="Polkadot"
        ),
        "MATIC/USDT": SymbolMetadata(
            name="MATIC/USDT",
            exchange="binance",
            asset_class="crypto",
            base_currency="MATIC",
            quote_currency="USDT",
            min_order_size=1.0,
            tick_size=0.0001,
            display_name="Polygon"
        ),
    }
    
    # US Stocks (FAANG + Popular)
    STOCKS_US = {
        "AAPL": SymbolMetadata(
            name="AAPL",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="AAPL",
            quote_currency="USD",
            min_order_size=0.001,  # Fractional shares
            tick_size=0.01,
            display_name="Apple Inc."
        ),
        "MSFT": SymbolMetadata(
            name="MSFT",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="MSFT",
            quote_currency="USD",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="Microsoft Corp."
        ),
        "GOOGL": SymbolMetadata(
            name="GOOGL",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="GOOGL",
            quote_currency="USD",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="Alphabet Inc."
        ),
        "AMZN": SymbolMetadata(
            name="AMZN",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="AMZN",
            quote_currency="USD",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="Amazon.com Inc."
        ),
        "TSLA": SymbolMetadata(
            name="TSLA",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="TSLA",
            quote_currency="USD",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="Tesla Inc."
        ),
        "NVDA": SymbolMetadata(
            name="NVDA",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="NVDA",
            quote_currency="USD",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="NVIDIA Corp."
        ),
        "META": SymbolMetadata(
            name="META",
            exchange="nasdaq",
            asset_class="stock_us",
            base_currency="META",
            quote_currency="USD",
            min_order_size=0.001,
            tick_size=0.01,
            display_name="Meta Platforms Inc."
        ),
    }
    
    # European Stocks
    STOCKS_EU = {
        "AIR.PA": SymbolMetadata(
            name="AIR.PA",
            exchange="euronext",
            asset_class="stock_eu",
            base_currency="AIR",
            quote_currency="EUR",
            min_order_size=1.0,
            tick_size=0.01,
            display_name="Airbus SE"
        ),
        "SAP.DE": SymbolMetadata(
            name="SAP.DE",
            exchange="xetra",
            asset_class="stock_eu",
            base_currency="SAP",
            quote_currency="EUR",
            min_order_size=1.0,
            tick_size=0.01,
            display_name="SAP SE"
        ),
        "ASML.AS": SymbolMetadata(
            name="ASML.AS",
            exchange="euronext",
            asset_class="stock_eu",
            base_currency="ASML",
            quote_currency="EUR",
            min_order_size=1.0,
            tick_size=0.01,
            display_name="ASML Holding NV"
        ),
    }
    
    # Forex Pairs
    FOREX = {
        "EUR/USD": SymbolMetadata(
            name="EUR/USD",
            exchange="forex",
            asset_class="forex",
            base_currency="EUR",
            quote_currency="USD",
            min_order_size=1000.0,  # Mini lot
            tick_size=0.00001,
            display_name="Euro / US Dollar"
        ),
        "GBP/USD": SymbolMetadata(
            name="GBP/USD",
            exchange="forex",
            asset_class="forex",
            base_currency="GBP",
            quote_currency="USD",
            min_order_size=1000.0,
            tick_size=0.00001,
            display_name="British Pound / US Dollar"
        ),
        "USD/JPY": SymbolMetadata(
            name="USD/JPY",
            exchange="forex",
            asset_class="forex",
            base_currency="USD",
            quote_currency="JPY",
            min_order_size=1000.0,
            tick_size=0.001,
            display_name="US Dollar / Japanese Yen"
        ),
    }
    
    @classmethod
    def get_all_symbols(cls) -> dict[str, SymbolMetadata]:
        """Return all registered symbols."""
        return {
            **cls.CRYPTO,
            **cls.STOCKS_US,
            **cls.STOCKS_EU,
            **cls.FOREX
        }
    
    @classmethod
    def get_symbol(cls, symbol: str) -> SymbolMetadata | None:
        """Get metadata for a specific symbol."""
        return cls.get_all_symbols().get(symbol)
    
    @classmethod
    def get_by_asset_class(cls, asset_class: str) -> dict[str, SymbolMetadata]:
        """Get all symbols of a specific asset class."""
        all_symbols = cls.get_all_symbols()
        return {
            sym: meta for sym, meta in all_symbols.items()
            if meta["asset_class"] == asset_class
        }
    
    @classmethod
    def is_valid_symbol(cls, symbol: str) -> bool:
        """Check if a symbol is registered."""
        return symbol in cls.get_all_symbols()


# Convenience function for quick lookup
def get_symbol_info(symbol: str) -> SymbolMetadata | None:
    """Get symbol metadata."""
    return SymbolRegistry.get_symbol(symbol)
