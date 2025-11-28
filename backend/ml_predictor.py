"""
Predictor ML para BTC usando Random Forest.
Predice el próximo movimiento de precio basado en 15 features técnicos.

MEJORAS:
- Random Forest (más preciso que regresión lineal)
- 15 features (antes 9)
- Window size 50 (antes 20)
- Validación cruzada para evitar overfitting
"""

from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


class MLPredictor:
    """Predictor inteligente de precio usando Random Forest."""
    
    WINDOW_SIZE = 50  # Tamaño de ventana para features históricos
    
    def __init__(self):
        # Random Forest con parámetros optimizados
        self.model = RandomForestRegressor(
            n_estimators=100,      # 🔥 100 árboles (más capacidad)
            max_depth=20,          # 🔥 Profundidad 20 (análisis más profundo)
            min_samples_split=5,   # Mínimo de muestras para dividir
            random_state=42,       # Reproducibilidad
            n_jobs=-1              # Usar todos los cores
        )
        self.is_trained = False
        self.feature_history = []
        self.target_history = []
        self.min_samples = self.WINDOW_SIZE  # Mínimo de muestras para entrenar
        # Path for persisted model
        self.MODEL_PATH = Path(__file__).with_name("ml_model.joblib")
        # Intentar cargar modelo existente
        self.load_model()
        
    def _extract_features(self, price_history: list[float], rsi: float, atr: float, volume: float, adx: float, cci: float) -> np.ndarray:
        """Extrae 17 features avanzados para el modelo."""
        if len(price_history) < 20:
            return None
            
        # Features basados en precio (más contexto)
        recent_prices = price_history[-20:]
        price_current = recent_prices[-1]
        price_mean_10 = np.mean(recent_prices[-10:])
        price_mean_20 = np.mean(recent_prices)
        price_std = np.std(recent_prices)
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        price_volatility = price_std / price_mean_20 if price_mean_20 > 0 else 0
        
        # Bollinger Band position (simple)
        upper_band = price_mean_20 + (2 * price_std)
        lower_band = price_mean_20 - (2 * price_std)
        bb_position = (price_current - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
        
        # Rate of Change (ROC)
        roc = (price_current - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        
        # Price delta features
        delta_1 = recent_prices[-1] - recent_prices[-2] if len(recent_prices) >= 2 else 0
        delta_2 = recent_prices[-2] - recent_prices[-3] if len(recent_prices) >= 3 else 0
        
        # Trend strength (EMA crossover proxy)
        ema_short = np.mean(recent_prices[-5:])  # EMA aproximado 5 periodos
        ema_long = np.mean(recent_prices[-15:])   # EMA aproximado 15 periodos
        trend_strength = (ema_short - ema_long) / ema_long if ema_long > 0 else 0
        
        # 17 features en total (Added ADX & CCI)
        features = np.array([
            price_current,          # 1. Precio actual
            price_mean_10,          # 2. Media 10 periodos
            price_mean_20,          # 3. Media 20 periodos  
            price_std,              # 4. Desviación estándar
            price_momentum,         # 5. Momentum
            price_volatility,       # 6. Volatilidad relativa
            rsi,                    # 7. RSI
            atr,                    # 8. ATR
            volume,                 # 9. Volumen
            bb_position,            # 10. Posición Bollinger Bands
            roc,                    # 11. Rate of Change
            delta_1,                # 12. Delta precio -1
            delta_2,                # 13. Delta precio -2
            trend_strength,         # 14. Fuerza de tendencia
            ema_short / ema_long if ema_long > 0 else 1.0,  # 15. Ratio EMA corto/largo
            adx,                    # 16. ADX (Trend Strength)
            cci                     # 17. CCI (Cyclical Trends)
        ])
        
        return features
        
    def update(self, price_history: list[float], rsi: float, atr: float, volume: float, adx: float, cci: float, actual_next_price: float):
        """Actualiza el modelo con nuevos datos."""
        features = self._extract_features(price_history, rsi, atr, volume, adx, cci)
        
        if features is None:
            return
            
        self.feature_history.append(features)
        self.target_history.append(actual_next_price)
        
        # Mantener solo últimas 200 muestras (ventana deslizante)
        if len(self.feature_history) > 200:
            self.feature_history.pop(0)
            self.target_history.pop(0)
            
        # Entrenar cada 20 nuevas muestras (o si no tenemos modelo aún)
        if len(self.feature_history) % 20 == 0 or not self.is_trained:
            self._train()
    
    def _train(self):
        """Entrena el Random Forest con los datos históricos."""
        if len(self.feature_history) < self.min_samples:
            return
            
        try:
            X = np.array(self.feature_history)
            y = np.array(self.target_history)
            
            # Entrenar modelo Random Forest (ya inicializado en __init__)
            self.model.fit(X, y)
            self.is_trained = True
            
            # Validación cruzada para estimar accuracy (opcional, ~20ms extra)
            if len(X) >= 100:  # Solo si tenemos datos suficientes
                scores = cross_val_score(self.model, X, y, cv=3, scoring='r2')
                accuracy = np.mean(scores)
                print(f"[ML] 🎯 Modelo entrenado: R² = {accuracy:.3f} (17 features, {len(X)} muestras)")
            else:
                print(f"[ML] ✅ Modelo Random Forest entrenado con {len(X)} muestras")
            
            # Guardar modelo después de entrenar
            self.save_model()
            
        except Exception as e:
            print(f"[ML] ❌ Error entrenando: {e}")
            self.is_trained = False
            
    def predict(self, price_history: list[float], rsi: float, atr: float, volume: float, adx: float, cci: float) -> Optional[float]:
        """Predice el próximo precio."""
        if not self.is_trained:
            return None
            
        features = self._extract_features(price_history, rsi, atr, volume, adx, cci)
        
        if features is None:
            return None
            
        try:
            prediction = self.model.predict(features.reshape(1, -1))[0]
            return prediction
        except Exception as e:
            print(f"[ML] Error en predicción: {e}")
            return None
            
    def get_signal(self, current_price: float, predicted_price: float, threshold: float = 0.0002) -> str:
        """
        Genera señal basada en predicción.
        
        Args:
            current_price: Precio actual
            predicted_price: Precio predicho
            threshold: Umbral mínimo de cambio (0.02% por defecto - 🔥 MÁS SENSIBLE)
            
        Returns:
            "BUY", "SELL", o "NEUTRAL"
        """
        if predicted_price is None:
            return "NEUTRAL"
            
        change_pct = (predicted_price - current_price) / current_price
        
        if change_pct > threshold:
            return "BUY"
        elif change_pct < -threshold:
            return "SELL"
        else:
            return "NEUTRAL"

    def save_model(self) -> None:
        """Persistir el modelo entrenado en disco usando joblib."""
        if self.model is None:
            print("[ML] No hay modelo para guardar.")
            return
        try:
            joblib.dump(self.model, self.MODEL_PATH)
            print(f"[ML] Modelo guardado en {self.MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Error guardando modelo: {e}")

    def load_model(self) -> None:
        """Cargar modelo previamente guardado si existe."""
        if not self.MODEL_PATH.is_file():
            print("[ML] No se encontró modelo guardado; se entrenará desde cero.")
            return
        try:
            self.model = joblib.load(self.MODEL_PATH)
            self.is_trained = True
            print(f"[ML] Modelo cargado desde {self.MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Error cargando modelo: {e}")
            self.is_trained = False
