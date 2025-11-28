"""
API endpoint para obtener predicciones de velas en tiempo real.
"""

# Añadir al final de api.py (después del último endpoint)

@app.get("/api/prediction")
def get_prediction(symbol: str = "BTC/USDT"):
    """Obtiene la predicción de la siguiente vela para un símbolo."""
    try:
        if _strategy_instance is None:
            return {"error": "Strategy not initialized"}
        
        # Normalizar símbolo
        symbol = symbol.replace("_", "/")
        
        # Obtener predicción actual
        prediction = _strategy_instance.predictions.get(symbol)
        
        if prediction is None:
            return {
                "symbol": symbol,
                "has_prediction": False,
                "message": "Esperando suficientes velas para predicción (mínimo 30)"
            }
        
        # Obtener precisión histórica del predictor
        predictor_stats = _strategy_instance.candle_predictor.get_stats()
        
        # Precio actual
        current_price = _strategy_instance.last_price.get(symbol, 0)
        
        # Calcular cambio esperado
        expected_change = prediction.predicted_close - current_price
        expected_change_pct = (expected_change / current_price * 100) if current_price > 0 else 0
        
        return {
            "symbol": symbol,
            "has_prediction": True,
            "current_price": round(current_price, 2),
            "prediction": {
                "time": prediction.time,
                "open": round(prediction.predicted_open, 2),
                "high": round(prediction.predicted_high, 2),
                "low": round(prediction.predicted_low, 2),
                "close": round(prediction.predicted_close, 2),
                "confidence": round(prediction.confidence * 100, 2),
                "method": prediction.prediction_method,
                "direction": "UP" if expected_change > 0 else "DOWN",
                "expected_change": round(expected_change, 2),
                "expected_change_pct": round(expected_change_pct, 2)
            },
            "predictor_stats": predictor_stats
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
