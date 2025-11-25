
"use client";

"use client";

import { useEffect, useState, useRef } from "react";
import { createChart, ColorType, IChartApi, CandlestickSeries, LineSeries, AreaSeries, HistogramSeries } from 'lightweight-charts';

const steps = [
  {
    title: "01 · Frontend",
    description:
      "Next.js + Tailwind listos para UI de control humano (HITL) con dashboards, alertas y botones críticos.",
    checklist: ["Dashboard base", "Tema accesible", "Botón STOP rojo"],
  },
  {
    title: "02 · Backend",
    description:
      "Módulos Python para datos históricos, risk engine, news analyzer y ejecución segura.",
    checklist: ["data_collector", "risk_strategy", "broker_api_handler"],
  },
  {
    title: "03 · DevOps & Docs",
    description:
      "Prompts, scripts y flujos documentados para reutilizar la fábrica en cada nuevo proyecto.",
    checklist: ["Manual en docs/", "Variables .env", "Playbooks de despliegue"],
  },
];

interface BacktestResult {
  final_capital: number;
  max_drawdown: number;
  trades: number;
}

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<string>("Checking...");
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [loadingBacktest, setLoadingBacktest] = useState(false);
  const [pnl, setPnl] = useState<{
    realized_pnl: number;
    unrealized_pnl: number;
    position_size: number;
    entry_price: number;
    current_price: number;
  } | null>(null);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [balance, setBalance] = useState<{ USDT: number, BTC: number, mode?: string } | null>(null);
  const lastTrade = trades.length > 0 ? trades[trades.length - 1] : null;
  const [tradingEnabled, setTradingEnabled] = useState(true);
  const [memoryStats, setMemoryStats] = useState<any>(null);
  const [priceHistory, setPriceHistory] = useState<{ time: number, price: number }[]>([]);
  const [currentRSI, setCurrentRSI] = useState<number | null>(null);
  const [currentATR, setCurrentATR] = useState<number | null>(null);
  const [currentMACD, setCurrentMACD] = useState<number | null>(null);
  const [showRSI, setShowRSI] = useState(true);
  const [showATR, setShowATR] = useState(true);
  const [showMACD, setShowMACD] = useState(true);
  const [mlPrediction, setMlPrediction] = useState<any>(null);
  const [marketStatus, setMarketStatus] = useState<any>(null);
  const [riskStats, setRiskStats] = useState<any>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [sentiment, setSentiment] = useState<any>(null);
  const [aiAdvice, setAiAdvice] = useState<any>(null);
  const [position, setPosition] = useState<any>(null);
  const [activeTrades, setActiveTrades] = useState<any[]>([]);

  // 🐱 EL GATO Intelligence States
  const [elGatoStatus, setElGatoStatus] = useState<any>(null);
  const [elGatoDailyProgress, setElGatoDailyProgress] = useState<any>(null);
  const [elGatoRecommendation, setElGatoRecommendation] = useState<string>("");

  // Symbol Selection
  const [selectedSymbol, setSelectedSymbol] = useState("BTC/USDT");
  const [tradingMode, setTradingMode] = useState("demo"); // demo, testnet, real

  // Trading Configuration
  const [tradeSize, setTradeSize] = useState(0.001);
  const [brokerFee, setBrokerFee] = useState(0.1);
  const [stopLossEnabled, setStopLossEnabled] = useState(false);
  const [stopLossPercent, setStopLossPercent] = useState(2.0);
  const [takeProfitEnabled, setTakeProfitEnabled] = useState(false);
  const [takeProfitPercent, setTakeProfitPercent] = useState(3.0);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const rsiSeriesRef = useRef<any>(null);
  const atrSeriesRef = useRef<any>(null);
  const macdSeriesRef = useRef<any>(null);

  const runBacktest = async () => {
    setLoadingBacktest(true);
    setBacktestResult(null);
    try {
      const res = await fetch("/api/backtest", {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        setBacktestResult(data);
      }
    } catch (error) {
      console.error("Backtest failed", error);
    } finally {
      setLoadingBacktest(false);
    }
  };

  const executeTrade = async (side: "BUY" | "SELL") => {
    // Calculate trade details
    const currentPrice = livePrice || 86000;
    const totalCost = tradeSize * currentPrice;
    const fee = totalCost * (brokerFee / 100);
    const slPrice = stopLossEnabled ? currentPrice * (1 - stopLossPercent / 100) : null;
    const tpPrice = takeProfitEnabled ? currentPrice * (1 + takeProfitPercent / 100) : null;

    // Show confirmation
    const confirmMsg = `📊 Resumen del Trade\n\n` +
      `Símbolo: ${selectedSymbol}\n` +
      `Lado: ${side}\n` +
      `Cantidad: ${tradeSize.toFixed(4)} (base)\n` +
      `Precio: $${currentPrice.toFixed(2)}\n` +
      `Total: $${totalCost.toFixed(2)}\n` +
      `Comisión (${brokerFee}%): $${fee.toFixed(2)}\n` +
      (stopLossEnabled ? `Stop Loss: $${slPrice?.toFixed(2)} (-${stopLossPercent}%)\n` : '') +
      (takeProfitEnabled ? `Take Profit: $${tpPrice?.toFixed(2)} (+${takeProfitPercent}%)\n` : '') +
      `\n¿Ejecutar orden?`;

    if (!confirm(confirmMsg)) return;

    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: selectedSymbol.replace("/", "_"),
          side: side,
          size: tradeSize,
          stop_loss: slPrice,
          take_profit: tpPrice,
        }),
      });
      const data = await res.json();

      // Verificar si la orden fue exitosa
      if (data.status === "FILLED" || data.status === "SIMULATED") {
        alert(`✅ Orden ${side} ejecutada!\nSímbolo: ${selectedSymbol}\nID: ${data.id}\nPrecio: ~$${currentPrice.toFixed(2)}`);
      } else if (data.error) {
        alert(`❌ Error: ${data.error}`);
      } else {
        alert(`⚠️ Respuesta: ${data.status || data.message || 'Unknown'}`);
      }
    } catch (error) {
      alert("❌ Error de conexión con el backend");
    }
  };

  const toggleTrading = async () => {
    try {
      const newStatus = !tradingEnabled;
      const res = await fetch("/api/trading/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: newStatus }),
      });
      if (res.ok) {
        setTradingEnabled(newStatus);
        // Clear AI advice when resuming auto-trading
        if (newStatus) {
          setAiAdvice(null);
        }
      }
    } catch (error) {
      console.error("Error toggling trading:", error);
    }
  };

  const getAIAdvice = async () => {
    try {
      const res = await fetch("/api/advice");
      if (res.ok) {
        const data = await res.json();
        setAiAdvice(data);
      }
    } catch (error) {
      console.error("Error getting AI advice:", error);
    }
  };

  // Auto-refresh advice when panel is open
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (aiAdvice) {
      interval = setInterval(getAIAdvice, 3000); // Refresh every 3 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [aiAdvice]);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch("/api/health");
        if (res.ok) {
          const data = await res.json();
          setBackendStatus(data.status === "online" ? "Online 🟢" : "Offline 🔴");
        } else {
          setBackendStatus("Offline 🔴");
        }
      } catch (error) {
        setBackendStatus("Offline 🔴");
      }
    };

    const fetchBacktestResults = async () => {
      try {
        const res = await fetch("/api/backtest");
        if (res.ok) {
          const data = await res.json();
          setBacktestResult(data);
        }
      } catch (error) {
        console.error("Failed to fetch backtest results", error);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const res = await fetch("/api/price");
        if (res.ok) {
          const data = await res.json();
          setLivePrice(data.price);
        }
      } catch (error) {
        // Silently fail
      }
    };

    fetchPrice();
    const interval = setInterval(fetchPrice, 1000); // Update every second
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res = await fetch("/api/trades");
        if (res.ok) {
          const data = await res.json();
          setTrades(data.trades || []);
        }
      } catch (error) {
        // Silently fail
      }
    };

    fetchTrades();
    fetchTrades();
    const interval = setInterval(fetchTrades, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  useEffect(() => {
    const fetchPosition = async () => {
      try {
        const res = await fetch("/api/position");
        if (res.ok) {
          const data = await res.json();
          setPosition(data);
        }
      } catch (error) {
        // Silently fail
      }
    };

    fetchPosition();
    const interval = setInterval(fetchPosition, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  useEffect(() => {
    const fetchActiveTrades = async () => {
      try {
        const res = await fetch("/api/trades/active");
        if (res.ok) {
          const data = await res.json();
          setActiveTrades(data.trades || []);
        }
      } catch (error) {
        // Silently fail
      }
    };

    fetchActiveTrades();
    const interval = setInterval(fetchActiveTrades, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  useEffect(() => {
    const fetchMlPrediction = async () => {
      try {
        const res = await fetch("/api/ml/prediction");
        if (res.ok) {
          const data = await res.json();
          setMlPrediction(data);
        }
      } catch (error) { }
    };

    const fetchMarketStatus = async () => {
      try {
        const res = await fetch("/api/market/status");
        if (res.ok) {
          const data = await res.json();
          setMarketStatus(data);
        }
      } catch (error) { }
    };

    const fetchRiskStats = async () => {
      try {
        const res = await fetch("/api/risk/stats");
        if (res.ok) {
          const data = await res.json();
          setRiskStats(data);
        }
      } catch (error) { }
    };

    const fetchSystemHealth = async () => {
      try {
        const res = await fetch("/api/system/health");
        if (res.ok) {
          const data = await res.json();
          setSystemHealth(data);
        }
      } catch (error) { }
    };

    const fetchSentiment = async () => {
      try {
        const res = await fetch("/api/sentiment");
        if (res.ok) {
          const data = await res.json();
          setSentiment(data);
        }
      } catch (error) { }
    };

    // 🐱 EL GATO Fetch Functions
    const fetchElGatoStatus = async () => {
      try {
        const res = await fetch("/api/el-gato/status");
        if (res.ok) {
          const data = await res.json();
          setElGatoStatus(data);
        }
      } catch (error) { }
    };

    const fetchElGatoDailyProgress = async () => {
      try {
        const res = await fetch("/api/el-gato/daily-progress");
        if (res.ok) {
          const data = await res.json();
          setElGatoDailyProgress(data);
        }
      } catch (error) { }
    };

    const fetchElGatoRecommendation = async () => {
      try {
        const res = await fetch("/api/el-gato/recommendation");
        if (res.ok) {
          const data = await res.json();
          setElGatoRecommendation(data.recommendation || "");
        }
      } catch (error) { }
    };

    fetchMlPrediction();
    fetchMarketStatus();
    fetchRiskStats();
    fetchSystemHealth();
    fetchSentiment();
    // 🐱 EL GATO Initial Fetch
    fetchElGatoStatus();
    fetchElGatoDailyProgress();
    fetchElGatoRecommendation();
    const interval = setInterval(() => {
      fetchMlPrediction();
      fetchMarketStatus();
      fetchRiskStats();
      fetchSystemHealth();
      fetchSentiment();
      // 🐱 EL GATO Polling  
      fetchElGatoStatus();
      fetchElGatoDailyProgress();
      fetchElGatoRecommendation();
    }, 2000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  useEffect(() => {
    const fetchBalance = async () => {
      try {
        const res = await fetch("/api/balance");
        if (res.ok) {
          const data = await res.json();
          setBalance(data);
        }
      } catch (error) {
        // Silently fail
      }
    };

    const fetchTradingStatus = async () => {
      try {
        const res = await fetch("/api/trading/status");
        if (res.ok) {
          const data = await res.json();
          setTradingEnabled(data.enabled);
        }
        // Fetch mode as well
        const modeRes = await fetch("/api/trading/mode");
        if (modeRes.ok) {
          const modeData = await modeRes.json();
          setTradingMode(modeData.mode);
        }
      } catch (error) {
        // Silently fail
      }
    };

    const fetchPnl = async () => {
      try {
        const res = await fetch("/api/pnl");
        if (res.ok) {
          const data = await res.json();
          setPnl(data);
        }
      } catch (error) {
        // Silently fail
      }
    };

    const fetchMemory = async () => {
      try {
        const res = await fetch("/api/memory");
        if (res.ok) {
          const data = await res.json();
          setMemoryStats(data);
        }
      } catch (error) { }
    };



    const fetchCandles = async () => {
      try {
        const res = await fetch("/api/candles");
        if (res.ok) {
          const data = await res.json();
          if (candlestickSeriesRef.current && data.candles.length > 0) {
            // Eliminar duplicados de tiempo para evitar errores de lightweight-charts
            const uniqueCandles = data.candles.reduce((acc: any[], current: any) => {
              const x = acc.find((item: any) => item.time === current.time);
              if (!x) {
                return acc.concat([current]);
              } else {
                return acc;
              }
            }, []);

            candlestickSeriesRef.current.setData(uniqueCandles);

            // Force auto-scaling to fit new data range
            if (chartRef.current) {
              chartRef.current.priceScale('right').applyOptions({
                autoScale: true,
              });
            }

            // Agregar marcadores de trades
            const markers = trades.map(t => {
              // Ajustar tiempo del trade al inicio de la vela de 5s correspondiente
              const tradeTime = Math.floor(t.time);
              const candleTime = tradeTime - (tradeTime % 5);

              return {
                time: candleTime,
                position: t.side === 'BUY' ? 'belowBar' : 'aboveBar',
                color: t.side === 'BUY' ? '#22c55e' : '#ef4444',
                shape: t.side === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: t.side === 'BUY' ? 'BUY' : 'SELL',
              };
            });
            candlestickSeriesRef.current.setMarkers(markers);
          }
        }
      } catch (error) {
        // Silently fail
      }
    };

    const fetchIndicators = async () => {
      try {
        const res = await fetch("/api/indicators");
        if (res.ok) {
          // Process indicator data  
          const data = await res.json();

          const currentPrice = livePrice || 90000;

          // RSI: Use raw values (0-100)
          if (rsiSeriesRef.current && data.rsi && data.rsi.length > 0) {
            if (showRSI) {
              rsiSeriesRef.current.setData(data.rsi);
            } else {
              rsiSeriesRef.current.setData([]);
            }
            setCurrentRSI(data.rsi[data.rsi.length - 1]?.value || null);
          }

          // ATR: Use raw values
          if (atrSeriesRef.current && data.atr && data.atr.length > 0) {
            if (showATR) {
              atrSeriesRef.current.setData(data.atr);
            } else {
              atrSeriesRef.current.setData([]);
            }
            const lastATR = data.atr[data.atr.length - 1]?.value || 0;
            setCurrentATR((lastATR / currentPrice) * 100);
          }

          // MACD: Use raw histogram values with colors
          if (macdSeriesRef.current && data.macd && data.macd.length > 0) {
            if (showMACD) {
              const macdData = data.macd.map((m: any) => ({
                time: m.time,
                value: m.histogram,
                color: m.histogram >= 0 ? '#22c55e' : '#ef4444'
              }));
              macdSeriesRef.current.setData(macdData);
            } else {
              macdSeriesRef.current.setData([]);
            }
            setCurrentMACD(data.macd[data.macd.length - 1]?.histogram || null);
          }
        }
      } catch (error) {
        // Silently fail
      }
    };

    fetchBalance();
    fetchTradingStatus();
    fetchPnl();
    fetchMemory();
    fetchIndicators();

    const balanceInterval = setInterval(fetchBalance, 5000);
    const statusInterval = setInterval(fetchTradingStatus, 1000);
    const candleInterval = setInterval(fetchCandles, 1000);
    const pnlInterval = setInterval(fetchPnl, 1000);
    const memoryInterval = setInterval(fetchMemory, 2000);
    const indicatorInterval = setInterval(fetchIndicators, 2000);

    return () => {
      clearInterval(balanceInterval);
      clearInterval(statusInterval);
      clearInterval(candleInterval);
      clearInterval(pnlInterval);
      clearInterval(memoryInterval);
      clearInterval(indicatorInterval);
    };
  }, [trades, selectedSymbol]);

  // Inicializar gráfico
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Limpiar contenedor por si acaso (evita duplicados en Strict Mode)
    chartContainerRef.current.innerHTML = '';

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0d0d0d' }, // Darker, richer black
        textColor: '#a8a8a8', // Softer gray for text
        fontSize: 12,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      },
      grid: {
        vertLines: {
          color: '#1a1a2e', // Subtle dark blue-gray
          style: 1,
          visible: true,
        },
        horzLines: {
          color: '#1a1a2e',
          style: 1,
          visible: true,
        },
      },
      width: chartContainerRef.current.clientWidth,
      height: 550, // Slightly taller for better visibility
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
        borderColor: '#252538',
        borderVisible: true,
        rightOffset: 5,
        barSpacing: 8, // More space between candles
        minBarSpacing: 4,
        fixLeftEdge: false,
        fixRightEdge: false,
        visible: true, // ✅ Ensure time labels are visible
      },
      rightPriceScale: {
        borderColor: '#252538',
        borderVisible: true,
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
        autoScale: true,
        visible: true, // ✅ Ensure price labels are visible
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: '#4488ff',
          style: 2, // Dashed line
          labelBackgroundColor: '#4488ff',
        },
        horzLine: {
          width: 1,
          color: '#4488ff',
          style: 2,
          labelBackgroundColor: '#4488ff',
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    chartRef.current = chart;

    // 🕯️ Premium Candlestick Series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00ff88', // Vibrant green
      downColor: '#ff4466', // Vibrant red
      borderVisible: true,
      wickUpColor: '#00ff88',
      wickDownColor: '#ff4466',
      borderUpColor: '#00ff88',
      borderDownColor: '#ff4466',
      priceScaleId: 'right',
    });

    // RSI Area - separate scale with gradient shadow
    const rsiSeries = chart.addSeries(AreaSeries, {
      topColor: 'rgba(168, 85, 247, 0.4)', // Purple with transparency
      bottomColor: 'rgba(168, 85, 247, 0)', // Fully transparent at bottom
      lineColor: '#a855f7', // Solid purple line
      lineWidth: 3,
      priceScaleId: 'rsi',
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 6,
    });

    // Configure RSI scale (0-100)
    chart.priceScale('rsi').applyOptions({
      scaleMargins: {
        top: 0.02,
        bottom: 0.78,
      },
      borderVisible: false,
    });

    // ATR Area - separate scale with gradient shadow (BLUE)
    const atrSeries = chart.addSeries(AreaSeries, {
      topColor: 'rgba(59, 130, 246, 0.4)', // Blue with transparency
      bottomColor: 'rgba(59, 130, 246, 0)', // Fully transparent at bottom
      lineColor: '#3b82f6', // Solid blue line
      lineWidth: 3,
      priceScaleId: 'atr',
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 6,
    });

    // Configure ATR scale
    chart.priceScale('atr').applyOptions({
      scaleMargins: {
        top: 0.82,
        bottom: 0.12,
      },
      borderVisible: false,
    });

    // MACD Histogram - separate scale
    const macdSeries = chart.addSeries(HistogramSeries, {
      color: '#26a69a',
      priceScaleId: 'macd',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Configure MACD scale
    chart.priceScale('macd').applyOptions({
      scaleMargins: {
        top: 0.92,
        bottom: 0.02,
      },
      borderVisible: false,
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    rsiSeriesRef.current = rsiSeries;
    atrSeriesRef.current = atrSeries;
    macdSeriesRef.current = macdSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);


  return (
    <div className="min-h-screen bg-black text-white">
      <main className="mx-auto flex w-full max-w-[98%] flex-col gap-10 px-4 py-8 lg:py-10">
        <header className="space-y-6 rounded-2xl border border-white/10 bg-white/5 p-8 shadow-panel backdrop-blur">
          <div className="flex justify-between items-center">
            <p className="text-sm uppercase tracking-[0.3em] text-gray-400 font-medium">
              Vibe Factory
            </p>
            <div className="flex items-center gap-2 text-sm font-mono border border-white/10 px-3 py-1 rounded-full bg-black/20">
              <span>Backend:</span>
              <span className={backendStatus.includes("Online") ? "text-green-400" : "text-red-400"}>
                {backendStatus}
              </span>
            </div>
          </div>
          <h1 className="text-4xl font-semibold leading-tight md:text-5xl">
            Configuración base para construir apps guiadas por IA en minutos.
          </h1>
          <p className="max-w-3xl text-lg text-gray-300">
            Esta vista resume el estado de la fábrica: frontend listo, módulos
            backend definidos y documentación en marcha. Usa esta página como
            panel central para seguir los pasos y saber qué tocar a
            continuación.
          </p>
          <div className="flex flex-wrap gap-4">
            <a
              href="https://nextjs.org/docs"
              target="_blank"
              rel="noreferrer"
              className="rounded-full bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90"
            >
              Ver guía Next.js
            </a>
            <a
              href="https://tailwindcss.com/docs"
              target="_blank"
              rel="noreferrer"
              className="rounded-full border border-foreground/20 px-6 py-3 text-sm font-semibold hover:bg-foreground/5"
            >
              Customizar tema
            </a>
          </div>
        </header>

        <section className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {steps.map((step) => (
            <article
              key={step.title}
              className="flex flex-col gap-4 rounded-2xl border border-foreground/10 bg-white/5 p-6 shadow-panel backdrop-blur"
            >
              <div>
                <p className="text-sm font-semibold text-primary">
                  {step.title}
                </p>
                <p className="mt-2 text-base text-gray-300">
                  {step.description}
                </p>
              </div>
              <ul className="space-y-2 text-sm text-gray-400">
                {step.checklist.map((item) => (
                  <li
                    key={item}
                    className="flex items-center gap-2 rounded-lg bg-white/5 px-3 py-2"
                  >
                    <span className="h-2 w-2 rounded-full bg-success" />
                    {item}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/5 p-8 shadow-panel backdrop-blur">
          <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-2xl font-semibold flex items-center gap-3">
                <span className="relative flex h-3 w-3">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${tradingEnabled ? 'bg-green-400' : 'bg-yellow-400'}`}></span>
                  <span className={`relative inline-flex rounded-full h-3 w-3 ${tradingEnabled ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
                </span>
                Live Trading Engine
              </h2>
              <p className="text-gray-400 font-medium mt-1">
                {balance?.mode === 'real'
                  ? "⚠️ MODO REAL: Operando con fondos reales en el Exchange."
                  : "🛡️ MODO DEMO: Simulando operaciones (Paper Trading)."}
              </p>
            </div>

            <div className="flex gap-4">
              {/* Mode Indicator */}
              <div className={`px-4 py-2 rounded-lg border flex items-center gap-2 font-mono text-sm ${balance?.mode === 'real'
                ? 'bg-red-500/10 border-red-500/50 text-red-400'
                : 'bg-blue-500/10 border-blue-500/50 text-blue-400'
                }`}>
                {balance?.mode === 'real' ? '🔴 REAL MONEY' : '🔵 DEMO MODE'}
              </div>

              <button
                onClick={toggleTrading}
                className={`rounded-full px-8 py-3 font-bold text-white transition hover:opacity-90 ${tradingEnabled ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                  }`}
              >
                {tradingEnabled ? "⏹ STOP TRADING" : "▶ START TRADING"}
              </button>

              <button
                onClick={async () => {
                  if (!confirm("⚠️ ¿ESTÁS SEGURO? Esto venderá TODO tu BTC al precio de mercado actual.")) return;
                  try {
                    const res = await fetch('/api/trade/close', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ symbol: "BTC_USDT" }),
                    });
                    const data = await res.json();
                    if (data.status === "FILLED" || data.status === "SIMULATED") {
                      alert(`✅ POSICIÓN CERRADA\nPrecio: $${data.price}\nID: ${data.id}`);
                    } else {
                      alert(`❌ Error: ${data.message || "No se pudo cerrar"}`);
                    }
                  } catch (e) {
                    alert("❌ Error de conexión");
                  }
                }}
                className="rounded-full bg-orange-600 px-6 py-3 font-bold text-white transition hover:bg-orange-700 flex items-center gap-2"
                title="Vender todo el BTC inmediatamente (PÁNICO)"
              >
                🚨 CLOSE ALL
              </button>
            </div>
          </div>

          {/* Stats Row */}
          <div className="mt-8 grid gap-4 border-t border-white/10 pt-8 sm:grid-cols-4">
            <div className="rounded-xl bg-black/20 p-4">
              <p className="text-sm text-gray-400 mb-1 font-medium">Balance USDT</p>
              <p className="text-2xl font-mono text-green-400">
                ${balance?.USDT.toLocaleString(undefined, { minimumFractionDigits: 2 }) || '---'}
              </p>
            </div>
            <div className="rounded-xl bg-black/20 p-4">
              <p className="text-sm text-gray-400 mb-1 font-medium">Balance BTC</p>
              <p className="text-2xl font-mono text-yellow-400">
                {balance?.BTC.toFixed(6) || '---'}
              </p>
            </div>
            <div className="rounded-xl bg-black/20 p-4">
              <p className="text-sm text-gray-400 mb-1 font-medium">PnL Realizado</p>
              <p className={`text-2xl font-mono ${pnl?.realized_pnl && pnl.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${pnl?.realized_pnl.toFixed(2) || '0.00'}
              </p>
            </div>
            <div className="rounded-xl bg-black/20 p-4">
              <p className="text-sm text-gray-400 mb-1 font-medium">Trades Hoy</p>
              <p className="text-2xl font-mono text-blue-400">
                {systemHealth?.trades_today || 0}
              </p>
            </div>
          </div>

          {/* Position Status Card */}
          {position && (
            <div className={`mt-6 rounded-2xl border p-6 transition-all duration-300 ${position.is_open
              ? 'bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border-yellow-500/30'
              : 'bg-black/20 border-white/10'
              }`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold flex items-center gap-2">
                  {position.is_open ? (
                    <>
                      <span className="relative flex h-2.5 w-2.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-yellow-500"></span>
                      </span>
                      🟢 POSICIÓN ABIERTA
                    </>
                  ) : (
                    <>🔴 SIN POSICIÓN</>
                  )}
                </h3>
                {position.is_open && (
                  <div className={`px-3 py-1 rounded-lg font-mono text-sm font-bold ${position.unrealized_pnl_usd >= 0
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-red-500/20 text-red-400 border border-red-500/30'
                    }`}>
                    {position.unrealized_pnl_usd >= 0 ? '+' : ''}${position.unrealized_pnl_usd.toFixed(2)}
                    {' '}({position.unrealized_pnl_pct >= 0 ? '+' : ''}{position.unrealized_pnl_pct.toFixed(2)}%)
                  </div>
                )}
              </div>

              {position.is_open && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="rounded-xl bg-black/30 p-3">
                    <p className="text-xs text-gray-400 mb-1">Tamaño</p>
                    <p className="text-lg font-mono text-yellow-400">{position.size.toFixed(6)} BTC</p>
                  </div>
                  <div className="rounded-xl bg-black/30 p-3">
                    <p className="text-xs text-gray-400 mb-1">Precio Entrada</p>
                    <p className="text-lg font-mono text-blue-400">${position.entry_price.toLocaleString()}</p>
                  </div>
                  <div className="rounded-xl bg-black/30 p-3">
                    <p className="text-xs text-gray-400 mb-1">Precio Actual</p>
                    <p className="text-lg font-mono text-purple-400">${position.current_price.toLocaleString()}</p>
                  </div>
                  <div className="rounded-xl bg-black/30 p-3">
                    <p className="text-xs text-gray-400 mb-1">PnL No Realizado</p>
                    <p className={`text-lg font-mono font-bold ${position.unrealized_pnl_usd >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                      {position.unrealized_pnl_usd >= 0 ? '+' : ''}${position.unrealized_pnl_usd.toFixed(2)}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Individual Trades Table */}
        {activeTrades.length > 0 && (
          <section className="mt-6">
            <h3 className="text-2xl font-bold mb-4">📊 Trades Activos</h3>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-gray-400 border-b border-white/10">
                    <th className="text-left py-3 px-4">ID</th>
                    <th className="text-left py-3 px-4">Lado</th>
                    <th className="text-right py-3 px-4">Tamaño</th>
                    <th className="text-right py-3 px-4">Entrada</th>
                    <th className="text-right py-3 px-4">Actual</th>
                    <th className="text-right py-3 px-4">PnL</th>
                    <th className="text-center py-3 px-4">Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {activeTrades.filter(t => t.symbol === undefined || t.symbol === selectedSymbol.replace("/", "_")).map((trade) => (
                    <tr key={trade.id} className="border-b border-white/5 hover:bg-white/5 transition">
                      <td className="py-3 px-4 font-mono">#{trade.id}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${trade.side === 'LONG'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                          }`}>
                          {trade.side}
                        </span>
                      </td>
                      <td className="text-right py-3 px-4 font-mono text-yellow-400">
                        {trade.size.toFixed(4)} BTC
                      </td>
                      <td className="text-right py-3 px-4 font-mono text-blue-400">
                        ${trade.entry_price.toLocaleString()}
                      </td>
                      <td className="text-right py-3 px-4 font-mono text-purple-400">
                        ${trade.current_price.toLocaleString()}
                      </td>
                      <td className={`text-right py-3 px-4 font-mono font-bold ${trade.unrealized_pnl_usd >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                        {trade.unrealized_pnl_usd >= 0 ? '+' : ''}${trade.unrealized_pnl_usd.toFixed(2)}
                        <span className="text-xs ml-2">
                          ({trade.unrealized_pnl_pct >= 0 ? '+' : ''}{trade.unrealized_pnl_pct.toFixed(2)}%)
                        </span>
                      </td>
                      <td className="text-center py-3 px-4">
                        <div className="flex gap-2 justify-center">
                          <button
                            onClick={async () => {
                              if (!confirm(`¿Cerrar trade #${trade.id}?`)) return;
                              try {
                                const res = await fetch(`/api/trades/close/${trade.id}`, { method: 'POST' });
                                const data = await res.json();
                                if (data.success) {
                                  alert(`✅ Trade cerrado con PnL: $${data.trade.pnl.toFixed(2)}`);
                                }
                              } catch (e) {
                                alert("❌ Error cerrando trade");
                              }
                            }}
                            className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm font-bold transition"
                          >
                            Cerrar
                          </button>
                          <button
                            onClick={async () => {
                              try {
                                const res = await fetch(`/api/trades/reverse/${trade.id}`, { method: 'POST' });
                                const data = await res.json();
                                if (data.success) {
                                  alert(`🔄 Trade invertido: ${data.trade.side}`);
                                }
                              } catch (e) {
                                alert("❌ Error invirtiendo trade");
                              }
                            }}
                            className="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm font-bold transition"
                          >
                            🔄 Invertir
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Trading Terminal Section - ALWAYS VISIBLE */}
        <section className="grid gap-6 lg:grid-cols-4">
          {/* Main Chart & Controls Area (3/4 width) */}
          <div className="lg:col-span-3 space-y-4">
            {/* Chart Card - simple container, 3D effects inside */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-1 shadow-lg backdrop-blur overflow-hidden relative">
              {/* Chart with internal glow effect */}
              <div
                ref={chartContainerRef}
                className="w-full h-[500px] bg-black/20 rounded-xl"
                style={{
                  filter: 'drop-shadow(0 0 20px rgba(147, 51, 234, 0.15)) drop-shadow(0 0 15px rgba(59, 130, 246, 0.15))'
                }}
              />
              {/* Force yellow text for chart labels */}
              <style jsx>{`
                div[ref='chartContainerRef'] * {
                  color: #fbbf24 !important;
                }
              `}</style>
              {/* Live Price Overlay - BIGGER and YELLOW */}
              <div className="absolute top-4 left-4 flex items-center gap-2">
                <div className="bg-black/80 backdrop-blur-md px-4 py-2.5 rounded-lg border border-yellow-500/30 shadow-lg">
                  <span className="text-xs text-gray-400 mr-2 uppercase">{selectedSymbol}</span>
                  <span className="font-mono font-bold text-2xl text-yellow-400">${livePrice?.toFixed(2) || '---'}</span>
                </div>
                {/* LIVE Indicator with glow effect */}
                <div className="bg-green-500/20 backdrop-blur-md px-3 py-2.5 rounded-lg border border-green-500/50 flex items-center gap-2 shadow-[0_0_20px_rgba(34,197,94,0.3)]">
                  <div className="w-2.5 h-2.5 bg-green-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.8)]"></div>
                  <span className="text-sm font-bold text-green-400">LIVE</span>
                </div>
              </div>
            </div>

            {/* Indicator Legend with Toggles */}
            <div className="flex items-center justify-center gap-6 py-2 px-4 bg-black/20 rounded-lg border border-white/10">
              <div
                className={`flex items-center gap-2 px-2 py-1 rounded transition-colors cursor-pointer ${showRSI ? 'bg-white/5' : 'opacity-50 hover:opacity-100'}`}
                onClick={() => setShowRSI(!showRSI)}
                title="Click to toggle RSI graph visibility"
              >
                <div className={`w-3 h-0.5 rounded-full shadow-[0_0_8px_rgba(147,51,234,0.6)] ${showRSI ? 'bg-purple-500' : 'bg-gray-500'}`}></div>
                <span className="text-xs text-purple-300 font-mono">RSI</span>
                <span className="text-xs text-gray-400 font-medium">Momentum</span>
                <span className="text-sm font-mono font-bold text-purple-400 ml-1">
                  {currentRSI !== null ? `${currentRSI.toFixed(1)}%` : '--'}
                </span>
                <span className="text-xs ml-1 text-foreground/30">{showRSI ? '👁️' : '🚫'}</span>
              </div>
              <div
                className={`flex items-center gap-2 px-2 py-1 rounded transition-colors cursor-pointer ${showATR ? 'bg-white/5' : 'opacity-50 hover:opacity-100'}`}
                onClick={() => setShowATR(!showATR)}
                title="Click to toggle ATR graph visibility"
              >
                <div className={`w-3 h-0.5 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.6)] ${showATR ? 'bg-blue-500' : 'bg-gray-500'}`}></div>
                <span className="text-xs text-blue-300 font-mono">ATR</span>
                <span className="text-xs text-gray-400 font-medium">Volatilidad</span>
                <span className="text-sm font-mono font-bold text-blue-400 ml-1">
                  {currentATR !== null ? `${currentATR.toFixed(2)}%` : '--'}
                </span>
                <span className="text-xs ml-1 text-foreground/30">{showATR ? '👁️' : '🚫'}</span>
              </div>
              <div
                className={`flex items-center gap-2 px-2 py-1 rounded transition-colors cursor-pointer ${showMACD ? 'bg-white/5' : 'opacity-50 hover:opacity-100'}`}
                onClick={() => setShowMACD(!showMACD)}
                title="Click to toggle MACD histogram visibility"
              >
                <div className={`w-3 h-0.5 rounded-full shadow-[0_0_8px_rgba(34,197,94,0.6)] ${showMACD ? 'bg-green-500' : 'bg-gray-500'}`}></div>
                <span className="text-xs text-green-300 font-mono">MACD</span>
                <span className="text-xs text-gray-400 font-medium">Tendencia</span>
                <span className="text-sm font-mono font-bold text-green-400 ml-1">
                  {currentMACD !== null ? `${currentMACD.toFixed(2)}` : '--'}
                </span>
                <span className="text-xs ml-1 text-foreground/30">{showMACD ? '👁️' : '🚫'}</span>
              </div>
            </div>

            {/* Trading Control Panel (Below Chart) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

              {/* 1. PnL & Position Info */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur flex flex-col justify-center">
                <div className="flex justify-between items-start mb-1">
                  <p className="text-xs uppercase tracking-wider text-gray-400 font-medium">Unrealized PnL</p>
                  {balance && <span className="text-xs font-mono text-yellow-400">Bal: {balance.BTC.toFixed(4)} BTC</span>}
                </div>
                {pnl ? (
                  <>
                    <div className={`text-3xl font-bold font-mono ${pnl.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pnl.unrealized_pnl >= 0 ? '+' : ''}{pnl.unrealized_pnl.toFixed(2)} <span className="text-sm text-gray-400">USDT</span>
                    </div>
                    <div className="flex justify-between text-sm mt-2">
                      <span className="text-gray-400">ROE: <span className={pnl.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                        {pnl.entry_price > 0 ? ((pnl.unrealized_pnl / (pnl.entry_price * Math.abs(pnl.position_size))) * 100).toFixed(2) : '0.00'}%
                      </span></span>
                      <span className="text-gray-400">Size: {Math.abs(pnl.position_size).toFixed(3)} BTC</span>
                    </div>
                  </>
                ) : (
                  <div className="text-2xl font-mono text-foreground/30">--.--</div>
                )}
              </div>

              {/* Symbol Selector */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur space-y-4">

                {/* Trading Mode Selector */}
                <div>
                  <label className="text-xs text-white/60 block mb-2">🎮 Trading Mode</label>
                  <select
                    value={tradingMode}
                    onChange={async (e) => {
                      const newMode = e.target.value;
                      setTradingMode(newMode);
                      try {
                        await fetch("/api/trading/mode", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ mode: newMode }),
                        });
                        alert(`Mode switched to ${newMode.toUpperCase()}`);
                      } catch (err) {
                        console.error("Failed to set mode:", err);
                      }
                    }}
                    className={`w-full px-3 py-2 border rounded font-mono text-sm focus:outline-none cursor-pointer ${tradingMode === 'real' ? 'bg-red-900/30 border-red-500/50 text-red-400' :
                      tradingMode === 'testnet' ? 'bg-yellow-900/30 border-yellow-500/50 text-yellow-400' :
                        tradingMode === 'coinbase' ? 'bg-cyan-900/30 border-cyan-500/50 text-cyan-400' :
                          'bg-black/30 border-white/10 text-white'
                      }`}
                  >
                    <option value="demo">📝 Paper Trading (Internal Sim)</option>
                    <option value="testnet">🧪 Binance Testnet (Sandbox)</option>
                    <option value="real">🚀 Binance Real (Mainnet)</option>
                    <option value="coinbase">🪙 Coinbase (BTC, ETH, etc.)</option>
                  </select>
                </div>

                {/* 🐱 EL GATO Intelligence Dashboard */}
                {elGatoStatus && (
                  <div className="border-t border-white/10 pt-4">
                    <div className="text-white/80 mb-3">
                      <span className="text-lg">🐱</span>
                      <span className="ml-2 font-bold">EL GATO</span>
                      <span className="ml-2 text-xs text-white/50">v{elGatoStatus.version}</span>
                    </div>

                    {/* IQ & Tier */}
                    <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg p-3 mb-3 border border-blue-500/30">
                      <div className="flex justify-between items-center mb-2">
                        <div>
                          <div className="text-xs text-white/50">Intelligence Level</div>
                          <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
                            IQ {Math.round(elGatoStatus.iq_level)}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-white/50">Tier {elGatoStatus.evolution_tier}/10</div>
                          <div className="text-sm font-semibold text-yellow-400">{elGatoStatus.tier_name}</div>
                        </div>
                      </div>
                      <div className="w-full bg-black/30 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${Math.min((elGatoStatus.iq_level % 100), 100)}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Daily Progress */}
                    {elGatoDailyProgress && (
                      <div className="bg-white/5 rounded-lg p-3 mb-3">
                        <div className="text-xs text-white/50 mb-1">Daily Objective</div>
                        <div className="flex justify-between items-baseline mb-2">
                          <div className="text-lg font-bold text-green-400">
                            ${elGatoDailyProgress.current_profit?.toFixed(2) || '0.00'}
                          </div>
                          <div className="text-xs text-white/50">
                            / ${elGatoDailyProgress.daily_target?.toFixed(0) || '0'}
                          </div>
                        </div>
                        <div className="w-full bg-black/30 rounded-full h-2 mb-1">
                          <div
                            className={`h-2 rounded-full transition-all duration-500 ${(elGatoDailyProgress.progress_pct || 0) >= 100
                              ? 'bg-gradient-to-r from-green-500 to-emerald-500'
                              : 'bg-gradient-to-r from-yellow-500 to-orange-500'
                              }`}
                            style={{ width: `${Math.min(elGatoDailyProgress.progress_pct || 0, 100)}%` }}
                          ></div>
                        </div>
                        <div className="text-xs text-white/40">
                          {elGatoDailyProgress.progress_pct?.toFixed(1) || '0.0'}% complete
                        </div>
                      </div>
                    )}

                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                      <div className="bg-white/5 rounded p-2">
                        <div className="text-white/50">Trades</div>
                        <div className="font-bold">{elGatoStatus.trades_executed || 0}</div>
                      </div>
                      <div className="bg-white/5 rounded p-2">
                        <div className="text-white/50">Win Rate</div>
                        <div className="font-bold text-green-400">{elGatoStatus.win_rate?.toFixed(1) || '0.0'}%</div>
                      </div>
                    </div>

                    {/* Capabilities */}
                    <div className="text-xs">
                      <div className="text-white/50 mb-1">Active Capabilities ({elGatoStatus.unlocked_capabilities?.length || 0})</div>
                      <div className="flex flex-wrap gap-1">
                        {(elGatoStatus.unlocked_capabilities || []).slice(0, 3).map((cap: string, idx: number) => (
                          <span key={idx} className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs">
                            {cap.replace(/_/g, ' ')}
                          </span>
                        ))}
                        {(elGatoStatus.unlocked_capabilities?.length || 0) > 3 && (
                          <span className="px-2 py-1 bg-white/10 text-white/50 rounded text-xs">
                            +{(elGatoStatus.unlocked_capabilities?.length || 0) - 3} more
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Recommendation */}
                    {elGatoRecommendation && (
                      <div className="mt-3 bg-purple-500/10 border border-purple-500/30 rounded p-2 text-xs text-purple-200">
                        💡 {elGatoRecommendation}
                      </div>
                    )}
                  </div>
                )}

                {/* Symbol Selector */}
                <div>
                  <label className="text-xs text-white/60 block mb-2">📊 Select Cryptocurrency</label>
                  <select
                    value={selectedSymbol}
                    onChange={async (e) => {
                      const newSymbol = e.target.value;
                      setSelectedSymbol(newSymbol);

                      // Reset trade size based on asset class (heuristic)
                      if (newSymbol.includes("BTC") || newSymbol.includes("ETH")) {
                        setTradeSize(0.001);
                      } else {
                        setTradeSize(10.0); // Default for cheaper assets
                      }

                      // Notificar al backend para cambiar el feed de datos
                      try {
                        await fetch("/api/set_symbol", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ symbol: newSymbol }),
                        });
                        // Resetear datos visuales
                        setMlPrediction(null);
                        if (candlestickSeriesRef.current) candlestickSeriesRef.current.setData([]);
                      } catch (err) {
                        console.error("Failed to set symbol:", err);
                      }
                    }}
                    className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded text-white font-mono text-sm focus:border-blue-500 focus:outline-none cursor-pointer"
                  >
                    <option value="BTC/USDT">₿ Bitcoin (BTC)</option>
                    <option value="ETH/USDT">Ξ Ethereum (ETH)</option>
                    <option value="BNB/USDT">⬥ Binance Coin (BNB)</option>
                    <option value="SOL/USDT">◎ Solana (SOL)</option>
                    <option value="XRP/USDT">✕ Ripple (XRP)</option>
                    <option value="ADA/USDT">₳ Cardano (ADA)</option>
                    <option value="AVAX/USDT">🔺 Avalanche (AVAX)</option>
                    <option value="DOGE/USDT">Ð Dogecoin (DOGE)</option>
                    <option value="DOT/USDT">● Polkadot (DOT)</option>
                    <option value="MATIC/USDT">⬡ Polygon (MATIC)</option>
                  </select>
                  <div className="text-xs text-white/40 mt-2">
                    Trading: <span className="text-blue-400 font-bold">{selectedSymbol}</span>
                  </div>
                </div>
              </div>

              {/* 2. Trading Configuration */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur flex flex-col gap-3">
                <h4 className="text-sm font-bold text-white/80">⚙️ Configuración de Trading</h4>

                {/* Trade Size */}
                <div>
                  <label className="text-xs text-white/60">Cantidad (BTC)</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={tradeSize}
                    onChange={(e) => setTradeSize(parseFloat(e.target.value) || 0)}
                    className="w-full mt-1 px-3 py-2 bg-black/30 border border-white/10 rounded text-white font-mono text-sm focus:border-blue-500 focus:outline-none"
                  />
                  <div className="text-xs text-white/40 mt-1">
                    ≈ ${((tradeSize * (livePrice || 86000))).toFixed(2)} USD
                  </div>
                </div>

                {/* Broker Fee */}
                <div>
                  <label className="text-xs text-white/60">Comisión del Broker (%)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={brokerFee}
                    onChange={(e) => setBrokerFee(parseFloat(e.target.value) || 0)}
                    className="w-full mt-1 px-3 py-2 bg-black/30 border border-white/10 rounded text-white font-mono text-sm focus:border-blue-500 focus:outline-none"
                  />
                </div>

                {/* Stop Loss */}
                <div className="flex items-center gap-2 p-2 bg-black/20 rounded">
                  <input
                    type="checkbox"
                    checked={stopLossEnabled}
                    onChange={(e) => setStopLossEnabled(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <div className="flex-1">
                    <label className="text-xs text-red-400 font-bold">Stop Loss</label>
                    <input
                      type="number"
                      step="0.1"
                      disabled={!stopLossEnabled}
                      value={stopLossPercent}
                      onChange={(e) => setStopLossPercent(parseFloat(e.target.value) || 0)}
                      className="w-full mt-1 px-2 py-1 bg-black/30 border border-red-500/30 rounded text-red-400 font-mono text-xs disabled:opacity-30 focus:border-red-500 focus:outline-none"
                      placeholder="%"
                    />
                  </div>
                </div>

                {/* Take Profit */}
                <div className="flex items-center gap-2 p-2 bg-black/20 rounded">
                  <input
                    type="checkbox"
                    checked={takeProfitEnabled}
                    onChange={(e) => setTakeProfitEnabled(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <div className="flex-1">
                    <label className="text-xs text-green-400 font-bold">Take Profit</label>
                    <input
                      type="number"
                      step="0.1"
                      disabled={!takeProfitEnabled}
                      value={takeProfitPercent}
                      onChange={(e) => setTakeProfitPercent(parseFloat(e.target.value) || 0)}
                      className="w-full mt-1 px-2 py-1 bg-black/30 border border-green-500/30 rounded text-green-400 font-mono text-xs disabled:opacity-30 focus:border-green-500 focus:outline-none"
                      placeholder="%"
                    />
                  </div>
                </div>
              </div>

              {/* 3. Manual Controls */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur flex flex-col gap-2">
                <div className="flex gap-2">
                  <button
                    onClick={() => executeTrade("BUY")}
                    className="flex-1 bg-green-500/20 hover:bg-green-500/30 text-green-400 border border-green-500/50 py-3 rounded-lg font-bold transition-all active:scale-95"
                  >
                    BUY / LONG
                  </button>
                  <button
                    onClick={() => executeTrade("SELL")}
                    className="flex-1 bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/50 py-3 rounded-lg font-bold transition-all active:scale-95"
                  >
                    SELL / SHORT
                  </button>
                </div>
                <button
                  onClick={toggleTrading}
                  className={`w-full py-2 rounded-lg font-mono text-sm border transition-all ${tradingEnabled
                    ? "bg-red-500/10 text-red-400 border-red-500/30 hover:bg-red-500/20"
                    : "bg-green-500/10 text-green-400 border-green-500/30 hover:bg-green-500/20"
                    }`}
                >
                  {tradingEnabled ? "🛑 PAUSE AUTO-TRADING" : "▶ RESUME AUTO-TRADING"}
                </button>

                {/* AI Advice Button - Only visible when trading is paused */}
                {!tradingEnabled && (
                  <button
                    onClick={getAIAdvice}
                    className="w-full py-2 rounded-lg font-mono text-sm border bg-purple-500/10 text-purple-400 border-purple-500/30 hover:bg-purple-500/20 transition-all"
                  >
                    🤖 OBTENER CONSEJO DE IA
                  </button>
                )}
              </div>

              {/* 3. Last Trade Info */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur flex flex-col justify-center">
                <p className="text-xs uppercase tracking-wider text-gray-400 font-medium mb-1">Last Trade</p>
                {lastTrade ? (
                  <>
                    <div className="text-3xl font-bold font-mono">
                      <span className={lastTrade.side === 'BUY' ? 'text-green-400' : 'text-red-400'}>
                        {lastTrade.side}
                      </span>
                      <span className="text-sm text-gray-500 ml-2">@ {lastTrade.price?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-sm mt-2">
                      <span className="text-gray-400">Size: {lastTrade.size?.toFixed(3)} BTC</span>
                      <span className="text-gray-400">
                        {lastTrade.time ? new Date(lastTrade.time * 1000).toLocaleTimeString() : 'N/A'}
                      </span>
                    </div>
                  </>
                ) : (
                  <p className="py-8 text-center text-foreground/40">No trade yet</p>
                )}
              </div>
            </div>

            {/* AI Advice Panel */}
            {aiAdvice && (
              <div className="rounded-xl border border-purple-500/30 bg-purple-500/5 p-4 backdrop-blur">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold text-purple-300 flex items-center gap-2">
                    🤖 Consejo de IA
                    <span className={`text-xs px-2 py-0.5 rounded ${aiAdvice.confidence === 'high' ? 'bg-green-500/20 text-green-400' :
                      aiAdvice.confidence === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                      {aiAdvice.confidence === 'high' ? 'Alta confianza' :
                        aiAdvice.confidence === 'medium' ? 'Media confianza' : 'Baja confianza'}
                    </span>
                  </h3>
                  <button
                    onClick={() => setAiAdvice(null)}
                    className="text-foreground/40 hover:text-foreground/80 transition-colors text-lg"
                  >
                    ✕
                  </button>
                </div>
                <p className="text-sm text-white mb-3 leading-relaxed whitespace-pre-line font-medium">
                  {aiAdvice.advice}
                </p>

                {/* Trade Setup Box */}
                {aiAdvice.action !== 'wait' && aiAdvice.trade_setup && (
                  <div className="mb-3 p-3 bg-black/40 rounded border border-white/10 grid grid-cols-3 gap-2 text-center">
                    <div>
                      <span className="text-[10px] text-yellow-400/70 uppercase block">Entry</span>
                      <span className="font-mono text-yellow-400 font-bold">${aiAdvice.trade_setup.entry}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-red-400/70 uppercase block">Stop Loss</span>
                      <span className="font-mono text-red-400 font-bold">${aiAdvice.trade_setup.sl}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-green-400/70 uppercase block">Take Profit</span>
                      <span className="font-mono text-green-400 font-bold">${aiAdvice.trade_setup.tp}</span>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between text-xs flex-wrap gap-2">
                  <div className="flex gap-3">
                    <span className="text-purple-300 font-mono">RSI: {aiAdvice.indicators.rsi}</span>
                    <span className="text-blue-300 font-mono">ATR: {aiAdvice.indicators.atr}</span>
                    <span className="text-white/60">{aiAdvice.indicators.trend} / {aiAdvice.indicators.volatility}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {aiAdvice.win_probability && (
                      <span className="px-2 py-1 bg-yellow-500/10 text-yellow-400 rounded font-mono font-bold border border-yellow-500/20">
                        🎯 {aiAdvice.win_probability}% Win Prob
                      </span>
                    )}
                    <span className={`px-3 py-1 rounded font-mono font-bold ${aiAdvice.action === 'buy' ? 'bg-green-500/20 text-green-400' :
                      aiAdvice.action === 'sell' ? 'bg-red-500/20 text-red-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                      {aiAdvice.action === 'buy' ? '📈 COMPRAR' :
                        aiAdvice.action === 'sell' ? '📉 VENDER' : '⏸️ ESPERAR'}
                    </span>
                  </div>
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* 4. AI Brain Dashboard (Enhanced) */}
              <div className="md:col-span-3 rounded-xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-blue-500/10 p-4 backdrop-blur">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/20 rounded-lg">
                      <span className="text-2xl">🧠</span>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-purple-300">AI Learning System</p>
                      <p className="text-xs text-purple-200/50">Self-Optimizing Trading Bot</p>
                    </div>
                    {marketStatus && (
                      <div className={`ml-4 px-3 py-1 rounded-full border text-xs font-mono flex items-center gap-2 ${marketStatus.is_peak
                        ? 'bg-yellow-500/20 border-yellow-500/30 text-yellow-300'
                        : 'bg-blue-500/20 border-blue-500/30 text-blue-300'
                        }`}>
                        <span>{marketStatus.is_peak ? '🔥 PEAK HOURS' : '🌙 OFF HOURS'}</span>
                        <span className="opacity-50">|</span>
                        <span>{marketStatus.aggressiveness}x Aggro</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* ML Prediction Panel */}
                <div className="mb-4 bg-black/20 rounded-lg p-4 border border-purple-500/20">
                  <h3 className="text-sm font-bold text-purple-300 mb-3 flex items-center gap-2">
                    <span>🤖</span> Predicción en Tiempo Real
                    {mlPrediction?.is_trained && <span className="text-[10px] bg-green-900/50 text-green-300 px-2 py-0.5 rounded-full border border-green-500/30">Modelo Entrenado</span>}
                  </h3>

                  {mlPrediction ? (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-black/30 p-3 rounded-lg text-center border border-white/5">
                        <div className="text-gray-400 text-xs mb-1">Señal ML</div>
                        <div className={`text-xl font-bold ${mlPrediction.signal === 'BUY' ? 'text-green-400' :
                          mlPrediction.signal === 'SELL' ? 'text-red-400' :
                            'text-gray-400'
                          }`}>
                          {mlPrediction.signal}
                        </div>
                      </div>
                      <div className="bg-black/30 p-3 rounded-lg text-center border border-white/5">
                        <div className="text-gray-400 text-xs mb-1">Precio Predicho</div>
                        <div className="text-xl font-mono text-blue-300">
                          ${mlPrediction.predicted_price?.toFixed(2) || '---'}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 py-2 text-sm">
                      Cargando modelo...
                    </div>
                  )}
                </div>

                {memoryStats && (
                  <div className="px-3 py-1 bg-purple-500/20 rounded-full border border-purple-500/30">
                    <span className="text-xs text-purple-300 font-mono">
                      {memoryStats.total_trades || 0} Experiences
                    </span>
                  </div>
                )}


                {/* Risk Manager Panel */}
                <div className="mb-4 bg-black/20 rounded-lg p-4 border border-red-500/20">
                  <h3 className="text-sm font-bold text-red-300 mb-3 flex items-center gap-2">
                    <span>🛡️</span> Risk Manager Shield
                    {riskStats && (
                      <span className={`text-[10px] px-2 py-0.5 rounded-full border ${riskStats.drawdown_state === 'NORMAL' ? 'bg-green-900/50 text-green-300 border-green-500/30' :
                        riskStats.drawdown_state === 'STOP' ? 'bg-red-900/50 text-red-300 border-red-500/30' :
                          'bg-yellow-900/50 text-yellow-300 border-yellow-500/30'
                        }`}>
                        {riskStats.drawdown_state}
                      </span>
                    )}
                  </h3>

                  {riskStats ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-black/30 p-2 rounded border border-white/5 text-center">
                        <div className="text-[10px] text-gray-400">Trailing Stops</div>
                        <div className="text-lg font-mono font-bold text-blue-300">
                          {riskStats.trailing_stops_active}
                        </div>
                      </div>
                      <div className="bg-black/30 p-2 rounded border border-white/5 text-center">
                        <div className="text-[10px] text-gray-400">Pyramids</div>
                        <div className="text-lg font-mono font-bold text-purple-300">
                          {riskStats.pyramid_positions}
                        </div>
                      </div>
                      <div className="bg-black/30 p-2 rounded border border-white/5 text-center">
                        <div className="text-[10px] text-gray-400">Drawdown</div>
                        <div className={`text-lg font-mono font-bold ${riskStats.daily_drawdown > 5 ? 'text-red-400' : 'text-green-400'
                          }`}>
                          {riskStats.daily_drawdown}%
                        </div>
                      </div>
                      <div className="bg-black/30 p-2 rounded border border-white/5 text-center">
                        <div className="text-[10px] text-gray-400">Peak Balance</div>
                        <div className="text-lg font-mono font-bold text-gray-300">
                          ${riskStats.peak_balance}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 py-2 text-sm">
                      Cargando stats...
                    </div>
                  )}
                </div>

                {memoryStats && memoryStats.total_trades > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {/* Win Rate with tooltip */}
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Win Rate: Porcentaje de trades ganadores. ≥50% = más victorias que derrotas"
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Win Rate ℹ️</p>
                      <p className={`text-2xl font-mono font-bold ${memoryStats.win_rate >= 50 ? 'text-green-400' : 'text-yellow-400'
                        }`}>
                        {memoryStats.win_rate}%
                      </p>
                      <p className="text-xs text-foreground/40 mt-1">
                        {memoryStats.total_wins}W / {memoryStats.total_losses}L
                      </p>
                    </div>

                    {/* Average PnL with tooltip */}
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Avg PnL: Ganancia o pérdida promedio por cada operación realizada."
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Avg PnL/Trade ℹ️</p>
                      <p className={`text-2xl font-mono font-bold ${memoryStats.average_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                        ${memoryStats.average_pnl}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Per trade</p>
                    </div>

                    {/* Best Context with tooltip */}
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Best Setup: La condición de mercado (Tendencia/Volatilidad) donde el bot gana más frecuentemente."
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Best Setup ℹ️</p>
                      <p className="text-lg font-mono font-bold text-green-400">
                        {memoryStats.best_context?.context || 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {memoryStats.best_context ? `${memoryStats.best_context.win_rate.toFixed(0)}% WR` : ''}
                      </p>
                    </div>

                    {/* Worst Context with tooltip */}
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Avoid: La condición de mercado donde el bot suele perder. El bot intentará evitar operar aquí."
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Avoid ℹ️</p>
                      <p className="text-lg font-mono font-bold text-red-400">
                        {memoryStats.worst_context?.context || 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {memoryStats.worst_context ? `${memoryStats.worst_context.win_rate.toFixed(0)}% WR` : ''}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-foreground/30 text-sm">
                    Waiting for first trade experience...
                  </div>
                )}
              </div>

              {/* Market Sentiment Dashboard */}
              {sentiment && (
                <div className="md:col-span-3 rounded-xl border border-purple-500/30 bg-gradient-to-r from-purple-500/10 to-pink-500/10 p-4 backdrop-blur mb-4">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-purple-500/20 rounded-lg">
                        <span className="text-2xl">📰</span>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-purple-300">Market Sentiment</p>
                        <p className="text-xs text-purple-200/50">Análisis en Tiempo Real</p>
                      </div>
                    </div>
                    <div className={`px-3 py-1 rounded-full border text-sm font-bold ${sentiment.overall === 'bullish'
                      ? 'bg-green-900/50 text-green-300 border-green-500/30'
                      : sentiment.overall === 'bearish'
                        ? 'bg-red-900/50 text-red-300 border-red-500/30'
                        : 'bg-gray-900/50 text-gray-300 border-gray-500/30'
                      }`}>
                      {sentiment.overall === 'bullish' && '🐂 BULLISH'}
                      {sentiment.overall === 'bearish' && '🐻 BEARISH'}
                      {sentiment.overall === 'neutral' && '😐 NEUTRAL'}
                    </div>
                  </div>

                  {/* Sentiment Score Bar */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-400">Sentiment Score</span>
                      <span className="text-xs font-mono text-purple-300">{sentiment.score.toFixed(2)}</span>
                    </div>
                    <div className="h-2 bg-black/30 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-500 ${sentiment.score > 0 ? 'bg-gradient-to-r from-green-500 to-green-400' : 'bg-gradient-to-r from-red-500 to-red-400'
                          }`}
                        style={{
                          width: `${Math.abs(sentiment.score) * 50}%`,
                          marginLeft: sentiment.score < 0 ? `${50 + (sentiment.score * 50)}%` : '50%'
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                      <span>-1.0 (Bearish)</span>
                      <span>0.0</span>
                      <span>+1.0 (Bullish)</span>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <div className="bg-black/20 rounded-lg p-2 text-center">
                      <div className="text-[10px] text-gray-400">Confidence</div>
                      <div className={`text-sm font-mono font-bold ${sentiment.confidence === 'high' ? 'text-green-400' :
                        sentiment.confidence === 'medium' ? 'text-yellow-400' :
                          'text-gray-400'
                        }`}>
                        {sentiment.confidence.toUpperCase()}
                      </div>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2 text-center">
                      <div className="text-[10px] text-gray-400">Trend</div>
                      <div className="text-sm font-mono font-bold text-purple-300">
                        {sentiment.trend === 'improving' && '📈'}
                        {sentiment.trend === 'worsening' && '📉'}
                        {sentiment.trend === 'stable' && '➡️'}
                        {' '}{sentiment.trend.toUpperCase()}
                      </div>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2 text-center">
                      <div className="text-[10px] text-gray-400">News</div>
                      <div className="text-sm font-mono font-bold text-blue-300">
                        {sentiment.news_count}
                      </div>
                    </div>
                  </div>

                  {/* Recent Events */}
                  {sentiment.recent_events && sentiment.recent_events.length > 0 && (
                    <div className="mt-4">
                      <div className="text-xs text-gray-400 mb-2 font-bold">📡 Latest News:</div>
                      <div className="space-y-2">
                        {sentiment.recent_events.slice(0, 3).map((news: any, i: number) => (
                          <div key={i} className="bg-black/20 rounded p-2 text-xs hover:bg-black/30 transition-colors">
                            <div className="flex items-start gap-2">
                              <span className="text-lg flex-shrink-0">
                                {news.sentiment?.sentiment === 'positive' && '🟢'}
                                {news.sentiment?.sentiment === 'negative' && '🔴'}
                                {news.sentiment?.sentiment === 'neutral' && '🟡'}
                              </span>
                              <div className="flex-1 min-w-0">
                                <p className="text-gray-300 line-clamp-2">{news.title}</p>
                                <div className="flex items-center gap-2 mt-1">
                                  <span className="text-[10px] text-gray-500">{news.source}</span>
                                  <span className="text-[10px] text-purple-400">
                                    Score: {news.sentiment?.score?.toFixed(2)}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* System Health Dashboard */}
              {systemHealth && (
                <div className="md:col-span-3 rounded-xl border border-green-500/30 bg-gradient-to-r from-green-500/10 to-emerald-500/10 p-4 backdrop-blur mb-4">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-500/20 rounded-lg">
                        <span className="text-2xl">⚡</span>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-green-300">System Health</p>
                        <p className="text-xs text-green-200/50">Status & Performance Monitoring</p>
                      </div>
                    </div>
                    <div className={`px-3 py-1 rounded-full border ${systemHealth.status === 'online'
                      ? 'bg-green-900/50 text-green-300 border-green-500/30'
                      : 'bg-red-900/50 text-red-300 border-red-500/30'
                      }`}>
                      <span className="text-xs font-mono">
                        {systemHealth.status === 'online' ? '🟢 ONLINE' : '🔴 OFFLINE'}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div className="bg-black/20 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-400 mb-1">Uptime</div>
                      <div className="text-lg font-mono font-bold text-green-300">
                        {Math.floor(systemHealth.uptime / 60)}m
                      </div>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-400 mb-1">Experiencias</div>
                      <div className="text-lg font-mono font-bold text-blue-300">
                        {systemHealth.memory_experiences}
                      </div>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-400 mb-1">Confianza</div>
                      <div className={`text-lg font-mono font-bold ${systemHealth.learning_confidence === 'HIGH' ? 'text-green-400' :
                        systemHealth.learning_confidence === 'MEDIUM' ? 'text-yellow-400' :
                          'text-gray-400'
                        }`}>
                        {systemHealth.learning_confidence}
                      </div>
                    </div>
                    <div className="bg-black/20 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-400 mb-1">RAM</div>
                      <div className="text-lg font-mono font-bold text-purple-300">
                        {systemHealth.memory_mb?.toFixed(0) || 0}MB
                      </div>
                    </div>
                  </div>

                  {/* Quick Training Button */}
                  <div className="bg-black/20 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-bold text-yellow-300">🚀 Entrenamiento Rápido</p>
                      <p className="text-xs text-gray-400">Aprende de 100 trades históricos en segundos</p>
                    </div>
                    <button
                      onClick={async () => {
                        setTrainingInProgress(true);
                        try {
                          const res = await fetch('/api/training/run', { method: 'POST' });
                          const data = await res.json();
                          alert(data.success ? `✅ ${data.message}` : `❌ ${data.message}`);
                        } catch (error) {
                          alert('❌ Error iniciando entrenamiento');
                        }
                        setTrainingInProgress(false);
                      }}
                      disabled={trainingInProgress}
                      className="px-4 py-2 bg-yellow-600/20 text-yellow-400 border border-yellow-600/50 rounded-lg hover:bg-yellow-600/30 disabled:opacity-50 text-sm font-bold"
                    >
                      {trainingInProgress ? '⏳ Entrenando...' : '🎯 Entrenar Ahora'}
                    </button>
                  </div>
                </div>
              )}

              {/* 5. Performance Dashboard (New) */}
              {memoryStats && memoryStats.total_trades > 0 && (
                <div className="md:col-span-3 rounded-xl border border-blue-500/30 bg-gradient-to-r from-blue-500/10 to-cyan-500/10 p-4 backdrop-blur mt-4">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-blue-500/20 rounded-lg">
                      <span className="text-2xl">📊</span>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-blue-300">Performance Dashboard</p>
                      <p className="text-xs text-blue-200/50">Real-time Metrics & Equity Curve</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Profit Factor: Relación entre Ganancia Bruta y Pérdida Bruta.&#10;> 1.0: Estrategia Rentable&#10;> 1.5: Estrategia Excelente&#10;< 1.0: Estrategia Perdedora"
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Profit Factor ℹ️</p>
                      <p className={`text-xl font-mono font-bold ${memoryStats.profit_factor >= 1.5 ? 'text-green-400' : memoryStats.profit_factor >= 1 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {memoryStats.profit_factor}
                      </p>
                    </div>
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Net Profit: Ganancia Neta Real (Ganancia Bruta - Pérdida Bruta). Lo que realmente has ganado."
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Net Profit ℹ️</p>
                      <p className={`text-xl font-mono font-bold ${memoryStats.net_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${memoryStats.net_profit}
                      </p>
                    </div>
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Gross Profit: Suma total de todas las operaciones ganadoras, sin restar las pérdidas."
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Gross Profit ℹ️</p>
                      <p className="text-xl font-mono font-bold text-green-400/80">
                        ${memoryStats.gross_profit}
                      </p>
                    </div>
                    <div
                      className="bg-black/20 rounded-lg p-3 hover:bg-black/30 transition-colors cursor-help"
                      title="Gross Loss: Suma total de todas las operaciones perdedoras (en valor absoluto)."
                    >
                      <p className="text-xs text-gray-400 mb-1 font-medium">Gross Loss ℹ️</p>
                      <p className="text-xl font-mono font-bold text-red-400/80">
                        -${memoryStats.gross_loss}
                      </p>
                    </div>
                  </div>

                  {/* Cumulative PnL Chart (Simple CSS Bar/Line representation) */}
                  <div className="bg-black/20 rounded-lg p-4 h-48 flex flex-col relative overflow-hidden">
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-xs text-yellow-400 font-mono font-bold">📈 Equity Curve (Ganancia Acumulada)</div>
                      {memoryStats.cumulative_pnl && memoryStats.cumulative_pnl.length > 0 && (
                        <div className="text-xs text-gray-400 font-mono">
                          {memoryStats.cumulative_pnl.length} trades
                        </div>
                      )}
                    </div>

                    {/* Chart Area */}
                    <div className="flex-1 relative">
                      {/* Zero Line */}
                      <div className="absolute w-full h-px bg-white/20 top-1/2 left-0 z-0">
                        <span className="absolute right-0 -top-3 text-[10px] text-gray-500">$0</span>
                      </div>

                      {/* Real-time Line Overlay */}
                      <svg className="absolute top-0 left-0 w-full h-full z-20 pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                        {memoryStats.cumulative_pnl && memoryStats.cumulative_pnl.length > 0 && (
                          <>
                            {/* Gradient fill under the line */}
                            <defs>
                              <linearGradient id="equityGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stopColor="#22c55e" stopOpacity="0.3" />
                                <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
                              </linearGradient>
                            </defs>
                            {/* Filled area */}
                            <polygon
                              points={`0,50 ${memoryStats.cumulative_pnl.map((point: any, i: number) => {
                                const maxVal = Math.max(...memoryStats.cumulative_pnl.map((p: any) => Math.abs(p.pnl)), 10);
                                const x = (i / (memoryStats.cumulative_pnl.length - 1 || 1)) * 100;
                                const y = 50 - (point.pnl / maxVal * 45);
                                return `${x},${y}`;
                              }).join(' ')} 100,50`}
                              fill="url(#equityGradient)"
                            />
                            {/* Main line */}
                            <polyline
                              points={memoryStats.cumulative_pnl.map((point: any, i: number) => {
                                const maxVal = Math.max(...memoryStats.cumulative_pnl.map((p: any) => Math.abs(p.pnl)), 10);
                                const x = (i / (memoryStats.cumulative_pnl.length - 1 || 1)) * 100;
                                const y = 50 - (point.pnl / maxVal * 45);
                                return `${x},${y}`;
                              }).join(' ')}
                              fill="none"
                              stroke="#22c55e"
                              strokeWidth="1"
                              vectorEffect="non-scaling-stroke"
                            />
                          </>
                        )}
                      </svg>

                      {/* Bar chart representation */}
                      <div className="flex items-end gap-1 w-full h-full z-10">
                        {memoryStats.cumulative_pnl && memoryStats.cumulative_pnl.length > 0 ? (
                          memoryStats.cumulative_pnl.map((point: any, i: number) => {
                            const maxVal = Math.max(...memoryStats.cumulative_pnl.map((p: any) => Math.abs(p.pnl)), 10);
                            const height = Math.min(Math.abs(point.pnl) / maxVal * 45, 45);
                            const isFirst = i === 0;
                            const isLast = i === memoryStats.cumulative_pnl.length - 1;
                            const showTimeLabel = isFirst || isLast || (i % Math.floor(memoryStats.cumulative_pnl.length / 4) === 0);

                            return (
                              <div key={i} className="flex-1 min-w-[2px] relative group">
                                <div
                                  className={`w-full rounded-t transition-all duration-500 ${point.pnl >= 0 ? 'bg-green-500/30' : 'bg-red-500/30'}`}
                                  style={{
                                    height: `${Math.max(height, 3)}%`,
                                    marginBottom: point.pnl >= 0 ? '50%' : `${50 - height}%`
                                  }}
                                >
                                  {/* Tooltip on hover */}
                                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-black/90 rounded text-[10px] whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-30">
                                    <div className={`font-bold ${point.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                      ${point.pnl.toFixed(2)}
                                    </div>
                                    <div className="text-gray-400">Trade #{point.trade}</div>
                                  </div>
                                </div>
                                {/* Time label */}
                                {showTimeLabel && (
                                  <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] text-gray-500 whitespace-nowrap">
                                    #{point.trade}
                                  </div>
                                )}
                              </div>
                            );
                          })
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">
                            No trades yet to display chart
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Pattern Performance */}
                  {memoryStats && memoryStats.pattern_stats && memoryStats.pattern_stats.length > 0 && (
                    <div className="mt-4 bg-black/20 rounded-lg p-4 border border-orange-500/20">
                      <h3 className="text-sm font-bold text-orange-300 mb-3 flex items-center gap-2">
                        <span>🕯️</span> Performance por Patrón
                      </h3>
                      <div className="space-y-2">
                        {memoryStats.pattern_stats.map((stat: any, i: number) => (
                          <div key={i} className="flex items-center justify-between bg-black/30 p-2 rounded text-xs">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-gray-300">{stat.pattern}</span>
                              <span className={`px-1.5 py-0.5 rounded text-[10px] ${stat.confidence === 'HIGH' ? 'bg-green-900 text-green-300' :
                                stat.confidence === 'MEDIUM' ? 'bg-yellow-900 text-yellow-300' :
                                  'bg-gray-800 text-gray-400'
                                }`}>{stat.confidence}</span>
                            </div>
                            <div className="flex items-center gap-4">
                              <span className={stat.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}>
                                {stat.win_rate}% WR
                              </span>
                              <span className={`font-mono w-16 text-right ${stat.avg_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                ${stat.avg_pnl}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Trade History (1/4 width) - NOW WITH INDIVIDUAL TRADES */}
          <div className="lg:col-span-1 space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur h-full flex flex-col">
              <h3 className="text-lg font-semibold mb-4">📊 Trades Activos</h3>
              <div className="flex-grow overflow-y-auto custom-scrollbar h-[500px]">
                <div className="space-y-2">
                  {activeTrades.length > 0 ? (
                    activeTrades.map((trade) => (
                      <div key={trade.id} className="bg-black/30 p-3 rounded-lg border border-white/5 hover:border-white/10 transition">
                        {/* Trade Info Row */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className={trade.side === 'LONG' ? 'text-green-400 text-xl' : 'text-red-400 text-xl'}>
                              {trade.side === 'LONG' ? '↗' : '↘'}
                            </span>
                            <div>
                              <div className="text-sm font-bold">{trade.side}</div>
                              <div className="text-xs text-gray-400 font-mono">{trade.size.toFixed(4)} BTC</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs text-gray-400">@ ${trade.entry_price.toLocaleString()}</div>
                            <div className="text-xs text-gray-500">{new Date(trade.opened_at).toLocaleTimeString()}</div>
                          </div>
                        </div>

                        {/* PnL Display */}
                        <div className={`text-center py-2 rounded mb-2 font-mono font-bold ${trade.unrealized_pnl_usd >= 0
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                          }`}>
                          {trade.unrealized_pnl_usd >= 0 ? '+' : ''}${trade.unrealized_pnl_usd.toFixed(2)}
                          <span className="text-xs ml-2">
                            ({trade.unrealized_pnl_pct >= 0 ? '+' : ''}{trade.unrealized_pnl_pct.toFixed(2)}%)
                          </span>
                        </div>

                        {/* Action Buttons */}
                        <div className="flex gap-2">
                          <button
                            onClick={async () => {
                              if (!confirm(`🛑 ¿Cerrar trade #${trade.id}?`)) return;
                              try {
                                const res = await fetch(`/api/trades/close/${trade.id}`, { method: 'POST' });
                                const data = await res.json();
                                if (data.success) {
                                  // Recargar lista inmediatamente
                                  const tradesRes = await fetch("/api/trades/active");
                                  if (tradesRes.ok) {
                                    const tradesData = await tradesRes.json();
                                    setActiveTrades(tradesData.trades || []);
                                  }
                                  alert(`✅ Trade #${trade.id} cerrado\nPnL: $${data.trade.pnl.toFixed(2)}`);
                                }
                              } catch (e) {
                                alert("❌ Error");
                              }
                            }}
                            className="flex-1 bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-xs font-bold transition"
                            title="Cerrar este trade"
                          >
                            STOP
                          </button>
                          <button
                            onClick={async () => {
                              try {
                                const res = await fetch(`/api/trades/reverse/${trade.id}`, { method: 'POST' });
                                const data = await res.json();
                                if (data.success) {
                                  // Recargar lista inmediatamente
                                  const tradesRes = await fetch("/api/trades/active");
                                  if (tradesRes.ok) {
                                    const tradesData = await tradesRes.json();
                                    setActiveTrades(tradesData.trades || []);
                                  }
                                  alert(`🔄 Trade #${trade.id} → ${data.trade.side}`);
                                }
                              } catch (e) {
                                alert("❌ Error");
                              }
                            }}
                            className="flex-1 bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded text-xs font-bold transition"
                            title="Invertir dirección (LONG↔SHORT)"
                          >
                            ⟲
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-center text-sm text-foreground/40 mt-8">
                      No hay trades activos...
                      <br />
                      <span className="text-xs">Haz clic en BUY para abrir uno</span>
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Backtest Section (Collapsed/Secondary) */}
        <section className="mt-8 border-t border-white/10 pt-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold opacity-50">Strategy Backtest</h2>
            <button
              onClick={runBacktest}
              disabled={loadingBacktest}
              className="px-4 py-2 bg-blue-600/20 text-blue-400 border border-blue-600/50 rounded-lg hover:bg-blue-600/30 disabled:opacity-50 text-sm"
            >
              {loadingBacktest ? "Running Simulation..." : "Run Backtest"}
            </button>
          </div>
          {backtestResult && (
            <div className="grid grid-cols-3 gap-4 opacity-70">
              <div className="bg-white/5 p-3 rounded border border-white/10">
                <p className="text-xs text-foreground/50">Final Capital</p>
                <p className="font-mono">${backtestResult.final_capital.toFixed(2)}</p>
              </div>
              <div className="bg-white/5 p-3 rounded border border-white/10">
                <p className="text-xs text-foreground/50">Max Drawdown</p>
                <p className="font-mono text-red-400">{(backtestResult.max_drawdown * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-white/5 p-3 rounded border border-white/10">
                <p className="text-xs text-foreground/50">Total Trades</p>
                <p className="font-mono">{backtestResult.trades}</p>
              </div>
            </div>
          )}
        </section>
      </main>
    </div >
  );
}
