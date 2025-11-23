
"use client";

"use client";

import { useEffect, useState, useRef } from "react";
import { createChart, ColorType, IChartApi, CandlestickSeries } from 'lightweight-charts';

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
  const [balance, setBalance] = useState<{ USDT: number, BTC: number } | null>(null);
  const lastTrade = trades.length > 0 ? trades[trades.length - 1] : null;
  const [tradingEnabled, setTradingEnabled] = useState(true);
  const [memoryStats, setMemoryStats] = useState<any>(null);
  const [priceHistory, setPriceHistory] = useState<{ time: number, price: number }[]>([]);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);

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
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: "BTC_USD",
          side: side,
          size: 0.001, // Tamaño fijo para demo
          stop_loss: null,
          take_profit: null,
        }),
      });
      const data = await res.json();

      // Verificar si la orden fue exitosa
      if (data.status === "FILLED" || data.status === "SIMULATED") {
        alert(`✅ Orden ${side} ejecutada!\nID: ${data.id}\nPrecio: ~$${livePrice?.toFixed(2) || 'N/A'}`);
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
      const res = await fetch("/api/trading/toggle", {
        method: "POST",
      });
      const data = await res.json();
      setTradingEnabled(data.enabled);
    } catch (error) {
      console.error("Failed to toggle trading", error);
    }
  };

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
  }, []);

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
    const interval = setInterval(fetchTrades, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);

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

    fetchBalance();
    fetchTradingStatus();
    fetchPnl();
    fetchMemory();

    const balanceInterval = setInterval(fetchBalance, 5000);
    const statusInterval = setInterval(fetchTradingStatus, 1000);
    const candleInterval = setInterval(fetchCandles, 1000);
    const pnlInterval = setInterval(fetchPnl, 1000);
    const memoryInterval = setInterval(fetchMemory, 2000);

    return () => {
      clearInterval(balanceInterval);
      clearInterval(statusInterval);
      clearInterval(candleInterval);
      clearInterval(pnlInterval);
      clearInterval(memoryInterval);
    };
  }, [trades]);

  // Inicializar gráfico
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Limpiar contenedor por si acaso (evita duplicados en Strict Mode)
    chartContainerRef.current.innerHTML = '';

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#ffffff10' },
        horzLines: { color: '#ffffff10' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

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
    <div className="min-h-screen bg-background text-foreground">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-6 py-16 lg:py-20">
        <header className="space-y-6 rounded-2xl border border-white/10 bg-white/5 p-8 shadow-panel backdrop-blur">
          <div className="flex justify-between items-center">
            <p className="text-sm uppercase tracking-[0.3em] text-foreground/70">
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
          <p className="max-w-3xl text-lg text-foreground/80">
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
                <p className="mt-2 text-base text-foreground/90">
                  {step.description}
                </p>
              </div>
              <ul className="space-y-2 text-sm text-foreground/80">
                {step.checklist.map((item) => (
                  <li
                    key={item}
                    className="flex items-center gap-2 rounded-lg bg-foreground/5 px-3 py-2"
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
              <h2 className="text-2xl font-semibold">Simulation Engine</h2>
              <p className="text-foreground/70">
                Run a historical backtest using the current strategy configuration.
              </p>
            </div>
            <button
              onClick={runBacktest}
              disabled={loadingBacktest || backendStatus.includes("Offline")}
              className="rounded-full bg-primary px-8 py-3 font-bold text-primary-foreground transition hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loadingBacktest ? "Running Simulation..." : "▶ Run Backtest"}
            </button>
          </div>

          {backtestResult && (
            <div className="mt-8 grid gap-4 border-t border-white/10 pt-8 sm:grid-cols-3">
              <div className="rounded-xl bg-black/20 p-4">
                <p className="text-sm text-foreground/60">Final Capital</p>
                <p className="text-2xl font-mono text-green-400">
                  ${backtestResult.final_capital.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
              </div>
              <div className="rounded-xl bg-black/20 p-4">
                <p className="text-sm text-foreground/60">Max Drawdown</p>
                <p className="text-2xl font-mono text-red-400">
                  {(backtestResult.max_drawdown * 100).toFixed(2)}%
                </p>
              </div>
              <div className="rounded-xl bg-black/20 p-4">
                <p className="text-sm text-foreground/60">Total Trades</p>
                <p className="text-2xl font-mono text-blue-400">
                  {backtestResult.trades}
                </p>
              </div>
            </div>
          )}
        </section>

        {/* Trading Terminal Section - ALWAYS VISIBLE */}
        <section className="grid gap-6 lg:grid-cols-4">
          {/* Main Chart & Controls Area (3/4 width) */}
          <div className="lg:col-span-3 space-y-4">
            {/* Chart Card */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-1 shadow-panel backdrop-blur overflow-hidden relative">
              <div ref={chartContainerRef} className="w-full h-[500px] bg-black/20" />
              {/* Live Price Overlay */}
              <div className="absolute top-4 left-4 bg-black/40 backdrop-blur px-3 py-1 rounded border border-white/10">
                <span className="text-xs text-foreground/50 mr-2">BTC/USDT</span>
                <span className="font-mono font-bold text-lg">${livePrice?.toFixed(2) || '---'}</span>
              </div>
            </div>

            {/* Trading Control Panel (Below Chart) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

              {/* 1. PnL & Position Info */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur flex flex-col justify-center">
                <div className="flex justify-between items-start mb-1">
                  <p className="text-xs uppercase tracking-wider text-foreground/50">Unrealized PnL</p>
                  {balance && <span className="text-xs font-mono text-yellow-400">Bal: {balance.BTC.toFixed(4)} BTC</span>}
                </div>
                {pnl ? (
                  <>
                    <div className={`text-3xl font-bold font-mono ${pnl.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pnl.unrealized_pnl >= 0 ? '+' : ''}{pnl.unrealized_pnl.toFixed(2)} <span className="text-sm text-foreground/50">USDT</span>
                    </div>
                    <div className="flex justify-between text-sm mt-2">
                      <span className="text-foreground/60">ROE: <span className={pnl.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                        {pnl.entry_price > 0 ? ((pnl.unrealized_pnl / (pnl.entry_price * Math.abs(pnl.position_size))) * 100).toFixed(2) : '0.00'}%
                      </span></span>
                      <span className="text-foreground/60">Size: {Math.abs(pnl.position_size).toFixed(3)} BTC</span>
                    </div>
                  </>
                ) : (
                  <div className="text-2xl font-mono text-foreground/30">--.--</div>
                )}
              </div>

              {/* 2. Manual Controls */}
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
              </div>

              {/* 3. Last Trade Info */}
              <div className="rounded-xl border border-white/10 bg-white/5 p-4 backdrop-blur flex flex-col justify-center">
                <p className="text-xs uppercase tracking-wider text-foreground/50 mb-1">Last Trade</p>
                {lastTrade ? (
                  <>
                    <div className="text-3xl font-bold font-mono">
                      <span className={lastTrade.side === 'BUY' ? 'text-green-400' : 'text-red-400'}>
                        {lastTrade.side}
                      </span>
                      <span className="text-sm text-foreground/50 ml-2">@ {lastTrade.price?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-sm mt-2">
                      <span className="text-foreground/60">Size: {lastTrade.size?.toFixed(3)} BTC</span>
                      <span className="text-foreground/60">
                        {lastTrade.time ? new Date(lastTrade.time * 1000).toLocaleTimeString() : 'N/A'}
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="text-2xl font-mono text-foreground/30">No trades</div>
                )}
              </div>

              {/* 4. Brain Status (New) */}
              <div className="md:col-span-3 rounded-xl border border-white/10 bg-purple-500/10 p-3 backdrop-blur flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-purple-500/20 rounded-lg">
                    <span className="text-xl">🧠</span>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-wider text-purple-300 font-bold">AI Memory</p>
                    <p className="text-sm text-purple-200/70">Learning from every trade...</p>
                  </div>
                </div>
                <div className="flex gap-6 text-right">
                  <div>
                    <p className="text-xs text-purple-300/50">Memories</p>
                    <p className="font-mono text-xl text-purple-300">{memoryStats?.total_memories || 0}</p>
                  </div>
                  <div>
                    <p className="text-xs text-purple-300/50">Last Lesson</p>
                    <p className="font-mono text-xl text-purple-300">{memoryStats?.recent_outcome || "Waiting..."}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Trade History (1/4 width) */}
          <div className="lg:col-span-1 space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur h-full flex flex-col">
              <h3 className="text-lg font-semibold mb-4">Trade History</h3>
              <div className="flex-grow overflow-y-auto custom-scrollbar h-[500px]">
                <div className="space-y-2">
                  {trades.length > 0 ? (
                    trades.slice().reverse().map((trade) => (
                      <div key={trade.id} className="flex items-center justify-between text-sm bg-black/20 p-2 rounded">
                        <div className="flex items-center gap-2">
                          <span title={trade.source === 'MANUAL' ? "Manual Trade" : "AI Auto-Trade"}>
                            {trade.source === 'MANUAL' ? '👤' : '🤖'}
                          </span>
                          <span className={trade.side === 'BUY' ? 'text-green-400' : 'text-red-400'}>
                            {trade.side}
                          </span>
                        </div>
                        <span className="font-mono text-foreground/70">
                          @{trade.price?.toFixed(2)}
                        </span>
                        <span className="text-xs text-foreground/50">
                          {new Date(trade.time * 1000).toLocaleTimeString()}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="text-center text-sm text-foreground/40">
                      No trades yet...
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
    </div>
  );
}
