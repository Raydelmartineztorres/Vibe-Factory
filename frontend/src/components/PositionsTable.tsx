import React from 'react';

interface Position {
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    pnl: number;
    pnlPercent: number;
}

interface PositionsTableProps {
    positions: Position[];
}

export default function PositionsTable({ positions }: PositionsTableProps) {
    if (!positions || positions.length === 0) {
        return (
            <div className="text-center py-8 text-gray-500 bg-gray-900/30 rounded-lg border border-gray-800">
                <p>No active positions</p>
            </div>
        );
    }

    return (
        <div className="overflow-x-auto rounded-lg border border-gray-800 bg-gray-900/50">
            <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-400 uppercase bg-gray-900/80 border-b border-gray-800">
                    <tr>
                        <th className="px-4 py-3">Asset</th>
                        <th className="px-4 py-3 text-right">Size</th>
                        <th className="px-4 py-3 text-right">Entry</th>
                        <th className="px-4 py-3 text-right">Price</th>
                        <th className="px-4 py-3 text-right">PnL</th>
                    </tr>
                </thead>
                <tbody>
                    {positions.map((pos) => (
                        <tr key={pos.symbol} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                            <td className="px-4 py-3 font-medium text-white flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full ${pos.size > 0 ? 'bg-green-500' : 'bg-red-500'}`}></div>
                                {pos.symbol}
                            </td>
                            <td className={`px-4 py-3 text-right ${pos.size > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {pos.size > 0 ? '+' : ''}{pos.size.toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-right text-gray-300">
                                ${pos.entryPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </td>
                            <td className="px-4 py-3 text-right text-gray-300">
                                ${pos.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            </td>
                            <td className={`px-4 py-3 text-right font-bold ${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                <div className="flex flex-col items-end">
                                    <span>{pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}</span>
                                    <span className="text-xs opacity-70">({pos.pnlPercent >= 0 ? '+' : ''}{pos.pnlPercent.toFixed(2)}%)</span>
                                </div>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
