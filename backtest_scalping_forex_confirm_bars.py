# -*- coding: utf-8 -*-
"""
اختبار سكالبينج على بيانات فوركس حقيقية (ساعية، مُعاد تجميعها إلى شموع أقصر
هنا نستخدم البيانات الساعية مباشرة بدل التجميع اليومي — أقرب لطبيعة السكالبينج
الحقيقية من الشموع اليومية).

المنطق المطبَّق (بالضبط كما طُلب):
  1) دخول شراء/بيع بمحرك اتجاه سريع (فترات قصيرة، مناسبة للسكالبينج).
  2) بمجرد تحرك السعر لصالح المركز بنسبة معيّنة → الوقف يُنقل تلقائياً
     لنقطة الدخول (breakeven) — الصفقة تصبح "بلا مخاطرة" من هذه اللحظة.
  3) بعدها الوقف يتحرك (trailing) خلف السعر باستمرار، يقفل مزيداً من الربح
     كل ما استمر الاتجاه، بدون أي تدخل يدوي — حتى يُغلق المركز تلقائياً
     عند أول ارتداد بمقدار مسافة التتبّع.
تكلفة سبريد حقيقية على كل صفقة (كما في الاختبار السابق).
"""
import numpy as np
import pandas as pd
from scipy import stats

from hft_agent_v3_scalping import FixedHFTAgent, SimulatedDataProvider

np.random.seed(42)

PAIRS = {
    "EURUSD": 100000.0, "GBPUSD": 100000.0, "USDJPY": 100000.0,
    "AUDUSD": 100000.0, "USDCAD": 100000.0, "USDCHF": 100000.0, "XAUUSD": 100.0,
}
SPREAD_BPS = {
    "EURUSD": 1.2, "GBPUSD": 1.6, "USDJPY": 1.5,
    "AUDUSD": 1.6, "USDCAD": 1.8, "USDCHF": 1.8, "XAUUSD": 2.5,
}

# نأخذ آخر سنتين فقط من البيانات الساعية لكل زوج (حجم بيانات كبير جداً لتشغيل
# ساعي كامل على 10 سنوات x7 أزواج ضمن وقت معقول) — عيّنة حديثة كافية للتقييم.
HOURS_WINDOW = 24 * 365  # آخر ~6 أشهر من البيانات الساعية لكل زوج
CONFIRM_BARS = 3  # عدد الشموع المتتالية المطلوبة لتأكيد الإشارة قبل الدخول/الخروج


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min()) * 100


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


results = []

for symbol, divisor in PAIRS.items():
    raw = pd.read_csv(f"fx_{symbol}.csv")
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").tail(HOURS_WINDOW).reset_index(drop=True)

    prices = (raw["close"].values / divisor).tolist()
    vols = raw["tick_volume"].values.astype(int).tolist()
    spread_frac = SPREAD_BPS[symbol] / 10000.0

    agent = FixedHFTAgent(
        agent_id=f"scalp_{symbol}",
        data_provider=SimulatedDataProvider([symbol], [prices[0]]),
        symbols=[symbol],
        config={
            "initial_balance": 100000,
            # ✅ التعديل المقترح: نفس فترات السكالبينج السريعة (4/12) لكن مع
            # فلتر تأكيد (confirm_bars) يمنع محرك الإشارة من الخروج على أول
            # ارتداد لحظي — يعطي الصفقة فرصة أكبر تصل لمنطقة الربح قبل ما
            # إشارة عكسية سريعة تقفلها.
            "trend": {"fast_period": 4, "slow_period": 12, "momentum_period": 4, "confirm_bars": CONFIRM_BARS},
            "risk": {
                "max_position_pct": 0.10,
                "stop_loss_pct": 0.006,           # وقف أولي أضيق (سكالبينج)
                "breakeven_trigger_pct": 0.004,   # ينقل الوقف لنقطة الدخول بعد ربح 0.4%
                "breakeven_buffer_pct": spread_frac * 1.5,  # هامش يغطي السبريد تقريباً
                "trailing_distance_pct": 0.004,   # يتبع السعر بمسافة 0.4%
            },
        },
    )

    equity_curve, bh_curve = [], []
    bh_units = 100000 / prices[0]
    n_stop_exits, n_breakeven_exits, n_signal_exits = 0, 0, 0

    for price, vol in zip(prices, vols):
        with agent.lock:
            agent.price_history[symbol].append(float(price))
            agent.volume_history[symbol].append(int(vol))
            ind = agent._calculate_indicators(symbol)
            agent.indicators_history[symbol].append(ind)
            # ✅ تقليم السجل التاريخي (كما يفعل update_market_data الأصلي) لتفادي
            # تباطؤ O(n^2) على بيانات ساعية طويلة، مع إبقاء نافذة كافية للمؤشرات
            for hist in (agent.price_history[symbol], agent.volume_history[symbol], agent.indicators_history[symbol]):
                if len(hist) > 300:
                    del hist[0]

        if len(agent.price_history[symbol]) >= 30:
            signal = agent.generate_signal(symbol)
            if not agent.risk_manager.trading_halted and not agent.risk_manager.check_circuit_breaker(
                agent.daily_pnl, agent.initial_balance
            ):
                if signal["action"] == "buy" and signal["confidence"] >= 60:
                    agent._execute_buy(symbol, price * (1 + spread_frac), signal)
                elif signal["action"] == "sell" and signal["confidence"] >= 60 and symbol in agent.positions:
                    n_signal_exits += 1
                    agent._execute_sell(symbol, price * (1 - spread_frac), signal)

            had_position = symbol in agent.positions
            was_breakeven = agent.positions.get(symbol, {}).get("breakeven_active", False)
            agent.check_stop_losses()
            if had_position and symbol not in agent.positions:
                if was_breakeven:
                    n_breakeven_exits += 1
                else:
                    n_stop_exits += 1

        pos_qty = agent.positions.get(symbol, {}).get("quantity", 0)
        equity_curve.append(agent.balance + pos_qty * price)
        bh_curve.append(bh_units * price)

    if symbol in agent.positions:
        agent._force_close(symbol, prices[-1] * (1 - spread_frac))
        equity_curve[-1] = agent.balance

    equity_curve = np.array(equity_curve)
    bh_curve = np.array(bh_curve)
    strat_h = np.diff(equity_curve) / equity_curve[:-1]
    bh_h = np.diff(bh_curve) / bh_curve[:-1]

    perf = agent.calculate_performance()
    strat_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100
    bh_ret = (bh_curve[-1] / bh_curve[0] - 1) * 100
    hours_per_year = 24 * 365

    results.append({
        "pair": symbol, "hours": len(prices), "trades": perf["total_trades"],
        "win_rate_pct": perf.get("win_rate", 0),
        "exits_stop_loss": n_stop_exits, "exits_trailing_profit": n_breakeven_exits, "exits_signal_reversal": n_signal_exits,
        "strategy_return_pct": strat_ret, "buy_hold_return_pct": bh_ret, "gap_pct": strat_ret - bh_ret,
        "strategy_sharpe": sharpe_ratio(strat_h, hours_per_year), "buy_hold_sharpe": sharpe_ratio(bh_h, hours_per_year),
        "strategy_max_dd_pct": max_drawdown(equity_curve), "buy_hold_max_dd_pct": max_drawdown(bh_curve),
    })
    print(f"{symbol:8s} | ساعات: {len(prices):6d} | صفقات: {perf['total_trades']:4d} | فوز: {perf.get('win_rate',0):5.1f}% | "
          f"خروج بوقف أولي: {n_stop_exits:3d} | خروج بوقف متحرك(ربح): {n_breakeven_exits:3d} | "
          f"استراتيجية: {strat_ret:7.2f}% | Buy&Hold: {bh_ret:7.2f}% | "
          f"Sharpe(است.): {sharpe_ratio(strat_h, hours_per_year):5.2f} | MaxDD(است.): {max_drawdown(equity_curve):6.2f}% | MaxDD(B&H): {max_drawdown(bh_curve):7.2f}%")

res_df = pd.DataFrame(results)
print("\n" + "=" * 110)
print(f"عدد الأزواج: {len(res_df)}")
print(f"متوسط عائد الاستراتيجية: {res_df['strategy_return_pct'].mean():.2f}%  |  متوسط Buy&Hold: {res_df['buy_hold_return_pct'].mean():.2f}%")
print(f"متوسط نسبة الفوز: {res_df['win_rate_pct'].mean():.1f}%")
print(f"متوسط Sharpe (استراتيجية): {res_df['strategy_sharpe'].mean():.2f}  |  (B&H): {res_df['buy_hold_sharpe'].mean():.2f}")
print(f"متوسط أقصى تراجع (استراتيجية): {res_df['strategy_max_dd_pct'].mean():.2f}%  |  (B&H): {res_df['buy_hold_max_dd_pct'].mean():.2f}%")
print(f"إجمالي الخروج بوقف أولي (خسارة): {res_df['exits_stop_loss'].sum()}  |  إجمالي الخروج بوقف متحرك (ربح محمي): {res_df['exits_trailing_profit'].sum()}")

res_df.to_csv("backtest_scalping_forex_results.csv", index=False)
print("\nتم حفظ النتائج في backtest_scalping_forex_results.csv")
