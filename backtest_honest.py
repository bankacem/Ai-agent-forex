"""
باكتيست صادق ومتعدد الأصول لمنطق FixedHFTAgent بعد استبدال محرك الذكاء
الاصطناعي الوهمي بمحرك اتجاه (MA Crossover) وإصلاح خلل الخروج الجزئي.
بيانات حقيقية (Kaggle/Plotly: all_stocks_5yr.csv)، عيّنة عشوائية غير منتقاة،
نفس الـ seed المستخدم سابقاً للمقارنة العادلة، مقارنة بمعيار Buy & Hold،
واختبار دلالة إحصائية (t-test).
"""
import random
import numpy as np
import pandas as pd
from scipy import stats

from hft_agent import FixedHFTAgent, SimulatedDataProvider, RiskManager

random.seed(42)
np.random.seed(42)

df = pd.read_csv("all_stocks_5yr.csv")
df["date"] = pd.to_datetime(df["date"])

symbols_all = df["Name"].unique().tolist()
random.shuffle(symbols_all)
sample_symbols = symbols_all[:25]

results = []

for symbol in sample_symbols:
    sdf = df[df["Name"] == symbol].sort_values("date")
    if len(sdf) < 200:
        continue
    prices = sdf["close"].tolist()
    volumes = sdf["volume"].tolist()

    agent = FixedHFTAgent(
        agent_id=f"bt_{symbol}",
        data_provider=SimulatedDataProvider([symbol], [prices[0]]),  # غير مستخدم؛ نغذي الأسعار يدوياً
        symbols=[symbol],
        config={"initial_balance": 100000, "risk": {}},
    )

    # تغذية الأسعار التاريخية الحقيقية تسلسلياً (كأنها تِكات سوقية حقيقية)
    for price, vol in zip(prices, volumes):
        with agent.lock:
            agent.price_history[symbol].append(float(price))
            agent.volume_history[symbol].append(int(vol))
            ind = agent._calculate_indicators(symbol)
            agent.indicators_history[symbol].append(ind)

        if len(agent.price_history[symbol]) >= 50:
            signal = agent.generate_signal(symbol)
            agent.execute_signal(symbol, signal)
            agent.check_stop_losses()

    # إغلاق أي مركز مفتوح في نهاية الفترة بسعر الإغلاق الأخير لحساب عائد عادل
    if symbol in agent.positions:
        agent._force_close(symbol, prices[-1])

    perf = agent.calculate_performance()
    strategy_return = (agent.balance - agent.initial_balance) / agent.initial_balance * 100
    buy_hold_return = (prices[-1] - prices[0]) / prices[0] * 100

    results.append({
        "symbol": symbol,
        "trades": perf["total_trades"],
        "win_rate": perf.get("win_rate", 0),
        "strategy_return_pct": strategy_return,
        "buy_hold_return_pct": buy_hold_return,
        "gap_pct": strategy_return - buy_hold_return,
    })
    print(f"{symbol:6s} | صفقات: {perf['total_trades']:4d} | استراتيجية: {strategy_return:8.2f}% | "
          f"Buy&Hold: {buy_hold_return:8.2f}% | الفارق: {strategy_return - buy_hold_return:8.2f}")

res_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print(f"عدد الأسهم المختبَرة: {len(res_df)}")
print(f"متوسط عائد الاستراتيجية: {res_df['strategy_return_pct'].mean():.2f}%")
print(f"متوسط عائد Buy & Hold:   {res_df['buy_hold_return_pct'].mean():.2f}%")
print(f"متوسط الفارق:            {res_df['gap_pct'].mean():.2f} نقطة مئوية")
print(f"عدد الأسهم التي تفوقت فيها الاستراتيجية: {(res_df['gap_pct'] > 0).sum()} من {len(res_df)}")

t_stat, p_val = stats.ttest_1samp(res_df["gap_pct"], 0)
print(f"\nt-statistic (الفارق مقابل صفر): {t_stat:.3f}, p-value: {p_val:.4f}")
if p_val < 0.05:
    if t_stat > 0:
        print("=> دلالة إحصائية على تفوق الاستراتيجية على Buy & Hold.")
    else:
        print("=> دلالة إحصائية على أن الاستراتيجية أسوأ من Buy & Hold.")
else:
    print("=> لا يوجد دليل إحصائي كافٍ على وجود فارق حقيقي (قد يكون التذبذب صدفة).")

res_df.to_csv("backtest_results_honest.csv", index=False)
print("\nتم حفظ النتائج التفصيلية في backtest_results_honest.csv")
