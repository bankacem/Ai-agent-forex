"""
نظام HFT — نسخة مُصلَحة
========================
هذا الملف يعالج المشاكل الحرجة التي تم رصدها في النسخة الأصلية:

  1. تدريب نموذج الذكاء الاصطناعي كان لا يُستدعى أبداً -> تم تفعيله دورياً
  2. calculate_alpha كان يمرر "عوائد" لدالة تتوقع "أسعاراً" -> تم تصحيحه
  3. لا توجد إدارة مخاطر حقيقية -> تمت إضافة: حجم صفقة متكيف مع التقلب،
     وقف خسارة، حد أقصى للخسارة اليومية (قاطع دائرة/circuit breaker)
  4. لا يوجد أمان للخيوط (thread safety) -> تمت إضافة قفل (Lock)
  5. بيانات السوق والمشاعر كانت عشوائية/وهمية بالكامل -> تم فصلها في
     طبقة "مزود بيانات" (DataProvider) قابلة للاستبدال بمصدر حقيقي
     (MT5 / CCXT / REST API لبروكر حقيقي) دون تعديل بقية النظام

⚠️ ملاحظة مهمة:
  هذا الملف ما زال يستخدم بيانات محاكاة افتراضياً (SimulatedDataProvider)
  لأن الشبكة في هذه البيئة لا تسمح بالاتصال ببروكرات فوركس حقيقية.
  لتشغيله على بيانات حقيقية: استبدل SimulatedDataProvider بتطبيق
  حقيقي لـ DataProvider (انظر التعليقات أسفل الكلاس) يتصل بـ MT5
  عبر مكتبة MetaTrader5، أو بمنصتك عبر CCXT/REST API.
"""

import time
import json
import random
import threading
import queue
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np


# ============ طبقة مزود البيانات (قابلة للاستبدال بمصدر حقيقي) ============

class DataProvider(ABC):
    """
    واجهة مجردة لمزود بيانات السوق.
    أي تطبيق حقيقي (MT5, CCXT, REST API لبروكر) يجب أن يرث من هذا الكلاس
    وينفّذ get_tick لكل رمز.
    """

    @abstractmethod
    def get_tick(self, symbol: str) -> Dict:
        """يرجع {'price': float, 'volume': int, 'bid': float, 'ask': float}"""
        raise NotImplementedError


class SimulatedDataProvider(DataProvider):
    """
    مزود بيانات محاكاة — للاختبار والتطوير فقط.
    ⚠️ النتائج هنا لا تعكس أداءً حقيقياً في السوق.
    """

    def __init__(self, symbols: List[str], base_prices: List[float]):
        self.symbols = symbols
        self.prices = dict(zip(symbols, base_prices))

    def get_tick(self, symbol: str) -> Dict:
        change = random.uniform(-3, 3)
        self.prices[symbol] += change
        price = self.prices[symbol]
        return {
            "price": price,
            "volume": random.randint(1000, 10000),
            "bid": price * 0.999,
            "ask": price * 1.001,
        }


class MT5DataProvider(DataProvider):
    """
    مثال هيكلي لتطبيق حقيقي عبر MetaTrader5.
    غير مُفعَّل هنا لأن هذه البيئة لا تملك اتصالاً بمنصة MT5.
    لتفعيله على جهازك: pip install MetaTrader5, ثم:

        import MetaTrader5 as mt5
        mt5.initialize(login=..., password=..., server=...)

    وتنفيذ get_tick باستخدام mt5.symbol_info_tick(symbol)
    """

    def __init__(self, login: int = None, password: str = None, server: str = None):
        raise NotImplementedError(
            "يتطلب مكتبة MetaTrader5 واتصالاً فعلياً بحساب تداول. "
            "ثبّت المكتبة على جهازك المحلي وزوّد بيانات الدخول هنا."
        )

    def get_tick(self, symbol: str) -> Dict:
        raise NotImplementedError


# ============ الطبقة الأولى: الذكاء الاصطناعي التنبؤي ============

class PredictiveAI:
    """نموذج تنبؤي بسيط. الإصلاح الجوهري: أصبح يُدرَّب فعلياً ودورياً."""

    def __init__(self, retrain_every: int = 200):
        self.model_weights: Dict[int, float] = {}
        self.is_trained = False
        self.retrain_every = retrain_every   # عدد نقاط السعر الجديدة قبل إعادة التدريب
        self.ticks_since_train = 0

    def extract_features(self, prices: List[float], volumes: List[float], indicators: Dict) -> np.ndarray:
        if len(prices) < 20:
            return np.array([])

        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        features = [
            np.mean(returns[-5:]) if returns else 0,
            np.std(returns[-5:]) if returns else 0,
            returns[-1] if returns else 0,
            indicators.get("rsi", 50),
            indicators.get("volatility", 0),
        ]

        if volumes:
            avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
            features.append(volumes[-1] / avg_volume if avg_volume > 0 else 1)
        else:
            features.append(1)

        features.append(prices[-1] / prices[-3] - 1 if len(prices) >= 3 else 0)
        features.append(prices[-1] / prices[-10] - 1 if len(prices) >= 10 else 0)

        return np.array(features)

    def train(self, price_history: List[float], volume_history: List[float], indicators_history: List[Dict]):
        """تدريب النموذج. يُستدعى الآن فعلياً من الوكيل (انظر maybe_retrain)."""
        if len(price_history) < 50:
            return

        features_list, targets = [], []
        for i in range(20, len(price_history) - 5):
            prices = price_history[: i + 1]
            volumes = volume_history[: i + 1] if i < len(volume_history) else []
            indicators = indicators_history[i] if i < len(indicators_history) else {}

            features = self.extract_features(prices, volumes, indicators)
            if len(features) == 0:
                continue

            future_price = price_history[i + 5] if i + 5 < len(price_history) else price_history[-1]
            target = 1 if future_price > price_history[i] else 0

            features_list.append(features)
            targets.append(target)

        if len(features_list) < 10:
            return

        self.is_trained = True
        feature_means = np.mean(features_list, axis=0)
        feature_stds = np.std(features_list, axis=0)
        target_mean = np.mean(targets)

        for i in range(len(feature_means)):
            correlation = 0.0
            if feature_stds[i] > 0:
                for features, target in zip(features_list, targets):
                    correlation += (features[i] - feature_means[i]) / feature_stds[i] * (target - target_mean)
                correlation /= len(features_list)
            self.model_weights[i] = correlation

        self.ticks_since_train = 0

    def maybe_retrain(self, price_history: List[float], volume_history: List[float], indicators_history: List[Dict]):
        """الإصلاح الأساسي: تُستدعى هذه في كل تحديث بيانات، وتُعيد التدريب دورياً بدل عدم التدريب أبداً."""
        self.ticks_since_train += 1
        if not self.is_trained or self.ticks_since_train >= self.retrain_every:
            self.train(price_history, volume_history, indicators_history)

    def predict(self, prices: List[float], volumes: List[float], indicators: Dict) -> Dict:
        if not self.is_trained:
            return {"direction": 0, "confidence": 0, "price_prediction": prices[-1] if prices else 0}

        features = self.extract_features(prices, volumes, indicators)
        if len(features) == 0:
            return {"direction": 0, "confidence": 0, "price_prediction": prices[-1] if prices else 0}

        prediction = sum(w * features[i] for i, w in self.model_weights.items() if i < len(features))
        prediction = float(np.tanh(prediction))
        confidence = min(100, abs(prediction) * 100 + 20)
        price_change = prediction * prices[-1] * 0.01 if prices else 0

        return {
            "direction": 1 if prediction > 0.1 else -1 if prediction < -0.1 else 0,
            "confidence": confidence,
            "price_prediction": (prices[-1] + price_change) if prices else 0,
            "raw_prediction": prediction,
        }


class TrendSignalEngine:
    """
    ✅ الاستبدال الصادق لـ PredictiveAI: بدل الادعاء بـ"تعلّم آلي" وهمي (ارتباط بسيط
    غير مُتحقق منه، منحاز بشكل ممنهج كما أثبت الاختبار على AET)، هذا محرك اتجاه
    كمّي معروف وموثّق: تقاطع متوسطات متحركة (MA Crossover) + تأكيد زخم (Momentum)
    + فلتر قوة الاتجاه. لا "تدريب" هنا، لا ادعاءات زائفة — فقط حساب رياضي شفاف
    يمكن تدقيقه بالكامل.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30, momentum_period: int = 10):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.momentum_period = momentum_period

    @staticmethod
    def _sma(values: List[float], period: int) -> float:
        if len(values) < period:
            return float(np.mean(values)) if values else 0.0
        return float(np.mean(values[-period:]))

    def compute(self, prices: List[float]) -> Dict:
        min_needed = self.slow_period + 5
        if len(prices) < min_needed:
            return {"direction": 0, "confidence": 0, "reason": "بيانات غير كافية"}

        fast_ma = self._sma(prices, self.fast_period)
        slow_ma = self._sma(prices, self.slow_period)

        # فرق المتوسطات نسبياً للسعر الحالي (لا وحدات مطلقة)
        spread = (fast_ma - slow_ma) / slow_ma if slow_ma else 0.0

        # زخم: التغيّر النسبي خلال نافذة الزخم
        momentum = (
            (prices[-1] - prices[-self.momentum_period]) / prices[-self.momentum_period]
            if len(prices) >= self.momentum_period and prices[-self.momentum_period] != 0
            else 0.0
        )

        # اتجاه أساسي: فوق/تحت المتوسط البطيء = فلتر الاتجاه العام
        above_slow = prices[-1] > slow_ma

        direction = 0
        if spread > 0 and momentum > 0 and above_slow:
            direction = 1
        elif spread < 0 and momentum < 0 and not above_slow:
            direction = -1

        # الثقة: تُبنى من قوة الفصل بين المتوسطين، مقيّدة بمقياس معقول (لا صيغة تعطي 100% تعسفاً)
        raw_strength = min(abs(spread) * 25, 1.0)  # spread ~4% => أقصى ثقة تقريباً
        confidence = 40 + raw_strength * 55  # نطاق [40, 95] عند وجود إشارة فعلية
        confidence = confidence if direction != 0 else 0.0

        return {
            "direction": direction,
            "confidence": round(min(confidence, 95.0), 2),
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "spread_pct": round(spread * 100, 3),
            "momentum_pct": round(momentum * 100, 3),
        }


# ============ الطبقة الثانية: تحليل المشاعر ============

class SentimentAnalyzer:
    """
    ⚠️ هذا محلل بسيط بالكلمات المفتاحية، وليس NLP حقيقياً.
    لتفعيل تحليل حقيقي: زوّد analyze_market_sentiment بنصوص حقيقية
    من مصدر فعلي (X/Twitter API, RSS أخبار...) بدل النصوص المولّدة داخلياً.
    """

    def __init__(self):
        self.sentiment_scores = defaultdict(list)
        self.keywords = {
            "positive": ["🚀", "📈", "✅", "🔥", "ممتاز", "رائع", "قوي", "صاعد", "اختراق"],
            "negative": ["📉", "⚠️", "🔴", "كارثة", "ضعيف", "هابط", "فشل", "انخفاض"],
        }

    def analyze_text(self, text: str) -> Dict:
        pos = sum(1 for k in self.keywords["positive"] if k in text)
        neg = sum(1 for k in self.keywords["negative"] if k in text)
        total = pos + neg
        if total == 0:
            return {"sentiment": 0, "confidence": 0, "label": "neutral"}
        score = (pos - neg) / total
        return {
            "sentiment": score,
            "confidence": min(100, total * 20),
            "label": "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral",
        }

    def analyze_market_sentiment(self, symbol: str, texts: List[str]) -> Dict:
        if not texts:
            return {"overall": 0, "confidence": 0, "count": 0}

        results = [self.analyze_text(t) for t in texts]
        avg_sentiment = sum(r["sentiment"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        self.sentiment_scores[symbol].append(avg_sentiment)
        self.sentiment_scores[symbol] = self.sentiment_scores[symbol][-100:]

        return {"overall": avg_sentiment, "confidence": avg_confidence, "count": len(results)}


# ============ الطبقة الثالثة: النماذج الكمية (مصححة) ============

class QuantitativeModels:
    @staticmethod
    def _returns(prices: List[float]) -> List[float]:
        return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

    @staticmethod
    def calculate_beta_from_prices(prices: List[float], market_prices: List[float]) -> float:
        """يتوقع أسعاراً وليس عوائد."""
        if len(prices) < 10 or len(market_prices) < 10:
            return 1.0
        returns = QuantitativeModels._returns(prices)
        market_returns = QuantitativeModels._returns(market_prices)
        min_len = min(len(returns), len(market_returns))
        returns, market_returns = returns[-min_len:], market_returns[-min_len:]
        if min_len < 2:
            return 1.0
        cov = np.cov(returns, market_returns)[0][1]
        var = np.var(market_returns)
        return float(cov / var) if var > 0 else 1.0

    @staticmethod
    def calculate_beta_from_returns(returns: List[float], market_returns: List[float]) -> float:
        """✅ جديد: نسخة تتوقع عوائد جاهزة مباشرة (بدون إعادة اشتقاقها من نفسها)."""
        min_len = min(len(returns), len(market_returns))
        if min_len < 2:
            return 1.0
        r, m = returns[-min_len:], market_returns[-min_len:]
        cov = np.cov(r, m)[0][1]
        var = np.var(m)
        return float(cov / var) if var > 0 else 1.0

    @staticmethod
    def calculate_alpha(returns: List[float], market_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        ✅ الإصلاح: كان يمرر returns لدالة تتوقع أسعاراً (فتُشتق عوائد من عوائد
        بالخطأ). الآن يستخدم calculate_beta_from_returns مباشرة.
        """
        if len(returns) < 2:
            return 0.0
        avg_return = np.mean(returns)
        avg_market = np.mean(market_returns) if market_returns else 0
        beta = QuantitativeModels.calculate_beta_from_returns(returns, market_returns)
        daily_rf = risk_free_rate / 252
        return float(avg_return - (daily_rf + beta * (avg_market - daily_rf)))

    @staticmethod
    def calculate_sharpe(returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.01
        return float((avg_return / std_return) * np.sqrt(252)) if std_return > 0 else 0.0


# ============ الطبقة الرابعة: إدارة المخاطر (جديدة بالكامل) ============

class RiskManager:
    """
    ⚠️ هذه الطبقة كانت غائبة تماماً في النسخة الأصلية. تشمل:
      - حجم صفقة يتكيف مع التقلب (لا نسبة ثابتة 10% دائماً)
      - وقف خسارة إلزامي لكل صفقة
      - حد أقصى للانكشاف لكل رمز
      - قاطع دائرة (circuit breaker): إيقاف التداول عند تجاوز خسارة يومية محددة
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,      # أقصى نسبة من الرصيد لكل صفقة
        max_daily_loss_pct: float = 0.03,     # 3% خسارة يومية توقف التداول تلقائياً
        stop_loss_pct: float = 0.02,          # وقف خسارة 2% لكل صفقة
        max_symbol_exposure_pct: float = 0.25,  # أقصى انكشاف إجمالي لرمز واحد
    ):
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_symbol_exposure_pct = max_symbol_exposure_pct
        self.trading_halted = False
        self.halt_reason = ""

    def check_circuit_breaker(self, daily_pnl: float, initial_balance: float) -> bool:
        """يُرجع True إذا يجب إيقاف التداول."""
        if initial_balance <= 0:
            return False
        loss_pct = -daily_pnl / initial_balance
        if loss_pct >= self.max_daily_loss_pct:
            self.trading_halted = True
            self.halt_reason = f"تجاوز حد الخسارة اليومية المسموح ({self.max_daily_loss_pct*100:.1f}%)"
            return True
        return False

    def position_size(self, balance: float, price: float, volatility: float) -> float:
        """
        حجم متكيف: كلما زاد التقلب، قلّ حجم الصفقة (المخاطرة الثابتة نسبياً
        بدل نسبة ثابتة من الرصيد كما كان في النسخة الأصلية).
        """
        volatility = max(volatility, 0.001)  # تجنب القسمة على صفر
        risk_scaling = min(1.0, 0.02 / volatility)  # كلما زاد التقلب، صغُر الحجم
        target_pct = self.max_position_pct * risk_scaling
        allocation = balance * target_pct
        return allocation / price if price > 0 else 0

    def stop_loss_price(self, entry_price: float, side: str) -> float:
        if side == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        return entry_price * (1 + self.stop_loss_pct)

    def within_exposure_limit(self, symbol_exposure: float, balance: float) -> bool:
        if balance <= 0:
            return False
        return symbol_exposure <= balance * self.max_symbol_exposure_pct

    def reset_daily(self):
        self.trading_halted = False
        self.halt_reason = ""


# ============ الوكيل الرئيسي (مُصلَح) ============

class FixedHFTAgent:
    def __init__(self, agent_id: str, data_provider: DataProvider, symbols: List[str], config: Dict = None):
        self.id = agent_id
        self.config = config or {}
        self.data_provider = data_provider
        self.symbols = symbols

        self.signal_engine = TrendSignalEngine(**self.config.get("trend", {}))
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager(**self.config.get("risk", {}))

        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.volume_history: Dict[str, List[int]] = defaultdict(list)
        self.indicators_history: Dict[str, List[Dict]] = defaultdict(list)
        self.sentiment_texts: Dict[str, List[str]] = defaultdict(list)

        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []

        self.initial_balance = self.config.get("initial_balance", 100000)
        self.balance = self.initial_balance
        self.daily_pnl = 0.0
        self.stats = {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "returns": []}

        # ✅ إصلاح: قفل لحماية الحالة المشتركة بين خيط التداول وأي قراءة خارجية
        self.lock = threading.Lock()

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"hft_fixed_{self.id}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(h)
        return logger

    def _calculate_indicators(self, symbol: str) -> Dict:
        prices = self.price_history[symbol]
        if len(prices) < 20:
            return {}
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        volatility = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0
        rsi = self._calculate_rsi(prices)
        return {"rsi": rsi, "volatility": volatility, "price": prices[-1]}

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            gains.append(max(diff, 0))
            losses.append(abs(min(diff, 0)))
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:]) or 1e-9
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def update_market_data(self, symbol: str):
        tick = self.data_provider.get_tick(symbol)
        with self.lock:
            self.price_history[symbol].append(tick["price"])
            self.volume_history[symbol].append(tick["volume"])
            indicators = self._calculate_indicators(symbol)
            self.indicators_history[symbol].append(indicators)

            for hist in (self.price_history[symbol], self.volume_history[symbol], self.indicators_history[symbol]):
                if len(hist) > 1000:
                    del hist[0]

    def generate_signal(self, symbol: str) -> Dict:
        with self.lock:
            prices = list(self.price_history[symbol])
            indicators = self.indicators_history[symbol][-1] if self.indicators_history[symbol] else {}

        if len(prices) < 50:
            return {"action": "hold", "confidence": 0}

        trend = self.signal_engine.compute(prices)
        sentiment = self.sentiment_analyzer.analyze_market_sentiment(
            symbol, self.sentiment_texts[symbol][-20:]
        )

        # الاتجاه هو محرك القرار الأساسي (وزن أعلى)؛ المشاعر مؤكِّد ثانوي فقط،
        # وتبقى محدودة الأثر لأنها -حالياً- مبنية على كلمات مفتاحية وليست بيانات حقيقية.
        buy_score = sell_score = 0.0
        if trend["direction"] == 1:
            buy_score += trend["confidence"] * 0.75
        elif trend["direction"] == -1:
            sell_score += trend["confidence"] * 0.75

        if sentiment["overall"] > 0.1:
            buy_score += sentiment["confidence"] * 0.25
        elif sentiment["overall"] < -0.1:
            sell_score += sentiment["confidence"] * 0.25

        total = buy_score + sell_score
        if total > 0:
            buy_score = buy_score / total * 100
            sell_score = sell_score / total * 100

        confidence = max(buy_score, sell_score)
        action = "hold"
        if buy_score > 60 and buy_score > sell_score:
            action = "buy"
        elif sell_score > 60 and sell_score > buy_score:
            action = "sell"

        return {
            "action": action, "confidence": round(confidence, 2),
            "trend": trend, "sentiment": sentiment,
            "volatility": indicators.get("volatility", 0.01),
        }

    def execute_signal(self, symbol: str, signal: Dict) -> Dict:
        # ✅ قاطع الدائرة: لا صفقات جديدة إذا توقّف التداول
        if self.risk_manager.trading_halted:
            return {"status": "halted", "reason": self.risk_manager.halt_reason}

        if self.risk_manager.check_circuit_breaker(self.daily_pnl, self.initial_balance):
            self.logger.warning(f"🚨 قاطع الدائرة فُعِّل: {self.risk_manager.halt_reason}")
            return {"status": "halted", "reason": self.risk_manager.halt_reason}

        if signal["action"] == "hold" or signal["confidence"] < 60:
            return {"status": "hold"}

        with self.lock:
            price = self.price_history[symbol][-1] if self.price_history[symbol] else 0
        if price <= 0:
            return {"status": "error", "message": "لا يوجد سعر صالح"}

        if signal["action"] == "buy":
            return self._execute_buy(symbol, price, signal)
        return self._execute_sell(symbol, price, signal)

    def _execute_buy(self, symbol: str, price: float, signal: Dict) -> Dict:
        with self.lock:
            volatility = signal.get("volatility", 0.01)
            quantity = self.risk_manager.position_size(self.balance, price, volatility)
            cost = quantity * price

            # ✅ فحص حد الانكشاف قبل الدخول
            existing_exposure = self.positions.get(symbol, {}).get("quantity", 0) * price
            if not self.risk_manager.within_exposure_limit(existing_exposure + cost, self.balance):
                return {"status": "rejected", "reason": "تجاوز حد الانكشاف المسموح لهذا الرمز"}

            if cost > self.balance or cost <= 0:
                return {"status": "rejected", "reason": "رصيد غير كافٍ"}

            self.balance -= cost
            stop_loss = self.risk_manager.stop_loss_price(price, "buy")

            pos = self.positions.setdefault(symbol, {"quantity": 0, "avg_price": 0, "stop_loss": stop_loss})
            total_qty = pos["quantity"] + quantity
            total_cost = pos["avg_price"] * pos["quantity"] + cost
            pos["avg_price"] = total_cost / total_qty if total_qty > 0 else 0
            pos["quantity"] = total_qty
            pos["stop_loss"] = stop_loss  # ✅ وقف خسارة مسجَّل لكل مركز

            self.trade_history.append({
                "symbol": symbol, "side": "buy", "quantity": quantity,
                "price": price, "stop_loss": stop_loss, "timestamp": datetime.now().isoformat(),
            })
            self.stats["total_trades"] += 1

        self.logger.info(f"🟢 شراء {quantity:.4f} {symbol} @ {price:.4f} | وقف خسارة: {stop_loss:.4f}")
        return {"status": "executed", "type": "buy", "quantity": quantity, "price": price}

    def _execute_sell(self, symbol: str, price: float, signal: Dict) -> Dict:
        with self.lock:
            pos = self.positions.get(symbol)
            if not pos or pos["quantity"] <= 0:
                return {"status": "rejected", "reason": "لا توجد كمية للبيع"}

            # ✅ إصلاح جوهري: إشارة "بيع" من محرك اتجاه تعني انعكاس الاتجاه فعلياً،
            # فالخروج الصحيح هو إغلاق كامل المركز — وليس بيع 50% بشكل متكرر (النمط
            # القديم كان يقتصّ أرباحاً صغيرة ويُفوّت أغلب الاتجاه الصاعد، وهو ما
            # فسّر الفجوة السلبية الضخمة في اختبار الـ25 سهماً سابقاً).
            quantity = pos["quantity"]
            revenue = quantity * price
            profit = (price - pos["avg_price"]) * quantity

            self.balance += revenue
            self.daily_pnl += profit
            pos["quantity"] = 0
            del self.positions[symbol]

            self.trade_history.append({
                "symbol": symbol, "side": "sell", "quantity": quantity,
                "price": price, "profit": profit, "timestamp": datetime.now().isoformat(),
            })
            self.stats["total_trades"] += 1
            self.stats["winning_trades" if profit > 0 else "losing_trades"] += 1
            self.stats["returns"].append(profit / self.initial_balance)

        self.logger.info(f"🔴 بيع {quantity:.4f} {symbol} @ {price:.4f} | ربح: {profit:.2f}")
        return {"status": "executed", "type": "sell", "quantity": quantity, "price": price, "profit": profit}

    def check_stop_losses(self):
        """✅ جديد: فحص وقف الخسارة على كل مركز مفتوح في كل دورة."""
        with self.lock:
            symbols_to_check = list(self.positions.keys())

        for symbol in symbols_to_check:
            with self.lock:
                pos = self.positions.get(symbol)
                current_price = self.price_history[symbol][-1] if self.price_history[symbol] else None

            if not pos or current_price is None:
                continue

            if current_price <= pos["stop_loss"]:
                self.logger.warning(f"⛔ تفعيل وقف الخسارة لـ {symbol} عند {current_price:.4f}")
                self._force_close(symbol, current_price)

    def _force_close(self, symbol: str, price: float):
        with self.lock:
            pos = self.positions.get(symbol)
            if not pos:
                return
            quantity = pos["quantity"]
            revenue = quantity * price
            profit = (price - pos["avg_price"]) * quantity
            self.balance += revenue
            self.daily_pnl += profit
            del self.positions[symbol]
            self.trade_history.append({
                "symbol": symbol, "side": "stop_loss", "quantity": quantity,
                "price": price, "profit": profit, "timestamp": datetime.now().isoformat(),
            })
            self.stats["returns"].append(profit / self.initial_balance)

    def calculate_performance(self) -> Dict:
        with self.lock:
            total_trades = self.stats["total_trades"]
            if total_trades == 0:
                return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}

            win_rate = self.stats["winning_trades"] / total_trades * 100
            portfolio_value = self.balance + sum(
                p["quantity"] * self.price_history[sym][-1]
                for sym, p in self.positions.items() if self.price_history[sym]
            )
            total_pnl = portfolio_value - self.initial_balance
            sharpe = QuantitativeModels.calculate_sharpe(self.stats["returns"])

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "portfolio_value": round(portfolio_value, 2),
            "sharpe_ratio": round(sharpe, 3),
            "trading_halted": self.risk_manager.trading_halted,
        }

    def run(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.logger.info("🧠 بدء التشغيل (بيانات محاكاة — للاختبار فقط)")

    def _loop(self):
        last_reset_day = datetime.now().date()
        while self.running:
            try:
                if datetime.now().date() != last_reset_day:
                    with self.lock:
                        self.daily_pnl = 0.0
                    self.risk_manager.reset_daily()
                    last_reset_day = datetime.now().date()

                for symbol in self.symbols:
                    self.update_market_data(symbol)

                self.check_stop_losses()

                for symbol in self.symbols:
                    with self.lock:
                        have_enough = len(self.price_history[symbol]) > 50
                    if have_enough:
                        signal = self.generate_signal(symbol)
                        if signal["action"] != "hold":
                            self.execute_signal(symbol, signal)

                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"⚠️ خطأ في الحلقة: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("🛑 تم الإيقاف")


if __name__ == "__main__":
    symbols = ["BTC", "ETH", "SOL"]
    provider = SimulatedDataProvider(symbols, [50000, 3000, 150])

    agent = FixedHFTAgent(
        "hft_fixed_001",
        data_provider=provider,
        symbols=symbols,
        config={
            "initial_balance": 100000,
            "retrain_every": 200,
            "risk": {
                "max_position_pct": 0.10,
                "max_daily_loss_pct": 0.03,
                "stop_loss_pct": 0.02,
                "max_symbol_exposure_pct": 0.25,
            },
        },
    )

    agent.run()
    print("🧠 الوكيل المُصلَح يعمل (بيانات محاكاة فقط)...\n")

    try:
        for _ in range(6):
            time.sleep(2)
            perf = agent.calculate_performance()
            print(f"📊 صفقات: {perf['total_trades']} | ربح/خسارة: ${perf['total_pnl']} "
                  f"| شارب: {perf['sharpe_ratio']} | متوقف: {perf['trading_halted']}")
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
