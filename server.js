import "dotenv/config";
import express from "express";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
import { loadModel, predict, fineTune } from "./src/model/model.js";
import { extractSequence } from "./src/features/features.js";
import crypto from "crypto";
import {
  fetchFullHistory,
  fetchFundingRate,
  fetchOpenInterest,
  fetchLongShortRatio,
  loadData,
  saveData,
} from "./src/data/historical.js";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: "50mb" }));

const PORT = process.env.PORT || 3001;
const SYMBOL = process.env.SYMBOL || "BTCUSDT";

const FINETUNE_EVERY = 15;
const FINETUNE_EPOCHS = 5;
const MAX_FEEDBACK_STORE = 500;
const FEEDBACK_FILE = "./data/trade_feedback.json";

let model = null;
let isRetraining = false;
let isFineTuning = false;
let lastTrainTime = null;
let lastFineTuneTime = null;
let predictionCount = 0;
let feedbackCount = 0;
let fineTuneCount = 0;

const loadFeedback = () => {
  try {
    if (fs.existsSync(FEEDBACK_FILE)) {
      return JSON.parse(fs.readFileSync(FEEDBACK_FILE, "utf-8"));
    }
  } catch {}
  return [];
};

const saveFeedback = (feedbacks) => {
  const dir = path.dirname(FEEDBACK_FILE);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(FEEDBACK_FILE, JSON.stringify(feedbacks, null, 2));
};

const init = async () => {
  await tf.setBackend("cpu");
  await tf.ready();
  console.log("✅ TensorFlow backend:", tf.getBackend());

  model = await loadModel("./saved_model");

  if (!model) {
    console.log("⚠️  Модель не найдена — запускаем первичное обучение...");
    await triggerRetraining();
  } else {
    console.log("✅ Модель загружена (LSTM + Attention)");
  }

  const storedFeedback = loadFeedback();
  feedbackCount = storedFeedback.length;
  console.log(`📊 Накоплено фидбеков от сделок: ${feedbackCount}`);

  scheduleRetraining();

  app.listen(PORT, () => {
    console.log(`🚀 ML Service запущен на порту ${PORT}`);
  });
};

const scheduleRetraining = () => {
  const RETRAIN_INTERVAL = 7 * 24 * 60 * 60 * 1000;
  setInterval(async () => {
    console.log("📅 Плановое переобучение модели...");
    await downloadFreshData();
    await triggerRetraining();
  }, RETRAIN_INTERVAL);
};

const downloadFreshData = async () => {
  console.log("📥 Обновляем рыночные данные...");
  try {
    const candles1h = await fetchFullHistory(SYMBOL, "1h", 5);
    saveData("candles_1h.json", candles1h);
    const candles4h = await fetchFullHistory(SYMBOL, "4h", 5);
    saveData("candles_4h.json", candles4h);
    const candles1d = await fetchFullHistory(SYMBOL, "1d", 5);
    saveData("candles_1d.json", candles1d);
    const funding = await fetchFundingRate(SYMBOL, 1000);
    saveData("funding_rate.json", funding);
    const oi = await fetchOpenInterest(SYMBOL, "1h", 500);
    saveData("open_interest.json", oi);
    const ls = await fetchLongShortRatio(SYMBOL, "1h", 500);
    saveData("long_short_ratio.json", ls);
    console.log("✅ Рыночные данные обновлены");
  } catch (err) {
    console.error("❌ Ошибка обновления данных:", err.message);
  }
};

const triggerRetraining = () => {
  return new Promise((resolve) => {
    if (isRetraining) {
      resolve();
      return;
    }
    isRetraining = true;
    console.log("🧠 Запуск полного переобучения...");
    const child = spawn("node", ["src/train/train.js"], {
      cwd: path.resolve(__dirname),
      stdio: "inherit",
      env: process.env,
    });
    child.on("close", async (code) => {
      isRetraining = false;
      lastTrainTime = new Date().toISOString();
      if (code === 0) {
        const newModel = await loadModel("./saved_model");
        if (newModel) {
          model = newModel;
          console.log("✅ Новая модель загружена!");
        }
      } else {
        console.error(`❌ Переобучение завершилось с ошибкой (код ${code})`);
      }
      resolve();
    });
  });
};

const triggerFineTune = async (feedbacks) => {
  if (!model || isFineTuning || isRetraining) return;
  if (feedbacks.length < 10) return;
  isFineTuning = true;
  console.log(`\n🔬 Онлайн дообучение на ${feedbacks.length} сделках...`);
  try {
    const recent = feedbacks
      .filter((f) => f.sequence && f.label !== undefined)
      .slice(-200);
    if (recent.length < 10) return;
    const X = recent.map((f) => f.sequence);
    const Y = recent.map((f) => {
      if (f.label === "BUY") return [1, 0, 0];
      if (f.label === "SELL") return [0, 0, 1];
      return [0, 1, 0];
    });
    const sampleWeights = recent.map((f) => (f.profitable ? 2.0 : 0.5));
    const result = await fineTune(model, X, Y, sampleWeights, FINETUNE_EPOCHS);
    lastFineTuneTime = new Date().toISOString();
    fineTuneCount++;
    console.log(
      `✅ Дообучение #${fineTuneCount} | loss: ${result.loss.toFixed(4)} | acc: ${(result.acc * 100).toFixed(1)}%`,
    );
    const { saveModel } = await import("./src/model/model.js");
    await saveModel(model, "./saved_model");
  } catch (err) {
    console.error("❌ Ошибка дообучения:", err.message);
  } finally {
    isFineTuning = false;
  }
};

// ─── /predict ─────────────────────────────────────────────────────────────
app.post("/predict", async (req, res) => {
  try {
    if (!model) {
      return res.status(503).json({ error: "Модель не загружена" });
    }
    const {
      candles1h,
      candles4h,
      candles1d,
      fundingRate,
      openInterest,
      longShortRatio,
    } = req.body;

    if (!candles1h || candles1h.length < 250) {
      return res.status(400).json({
        error: `Недостаточно свечей 1h: нужно 250+, получено ${candles1h?.length ?? 0}`,
      });
    }

    // 🔍 DEBUG: хеш входных свечей
    const candlesHash = crypto
      .createHash("md5")
      .update(JSON.stringify(candles1h))
      .digest("hex")
      .slice(0, 8);
    const lc = candles1h.at(-1);
    console.log(
      `🔍 candlesHash=${candlesHash} | lastCandle t=${new Date(lc.openTime).toISOString()} close=${lc.close} | len=${candles1h.length}`,
    );

    const sequence = extractSequence(
      candles1h,
      candles4h ?? [],
      candles1d ?? [],
      fundingRate ?? [],
      openInterest ?? [],
      longShortRatio ?? [],
    );

    if (!sequence) {
      return res
        .status(400)
        .json({ error: "Не удалось построить последовательность" });
    }

    // 🔍 DEBUG: хеш фичей
    const seqHash = crypto
      .createHash("md5")
      .update(JSON.stringify(sequence))
      .digest("hex")
      .slice(0, 8);
    console.log(`🔍 seqHash=${seqHash}`);

    const result = predict(model, sequence);
    predictionCount++;

    console.log(
      `📡 [#${predictionCount}] ${result.signal} | ` +
        `BUY=${(result.buy * 100).toFixed(1)}% ` +
        `HOLD=${(result.hold * 100).toFixed(1)}% ` +
        `SELL=${(result.sell * 100).toFixed(1)}% ` +
        `(conf: ${(result.confidence * 100).toFixed(1)}%)`,
    );

    res.json(result);
  } catch (err) {
    console.error("❌ Ошибка предсказания:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─── /feedback ────────────────────────────────────────────────────────────
app.post("/feedback", async (req, res) => {
  try {
    const {
      side,
      entryPrice,
      exitPrice,
      pnlUSDT,
      strategy,
      mlSignal,
      mlConfidence,
      closeReason,
      candles1h,
      candles4h,
      candles1d,
      fundingRate,
      openInterest,
      longShortRatio,
    } = req.body;

    if (!candles1h || candles1h.length < 250) {
      return res.status(400).json({ error: "Нужны свечи" });
    }

    const profitable = (pnlUSDT ?? 0) > 0;
    let label;
    if (profitable) {
      // сделка прибыльна → направление было правильным
      label = side; // "BUY" или "SELL"
    } else {
      // сделка убыточна → НЕ надо было входить, правильный ответ HOLD
      label = "HOLD";
    }

    let sequence = null;
    try {
      sequence = extractSequence(
        candles1h,
        candles4h ?? [],
        candles1d ?? [],
        fundingRate ?? [],
        openInterest ?? [],
        longShortRatio ?? [],
      );
    } catch (e) {
      console.warn("⚠️  sequence для фидбека не построен:", e.message);
    }

    const feedback = {
      timestamp: new Date().toISOString(),
      side,
      entryPrice,
      exitPrice,
      pnlUSDT,
      profitable,
      strategy,
      mlSignal,
      mlConfidence,
      closeReason,
      label,
      sequence,
    };

    const allFeedbacks = loadFeedback();
    allFeedbacks.push(feedback);
    const trimmed = allFeedbacks.slice(-MAX_FEEDBACK_STORE);
    saveFeedback(trimmed);
    feedbackCount = trimmed.length;

    const emoji = profitable ? "✅" : "❌";
    console.log(
      `\n${emoji} Фидбек | ${side} | PnL: ${pnlUSDT?.toFixed(2)} | ${strategy} | Лейбл: ${label}`,
    );

    res.json({ ok: true, feedbackCount, label, profitable });

    const feedbacksWithSequence = trimmed.filter((f) => f.sequence);
    if (
      feedbacksWithSequence.length >= FINETUNE_EVERY &&
      feedbacksWithSequence.length % FINETUNE_EVERY === 0
    ) {
      setTimeout(() => triggerFineTune(feedbacksWithSequence), 1000);
    }
  } catch (err) {
    console.error("❌ Ошибка фидбека:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ─── /status ─────────────────────────────────────────────────────────────
app.get("/status", (req, res) => {
  res.json({
    status: model ? "ready" : isRetraining ? "training" : "no_model",
    model: model ? "loaded" : "not_loaded",
    isRetraining,
    isFineTuning,
    lastTrainTime,
    lastFineTuneTime,
    predictionCount,
    feedbackCount,
    fineTuneCount,
    uptime: process.uptime(),
  });
});

app.post("/retrain", async (req, res) => {
  if (isRetraining) return res.json({ status: "already_training" });
  res.json({ status: "started" });
  await downloadFreshData();
  await triggerRetraining();
});

app.post("/finetune", async (req, res) => {
  if (isFineTuning) return res.json({ status: "already_fine_tuning" });
  if (!model) return res.json({ status: "no_model" });
  const feedbacks = loadFeedback().filter((f) => f.sequence);
  if (feedbacks.length < 10)
    return res.json({ status: "not_enough_feedback", count: feedbacks.length });
  res.json({ status: "started", feedbackCount: feedbacks.length });
  setTimeout(() => triggerFineTune(feedbacks), 100);
});

app.get("/health", (req, res) => {
  res.json({ ok: true, model: !!model, feedbackCount });
});

init().catch(console.error);
