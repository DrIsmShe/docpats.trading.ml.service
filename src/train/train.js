import "dotenv/config";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
import { loadData } from "../data/historical.js";
import { prepareTrainingData } from "../features/features.js";
import { createModel, saveModel, SEQ_LEN } from "../model/model.js";
import fs from "fs";
import path from "path";

const FEEDBACK_FILE = path.resolve("./data/trade_feedback.json");

// ─── Загрузка фидбеков от реальных сделок ─────────────────────────────────
const loadTradeFeedback = () => {
  try {
    if (fs.existsSync(FEEDBACK_FILE)) {
      const raw = JSON.parse(fs.readFileSync(FEEDBACK_FILE, "utf-8"));
      // Берём только те у которых есть sequence (фичи)
      return raw.filter((f) => f.sequence && f.label);
    }
  } catch (err) {
    console.warn("⚠️  Не удалось загрузить фидбеки:", err.message);
  }
  return [];
};

// ─── Перевод лейбла в one-hot ─────────────────────────────────────────────
const labelToOneHot = (label) => {
  if (label === "BUY") return [1, 0, 0];
  if (label === "SELL") return [0, 0, 1];
  return [0, 1, 0];
};

const run = async () => {
  console.log("🧠 Начало обучения LSTM модели...\n");
  console.log(`   Архитектура: LSTM + Attention`);
  console.log(`   Последовательность: ${SEQ_LEN} свечей\n`);

  await tf.setBackend("cpu");
  await tf.ready();
  console.log("✅ Backend:", tf.getBackend());

  // ─── Шаг 1 — Загрузка рыночных данных ───────────────
  console.log("📂 Загружаем рыночные данные...");
  const candles1h = loadData("candles_1h.json");
  const candles4h = loadData("candles_4h.json");
  const candles1d = loadData("candles_1d.json");
  const fundingRate = loadData("funding_rate.json");
  const openInterest = loadData("open_interest.json");
  const longShortRatio = loadData("long_short_ratio.json");

  if (!candles1h || !candles4h || !candles1d) {
    console.error("❌ Данные не найдены — сначала запусти download.js");
    process.exit(1);
  }

  console.log(`✅ Загружено:`);
  console.log(`   1h: ${candles1h.length} свечей`);
  console.log(`   4h: ${candles4h.length} свечей`);
  console.log(`   1d: ${candles1d.length} свечей`);
  console.log(`   Funding Rate: ${fundingRate?.length ?? 0}`);
  console.log(`   Open Interest: ${openInterest?.length ?? 0}`);
  console.log(`   Long/Short: ${longShortRatio?.length ?? 0}\n`);

  // ─── Шаг 2 — Загрузка реальных сделок бота ──────────
  console.log("📊 Загружаем фидбеки от реальных сделок бота...");
  const tradeFeedbacks = loadTradeFeedback();
  console.log(`✅ Фидбеков с данными: ${tradeFeedbacks.length}\n`);

  // ─── Шаг 3 — Подготовка рыночных последовательностей ─
  console.log("⚙️  Подготовка LSTM последовательностей из рыночных данных...");
  const { X: X_market, Y: Y_market } = prepareTrainingData(
    candles1h,
    candles4h,
    candles1d,
    fundingRate ?? [],
    openInterest ?? [],
    longShortRatio ?? [],
    4, // lookahead: 4 свечи
    0.008, // threshold: 0.8%
  );

  if (X_market.length < 500) {
    console.error("❌ Недостаточно рыночных данных для обучения");
    process.exit(1);
  }

  // ─── Шаг 4 — Объединяем рыночные данные + реальные сделки ──
  let X_all = [...X_market];
  let Y_all = [...Y_market];
  let sampleWeights = new Array(X_market.length).fill(1.0);

  if (tradeFeedbacks.length > 0) {
    const X_trades = tradeFeedbacks.map((f) => f.sequence);
    const Y_trades = tradeFeedbacks.map((f) => labelToOneHot(f.label));

    // Реальные сделки имеют повышенный вес:
    // - прибыльные → вес 3.0 (усиленно учимся)
    // - убыточные → вес 2.0 (тоже важно — учимся НЕ делать так)
    const tradeWeights = tradeFeedbacks.map((f) => (f.profitable ? 3.0 : 2.0));

    X_all = [...X_all, ...X_trades];
    Y_all = [...Y_all, ...Y_trades];
    sampleWeights = [...sampleWeights, ...tradeWeights];

    console.log(`✅ Добавлено реальных сделок: ${tradeFeedbacks.length}`);
    console.log(
      `   Прибыльных: ${tradeFeedbacks.filter((f) => f.profitable).length} (вес 3.0)`,
    );
    console.log(
      `   Убыточных: ${tradeFeedbacks.filter((f) => !f.profitable).length} (вес 2.0)`,
    );
    console.log(`   Итого примеров: ${X_all.length}\n`);
  }

  // ─── Шаг 5 — Train/Test split ───────────────────────
  // ВАЖНО: не перемешиваем — сохраняем временной порядок для рыночных данных
  // Сделки бота идут в конец (как самые новые данные)
  const splitIndex = Math.floor(X_market.length * 0.8);
  const X_train = X_all.slice(0, splitIndex);
  const Y_train = Y_all.slice(0, splitIndex);
  const weights_train = sampleWeights.slice(0, splitIndex);

  // В тест — оставшиеся рыночные данные (без фидбеков — чистая оценка)
  const X_test = X_market.slice(splitIndex);
  const Y_test = Y_market.slice(splitIndex);

  console.log(`📊 Train: ${X_train.length} | Test: ${X_test.length}\n`);

  // ─── Шаг 6 — Балансировка классов ───────────────────
  const buys = Y_train.filter((y) => y[0] === 1).length;
  const holds = Y_train.filter((y) => y[1] === 1).length;
  const sells = Y_train.filter((y) => y[2] === 1).length;
  const total = Y_train.length;

  const classWeight = {
    0: total / (3 * Math.max(buys, 1)),
    1: total / (3 * Math.max(holds, 1)),
    2: total / (3 * Math.max(sells, 1)),
  };

  console.log(`⚖️  Class weights:`);
  console.log(`   BUY  (${buys}):  ${classWeight[0].toFixed(2)}x`);
  console.log(`   HOLD (${holds}): ${classWeight[1].toFixed(2)}x`);
  console.log(`   SELL (${sells}): ${classWeight[2].toFixed(2)}x\n`);

  // ─── Шаг 7 — Создание тензоров ──────────────────────
  const featureCount = X_train[0][0].length;
  console.log(
    `✅ Размерность: [${X_train.length}, ${SEQ_LEN}, ${featureCount}]`,
  );

  const xTrain = tf.tensor3d(X_train);
  const yTrain = tf.tensor2d(Y_train);
  const xTest = tf.tensor3d(X_test);
  const yTest = tf.tensor2d(Y_test);

  // ─── Шаг 8 — Создание модели ────────────────────────
  console.log("🏗️  Создание LSTM модели...");
  const model = createModel(featureCount);
  model.summary();

  // ─── Шаг 9 — Обучение с Early Stopping ──────────────
  console.log("\n🚀 Начало обучения...\n");

  let bestValAcc = 0;
  let patienceCounter = 0;
  const PATIENCE = 8;
  const EPOCHS = 80;

  await model.fit(xTrain, yTrain, {
    epochs: EPOCHS,
    batchSize: 32,
    validationData: [xTest, yTest],
    shuffle: false,
    verbose: 0,
    classWeight,

    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const loss = logs.loss.toFixed(4);
        const acc = ((logs.acc ?? logs.accuracy ?? 0) * 100).toFixed(1);
        const valLoss = logs.val_loss.toFixed(4);
        const valAcc = ((logs.val_acc ?? logs.val_accuracy ?? 0) * 100).toFixed(
          1,
        );

        process.stdout.write(
          `Epoch ${String(epoch + 1).padStart(3)}/${EPOCHS} | ` +
            `loss: ${loss} | acc: ${acc}% | ` +
            `val_loss: ${valLoss} | val_acc: ${valAcc}%`,
        );

        const currentValAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
        if (currentValAcc > bestValAcc) {
          bestValAcc = currentValAcc;
          patienceCounter = 0;
          await saveModel(model, "./saved_model");
          process.stdout.write(" ← 💾\n");
        } else {
          patienceCounter++;
          process.stdout.write("\n");
          if (patienceCounter >= PATIENCE) {
            console.log(
              `\n⏹️  Early stopping на epoch ${epoch + 1} (patience=${PATIENCE})`,
            );
            model.stopTraining = true;
          }
        }
      },
    },
  });

  // ─── Шаг 10 — Финальная оценка ──────────────────────
  console.log("\n📈 Финальная оценка...");
  const evalResult = model.evaluate(xTest, yTest, { verbose: 0 });
  const testLoss = evalResult[0].dataSync()[0].toFixed(4);
  const testAcc = (evalResult[1].dataSync()[0] * 100).toFixed(1);

  console.log(`\n✅ Результаты:`);
  console.log(`   Test Loss:     ${testLoss}`);
  console.log(`   Test Accuracy: ${testAcc}%`);
  console.log(`   Best Val Acc:  ${(bestValAcc * 100).toFixed(1)}%`);
  if (tradeFeedbacks.length > 0) {
    console.log(`   Реальных сделок в обучении: ${tradeFeedbacks.length}`);
  }

  // ─── Шаг 11 — Анализ предсказаний ───────────────────
  console.log("\n🔍 Анализ на тестовых данных...");
  const predictions = model.predict(xTest);
  const predData = await predictions.array();
  const trueData = await yTest.array();

  let tp_buy = 0,
    fp_buy = 0,
    fn_buy = 0;
  let tp_sell = 0,
    fp_sell = 0,
    fn_sell = 0;

  for (let i = 0; i < predData.length; i++) {
    const pred = predData[i].indexOf(Math.max(...predData[i]));
    const true_ = trueData[i].indexOf(Math.max(...trueData[i]));

    if (pred === 0) {
      if (true_ === 0) tp_buy++;
      else fp_buy++;
    }
    if (true_ === 0 && pred !== 0) fn_buy++;
    if (pred === 2) {
      if (true_ === 2) tp_sell++;
      else fp_sell++;
    }
    if (true_ === 2 && pred !== 2) fn_sell++;
  }

  const buyPrecision = tp_buy / (tp_buy + fp_buy + 0.001);
  const sellPrecision = tp_sell / (tp_sell + fp_sell + 0.001);

  console.log(`   BUY  precision: ${(buyPrecision * 100).toFixed(1)}%`);
  console.log(`   SELL precision: ${(sellPrecision * 100).toFixed(1)}%`);

  if (parseFloat(testAcc) >= 55) {
    console.log(`\n🎉 Отличный результат! Модель готова.`);
  } else if (parseFloat(testAcc) >= 50) {
    console.log(`\n✅ Хороший результат. Модель можно использовать.`);
  } else {
    console.log(`\n⚠️  Точность низкая. Рекомендуется больше данных.`);
  }

  // ─── Очистка ─────────────────────────────────────────
  xTrain.dispose();
  yTrain.dispose();
  xTest.dispose();
  yTest.dispose();

  predictions.dispose();

  console.log("\n✅ Обучение завершено!");
  process.exit(0);
};

run().catch((err) => {
  console.error("❌ Ошибка обучения:", err.message);
  console.error(err.stack);
  process.exit(1);
});
