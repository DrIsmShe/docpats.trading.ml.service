import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

// ─── Архитектура ───────────────────────────────────────────────────────────
export const SEQ_LEN = 24; // 24 часа истории

export const createModel = (featureCount = 45) => {
  const model = tf.sequential();

  // LSTM 1 — краткосрочные паттерны
  model.add(
    tf.layers.lstm({
      inputShape: [SEQ_LEN, featureCount],
      units: 128,
      returnSequences: true,
      dropout: 0.2,
      recurrentDropout: 0.1,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }),
  );

  // LSTM 2 — долгосрочные паттерны
  model.add(
    tf.layers.lstm({
      units: 64,
      returnSequences: false,
      dropout: 0.2,
      recurrentDropout: 0.1,
    }),
  );

  // Классификационная голова
  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }),
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.1 }));

  // Выход: BUY / HOLD / SELL
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.0003),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

// ─── Сохранение модели ─────────────────────────────────────────────────────
export const saveModel = async (model, savePath = "./saved_model") => {
  const absolutePath = path.resolve(savePath);
  if (!fs.existsSync(absolutePath)) {
    fs.mkdirSync(absolutePath, { recursive: true });
  }

  const modelJSON = model.toJSON();
  fs.writeFileSync(
    path.join(absolutePath, "model.json"),
    JSON.stringify(modelJSON, null, 2),
  );

  const weights = model.getWeights();
  const weightData = weights.map((w) => ({
    name: w.name,
    shape: w.shape,
    data: Array.from(w.dataSync()),
  }));
  fs.writeFileSync(
    path.join(absolutePath, "weights.json"),
    JSON.stringify(weightData, null, 2),
  );

  const meta = {
    savedAt: new Date().toISOString(),
    seqLen: SEQ_LEN,
    featureCount: weights[0]?.shape[0] ?? 45,
    architecture: "LSTM",
  };
  fs.writeFileSync(
    path.join(absolutePath, "meta.json"),
    JSON.stringify(meta, null, 2),
  );

  console.log(`✅ Модель сохранена: ${absolutePath}`);
};

// ─── Загрузка модели ───────────────────────────────────────────────────────
export const loadModel = async (savePath = "./saved_model") => {
  try {
    const absolutePath = path.resolve(savePath);
    const weightsPath = path.join(absolutePath, "weights.json");
    const metaPath = path.join(absolutePath, "meta.json");

    if (!fs.existsSync(weightsPath)) {
      console.log("⚠️  Модель не найдена — нужно обучить");
      return null;
    }

    const weightData = JSON.parse(fs.readFileSync(weightsPath, "utf-8"));
    const meta = fs.existsSync(metaPath)
      ? JSON.parse(fs.readFileSync(metaPath, "utf-8"))
      : { featureCount: 45, architecture: "LSTM" };

    console.log(`📂 Загружаем модель: ${meta.architecture} (${meta.savedAt})`);

    const featureCount = meta.featureCount ?? 45;
    const model = createModel(featureCount);

    const weights = weightData.map((w) => tf.tensor(w.data, w.shape));
    model.setWeights(weights);

    console.log("✅ Модель загружена с диска");
    return model;
  } catch (err) {
    console.log("⚠️  Ошибка загрузки модели:", err.message);
    return null;
  }
};

// ─── Предсказание ──────────────────────────────────────────────────────────
export const predict = (model, featuresSequence) => {
  const input = tf.tensor3d([featuresSequence]); // [1, SEQ_LEN, featureCount]
  const output = model.predict(input);
  const probs = output.dataSync();
  input.dispose();
  output.dispose();

  const buy = probs[0];
  const hold = probs[1];
  const sell = probs[2];

  // Порог 50% — чёткая уверенность требуется
  const THRESHOLD = 0.5;
  let signal = "HOLD";
  if (buy > sell && buy > THRESHOLD) signal = "BUY";
  else if (sell > buy && sell > THRESHOLD) signal = "SELL";

  return {
    buy,
    hold,
    sell,
    signal,
    confidence: Math.max(buy, sell),
  };
};

// ─── Онлайн дообучение на реальных сделках бота ───────────────────────────
// X: [[SEQ_LEN × featureCount], ...] — последовательности фичей
// Y: [[1,0,0], ...] — правильные лейблы (BUY/HOLD/SELL)
// sampleWeights: [2.0, 0.5, ...] — прибыльные сделки имеют больший вес
export const fineTune = async (
  model,
  X,
  Y,
  sampleWeights = null,
  epochs = 5,
) => {
  if (!X || X.length < 5) throw new Error("Слишком мало данных для дообучения");

  // Используем низкий learning rate чтобы не сломать существующие знания
  model.compile({
    optimizer: tf.train.adam(0.00005), // очень малый LR для дообучения
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const xTensor = tf.tensor3d(X); // [N, SEQ_LEN, featureCount]
  const yTensor = tf.tensor2d(Y); // [N, 3]

  let swTensor = null;
  if (sampleWeights && sampleWeights.length === X.length) {
    swTensor = tf.tensor1d(sampleWeights);
  }

  let lastLoss = 0;
  let lastAcc = 0;

  try {
    const fitArgs = {
      epochs,
      batchSize: Math.min(16, X.length),
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          lastLoss = logs.loss;
          lastAcc = logs.acc ?? logs.accuracy ?? 0;
          process.stdout.write(
            `\r  FineTune epoch ${epoch + 1}/${epochs} | ` +
              `loss: ${lastLoss.toFixed(4)} | acc: ${(lastAcc * 100).toFixed(1)}%`,
          );
        },
      },
    };

    if (swTensor) {
      fitArgs.sampleWeight = swTensor;
    }

    await model.fit(xTensor, yTensor, fitArgs);
    console.log(""); // новая строка
  } finally {
    xTensor.dispose();
    yTensor.dispose();
    swTensor?.dispose();

    // Возвращаем нормальный LR для обычных предсказаний
    model.compile({
      optimizer: tf.train.adam(0.0003),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }

  return { loss: lastLoss, acc: lastAcc };
};
