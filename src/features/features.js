import {
  calcRSI,
  calcEMA,
  calcMACD,
  calcBB,
  calcATR,
  calcStochastic,
  calcADX,
} from "./indicators.js";

import { SEQ_LEN } from "../model/model.js";

// ─── Нормализация ──────────────────────────────────────────────────────────
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const normalize = (v, min, max) =>
  max === min ? 0.5 : clamp((v - min) / (max - min), 0, 1);

// ─── Извлечение 45 признаков из одной точки времени ───────────────────────
export const extractFeatures = (
  candles1h,
  candles4h,
  candles1d,
  fundingRate,
  openInterest,
  longShortRatio,
) => {
  if (candles1h.length < 200) return null;

  const closes1h = candles1h.map((c) => c.close);
  const closes4h = candles4h.map((c) => c.close);
  const closes1d = candles1d.map((c) => c.close);

  const last = candles1h.at(-1);
  const prev = candles1h.at(-2);
  const prev2 = candles1h.at(-3);
  const price = last.close;

  // ── Технические 1h ──────────────────────────────────
  const rsi14 = calcRSI(closes1h, 14);
  const rsi7 = calcRSI(closes1h, 7);
  const ema20 = calcEMA(closes1h, 20);
  const ema50 = calcEMA(closes1h, 50);
  const ema200 = calcEMA(closes1h, 200);
  const macd = calcMACD(closes1h);
  const bb = calcBB(closes1h, 20);
  const atr = calcATR(candles1h, 14);
  const stoch = calcStochastic(candles1h, 14);
  const adx = calcADX(candles1h, 14);

  const lastRSI14 = rsi14.at(-1) ?? 50;
  const lastRSI7 = rsi7.at(-1) ?? 50;
  const lastEMA20 = ema20.at(-1) ?? price;
  const lastEMA50 = ema50.at(-1) ?? price;
  const lastEMA200 = ema200.at(-1) ?? price;
  const lastMACD = macd.histogram.at(-1) ?? 0;
  const prevMACD = macd.histogram.at(-2) ?? 0;
  const lastBB = bb.at(-1);
  const lastATR = atr.at(-1) ?? 1;
  const lastStoch = stoch.at(-1) ?? 50;
  const lastADX = adx.at(-1) ?? 25;

  // ── Технические 4h ──────────────────────────────────
  const rsi4h = calcRSI(closes4h, 14);
  const ema20_4h = calcEMA(closes4h, 20);
  const ema50_4h = calcEMA(closes4h, 50);
  const lastRSI4h = rsi4h.at(-1) ?? 50;
  const lastEMA20_4h = ema20_4h.at(-1) ?? closes4h.at(-1);
  const lastEMA50_4h = ema50_4h.at(-1) ?? closes4h.at(-1);
  const price4h = closes4h.at(-1) ?? price;

  // ── Технические 1d ──────────────────────────────────
  const rsi1d = calcRSI(closes1d, 14);
  const ema20_1d = calcEMA(closes1d, 20);
  const lastRSI1d = rsi1d.at(-1) ?? 50;
  const lastEMA20_1d = ema20_1d.at(-1) ?? closes1d.at(-1);
  const price1d = closes1d.at(-1) ?? price;

  // ── Объём ────────────────────────────────────────────
  const volumes = candles1h.slice(-20).map((c) => c.volume);
  const avgVol = volumes.slice(0, -1).reduce((a, b) => a + b, 0) / 19;
  const volRatio = avgVol > 0 ? last.volume / avgVol : 1;

  // ── Рыночные данные (крипто-специфика) ───────────────
  const lastFunding = fundingRate?.at(-1)?.rate ?? 0;
  const avgFunding3d =
    (fundingRate?.slice(-9).reduce((a, b) => a + b.rate, 0) ?? 0) / 9;
  const lastOI = openInterest?.at(-1)?.openInterest ?? 0;
  const prevOI = openInterest?.at(-2)?.openInterest ?? lastOI;
  const oiChange = prevOI > 0 ? (lastOI - prevOI) / prevOI : 0;
  const lastLSRatio = longShortRatio?.at(-1)?.ratio ?? 1;

  // ── Свечные паттерны ─────────────────────────────────
  const candleBody = Math.abs(last.close - last.open);
  const candleRange = last.high - last.low || 1;
  const bodyRatio = candleBody / candleRange;
  const isBullish = last.close > last.open ? 1 : 0;
  const upperShadow =
    (last.high - Math.max(last.open, last.close)) / candleRange;
  const lowerShadow =
    (Math.min(last.open, last.close) - last.low) / candleRange;

  // ── Временные признаки ───────────────────────────────
  const date = new Date(last.openTime);
  const hourOfDay = date.getUTCHours() / 24;
  const dayOfWeek = date.getUTCDay() / 7;
  const isWeekend = date.getUTCDay() === 0 || date.getUTCDay() === 6 ? 1 : 0;

  // ── Ценовые изменения ────────────────────────────────
  const change1 = prev.close > 0 ? (last.close - prev.close) / prev.close : 0;
  const change2 =
    prev2?.close > 0 ? (last.close - prev2.close) / prev2.close : 0;
  const change4h =
    closes1h.length > 4 ? (last.close - closes1h.at(-5)) / closes1h.at(-5) : 0;
  const change24h =
    closes1h.length > 24
      ? (last.close - closes1h.at(-25)) / closes1h.at(-25)
      : 0;

  // ── 45 признаков ─────────────────────────────────────
  return [
    // Технические 1h (14)
    lastRSI14 / 100,
    lastRSI7 / 100,
    clamp((price - lastEMA20) / lastEMA20, -0.1, 0.1) / 0.1,
    clamp((price - lastEMA50) / lastEMA50, -0.2, 0.2) / 0.2,
    clamp((price - lastEMA200) / lastEMA200, -0.3, 0.3) / 0.3,
    clamp((lastEMA20 - lastEMA50) / lastEMA50, -0.1, 0.1) / 0.1,
    clamp((lastEMA50 - lastEMA200) / lastEMA200, -0.2, 0.2) / 0.2,
    clamp(lastMACD / price, -0.01, 0.01) / 0.01,
    clamp((lastMACD - prevMACD) / price, -0.005, 0.005) / 0.005,
    lastBB ? (price - lastBB.lower) / (lastBB.upper - lastBB.lower || 1) : 0.5,
    lastBB ? clamp(lastBB.width, 0, 0.1) / 0.1 : 0.02,
    clamp(lastATR / price, 0, 0.05) / 0.05,
    lastStoch / 100,
    clamp(lastADX, 0, 100) / 100,

    // Технические 4h (4)
    lastRSI4h / 100,
    clamp((price4h - lastEMA20_4h) / lastEMA20_4h, -0.1, 0.1) / 0.1,
    clamp((price4h - lastEMA50_4h) / lastEMA50_4h, -0.2, 0.2) / 0.2,
    clamp((lastEMA20_4h - lastEMA50_4h) / lastEMA50_4h, -0.1, 0.1) / 0.1,

    // Технические 1d (3)
    lastRSI1d / 100,
    clamp((price1d - lastEMA20_1d) / lastEMA20_1d, -0.2, 0.2) / 0.2,
    clamp(change24h, -0.2, 0.2) / 0.2,

    // Объём (2)
    clamp(volRatio, 0, 5) / 5,
    clamp(last.volume / (avgVol * 3), 0, 1),

    // Рыночные данные (5)
    clamp(lastFunding * 1000, -1, 1),
    clamp(avgFunding3d * 1000, -1, 1),
    clamp(oiChange * 10, -1, 1),
    clamp(lastLSRatio, 0, 2) / 2,
    lastOI > 0 ? clamp(lastOI / 100000, 0, 1) : 0.5,

    // Свечные паттерны (5)
    bodyRatio,
    isBullish,
    upperShadow,
    lowerShadow,
    clamp(candleBody / (lastATR || 1), 0, 1),

    // Временные (3)
    hourOfDay,
    dayOfWeek,
    isWeekend,

    // Ценовые изменения (4)
    clamp(change1, -0.1, 0.1) / 0.1,
    clamp(change2, -0.1, 0.1) / 0.1,
    clamp(change4h, -0.1, 0.1) / 0.1,
    clamp(change24h, -0.2, 0.2) / 0.2,

    // Дополнительные (5)
    closes1h.length > 48
      ? normalize(
          last.close,
          Math.min(...closes1h.slice(-48)),
          Math.max(...closes1h.slice(-48)),
        )
      : 0.5,
    closes1h.length > 48
      ? clamp(
          (last.close - Math.max(...closes1h.slice(-48))) / lastATR,
          -5,
          5,
        ) / 5
      : 0,
    clamp(Math.abs(change24h) * 10, 0, 1),
    lastFunding > 0.001 ? 1 : lastFunding < -0.001 ? 0 : 0.5,
    lastLSRatio > 1.5 ? 1 : lastLSRatio < 0.7 ? 0 : 0.5,
  ];
};

// ─── Подготовка ПОСЛЕДОВАТЕЛЬНОСТЕЙ для LSTM ──────────────────────────────
// Для каждого момента времени i создаём окно [i-SEQ_LEN .. i]
// Это позволяет LSTM видеть историю
export const prepareTrainingData = (
  candles1h,
  candles4h,
  candles1d,
  fundingRate,
  openInterest,
  longShortRatio,
  lookahead = 4, // уменьшили с 8 до 4 — более реалистичный горизонт
  threshold = 0.008, // уменьшили с 1.5% до 0.8% — больше сигналов
) => {
  const X = []; // [samples, SEQ_LEN, featureCount]
  const Y = []; // [samples, 3]

  console.log(
    `📊 Подготовка LSTM последовательностей: ${candles1h.length} свечей...`,
  );
  console.log(
    `   SEQ_LEN=${SEQ_LEN}, lookahead=${lookahead}, threshold=${threshold * 100}%`,
  );

  // Предвычисляем индексы
  let idx4h = 0;
  let idx1d = 0;
  let idxFunding = 0;
  let idxOI = 0;
  let idxLS = 0;

  // Кэш признаков — не пересчитываем одно и то же
  const featureCache = new Map();

  const getFeatures = (i) => {
    if (featureCache.has(i)) return featureCache.get(i);

    const slice1h = candles1h.slice(Math.max(0, i - 250), i + 1);
    const slice4h = candles4h.slice(Math.max(0, idx4h - 100), idx4h + 1);
    const slice1d = candles1d.slice(Math.max(0, idx1d - 60), idx1d + 1);
    const sliceFunding = fundingRate.slice(
      Math.max(0, idxFunding - 20),
      idxFunding + 1,
    );
    const sliceOI = openInterest.slice(Math.max(0, idxOI - 20), idxOI + 1);
    const sliceLS = longShortRatio.slice(Math.max(0, idxLS - 20), idxLS + 1);

    if (slice4h.length < 50 || slice1d.length < 30) return null;

    const features = extractFeatures(
      slice1h,
      slice4h,
      slice1d,
      sliceFunding,
      sliceOI,
      sliceLS,
    );
    featureCache.set(i, features);
    return features;
  };

  const startIdx = 200 + SEQ_LEN;

  for (let i = startIdx; i < candles1h.length - lookahead; i++) {
    const currentTime = candles1h[i].openTime;

    // Двигаем индексы
    while (
      idx4h + 1 < candles4h.length &&
      candles4h[idx4h + 1].openTime <= currentTime
    )
      idx4h++;
    while (
      idx1d + 1 < candles1d.length &&
      candles1d[idx1d + 1].openTime <= currentTime
    )
      idx1d++;
    while (
      idxFunding + 1 < fundingRate.length &&
      fundingRate[idxFunding + 1].time <= currentTime
    )
      idxFunding++;
    while (
      idxOI + 1 < openInterest.length &&
      openInterest[idxOI + 1].time <= currentTime
    )
      idxOI++;
    while (
      idxLS + 1 < longShortRatio.length &&
      longShortRatio[idxLS + 1].time <= currentTime
    )
      idxLS++;

    // Собираем SEQ_LEN последовательных векторов признаков
    const sequence = [];
    let valid = true;

    for (let j = i - SEQ_LEN + 1; j <= i; j++) {
      const features = getFeatures(j);
      if (!features) {
        valid = false;
        break;
      }
      sequence.push(features);
    }

    if (!valid || sequence.length !== SEQ_LEN) continue;

    // Метка — что произойдёт через lookahead свечей
    const currentPrice = candles1h[i].close;
    const futurePrice = candles1h[i + lookahead].close;
    const change = (futurePrice - currentPrice) / currentPrice;

    let label;
    if (change > threshold)
      label = [1, 0, 0]; // BUY
    else if (change < -threshold)
      label = [0, 0, 1]; // SELL
    else label = [0, 1, 0]; // HOLD

    X.push(sequence);
    Y.push(label);

    // Чистим кэш старых записей
    if (i % 500 === 0) {
      for (const key of featureCache.keys()) {
        if (key < i - SEQ_LEN - 10) featureCache.delete(key);
      }
      process.stdout.write(
        `\r  Прогресс: ${i}/${candles1h.length} (${X.length} примеров)`,
      );
    }
  }

  const buys = Y.filter((y) => y[0] === 1).length;
  const holds = Y.filter((y) => y[1] === 1).length;
  const sells = Y.filter((y) => y[2] === 1).length;

  console.log(`\n✅ Данные готовы: ${X.length} примеров`);
  console.log(`   BUY: ${buys} (${((buys / Y.length) * 100).toFixed(1)}%)`);
  console.log(`   HOLD: ${holds} (${((holds / Y.length) * 100).toFixed(1)}%)`);
  console.log(`   SELL: ${sells} (${((sells / Y.length) * 100).toFixed(1)}%)`);

  return { X, Y };
};

// ─── Извлечение последовательности для реального предсказания ─────────────
// Берёт последние SEQ_LEN точек и строит матрицу [SEQ_LEN × 45]
export const extractSequence = (
  candles1h,
  candles4h,
  candles1d,
  fundingRate,
  openInterest,
  longShortRatio,
) => {
  if (candles1h.length < 200 + SEQ_LEN) return null;

  const sequence = [];
  const n = candles1h.length;

  for (let i = n - SEQ_LEN; i < n; i++) {
    const slice1h = candles1h.slice(Math.max(0, i - 250), i + 1);

    // Для 4h, 1d — берём данные до текущего момента
    const currentTime = candles1h[i].openTime;
    const slice4h = candles4h
      .filter((c) => c.openTime <= currentTime)
      .slice(-100);
    const slice1d = candles1d
      .filter((c) => c.openTime <= currentTime)
      .slice(-60);
    const sliceFunding = (fundingRate ?? [])
      .filter((f) => f.time <= currentTime)
      .slice(-20);
    const sliceOI = (openInterest ?? [])
      .filter((o) => o.time <= currentTime)
      .slice(-20);
    const sliceLS = (longShortRatio ?? [])
      .filter((l) => l.time <= currentTime)
      .slice(-20);

    if (slice4h.length < 50 || slice1d.length < 30) return null;

    const features = extractFeatures(
      slice1h,
      slice4h,
      slice1d,
      sliceFunding,
      sliceOI,
      sliceLS,
    );
    if (!features) return null;

    sequence.push(features);
  }

  return sequence.length === SEQ_LEN ? sequence : null;
};
