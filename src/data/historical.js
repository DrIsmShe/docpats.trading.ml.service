import axios from "axios";
import fs from "fs";
import path from "path";

const BINANCE_URL = "https://api.binance.com/api/v3/klines";
const FUTURES_URL = "https://fapi.binance.com/fapi/v1";

// ─── Загрузка свечей ───────────────────────────────
export const fetchCandles = async (symbol, interval, limit = 1000) => {
  const response = await axios.get(BINANCE_URL, {
    params: { symbol, interval, limit },
  });

  return response.data.map((k) => ({
    openTime: k[0],
    open: parseFloat(k[1]),
    high: parseFloat(k[2]),
    low: parseFloat(k[3]),
    close: parseFloat(k[4]),
    volume: parseFloat(k[5]),
    closeTime: k[6],
  }));
};

// ─── Загрузка всей истории по частям ──────────────
export const fetchFullHistory = async (symbol, interval, years = 5) => {
  console.log(`📥 Загружаем историю ${symbol} ${interval} за ${years} лет...`);

  const allCandles = [];
  const msPerCandle = intervalToMs(interval);
  const totalCandles = Math.floor(
    (years * 365 * 24 * 60 * 60 * 1000) / msPerCandle,
  );
  const batchSize = 1000;
  const batches = Math.ceil(totalCandles / batchSize);

  let endTime = Date.now();

  for (let i = 0; i < batches; i++) {
    try {
      const response = await axios.get(BINANCE_URL, {
        params: {
          symbol,
          interval,
          limit: batchSize,
          endTime,
        },
      });

      const candles = response.data.map((k) => ({
        openTime: k[0],
        open: parseFloat(k[1]),
        high: parseFloat(k[2]),
        low: parseFloat(k[3]),
        close: parseFloat(k[4]),
        volume: parseFloat(k[5]),
        closeTime: k[6],
      }));

      if (candles.length === 0) break;

      allCandles.unshift(...candles);
      endTime = candles[0].openTime - 1;

      process.stdout.write(
        `\r  Прогресс: ${i + 1}/${batches} батчей (${allCandles.length} свечей)`,
      );

      // Пауза чтобы не превысить лимит Binance API
      await sleep(300);
    } catch (err) {
      console.error(`\n❌ Ошибка батча ${i}: ${err.message}`);
      await sleep(2000);
    }
  }

  console.log(`\n✅ Загружено ${allCandles.length} свечей`);
  return allCandles;
};

// ─── Загрузка Funding Rate ─────────────────────────
export const fetchFundingRate = async (symbol, limit = 1000) => {
  console.log(`📥 Загружаем Funding Rate ${symbol}...`);

  const allData = [];
  let endTime = Date.now();
  const batches = 10;

  for (let i = 0; i < batches; i++) {
    try {
      const response = await axios.get(`${FUTURES_URL}/fundingRate`, {
        params: { symbol, limit, endTime },
      });

      const data = response.data.map((f) => ({
        time: f.fundingTime,
        rate: parseFloat(f.fundingRate),
      }));

      if (data.length === 0) break;

      allData.unshift(...data);
      endTime = data[0].time - 1;

      await sleep(300);
    } catch (err) {
      console.error(`❌ Funding Rate ошибка: ${err.message}`);
      break;
    }
  }

  console.log(`✅ Загружено ${allData.length} записей Funding Rate`);
  return allData;
};

// ─── Загрузка Open Interest ────────────────────────
export const fetchOpenInterest = async (
  symbol,
  interval = "1h",
  limit = 500,
) => {
  console.log(`📥 Загружаем Open Interest ${symbol}...`);

  try {
    const response = await axios.get(
      "https://fapi.binance.com/futures/data/openInterestHist",
      {
        params: { symbol, period: interval, limit },
      },
    );

    return response.data.map((d) => ({
      time: d.timestamp,
      openInterest: parseFloat(d.sumOpenInterest),
      openInterestValue: parseFloat(d.sumOpenInterestValue),
    }));
  } catch (err) {
    console.error(`❌ Open Interest ошибка: ${err.message}`);
    return [];
  }
};

// ─── Загрузка Long/Short Ratio ─────────────────────
export const fetchLongShortRatio = async (
  symbol,
  interval = "1h",
  limit = 500,
) => {
  console.log(`📥 Загружаем Long/Short Ratio ${symbol}...`);

  try {
    const response = await axios.get(
      "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
      {
        params: { symbol, period: interval, limit },
      },
    );

    return response.data.map((d) => ({
      time: d.timestamp,
      ratio: parseFloat(d.longShortRatio),
      longAccount: parseFloat(d.longAccount),
      shortAccount: parseFloat(d.shortAccount),
    }));
  } catch (err) {
    console.error(`❌ Long/Short Ratio ошибка: ${err.message}`);
    return [];
  }
};

// ─── Сохранение данных на диск ─────────────────────
export const saveData = (filename, data) => {
  const dir = path.join(process.cwd(), "data");
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const filepath = path.join(dir, filename);
  fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
  console.log(`💾 Сохранено: ${filepath} (${data.length} записей)`);
};

// ─── Загрузка данных с диска ───────────────────────
export const loadData = (filename) => {
  const filepath = path.join(process.cwd(), "data", filename);
  if (!fs.existsSync(filepath)) return null;

  const raw = fs.readFileSync(filepath, "utf-8");
  return JSON.parse(raw);
};

// ─── Вспомогательные функции ───────────────────────
const intervalToMs = (interval) => {
  const map = {
    "1m": 60000,
    "5m": 300000,
    "15m": 900000,
    "1h": 3600000,
    "4h": 14400000,
    "1d": 86400000,
  };
  return map[interval] || 3600000;
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
