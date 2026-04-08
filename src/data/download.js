import "dotenv/config";
import {
  fetchFullHistory,
  fetchFundingRate,
  fetchOpenInterest,
  fetchLongShortRatio,
  saveData,
} from "./historical.js";

const SYMBOL = "BTCUSDT";

const run = async () => {
  console.log("🚀 Начинаем загрузку исторических данных...\n");

  try {
    // ─── 1. Свечи 1h (5 лет) ──────────────────────
    console.log("1️⃣  Загружаем свечи 1h...");
    const candles1h = await fetchFullHistory(SYMBOL, "1h", 5);
    saveData("candles_1h.json", candles1h);

    // ─── 2. Свечи 4h (5 лет) ──────────────────────
    console.log("\n2️⃣  Загружаем свечи 4h...");
    const candles4h = await fetchFullHistory(SYMBOL, "4h", 5);
    saveData("candles_4h.json", candles4h);

    // ─── 3. Свечи 1d (5 лет) ──────────────────────
    console.log("\n3️⃣  Загружаем свечи 1d...");
    const candles1d = await fetchFullHistory(SYMBOL, "1d", 5);
    saveData("candles_1d.json", candles1d);

    // ─── 4. Funding Rate ───────────────────────────
    console.log("\n4️⃣  Загружаем Funding Rate...");
    const funding = await fetchFundingRate(SYMBOL, 1000);
    saveData("funding_rate.json", funding);

    // ─── 5. Open Interest ──────────────────────────
    console.log("\n5️⃣  Загружаем Open Interest...");
    const openInterest = await fetchOpenInterest(SYMBOL, "1h", 500);
    saveData("open_interest.json", openInterest);

    // ─── 6. Long/Short Ratio ───────────────────────
    console.log("\n6️⃣  Загружаем Long/Short Ratio...");
    const lsRatio = await fetchLongShortRatio(SYMBOL, "1h", 500);
    saveData("long_short_ratio.json", lsRatio);

    // ─── Итог ──────────────────────────────────────
    console.log("\n✅ Все данные загружены!\n");
    console.log("📁 Файлы сохранены в папке data/:");
    console.log("   - candles_1h.json");
    console.log("   - candles_4h.json");
    console.log("   - candles_1d.json");
    console.log("   - funding_rate.json");
    console.log("   - open_interest.json");
    console.log("   - long_short_ratio.json");
  } catch (err) {
    console.error("❌ Ошибка загрузки:", err.message);
    process.exit(1);
  }
};

run();
