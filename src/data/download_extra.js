import "dotenv/config";
import {
  fetchOpenInterest,
  fetchLongShortRatio,
  saveData,
} from "./historical.js";

const SYMBOL = "BTCUSDT";

const run = async () => {
  console.log("🚀 Загружаем недостающие данные...\n");

  console.log("5️⃣  Загружаем Open Interest...");
  const openInterest = await fetchOpenInterest(SYMBOL, "1h", 500);
  saveData("open_interest.json", openInterest);

  console.log("\n6️⃣  Загружаем Long/Short Ratio...");
  const lsRatio = await fetchLongShortRatio(SYMBOL, "1h", 500);
  saveData("long_short_ratio.json", lsRatio);

  console.log("\n✅ Готово!");
};

run();
