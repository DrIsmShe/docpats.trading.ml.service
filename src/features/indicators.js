// ─── Технические индикаторы ────────────────────────

export const calcRSI = (closes, period = 14) => {
  const result = [];
  for (let i = period; i < closes.length; i++) {
    const slice = closes.slice(i - period, i);
    let gains = 0,
      losses = 0;
    for (let j = 1; j < slice.length; j++) {
      const diff = slice[j] - slice[j - 1];
      if (diff > 0) gains += diff;
      else losses += Math.abs(diff);
    }
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    result.push(100 - 100 / (1 + rs));
  }
  return result;
};

export const calcEMA = (values, period) => {
  const k = 2 / (period + 1);
  const result = [values[0]];
  for (let i = 1; i < values.length; i++) {
    result.push(values[i] * k + result[i - 1] * (1 - k));
  }
  return result;
};

export const calcMACD = (closes) => {
  const ema12 = calcEMA(closes, 12);
  const ema26 = calcEMA(closes, 26);
  const minLen = Math.min(ema12.length, ema26.length);
  const macdLine = [];
  for (let i = 0; i < minLen; i++) {
    macdLine.push(
      ema12[ema12.length - minLen + i] - ema26[ema26.length - minLen + i],
    );
  }
  const signal = calcEMA(macdLine, 9);
  const histogram = macdLine.slice(-signal.length).map((v, i) => v - signal[i]);
  return { macdLine, signal, histogram };
};

export const calcBB = (closes, period = 20, mult = 2) => {
  const result = [];
  for (let i = period - 1; i < closes.length; i++) {
    const slice = closes.slice(i - period + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(
      slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period,
    );
    result.push({
      upper: mean + mult * std,
      middle: mean,
      lower: mean - mult * std,
      width: (mult * 2 * std) / mean,
    });
  }
  return result;
};

export const calcATR = (candles, period = 14) => {
  const trs = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1];
    return Math.max(
      c.high - c.low,
      Math.abs(c.high - prev.close),
      Math.abs(c.low - prev.close),
    );
  });
  const result = [];
  for (let i = period - 1; i < trs.length; i++) {
    result.push(
      trs.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period,
    );
  }
  return result;
};

export const calcStochastic = (candles, period = 14) => {
  const result = [];
  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const high = Math.max(...slice.map((c) => c.high));
    const low = Math.min(...slice.map((c) => c.low));
    const k =
      high === low ? 50 : ((slice.at(-1).close - low) / (high - low)) * 100;
    result.push(k);
  }
  return result;
};

export const calcADX = (candles, period = 14) => {
  const result = [];
  for (let i = period * 2; i < candles.length; i++) {
    const slice = candles.slice(i - period * 2, i + 1);
    let plusDM = 0,
      minusDM = 0,
      tr = 0;
    for (let j = 1; j < slice.length; j++) {
      const curr = slice[j];
      const prev = slice[j - 1];
      const upMove = curr.high - prev.high;
      const downMove = prev.low - curr.low;
      if (upMove > downMove && upMove > 0) plusDM += upMove;
      if (downMove > upMove && downMove > 0) minusDM += downMove;
      tr += Math.max(
        curr.high - curr.low,
        Math.abs(curr.high - prev.close),
        Math.abs(curr.low - prev.close),
      );
    }
    const plusDI = tr > 0 ? (plusDM / tr) * 100 : 0;
    const minusDI = tr > 0 ? (minusDM / tr) * 100 : 0;
    const dx =
      plusDI + minusDI > 0
        ? (Math.abs(plusDI - minusDI) / (plusDI + minusDI)) * 100
        : 0;
    result.push(dx);
  }
  return result;
};
