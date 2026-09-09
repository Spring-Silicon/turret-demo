"use strict";

// Release the native boot cover only after the local UI has painted. Camera and
// model readiness is independent: each panel initially shows its own spinner.
window.springDemoReady = async () => {
  const token = new URLSearchParams(window.location.search).get("kiosk");
  if (!/^[a-f0-9]{32}$/.test(token || "")) return;
  if (document.fonts) await document.fonts.ready;
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const response = await fetch(`/kiosk-ready/${token}`, {method: "POST", signal: AbortSignal.timeout(2000)});
      if (response.ok) return;
    } catch (_) { /* Local frontend may still be restarting. */ }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
};
