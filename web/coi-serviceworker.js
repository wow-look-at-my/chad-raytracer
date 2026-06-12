// Cross-origin isolation shim for static hosts (e.g. GitHub Pages) that
// cannot send COOP/COEP headers. A service worker injects the headers so
// SharedArrayBuffer (and therefore WASM threads) becomes available.
// Loaded both as a window script and as the service worker itself.
if (typeof window === 'undefined') {
  // Service-worker context.
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', (e) => {
    const req = e.request;
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    e.respondWith(
      fetch(req).then((r) => {
        if (r.status === 0) return r;
        const headers = new Headers(r.headers);
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        return new Response(r.body, {
          status: r.status,
          statusText: r.statusText,
          headers,
        });
      })
    );
  });
} else if (!window.crossOriginIsolated && 'serviceWorker' in navigator &&
           window.isSecureContext) {
  const KEY = 'coi-reloaded';
  navigator.serviceWorker
    .register(document.currentScript.src)
    .then((reg) => {
      // Reload once so the page is fetched through the service worker.
      if (!navigator.serviceWorker.controller && !sessionStorage.getItem(KEY)) {
        sessionStorage.setItem(KEY, '1');
        reg.addEventListener('updatefound', () => {});
        const tryReload = () => {
          if (reg.active) location.reload();
          else setTimeout(tryReload, 50);
        };
        tryReload();
      }
    })
    .catch((err) => console.warn('coi-serviceworker registration failed:', err));
}
