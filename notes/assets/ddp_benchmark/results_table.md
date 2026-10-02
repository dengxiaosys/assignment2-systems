| Strategy | Calls/step | Step mean +/- repeat std (ms) | Sync/tail mean +/- repeat std (ms) | Sync/tail share | Speedup vs naive |
|---|---:|---:|---:|---:|---:|
| `naive` | 21 | 20.235 +/- 0.727 | 12.741 +/- 0.383 | 63.0% | 1.00x |
| `flat` | 1 | 8.509 +/- 0.535 | 1.660 +/- 0.271 | 19.4% | 2.38x |
| `overlap` | 21 | 15.376 +/- 0.887 | 6.405 +/- 0.482 | 41.6% | 1.32x |

> `naive` and `flat` report complete post-backward gradient synchronization. `overlap` reports only the exposed post-backward tail wait.
