# MercuryRec — web

The frontend for MercuryRec. Nine pages over the FastAPI service, including a
3D projection of the two-tower model's item embeddings.

## Running it

The API has to be up first; this application has no data of its own.

```bash
# from the repository root
uv run uvicorn mercury_rec.api.main:app --port 8000

# here
npm install
npm run dev
```

Point it somewhere else with `NEXT_PUBLIC_API_BASE`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev
```

## The rule this codebase is built around

**No number is ever invented.** Every figure on screen comes from the API or
from an evaluation artifact committed in this repository. Where a value is not
available the UI says which command produces it, rather than rendering a zero
that reads as a measurement.

This is enforced structurally rather than by discipline. `MetricTile` takes
`value: number | null` with no default, so omitting it is a type error and
passing `null` renders an em dash with a stated reason. There is no way to
make the component show a number nobody measured.

The same applies to provenance: responses carry a `data_provenance` field, and
`ProvenanceBadge` surfaces it so a reader can tell that behaviour is real
Retailrocket while merchants, regions, prices and verticals are synthesised,
without having to find the data card.

## Pages

| Route | What it shows |
| --- | --- |
| `/` | The funnel, with stage counts read from the running system |
| `/overview` | Live system state and the most recent committed evaluation |
| `/explorer` | One user, one context, the full ranked slate with SHAP attributions |
| `/galaxy` | 6,000 item embeddings in 3D, with the stages of a real request lit up |
| `/pipeline` | One request, stage by stage, with per-stage latency and membership |
| `/models` | Every model under one protocol, accuracy and coverage side by side |
| `/journey` | A session built one item at a time, and the same session across the day |
| `/experiments` | The offline A/B simulation, its bootstrap intervals and the promotion gate |
| `/monitoring` | Live latency and cache telemetry, plus offline feature drift |
| `/architecture` | How the system is arranged and which parts of that are verified |

## Notes on the 3D view

The projection is computed offline by `mercury viz project` and served as
base64 `Float32Array` buffers — roughly 75KB of binary against ~450KB of JSON
text for the same 6,000 points, decoded straight into a `BufferAttribute`.

The scene is a single `<points>` object with one BufferGeometry. Rendering
6,000 meshes is the usual way this goes wrong and costs most of the frame
budget before anything moves. The frame loop is on demand, so with auto-rotate
paused the GPU does nothing at all.

Under reduced motion, on a narrow viewport, or without WebGL, the same
coordinates render as a 2D scatter instead. It is not a placeholder: it keeps
the stage highlighting and drops only the third dimension.

## Checks

```bash
npm run lint         # eslint, zero warnings tolerated
npx tsc --noEmit     # types
npm run build        # production build
```
