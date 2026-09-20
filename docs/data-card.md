# Data Card — MercuryRec

Every number on this page was produced by running `mercury data build --preset full`
in this repository. Nothing is estimated, rounded for effect, or carried over from
another source. Re-running the command reproduces them exactly (`seed: 42`).

> **The dataset is hybrid.** Behaviour is real; the platform attributes around it
> are synthesised. The two are separated explicitly below, because presenting
> generated attributes as observed data would be dishonest and would make the
> evaluation meaningless.

---

## 1. Source

| | |
|---|---|
| Dataset | [Retailrocket ecommerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| Publisher | Retailrocket, via Kaggle |
| License | **CC BY-NC-SA 4.0** (Attribution · NonCommercial · ShareAlike) |
| Size | ~290 MB compressed, 852 MB extracted |
| Collected | 3 May 2015 – 18 September 2015 (4.5 months) |
| Redistributed here? | **No** — see below |

### Licensing consequences

CC BY-NC-SA 4.0 is a **NonCommercial + ShareAlike** license, and ShareAlike
extends to derived works. Three concrete consequences, all enforced in the repo
rather than merely noted:

1. **Neither raw nor processed data is committed.** `/data/` is git-ignored and
   the repository ships `mercury data download`, which fetches from the original
   source. Derived parquet is also excluded, because it is a derivative work and
   carries the same terms.
2. **Repository code is MIT.** Code is not a derivative of the data, so the two
   licenses coexist. This is stated in [`LICENSE`](../LICENSE).
3. **Non-commercial use only** for the data and anything derived from it. This
   project is a portfolio and engineering demonstration, which falls inside that
   term; anyone reusing it commercially would need different data.

Attribution: *Retailrocket ecommerce dataset*, published by Retailrocket under
CC BY-NC-SA 4.0.

---

## 2. What is real and what is synthesised

This is the most important section on the page.

### Real — taken directly from Retailrocket

| Field | Notes |
|---|---|
| `user_id` | Retailrocket `visitorid`, densely re-indexed |
| `item_id` | Retailrocket `itemid`, densely re-indexed |
| `ts` | Real event timestamps, epoch **milliseconds**, UTC |
| `event_type` | Real `view` / `addtocart` / `transaction` |
| `category_id` | Real category assignment from `item_properties` |
| `available_from` | Derived from the real `available` property |
| `session_id` | Derived from real timestamps by inactivity gap |
| `is_repeat` | Derived from the real purchase sequence |

**All behaviour is real.** Every interaction, and when it happened, comes from the
observed log. No events were generated, resampled, or up-weighted.

### Synthesised — generated because Retailrocket does not contain them

| Field | Why it had to be generated | How |
|---|---|---|
| `merchant_id` | Dataset has no merchant concept | Items assigned to a merchant **within their own vertical** |
| `vertical` | Dataset has no food/grocery/pharmacy structure | Root categories assigned greedily, balanced by item count |
| `region_id` | No geography at all | Zipf-distributed populations (s = 1.1) over 12 regions |
| `price`, `price_at_event` | Real price property is **hashed and unrecoverable** | Per-vertical log-normal |
| `base_rating` | No ratings | Beta(7, 2) scaled to [0, 5] |
| `device` | No device information | Weighted categorical, mobile-app dominant |
| `delivery_radius_km`, `avg_delivery_minutes` | No delivery model | Uniform over configured ranges |

Two properties make this reproducible rather than arbitrary:

- **Deterministic by id.** Every synthesised attribute is a hash of
  `(entity id, seed, per-attribute salt)`, not a sequential RNG draw. An item
  gets the same price on every machine, in every run, and in both presets — so
  `demo` is a genuine subset of `full`, not a differently-generated dataset.
- **Independent draws.** Distinct salts per attribute mean region and device are
  statistically independent; a shared salt would invent a correlation that the
  data never contained. There is a test asserting |ρ| < 0.05.

The prices are **plausible magnitudes, not recovered values**. Nothing in this
repository claims otherwise, and no price-based metric should be read as a
statement about real Retailrocket prices.

---

## 3. Scale, as measured

### Raw, as published

| | Count |
|---|---:|
| Events | 2,756,101 |
| Visitors | 1,407,580 |
| Items | 235,061 |
| Categories | 1,669 (25 root) |

Event mix: 2,664,312 views · 69,332 add-to-cart · 22,457 transactions.

### After the pipeline (`full` preset)

| Stage | Events | Users | Items |
|---|---:|---:|---:|
| Raw | 2,756,101 | 1,407,580 | 235,061 |
| After deduplication | 2,752,009 | — | — |
| **After k-core (k = 5)** | **777,981** | **65,684** | **36,044** |

**28.2% of events retained**, converged after **11 iterations**.

Final event mix: 717,749 views · 42,553 add-to-cart · 16,521 purchases ·
1,158 repeat purchases. Also produced: 720 merchants, 223,820 sessions,
1,524 items with no resolvable category.

### Why k-core filtering, and what it costs

Retailrocket is extremely sparse: **2.76M events across 1.41M visitors is a mean
of under 2 events per visitor**, and only ~11k visitors ever transacted.
Collaborative and neural retrieval models cannot learn a representation from a
single observation — most users would contribute nothing but noise.

So the interaction graph is reduced to its 5-core: every remaining user has ≥5
interactions and every remaining item has ≥5. It is applied iteratively, because
removing a sparse user can push an item below threshold and orphan another user
in turn; a single pass does not satisfy either constraint.

**This changes what the reported metrics describe.** They describe the active
core of the catalogue, not the full long tail. That is the standard protocol in
the recommendation literature, and it is stated here rather than buried.

---

## 4. Temporal split

Boundaries are global timestamps at event quantiles — strictly chronological, so
no model ever trains on an event that occurred after one it is scored on.

```
|------------- TRAIN 70% -------------|--- VAL 15% ---|--- TEST 15% ---|
                                 2015-07-29       2015-08-22
```

| | At the boundary | After `require_train_history` |
|---|---:|---:|
| Train | 70.0% | 544,587 events (90.2%) |
| Validation | 15.0% | 36,865 events (6.1%) |
| Test | 15.0% | 22,007 events (3.6%) |

**Both figures are reported because they answer different questions**, and quoting
only one would misdescribe the split. The chronological boundaries genuinely sit
at 70/15/15. But evaluation events belonging to users with no training history are
then dropped — such users cannot be personalised for, so scoring them in the main
comparison would measure the cold-start fallback while appearing to measure the
personalised models. That filtering is what shrinks the later windows to 90/6/4.

Users: **48,526** in train, **5,276** in validation, **3,552** in test.

> **Limitation worth stating plainly:** only 3,552 of 65,684 users appear in the
> test window. Most Retailrocket visitors never return, so the personalised
> evaluation rests on a much smaller population than the headline user count
> suggests. Confidence intervals on test metrics will be correspondingly wide.

---

## 5. Known limitations

1. **No impression log.** Retailrocket records only views, add-to-carts and
   transactions — not what was *shown*. True CTR is therefore not computable;
   CTR-style features are ratios over observed views, which is a different
   quantity and is named as such in the feature definitions.
2. **Opaque categories.** Category ids are hashed tokens with no labels. The
   vertical mapping imposes a consistent structure; it does not claim to know
   what any category contains.
3. **Unrecoverable prices.** The real price property is hashed, so prices are
   synthesised.
4. **No true geography.** Regions are synthetic labels, not real locations;
   "delivery distance" is a modelled quantity.
5. **Vertical shares deviate by a few points.** Root categories are indivisible,
   and the largest root holds ~18% of items — above the smallest vertical target
   of 10%. Exact targets are therefore arithmetically unreachable. Measured
   deviation: **max 2.6%**.
6. **Single current category per item.** Items can be recategorised over time;
   the pipeline takes the most recent category rather than doing an as-of lookup,
   because category is consumed as a static attribute.
7. **Dataset is from 2015** and reflects one retailer's traffic. It is not
   representative of on-demand commerce generally.

---

## 6. Reproducing this

```bash
uv run mercury data download          # fetches from Kaggle (needs API credentials)
uv run mercury data build --preset full
```

Requires a Kaggle API token at `~/.kaggle/kaggle.json`
([create one here](https://www.kaggle.com/settings)).

Outputs land in `data/processed/full/`, alongside a `METADATA.json` recording row
counts at every stage, split boundaries, config hashes and a dataset hash. Every
MLflow run records that dataset hash, so any metric can be traced back to the
exact data that produced it.

| Build | Value |
|---|---|
| Dataset hash | `c38478b5cb894be4` |
| `configs/data.yaml` hash | `55a5ea5bbe2c` |
| `configs/events.yaml` hash | `aeeea6e52449` |
| Seed | `42` |
| Build time | ~16 s (full), ~14 s (demo) |

A `demo` preset caps the build at 5,000 users / 3,000 items for fast iteration:
131,331 events, 4,644 users, 3,000 items.
