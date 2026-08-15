# EgoSelect

> **Capability-aware training-value curation for EgoVerse.**  
> Select the most informative demonstration episodes to maximize coverage and quality while minimizing redundancy under compute budgets.

[![Demo Video](https://img.shields.io/badge/YouTube-Watch%20Demo-red?logo=youtube)](https://www.youtube.com/watch?v=iOxlZ7C8pBA)
[![Project Slides](https://img.shields.io/badge/Gamma-Project%20Slides-blueviolet)](https://gamma.app/docs/Keep-30-of-the-data-Keep-the-region-coverage-qodfuf4zzznz8q7?mode=doc)


[![EgoSelect Demo](assets/demo.webp)](https://www.youtube.com/watch?v=iOxlZ7C8pBA)


---

## 💡 Overview & Problem

**Which demonstrations are actually worth spending training compute on?**

EgoVerse datasets contain rich multimodal motion but are often polluted with near-duplicate runs and idle-heavy sequences. **EgoSelect** ranks and filters episodes based on their **marginal training value** relative to already-selected data.

> **Note:** This framework focuses strictly on data curation and representation coverage. It does not train policies directly and makes no downstream policy performance claims.

---

## ⚙️ Methodology

EgoSelect runs an iterative greedy selection process over multimodal episode embeddings $z_i$ (combining DINOv2 visual PCA and standardized motion features). 

After each episode is selected into active subset $\mathcal{S}$, all remaining candidate episodes are rescored against $\mathcal{S}$. *No SQL ground-truth labels are used during scoring.*

### Objective Function

$$\text{Value}(i \mid \mathcal{S}) = 0.35 \cdot \text{Quality}(i) + 0.45 \cdot \text{CoverageGain}(i \mid \mathcal{S}) - 0.20 \cdot \text{Redundancy}(i \mid \mathcal{S})$$

| Component | Description |
| :--- | :--- |
| **Quality** | Frame validity, completeness, finite pose checks, non-stationary movement ratio, and temporal consistency. |
| **CoverageGain** | Discovery of new feature regions, representation distance to existing subset $\mathcal{S}$, and underrepresented-region bonuses. |
| **Redundancy** | Maximum cosine similarity between candidate $i$ and any item in selected subset $\mathcal{S}$. |

---

## 🏗️ Architecture Pipeline

```text
EgoVerse (SQL / S3)
       │
       ▼
Local Zarr Cache
       │
       ▼
Feature Extraction (DINOv2 + Motion Dynamics + Quality Checks)
       │
       ▼
Embedding Space z_i (2D PCA + Representation Regions)
       │
       ▼
Greedy EgoSelect Ranking (Dynamic Rescoring Loop)
       │
       ▼
Benchmark Evaluation (Equal-budget baselines + Synthetic feature corruption)
       │
       ▼
Payload Export (demo_payload.json)
       │
       ▼
Static Visualizer (React + Vite + SVG Canvas / Zero-API)
