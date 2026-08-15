export type Episode = {
  id: string;
  x: number;
  y: number;
  region: number;
  rank: number;
  quality: number;
  coverage_gain: number;
  redundancy: number;
  value: number;
  nearest: string;
  nearest_similarity: number | null;
  stationary: number;
  reason: string;
  task: string;
  lab: string;
  role: "keep_example" | "drop_example" | null;
};

export type CurvePoint = {
  k: number;
  fraction: number;
  coverage: number;
  quality: number;
  redundancy: number;
  visual_coverage: number;
  stationary: number;
};

export type MethodBlock = {
  name: string;
  coverage: number;
  quality: number;
  redundancy: number;
  visual_coverage: number;
  stationary: number;
  corrupt_retained?: number;
  corrupt_pool?: number;
};

export type Payload = {
  meta: {
    title: string;
    subtitle: string;
    n_episodes: number;
    n_regions: number;
    primary_budget: number;
    primary_keep: number;
    formula: string;
    weights: { alpha: number; beta: number; gamma: number };
    greedy_recomputed: boolean;
    keep_example: string;
    drop_example: string;
    keep_reason: string;
    drop_reason: string;
    headline: string;
    tasks: string[];
    labs: string[];
    n_rescore_passes: number;
  };
  episodes: Episode[];
  retention_curve: CurvePoint[];
  benchmark: {
    budget: number;
    n_keep: number;
    methods: MethodBlock[];
  };
  stress: {
    inject_rate: number;
    n_injected: number;
    n_keep: number;
    methods: MethodBlock[];
  };
};
