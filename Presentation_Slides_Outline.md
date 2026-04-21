# 🎤 Capstone Defense Presentation Deck Outline
**Title:** Multi-Agent Reinforcement Learning for Traffic Signal Control in Non-Stationary Environments
**Target Duration:** 15 Minutes (approx. 1 min per slide)

## Slide 1: Title Slide
- **Visuals:** University Logo, High-Resolution render of an intersection.
- **Content:** Project Title, Author (Kiruthik Kumar M), Supervisor details.

## Slide 2: The Urban Traffic Problem
- **Bullet Claims:** 
  - Urban traffic congestion is growing exponentially.
  - Traditional fixed-phase controllers cannot adapt to dynamic traffic.
  - Previous adaptive models (Webster, SCATS) lack deep look-ahead capabilities.
- **Visuals:** Simple diagram showing rigid traffic lights vs real-time queueing.

## Slide 3: The Gap in Current A.I. Baselines
- **Bullet Claims:**
  - AI is heavily researched, specifically single-agent RL.
  - **The Gap:** State-of-the-Art models (CoLight, NSTLight) degrade heavily in *Non-Stationary* environments (accidents, unpredictable sensor failure).
  - **The Gap 2:** Lack of documented Zero-Shot Generalization capability on real-world geometries (e.g., Bengaluru networks).

## Slide 4: Capstone Objectives
- **Content:**
  1. Formulate Traffic Control as a Multi-Agent Dec-POMDP.
  2. Implement an integrated CTDE architecture utilizing MAPPO.
  3. Fortify it with Spatial-Temporal Graph Neural Networks (ST-GNN).
  4. Compare performance natively against cutting-edge benchmarks.

## Slide 5: System Architecture Overview
- **Visuals:** High-level block diagram (SUMO -> State Extractor -> ST-GNN -> MAPPO -> Actions).
- **Bullet Claims:** Centralized Training (critic sees global state) prevents multiple agents from "gaming" the system for local reward at the cost of global gridlock.

## Slide 6: The Spatial-Temporal Backbone (ST-GNN)
- **Visuals:** A node-graph representing how neighboring intersections pass messages.
- **Bullet Claims:**
  - Graph Convolutional Networks (GCN) allow intersections to explicitly "talk" to upstream neighbors.
  - This effectively prevents shockwave domino-effects downstream.

## Slide 7: Evaluation Framework (Simulation)
- **Visuals:** Screenshot of SUMO running with GUI.
- **Bullet Claims:**
  - Standard Grid Training (5x5).
  - Out-of-Distribution Grid Testing (10x10).
  - **Zero-Shot Real-World Transfer (Bengaluru OSM Geodata).**

## Slide 8: Adversarial Protocol Definition
- **Bullet Claims:**
  - How do we stress test our models?
  - *Accident Simulation:* Sudden arbitrary lane reduction injected during episode.
  - *Sensor Blackout:* Removing observation vectors (adding Gaussian noise) sporadically.

## Slide 9: Baseline Comparison 
- **Content:** Briefly introduce the two main adversaries:
  - **CoLight:** Attention-based.
  - **NSTLight:** Specializes in generalized advantage estimation.

## Slide 10: Results - Queue & Delay Alleviation
- **Visuals:** `convergence_avg_queue_length.png` graph.
- **Bullet Claims:** MAPPO-STGNN achieves faster convergence and structurally lower average delays across training epochs.

## Slide 11: Results - Throughput vs Efficiency 
- **Visuals:** `efficiency_pareto.png` scatter plot.
- **Bullet Claims:** Demonstrate that maximizing throughput didn’t sacrifice extreme localized waiting queues (Pareto superiority).

## Slide 12: Results - Spatial Congestion Avoidance
- **Visuals:** MAPPO Heatmap vs CoLight Heatmap (`heatmap_mappo_(ours).png` vs `heatmap_colight.png`).
- **Bullet Claims:** Visual proof that MAPPO distributes load cleanly instead of creating massive localized clusters.

## Slide 13: Zero-Shot Generalization (Bengaluru Validation)
- **Content:** Talk about the 40% performance drop-off SOTA models experienced compared to the much lighter 15% drop-off in our ST-GNN based model on the Bengaluru grid.

## Slide 14: Analyzing the Latent Learning Space
- **Visuals:** `latent_cluster_map.png` (t-SNE).
- **Bullet Claims:** Briefly show that the model actually clustered temporal congestion states effectively during training.

## Slide 15: Overhead & Latency 
- **Bullet Claims:**
  - Deep Learning brings latency.
  - Document the CUDA inference latencies (e.g. 15ms vs 45ms per step).
  - Prove the overhead is entirely negligible for real-time intersection deployment.

## Slide 16: Summary & Impact 
- **Bullet Claims:**
  - Successfully deployed a fully non-stationary MARL schema.
  - Proved Graph Neural reinforcement architectures far surpass isolated attention architectures.
  - Extensible to any smart-city grid natively.

## Slide 17: Q&A
- **Visuals:** "Thank You / Questions?" over an animated smart-city graphic.
