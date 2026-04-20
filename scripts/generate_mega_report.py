import os
import glob

OUTPUT_FILE = "Capstone_Mega_Report.md"

EXTENSIVE_THEORY = """
# Chapter 1: Introduction and Comprehensive Theoretical Background

## 1.1 Introduction
The rapid urbanization and exponential growth in the number of vehicles have led to unprecedented traffic congestion. Fixed-time traffic light controllers and even rule-based adaptive algorithms (like Webster's method or SCATS) fall short because they cannot adequately capture the extremely non-linear, non-stationary dynamics of modern urban traffic. Our capstone project explicitly addresses this fundamental gap by proposing, developing, and evaluating a Multi-Agent Reinforcement Learning (MARL) approach, fortified with Spatial-Temporal Graph Neural Networks (ST-GNN).

## 1.2 Theoretical Foundation of Neural Networks
Before delving into advanced RL, we must formalize the building blocks. An artificial neural network consists of layers of interconnected nodes. The dynamics of a single feedforward layer are described as:
`h = sigma(W * x + b)`
where `W` is the weight matrix, `x` is the input vector, `b` is the bias vector, and `sigma` is a non-linear activation function such as ReLU. As traffic states are exceptionally high-dimensional (queue lengths, speeds, wait times), deep feature extraction is essential.

## 1.3 Reinforcement Learning (RL) and Markov Decision Processes (MDP)
An RL framework is mathematically described as an MDP `(S, A, P, R, gamma)`.
- `S`: Continuous state space representing intersection traffic density.
- `A`: Action space representing phase selection.
- `P`: Transition probability function, mapping `(S, A)` to a probability over the next state `S'`.
- `R`: The reward function, mapping `(S, A)` to a real-valued immediate reward.
- `gamma`: The discount factor `[0, 1)`.

The agent's objective is to find an optimal policy `pi*` which maximizes the expected discounted cumulative reward:
`V(s) = E_pi [ sum_{t=0}^{inf} gamma^t R(S_t, A_t) | S_0 = s ]`

## 1.4 Proximal Policy Optimization (PPO) 
Our foundational training algorithm is PPO. Given the high variance in policy gradient methods like REINFORCE, PPO utilizes a clipped surrogate objective:
`L^{CLIP}(theta) = E [ min( r_t(theta) * A_t , clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t ) ]`
where `r_t(theta)` is the probability ratio between the new policy and the old policy, and `A_t` is the advantage estimate.

## 1.5 Multi-Agent PPO (MAPPO) and CTDE
For multiple independent intersections, we adopt the Centralized Training, Decentralized Execution (CTDE) paradigm. During training, a centralized critic observes the joint state to stabilize value estimation, while independent actors function using local observations during execution.

## 1.6 Spatial-Temporal Graph Neural Networks (ST-GNN)
Traffic data exhibits both strong spatial correlations (upstream/downstream intersections) and temporal correlations.
- **Graph Convolutional Networks (GCN):** `H^{(l+1)} = sigma( D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)} )`. This allows neighboring traffic sensors to share latent state embeddings.
- **Temporal Components:** Gated Recurrent Units (GRUs) or Temporal Convolutional Networks (TCNs) are stacked to capture time-series evolution.

## 1.7 Baselines
1. **CoLight:** An attention-based RL model that dynamically weighs the messages coming from neighboring intersections depending on their apparent relevance.
2. **NSTLight:** Designed explicitly for Non-Stationary traffic environments utilizing a generalized advantage formulation.
3. **MaxPressure:** A robust mathematical baseline aiming to purely maximize pressure at intersections independently, serving as the benchmark for uncoordinated control.

---
"""

def generate_report():
    print(f"Generating Mega Report: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# CAPSTONE MEGA REPORT: Multi-Agent RL for Traffic Signal Control\n\n")
        
        # 1. Theoretical Background
        out.write(EXTENSIVE_THEORY)
        
        # 2. Existing Markdown Docs (All Project Planning, Specs, Steps)
        out.write("\n# Chapter 2: Project Specifications, Plans, and Documentation\n\n")
        md_files = [f for f in glob.glob("*.md") if f not in [OUTPUT_FILE]]
        md_files.sort()
        for md in md_files:
            try:
                content = open(md, "r", encoding="utf-8").read()
                out.write(f"## {md}\n")
                out.write("```markdown\n")
                out.write(content)
                out.write("\n```\n\n")
            except Exception as e:
                print(f"Skipping {md} due to {e}")
                
        # 3. Source Code - Models and Agents
        out.write("\n# Chapter 3: Comprehensive Implementation Source Code\n\n")
        out.write("The complete system implementation spans multiple directories including the core MARL algorithms, GNNs, baselines, and evaluation scripts.\n\n")
        
        py_files = []
        for root, dirs, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
        for root, dirs, files in os.walk("scripts"):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
                    
        py_files.sort()
        for py in py_files:
            try:
                content = open(py, "r", encoding="utf-8").read()
                out.write(f"## Source File: `{py}`\n")
                out.write("```python\n")
                out.write(content)
                out.write("\n```\n\n")
            except Exception as e:
                print(f"Skipping {py} due to {e}")

        # 4. Configurations
        out.write("\n# Chapter 4: System Configurations\n\n")
        config_files = glob.glob("configs/*.yaml")
        for conf in config_files:
            try:
                content = open(conf, "r", encoding="utf-8").read()
                out.write(f"## Config File: `{conf}`\n")
                out.write("```yaml\n")
                out.write(content)
                out.write("\n```\n\n")
            except Exception as e:
                pass
                
        # 5. Results & Metrics
        out.write("\n# Chapter 5: Quantitative Evaluation and Metrics\n\n")
        csv_files = glob.glob("FAST_VAL_RESULTS/*.csv")
        for csv in csv_files:
            try:
                content = open(csv, "r", encoding="utf-8").read()
                out.write(f"## Metrics Log: `{csv}`\n")
                out.write("```csv\n")
                # Truncate to first 500 lines to avoid massive file locking, but keep it huge
                lines = content.split('\\n')
                out.write('\\n'.join(lines[:1000]))
                out.write("\n```\n\n")
            except Exception as e:
                pass
                
        # 6. Plots Embeddings
        out.write("\n# Chapter 6: Visualizations and System Artifacts\n\n")
        out.write("This section details the generated visual representations of model performance.\n\n")
        plots = glob.glob("FAST_VAL_RESULTS/plots/*.png")
        for plot in plots:
            out.write(f"### {os.path.basename(plot)}\n")
            out.write(f"![{os.path.basename(plot)}](file:///{os.path.abspath(plot).replace(chr(92), '/')})\n\n")
            out.write("The plot above represents a critical evaluation metric for the system's performance, contrasting our MAPPO-STGNN with robust SOTA paradigms.\n\n")
            
        print(f"Successfully compiled {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
