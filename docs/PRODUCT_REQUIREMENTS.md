# Product Requirements: First Deployment Wedge

## Product Positioning

This project should initially be positioned as an advisory and shadow-mode traffic operations platform, not an autonomous traffic signal controller.

The first product wedge is corridor signal intelligence for a small managed road network where the operator needs better visibility, timing recommendations, and incident response support without replacing the existing controller stack.

## Initial Customer Profile

Primary target:
- University campus, industrial park, private township, airport landside road network, or municipal corridor with 5-20 signalized intersections.

Buyer or sponsor:
- Transportation operations manager, campus facilities director, smart-city program lead, or municipal traffic engineer.

Operator:
- Traffic engineer or operations staff responsible for signal timing, congestion monitoring, and incident response.

Why this customer first:
- Lower deployment friction than citywide public-road actuation.
- Easier access to field observations and manual timing plans.
- Advisory mode can create value before direct hardware control.
- Real-world operations matter, but regulatory and procurement burden is lower than broad municipal deployment.

## Problem Statement

Operators often lack continuous, trustworthy visibility into intersection performance. Retiming is manual, expensive, and usually based on periodic studies rather than persistent measurement. During demand shifts, incidents, events, and sensor faults, fixed timing plans can become stale before anyone acts.

The product should help operators answer:
- Where are queues building?
- Which intersections are degrading?
- Which timing plans or splits appear mismatched to current demand?
- What recommendation is the system making, and why?
- Is the recommendation safe under signal timing constraints?
- Did the recommendation improve delay, queue length, stops, throughput, or reliability?

## Initial Product Mode

Phase 1: Offline analysis
- Ingest historical traffic, detector, and timing data.
- Generate performance reports and candidate timing recommendations.

Phase 2: Shadow mode
- Consume live or near-live data.
- Generate recommendations without applying them.
- Compare recommendations against observed operations.

Phase 3: Advisory mode
- Present recommendations to an operator for approval.
- Export suggested timing changes or action plans.

Out of scope for the first customer:
- Fully autonomous field actuation.
- Emergency vehicle preemption control.
- Pedestrian safety-critical decision automation.
- Unsupervised reinforcement learning on live roads.
- Citywide optimization.

## Core User Workflows

1. Import or connect corridor data.
2. Review current intersection and corridor health.
3. Inspect queue, delay, throughput, stop, and travel-time trends.
4. Review recommendations with reasons and safety validation status.
5. Export a report for before/after review.
6. Run a shadow-mode comparison over a fixed pilot window.

## Required Capabilities

Data:
- CSV importer for historical detector counts, signal phase timing, and travel-time observations.
- Scenario metadata for intersections, approaches, lanes, signal phases, and timing plans.
- Provenance tracking for every dataset and derived metric.

Simulation:
- SUMO scenario calibration notes.
- Seeded benchmark harness with fixed-time, random, MaxPressure, and learned controllers.
- Confidence intervals across seeds before any performance claim.

Recommendation:
- Explicit action proposal format.
- Safety validator that rejects illegal or incomplete signal timing changes.
- Human-readable reason for each recommendation.

Dashboard:
- Corridor status overview.
- Intersection drilldown.
- Recommendation queue.
- Evidence status labels.
- Exportable pilot report.

Reliability:
- Structured logs for every recommendation.
- Fallback mode definition.
- Watchdog behavior for hardware-in-the-loop testing.

## Success Metrics

Operational:
- Mean delay.
- 95th percentile queue length.
- Stops per vehicle.
- Corridor travel time and reliability.
- Throughput.
- Operator intervention count.
- Recommendation acceptance rate.

System:
- Shadow-mode uptime.
- Data ingestion latency.
- Recommendation generation latency.
- Percentage of rejected recommendations and rejection reasons.
- Recovery time after process, network, or sensor failure.

Commercial:
- Operator saves time identifying poor timing periods.
- Pilot identifies measurable improvement opportunities.
- Buyer agrees the system can run in shadow or advisory mode for a longer pilot.

## Non-Negotiable Safety Requirements

- The model must never directly command a field signal without a rule-based validator.
- Minimum green, yellow, all-red, pedestrian, coordination, and max-cycle constraints must be represented explicitly.
- Every recommendation must have an audit record.
- The system must fail closed into advisory/no-action mode.
- Existing controller timing must remain the fallback.

## First Pilot Exit Criteria

A pilot is successful only if all of the following are true:
- The system runs in shadow or advisory mode for the agreed pilot window.
- Data provenance is complete enough to audit results.
- Recommendations are explainable to the operator.
- No safety rule violations pass validation.
- A before/after or opportunity report is produced with honest limitations.
- The customer wants either a longer shadow pilot or a supervised advisory deployment.
