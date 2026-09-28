---
name: 01-problem-definition
description: Understand the user's problem through conversational exploration
metadata:
  sources:
    - rl-problem-workflow (Stage 1 - problem understanding)
---

# Step 1: Problem Definition

Tell me about what you want to solve. Use multiple turns and ask for clarifications if required.

## Goal

Build a thorough understanding of the user's problem before any technical RL design begins. This step is purely about understanding the domain, objectives, dynamics, and constraints — not about defining states, actions, or rewards (later steps handle those).

## What to Explore

Through conversation, clarify the following:

### The Objective
- What is the user trying to optimize or achieve?
- Is there a single objective or multiple competing objectives?
- Is there a known analytical or closed-form solution for any part of the problem?

### The Dynamics
- What system is being controlled? What are its moving parts?
- What changes over time? What does the agent observe?
- What decisions does the agent make, and how often?
- Are decisions continuous or discrete in nature?

### The Constraints
- Are there hard constraints (must never violate) vs soft constraints (penalized)?
- Are there physical, regulatory, or domain-specific limits?
- What constitutes failure or an invalid outcome?

### The Context
- What domain is this in (finance, robotics, operations, etc.)?
- What scale are the quantities involved (magnitudes in single digits, thousands, millions)?
- Is there existing data, a simulator, or analytical models?
- Has the user tried other approaches?

### The Episode
- What defines the start and end of one decision-making episode?
- Is the horizon fixed or variable?
- What happens at the end of an episode?

## Conversational Guidelines

- Ask one or two focused questions at a time, not a long list
- Summarize understanding back to the user periodically
- Flag ambiguities or potential issues early
- If the user's description suggests multiple sub-problems, ask whether they want to tackle them together or separately
- Keep it simple — prefer the simplest framing that captures the problem

## Problem Definition Checklist

- [ ] Core objective clearly understood
- [ ] System dynamics described (what changes, what's observed, what's controlled)
- [ ] Decision type and frequency identified (continuous/discrete, per-step/periodic)
- [ ] Constraints and failure modes catalogued
- [ ] Domain context and scale of quantities understood
- [ ] Episode structure (start, end, horizon) understood
- [ ] Existing solutions or analytical benchmarks identified (if any)
- [ ] User's prior attempts or preferences noted
- [ ] Reference files mentioned by the user listed by path in the summary

## /suggest Feature

If the user types `/suggest` anywhere in their first message (e.g., "I want to look at
portfolio selection for goals-based wealth management, /suggest"), CARL will:
1. Search for relevant academic papers via Semantic Scholar
2. Synthesize 3–5 candidate RL problem formulations from those papers
3. Let the user choose one
4. Auto-generate and save a detailed problem specification to `refs/<slug>.md`

The generated file will be injected into this step as context. When you receive a message
containing content from a `carl-suggest`-sourced file (identified by `source: carl-suggest`
in its front matter), acknowledge the chosen formulation explicitly, then verify it covers
all items in the **Problem Definition Checklist** above. Only ask follow-up questions for
checklist items that remain genuinely ambiguous or missing.
