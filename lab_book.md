# Lab Book - `innovate` Library Development

## 2025-07-11

### Goal: Implement the Norton-Bass Model

Starting work on Phase 4 of the roadmap. The first task is to implement the Norton-Bass model for technological substitution.

### Task 1: `NortonBassModel` Class and `differential_equation` Method

**1. Research and Formulation:**
The Norton-Bass model has several formulations in the literature. After reviewing a few sources, I've decided to implement the system of differential equations as described in the documentation for the `Diffusion` package in R, which is a respected implementation. This formulation captures the cannibalization of the market of earlier generations by later ones.

For a system with `k` generations, where `y_i` is the cumulative adoption of generation `i`, the system of ODEs is:
- `dy_i/dt = (p_i + q_i * y_i / m_i) * (m_i - y_i - sum_{j=i+1 to k} y_j)` for `i = 1 to k-1`
- `dy_k/dt = (p_k + q_k * y_k / m_k) * (m_k - y_k)`

This seems robust and implementable.

**2. Implementation Plan:**
- The `differential_equation` method will take the state vector `y` (a list of cumulative adoptions for each generation) and the parameter vector `p` (a flattened list of `[p1, q1, m1, p2, q2, m2, ...]`).
- It will construct the system of ODEs based on the formulation above and return a list of the derivatives `[dy_1/dt, dy_2/dt, ...]`.
- I will update the `norton_bass.py` file with this implementation.
