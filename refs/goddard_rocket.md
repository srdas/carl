# Maximizing Rocket Altitude With Limited Fuel: The Goddard Rocket Problem.

**Introduction**

A single-stage rocket sits on a launch pad, pointed straight up. It carries a fixed structural (dry) mass plus a load of fuel, and its only job is to burn that fuel as cleverly as possible to reach the greatest altitude it can before gravity and drag win and it starts falling back down. This is a classical problem in the optimal-control literature (Goddard, 1919), where the "trick" is that burning fuel too fast wastes energy fighting atmospheric drag near the ground, while burning too slowly means fighting gravity for longer than necessary; the **unconstrained** optimal policy has three well-known phases: (i) a full-thrust boost, (ii) an intermediate drawdown phase where the thrust dwindles to zero while maximizing energy retention, and (iii) an unpowered coast to the maximum altitude.

Our version of the problem adds one twist that real launch vehicles actually face and that breaks the classical three-phase solution: a hard limit on dynamic pressure ("max-Q"). Real rockets throttle down through the region of peak aerodynamic loading to avoid structural failure; once that constraint is active, the max/drawdown/coast characterization no longer applies, and there is no simple closed-form policy to fall back on. You have to learn a throttle profile that respects the state-dependent limit from first principles, since the limit itself is randomized per rocket (see below).

**Physical setup.**

Let $h_t$ (m), $v_t$ (m/s), and $m_t$ (kg) be the rocket's altitude, vertical velocity, and mass at time $t$. The rocket has a fixed dry mass $m_{\text{dry}} = 1{,}000$ kg (structure plus empty tanks); everything above that is fuel. Gravity is constant, $g = 9.81\text{ m/s}^2$ (the altitudes reached here are small enough that flat-Earth, constant-$g$ physics is a fine approximation). The atmosphere thins exponentially with altitude,
$$\rho(h) = \rho_0 e^{-h/H}, \qquad \rho_0 = 1.225\text{ kg/m}^3,\ \ H = 8{,}500\text{ m},$$
and drag opposes the rocket's motion with magnitude
$$D(h,v) = c_d\,\rho(h)\,v\,|v|, \qquad \text{ where }c_d \text{ covers both aerodynamic and skin friction}.$$
The rocket's engine can produce thrust anywhere from zero up to $T_{\max}$, and burns fuel at a rate proportional to thrust, $\dot m = -T/c_e$, with effective exhaust velocity $c_e = 0.01 \, T_{\max}$ m/s. The engine cannot draw the rocket below its dry mass; once $m_t = m_{\text{dry}}$, thrust is forced to zero (out of fuel) regardless of the commanded throttle. Equations of motion:
$$\dot h = v, \qquad \dot v = \frac{T - D(h,v)}{m} - g, \qquad \dot m = -\frac{T}{c_e}.$$

**Dynamic pressure constraint (max-Q).** Define dynamic pressure as $q(h,v) = \tfrac{1}{2}\rho(h)\,v^2$ (a property of the trajectory's speed/altitude profile, not entangled with the randomized drag coefficient). Each episode imposes a hard ceiling $q_t \le Q_{\max}$ for all $t$, where $Q_{\max}$ is itself a randomized, fully-observed episode parameter (see below).

**Over-Q vibration penalty.** If the agent commands a throttle $u_t$ whose resulting thrust would drive $q_{t+\Delta t} > Q_{\max}$ (i.e., the commanded setting is beyond the max-Q envelope), the airframe vibrates and drag increases by 5% for that step. Concretely, on any step where the commanded thrust is clamped down by the constraint enforcement, the effective drag used in the equations of motion is $1.05\,D(h,v)$ instead of $D(h,v)$. Commanding a setting that would keep $q$ within the ceiling incurs no penalty.

**Action.** The agent's action each step is a throttle setting $u_t \in [0,1]$, giving thrust $T_t = u_t T_{\max}$.

**Observation.** All episodic and time-varying parameters are observable from the perspective of designing the MDP.

**Constraint enforcement.** The environment must enforce $q_t \le Q_{\max}$ at all times.

**Time discretization.** Pick a reasonable simulation step such as $\Delta t = 0.1$ s  to approximate the continuous dynamics and to query the policy.

**Episode termination.** The episode ends the moment the rocket reaches apogee, i.e., the first step where $v_t \le 0$ after having been positive (after that, altitude can only decrease, and the objective only cares about the peak).

**Domain randomization for training.** To learn a policy that generalizes rather than memorizing one rocket, randomize per training episode:
- initial fuel mass,
- max thrust,
- drag constant $c_d$,
- max-Q cap $Q_{\max}$,

with $h_0 = 0$, $v_0 = 0$ always.

## Use only for evaluation:

For testing and evaluation, freeze $T_{\max} = 200{,}000$ N and $c_d = 1.5$, and determine the maximum altitude reached (and the full throttle-vs-time profile that achieves it) for the following three initial fuel loads, each evaluated at three max-Q settings ($Q_{\max} \in \{15{,}000,\ 30{,}000,\ 45{,}000\}$ Pa, spanning tight to loose):

1. A fuel-light rocket: $m_0 = 1{,}500$ kg (500 kg of fuel).
2. The nominal rocket: $m_0 = 3{,}000$ kg (2,000 kg of fuel).
3. A fuel-heavy rocket: $m_0 = 5{,}000$ kg (4,000 kg of fuel).

For each of these nine (fuel load × $Q_{\max}$) combinations, also compare the learned throttle trajectory against the following two baselines:

a. A simple bang-bang heuristic that sticks to $T_{\max}$ until fuel runs out, then coasts to the top,

b. A brute force search policy that finds the approximate optimal solution numerically: it parameterizes the throttle as a piecewise-constant schedule (~30 segments spanning several multiples of the full-throttle burn time), simulates each candidate schedule with the same dynamics integrator the environment uses, and finds the combination that maximizes peak altitude. This schedule is solved separately for each evaluation scenario.

Report altitude lost to the constraint by comparing each case's peak altitude for the three policies.
