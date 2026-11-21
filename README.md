# GitHub README

## Escape from Phobos | Autonomous spacecraft guidance and control

Texas A&M SEDS developed a Python flight controller for *Escape from Phobos*, the 2026 Space Teams University competition. The simulated shuttle launches from Phobos, travels roughly 1.3 km through moving debris, and completes the mission when it comes within 50 m of a target spacecraft. **Our submitted score was 57.8.** [Flight demonstration](https://www.youtube.com/watch?v=huERMtVemqs) · [Submitted controller](https://github.com/JacobGainesMXET/TAMU-SEDS-Escape-from-Phobos-Competition-Submission-NASA-Texas-Space-Grant/blob/main/STU4-Solution-TAMU-SEDS/Controller.py)

Our first shuttle made it to the target without a collision. It also burned **129.14 kg** of propellant and scored **−62.9**. Watching the flight revealed a problem the success message missed. The hull spent much of the ascent turning toward the target, then firing thrusters to arrest that rotation. We redesigned the departure around the shuttle's bottom thrusters, predicted where its velocity would carry it under gravity, and gave the radar planner authority to interrupt a burn when a rock threatened that trajectory.

### How the controller flies

| Flight function | What the software does |
| --- | --- |
| Liftoff and attitude control | Builds closing speed with the bottom thrusters while holding the hull near its launch attitude. The nominal tilt command is limited to 7.5°. Angular-rate feedback damps the turn as it approaches the commanded orientation. |
| Navigation and estimation | Reads shuttle and target state in the Phobos body-fixed frame. Combines timestamped radar returns, shuttle motion, and a six-state Kalman filter for each tracked rock to estimate its 3D position and velocity. |
| Guidance | Predicts whether a gravity-curved coast will enter the 50 m arrival sphere. Compares that coast with a short additional burn and uses the measured fuel cost and predicted arrival time to choose between them. |
| Collision avoidance | Predicts closest approach on the intended path. A longer-range planner can shift burn timing or the aim point. A faster safety check can interrupt the current command if a nearby crossing develops. |
| Actuation | Converts the selected acceleration to shuttle-body coordinates and sends linear and angular acceleration requests through the competition API. The simulator's jet allocator selects the individual thrusters. |

### Finding the right time to cut thrust

For a successful, undamaged flight, the score is `50 + 12000/T − (100/75)F`, where `T` is total flight time in seconds and `F` is fuel consumed in kilograms. At a 200 s arrival, another kilogram must save roughly **4.4 s** to break even. Early thrust can earn that time back across a long coast. A burn seconds before arrival has far less opportunity to pay for itself. [Competition scoring rules](https://github.com/SimDynamX/STU4-Solution/wiki/2.-mission-success,-failure,-and-scoring)

We tested fixed cruise speeds and faster profiles with a fuel budget. The submitted controller compares two forecasts. It predicts arrival if the shuttle cuts thrust now, then repeats the prediction with two more seconds of burn. It learns the fuel cost per unit of velocity gained from measured burns with little rotation. Once the predicted coast passes far enough inside the arrival sphere, small changes in the forecast can leave the engines off. A predicted miss can trigger a short course correction. Because crossing the sphere completes the mission at any speed, a safe flyby can finish under zero commanded thrust.

### Matching the launch to the thruster geometry

The target begins roughly 94° from the shuttle's nose. Pointing the nose at it would require nearly a quarter-turn before settling. The bottom jets already push in a useful departure direction. On a clear ascent, the final controller biases its acceleration request just **0.05° forward of the body lift axis**, which lets the allocator combine upward acceleration and stabilizing torque through the bottom jets.

We tested that small change with identical initial conditions in a 60 Hz rigid-body simulation. The earlier low-tilt profile took **210.58 s** and used **38.12 kg**. The bottom-thruster profile took **200.70 s** and used **38.44 kg**. That bought **9.88 s for 0.32 kg**, raising the modeled score from **56.16 to 58.53**. Fuel spent through jets outside the bottom array fell from 0.806 to 0.034 kg. Measuring individual jet impulses made it possible to see what the allocator actually did with each command.

### Predicting a coast under gravity

The shuttle and target are observed in a frame that rotates with Phobos. Guidance starts with Phobos's inverse-square gravity and estimates the remaining acceleration from changes in the shuttle's measured velocity. During a coast it can observe all three components. During early powered flight it can also estimate the component perpendicular to the thrust direction. Each forecast uses the latest gravity estimate to predict the curved path into the arrival sphere.

We checked that approximation against recorded coasts and an independent offline reference using Mars and Phobos gravity, SPICE ephemerides, and rotating-frame terms. Across six recorded 90 s coasts, extrapolating the measured velocity in a straight line produced a **44.65 m median endpoint error**. The controller's gravity estimate reduced it to **2.13 m**. The full offline reference reached **0.69 m**. With a 50 m arrival radius, the simpler onboard estimate captured most of the improvement that mattered to guidance.

### Tracking asteroids and deciding when to dodge

Radar reports are noisy, arrive at different times, and change order as rocks pass one another. The tracker reconstructs the shuttle state at each measurement time and associates compatible returns with existing tracks using position, size, and radial velocity. A first return tells us how fast a rock is closing along the radar line of sight. Subsequent positions reveal its sideways motion. The filter carries the remaining uncertainty forward between observations and gives ambiguous matches less weight.

The route planner checks the direct path first and forecasts potential encounters up to **120 s** ahead. If a crossing threatens that path, it compares delaying a burn to let the rock pass, accelerating to pass ahead of it, and making a shallow steering or aim-point change. A separate safety check runs about every **0.25 s** and usually looks **30 s** ahead. It accounts for the size of both bodies, their motion between samples, radar uncertainty, and variation in delivered thrust. It can select a stronger escape maneuver when timing alone leaves too little clearance.

Early versions intervened whenever a broad uncertainty estimate overlapped the route. That could turn a harmless pass into a costly detour. The final controller keeps tracking an uncertain contact while the estimated path clears both hulls and a safety margin. When it does intervene, it evaluates alternate maneuvers against the full predicted uncertainty. Protection starts with the first usable radar returns because a blocked launch can leave very little time to act.

### Verification and flight results

We used debris-free flights to isolate propulsion efficiency, repeated synthetic encounters with identical radar noise to compare avoidance changes, and replayed recorded telemetry to inspect individual control decisions. The development tools recorded raw radar returns, track counts, predicted clearance and uncertainty, control modes, commanded acceleration, fuel, and source hashes. The team packaged the flight software into one `Controller.py`. The development suite passed **185 tests**, and the single-file version reproduced the development controller's linear and angular commands across **25,435 control ticks** in two modeled flights.

| Flight or benchmark | Score | Time | Fuel | Damage |
| --- | ---: | ---: | ---: | ---: |
| First successful Space Teams flight | −62.9 | 202.33 s | 129.14 kg | 0 J |
| Final controller, local Space Teams flight A | 55.2 | 220.46 s | 36.95 kg | 0 J |
| Final controller, local Space Teams flight B | 42.3 | 239.28 s | 43.37 kg | 0 J |
| Matched offline benchmark, clear field | 58.53 | 200.70 s | 38.44 kg | Contact damage unmodeled |

The two local flights used the same controller file as the submission. Flight A handled its encounters by coasting. Flight B also used powered escape and course correction. Both arrived undamaged, but the second took 18.82 s longer and used 6.42 kg more. Those logs show how much the debris geometry and the timing of an intervention can matter. The submitted 57.8 is reported separately from these development flights.

The hardest remaining case was a nearby rock first resolved after liftoff. In seven offline runs of the same late crossing with different radar noise, six cleared the modeled hulls. One overlapped by 0.64 m, and the tightest successful pass cleared by just 0.04 m. The offline model captures rigid-body motion and jet allocation. Space Teams supplies launchpad contact and impact damage, so a modeled gap that small leaves little confidence in a clean pass.

**Engineering tools** include Python, NumPy, Space Teams PRO, Kalman filtering, rigid-body simulation, SPICE-based trajectory analysis, telemetry replay, and automated regression tests. [Competition overview](https://github.com/SimDynamX/SpaceTeamsPRO/discussions/218)
