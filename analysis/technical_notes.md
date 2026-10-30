# Technical notes

I joined the encounter and shift data using the physician and shift IDs. I treated all timestamps as the same local time. The reference sample has 395,654 encounters with a positive assignment-to-order LOS. For the main analysis, I also required assignment to be at or after triage and departure to be at or after the order. That left 395,280 encounters. I kept assignments before the recorded shift start because the exercise says they can happen.

For the reference census, patients enter at triage. Assigned patients leave the count at their discharge order, and patients without an assignment leave at physical departure. When events happen at the same time, exits come first and arrivals are ordered by encounter ID. The arriving patient is included. This reproduces all 63 reference bins and counts. I also checked census separately for 120 encounters.

I made other measures for different parts of the visit. Waiting-room census ends at assignment or departure without assignment. Active treatment runs from assignment to the order. Physician workload counts the physician's other assigned patients at the time of assignment. Physical occupancy ends at departure. The alternative-measure checks exclude invalid intervals.

The models compare each census group with the group below 20. I add severity, physician, arrival time, shift timing, workload, active ED load, and staffing controls in stages. The code lists the spline knots. Arrival-time controls use hour of week and year-month. Standard errors are clustered by arrival date. Fixed effects are removed before estimating the regression, and the code checks that this step converged.

For workload, I compare patients across a physician's shifts and then within the same shift. The within-shift model uses shift effects in place of physician and calendar effects. I also check patients assigned during their recorded shift.

To look at patients who leave without an assignment, I adjust the observed LOS averages using the census, arrival hour, and severity of all arrivals. This depends on those observed factors capturing the differences between assigned and unassigned patients. Two arrivals in the 80-plus census group have no matching observed LOS cell. The results are descriptive and do not establish that crowding caused a change in LOS.
