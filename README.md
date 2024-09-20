_v1
Used for daily stock feature data input.
For example, processes 30 days of data.
Calculates factors using a window size of 10 days.
Supports rolling calculations.

_v1.1
Used for minute-by-minute stock feature data input over the past 20 days.
Supports rolling calculations.

_v1.2
Used for daily minute-by-minute stock feature data input.
Does not support rolling calculations.

_v2
Adds LSTM to _v1.1.
Removes rolling calculations.
