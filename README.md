# AlphaNet

**AlphaNet** is a tool designed to traverse and extract features from raw stock volume and price data in an end-to-end manner.

## Version Overview

### _v1

- **Data Input**: Daily stock feature data.
- **Data Window**: Processes data over a period of 30 days (e.g.).
- **Factor Calculation**: Calculates factors using a window size of 10 days.
- **Rolling Calculations**: Supports rolling calculations.

### _v1.1

- **Data Input**: Minute-by-minute stock feature data over the past 20 days.
- **Rolling Calculations**: Supports rolling calculations.

### _v1.2

- **Data Input**: Daily minute-by-minute stock feature data.
- **Rolling Calculations**: Does not support rolling calculations.

### _v2

- **Enhancement**: Incorporates LSTM and GRU networks into the _v1.1 framework.
- **Rolling Calculations**: Removes rolling calculations.

---

Feel free to explore each version to find the one that best suits your data analysis needs. For more details on implementation and usage, please refer to the documentation specific to each version.
