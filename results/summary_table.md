| instance   | method                   |    cost |   reference_cost |   gap_percent |   routes | valid   |   time_seconds |
|:-----------|:-------------------------|--------:|-----------------:|--------------:|---------:|:--------|---------------:|
| A-n32-k5   | Nearest Neighbor CVRP    | 1146.39 |              784 |         46.22 |        5 | Oui     |       0.00045  |
| A-n32-k5   | Nearest Neighbor + 2-opt | 1118.26 |              784 |         42.64 |        5 | Oui     |       0.001138 |
| A-n32-k5   | Clarke & Wright Savings  |  843.69 |              784 |          7.61 |        5 | Oui     |       0.00249  |
| A-n32-k5   | Clarke & Wright + 2-opt  |  830.67 |              784 |          5.95 |        5 | Oui     |       0.001692 |
| A-n32-k5   | Q-learning RL            |  967.14 |              784 |         23.36 |        5 | Oui     |       1.21966  |
| A-n32-k5   | Masked Policy RL         |  960.24 |              784 |         22.48 |        5 | Oui     |     353.991    |
| A-n32-k5   | Masked Policy RL + 2-opt |  884.87 |              784 |         12.87 |        5 | Oui     |       0.002397 |