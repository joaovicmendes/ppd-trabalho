# Results
Resultados em uma máquina com 2 cores físicos e 4 virtuais (HTT).

### Sample = 10.000.000

Program | Threads | Time | Computed Value |
--------|---------|------|----------|
pi.out| 1|0.24s|Pi: 3.1413516998291016|
pi-parallel.out| 1|0.24s|Pi: 3.1415500640869141|
pi-parallel.out| 4| 0.8s|Pi: 3.1423521041870117|
pi-parallel.out| 8| 0.8s|Pi: 3.1423392295837402|
pi-parallel.out|64| 0.8s|Pi: 3.1412992477416992|

### Sample = 1.000.000.000

Program | Threads | Time | Computed Value |
--------|---------|------|----------|
pi.out| 1|24.81s|Pi: 3.1415555477142334|
pi-parallel.out| 1|23.98s|Pi: 3.1415491104125977|
pi-parallel.out| 4| 8.11s|Pi: 3.1414895057678223|
pi-parallel.out| 8| 8.08s|Pi: 3.1413512229919434|
pi-parallel.out|64| 8.33s|Pi: 3.1416158676147461|


### Sample = 10.000.000.000

Program | Threads | Time | Computed Value |
--------|---------|------|----------|
pi.out| 1|33.95s|Pi: 3.1416199207305908|
pi-parallel.out| 1|33.70s|Pi: 3.1415755748748779|
pi-parallel.out| 4|11.57s|Pi: 3.1414873600006104|
pi-parallel.out| 8|11.55s|Pi: 3.1414546966552734|
pi-parallel.out|64|11.52s|Pi: 3.1415712833404541|
