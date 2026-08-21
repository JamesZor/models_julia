# Scottish Lower Leagues: Open-Play Signals, Penalties, & Own Goals (EDA Report)

**Date:** 2026-08-21  
**Tournament Segment:** Scottish League One (`#56`) & Scottish League Two (`#57`)  
**Sample Size:** 1,990 historical matches (3,980 team-matches across seasons 2021/22 – 2024/25)  
**Data Sources:** `sofascore.match_incidents`, `sofascore.matches`, and `bbc.match_officials` (100% referee coverage)

---

## 1. Executive Summary & Core Statistical Proofs

1. **Massive Noise Volume**:
   - **9.63% of ALL goals** (527 out of 5,475 total goals) are penalties (7.62%) or own goals (2.01%).
2. **Team Penalty Disparity ($p = 0.0093$)**:
   - Team penalty rates range from **0.055 to 0.243 penalties/match** ($\chi^2 = 51.17, p = 0.0093$).
   - However, year-over-year persistence is low ($r = +0.128$). Penalty drawing is driven by current attacking volume/box presence rather than a permanent independent talent.
3. **Referee Whistle Spread ($4.4\times$ Difference)**:
   - Penalty frequency varies dramatically by official:
     - **High-Whistle Referees**: Ross Hardie (0.475 pens/match), Scott Lambie (0.451), Craig Napier (0.444), Calum Scott (0.441).
     - **Conservative Referees**: Steven Reid (0.107 pens/match), Mike Roncone (0.120), Graham Beaton (0.136), Peter Stuart (0.163).
   - A match with a high-whistle referee is **$4.4\times$ more likely to see a penalty** than one with a conservative referee.
   - Referees exhibit significant variance in **Home Bias** (ranging from 16.7% to 73.3% home penalties).
4. **Own Goals are Pure Stochastic Noise ($p = 0.4031, r = +0.029$)**:
   - Team own goal counts pass Chi-squared uniformity ($\chi^2 = 31.25, p = 0.4031$).
   - Year-over-year correlation is zero ($r = +0.029$). No team has an intrinsic own-goal tendency.
5. **Variance Reduction**:
   - Stripping penalties and own goals reduces observation variance by **10.08%** and increases goal signal persistence from $r = +0.180$ to **$r = +0.228$ (+26.3% higher signal)**.

---

## 2. Detailed Empirical Tables

### A. Macro Event Totals
| Metric | Total Count | Per Match Rate | % of All Goals |
| :--- | :--- | :--- | :--- |
| **Total Matches Analyzed** | **1,990** | - | - |
| **Total Raw Goals** | **5,475** | **2.75** | 100.00% |
| - *Home Goals* | 2,912 | 1.46 | 53.19% |
| - *Away Goals* | 2,563 | 1.29 | 46.81% |
| **Total Penalties Awarded** | **543** | **0.27** | - |
| - *Penalties Scored* | 417 | 0.21 | **7.62%** |
| - *Penalties Missed* | 126 (23.2% fail rate) | 0.06 | - |
| - *Penalty Conversion Rate* | **76.8%** | - | - |
| **Total Own Goals** | **110** | **0.06** | **2.01%** |
| **Total Noise (Pens + OGs)** | **527** | **0.27** | **9.63%** |
| **Clean Open-Play Goals** | **4,948** | **2.49** | **90.37%** |

---

### B. Team-Level Penalty Disparity ($\chi^2 = 51.17, p = 0.0093$)

| Team | Matches | Pens Won | Pens Won/Pg | Pens Conceded | Conceded/Pg | Net Pens | Conv % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **East Kilbride** | 37 | 9 | **0.243** | 3 | 0.081 | +6 | 88.9% |
| **Bonnyrigg Rose** | 108 | 26 | **0.241** | 21 | 0.194 | +5 | 92.3% |
| **Montrose** | 199 | 43 | **0.216** | 19 | 0.095 | **+24** | 72.1% |
| **Dunfermline Athletic** | 36 | 7 | **0.194** | 1 | 0.028 | +6 | 71.4% |
| **Airdrieonians** | 92 | 16 | **0.174** | 12 | 0.130 | +4 | 62.5% |
| **Annan Athletic** | 199 | 34 | **0.171** | 30 | 0.151 | +4 | 79.4% |
| **Queen of the South** | 145 | 24 | **0.166** | 22 | 0.152 | +2 | 58.3% |
| **The Spartans FC** | 109 | 18 | **0.165** | 14 | 0.128 | +4 | 88.9% |
| **Stenhousemuir** | 197 | 30 | **0.152** | 15 | 0.076 | **+15** | 80.0% |
| **Edinburgh City** | 199 | 30 | 0.151 | 36 | 0.181 | -6 | 73.3% |
| **Stranraer** | 199 | 30 | 0.151 | 25 | 0.126 | +5 | 66.7% |
| **Cove Rangers** | 163 | 24 | 0.147 | 17 | 0.104 | +7 | 70.8% |
| **Albion Rovers** | 90 | 13 | 0.144 | 17 | 0.189 | -4 | 69.2% |
| **Dumbarton** | 199 | 28 | 0.141 | 39 | 0.196 | **-11** | 85.7% |
| **Falkirk** | 126 | 17 | 0.135 | 18 | 0.143 | -1 | 88.2% |
| **Peterhead** | 199 | 26 | 0.131 | 25 | 0.126 | +1 | 73.1% |
| **East Fife** | 199 | 23 | 0.116 | 32 | 0.161 | -9 | 87.0% |
| **Elgin City** | 199 | 23 | 0.116 | 37 | 0.186 | **-14** | 69.6% |
| **Stirling Albion** | 199 | 22 | 0.111 | 24 | 0.121 | -2 | 77.3% |
| **Forfar Athletic** | 199 | 20 | 0.101 | 26 | 0.131 | -6 | 95.0% |
| **Clyde** | 199 | 17 | 0.085 | 29 | 0.146 | **-12** | 82.4% |
| **Alloa Athletic** | 181 | 12 | 0.066 | 25 | 0.138 | **-13** | 91.7% |
| **Hamilton Academical**| 73 | 4 | **0.055** | 3 | 0.041 | +1 | 25.0% |

---

### C. Referee-Level Penalty Rates & Home Bias ($\ge 15$ Matches Officiated)

| Referee Name | Matches | Pens Total | Pens / Match | Home Pens | Away Pens | Home Pen % | Cards / Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ross Hardie** | 40 | 19 | **0.475** | 12 | 7 | 63.2% | 4.60 |
| **Scott Lambie** | 51 | 23 | **0.451** | 12 | 11 | 52.2% | 3.39 |
| **Craig Napier** | 27 | 12 | **0.444** | 4 | 8 | 33.3% | 2.81 |
| **Calum Scott** | 59 | 26 | **0.441** | 12 | 14 | 46.2% | 4.32 |
| **Cameron Stirling** | 20 | 8 | **0.400** | 3 | 5 | 37.5% | 4.35 |
| **Duncan Williams** | 93 | 36 | **0.387** | 23 | 13 | 63.9% | 4.41 |
| **Steven Kirkland** | 66 | 24 | **0.364** | 11 | 13 | 45.8% | 4.74 |
| **Gavin Duncan** | 44 | 15 | **0.341** | 7 | 8 | 46.7% | 3.41 |
| **Colin Whyte** | 44 | 15 | **0.341** | 11 | 4 | **73.3%** | 5.41 |
| **Lewis Hogarth** | 18 | 6 | **0.333** | 4 | 2 | **66.7%** | 6.11 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **Jordan Curran** | 47 | 7 | **0.149** | 4 | 3 | 57.1% | 4.51 |
| **Graham Beaton** | 22 | 3 | **0.136** | 3 | 0 | **100.0%** | 3.41 |
| **Connor Ashwood** | 15 | 2 | **0.133** | 1 | 1 | 50.0% | 5.13 |
| **Mike Roncone** | 25 | 3 | **0.120** | 0 | 3 | **0.0%** | 3.72 |
| **Steven Reid** | 28 | 3 | **0.107** | 3 | 0 | **100.0%** | 2.93 |

---

### D. Team Own Goals Conceded vs Benefited ($\chi^2 = 31.25, p = 0.4031$)

| Team | Matches | OG Conceded | Conceded/Pg | OG Benefited | Benefited/Pg | Net OGs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Edinburgh City** | 199 | 8 | 0.040 | 8 | 0.040 | 0 |
| **Stirling Albion** | 199 | 7 | 0.035 | 8 | 0.040 | +1 |
| **Stranraer** | 199 | 7 | 0.035 | 5 | 0.025 | -2 |
| **Cove Rangers** | 163 | 6 | 0.037 | 7 | 0.043 | +1 |
| **Peterhead** | 199 | 6 | 0.030 | 4 | 0.020 | -2 |
| **Dumbarton** | 199 | 5 | 0.025 | 8 | 0.040 | +3 |
| **Airdrieonians** | 92 | 4 | 0.043 | 6 | 0.065 | +2 |
| **Forfar Athletic** | 199 | 4 | 0.020 | 3 | 0.015 | -1 |
| **Elgin City** | 199 | 3 | 0.015 | 4 | 0.020 | +1 |
| **Alloa Athletic** | 181 | 2 | 0.011 | 4 | 0.022 | +2 |
| **Falkirk** | 126 | 2 | 0.016 | 4 | 0.032 | +2 |
| **Dunfermline** | 36 | 1 | 0.028 | 2 | 0.056 | +1 |
| **Stenhousemuir** | 197 | 1 | 0.005 | 6 | 0.030 | +5 |
| **East Fife** | 199 | 1 | 0.005 | 8 | 0.040 | **+7** |

> **Own Goal Homogeneity Test**: $\chi^2 = 31.25, p = 0.4031$. We fail to reject the null hypothesis of equal team own-goal propensity. Own goals occur at a constant rate of **1 every ~36 team-matches** across all clubs.

---

## 3. Modeling Conclusions & Takeaways

1. **Penalties are Attacking-Volume Proxies but Game-State Confounded**:
   - Dominant attacking teams win more penalties because they spend more time inside the 18-yard box.
   - However, training MCMC chains on raw penalty goals over-rewards teams that had outlier penalty fortune in short 4-week spans.
2. **Referee Assignment is a Potent In-Play / Matchday Context**:
   - While pre-game models should train on clean open-play signals, matchday live execution could incorporate referee whistle thresholds ($0.475$ vs $0.107$ pens/match) for in-play total goals markets.
3. **Own Goals Should Be 100% Excluded**:
   - Zero team correlation ($r = 0.029, p = 0.403$). Every own goal in the training set is pure noise that distorts team strength estimation.
