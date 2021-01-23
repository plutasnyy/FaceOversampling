set Months := {1..6};
set Products := {1..7};
set Machines := {"Grinder", "VertDrill", "HorizDrill", "Borer", "Planer"};

param machCap[Machines] := <"Grinder"> 4, <"VertDrill"> 2, <"HorizDrill"> 3, <"Borer"> 1, <"Planer"> 1;
param  maintenance[Months * Machines] :=
    <1, "Grinder"> 1,
    <2, "HorizDrill"> 2,
    <3, "Borer"> 1,
    <4, "VertDrill"> 1,
    <5, "Grinder"> 1,
    <5, "VertDrill"> 1,
    <6, "Planer"> 1,
    <6, "HorizDrill"> 1
    default 0;
param machCapInMo[<mo, ma> in Months * Machines] := machCap[ma] - maintenance[mo, ma];
param cost[Machines * Products]:=
             |    1,    2,    3,    4,	  5,   6,    7|
|"Grinder"	 |  0.5,  0.7,    0,	0,	0.3, 0.2,  0.5|
|"VertDrill" |  0.1,  0.2,    0,  0.3,    0, 0.6,    0|
|"HorizDrill"|  0.2,    0,  0.8,	0,    0,   0,  0.6|
|"Borer"	 | 0.05, 0.03,    0, 0.07,  0.1,   0, 0.08|
|"Planer"	 |	  0,    0, 0.01,	0, 0.05,   0, 0.05|;
param profit[Products] := <1> 10, <2> 6, <3> 8, <4> 4, <5> 11, <6> 9, <7> 3;
param storageCost := 0.5;
param storageCap := 100;
param storageEnd := 50;
param monthWorkhour := 24 * 2 * 8;
param demand[Months * Products] :=
   |   1,    2,   3,   4,    5,   6,   7|
|1 | 500, 1000, 300, 300,  800, 200, 100|
|2 | 600,  500, 200,   0,  400, 300, 150|
|3 | 300,  600,   0,   0,  500, 400, 100|
|4 | 200,  300, 400, 500,  200,   0, 100|
|5 |   0,  100, 500, 100, 1000, 300,   0|
|6 | 500,  500, 100, 300, 1100, 500,  60|;

var production[<m, p> in Months * Products] integer >= 0 <= demand[m, p] + storageCap;
var sell[<m, p> in Months * Products] integer >= 0 <= demand[m, p];
var storage[<m, p> in (Months union {0}) * Products] integer >= (if m == 6 then storageEnd else 0 end) <= (if m == 0 then 0 else storageCap end);

maximize profit:
    sum <m, p> in Months * Products: (profit[p] * sell[m, p] - storageCost * storage[m, p]);
