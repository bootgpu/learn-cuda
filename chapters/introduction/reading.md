::youtube[Chapter 1 video: Why GPUs]{id="CIDS0tQzGyc"}

This page collects the formulas from the video, so you can look them up while solving the problems.

## Speedup

```text
speedup = old time ÷ new time
```

A program that took 60 seconds and now takes 3 seconds has a speedup of 60 ÷ 3 = 20.

## Amdahl's Law

Take the old running time as 100%, split into a serial part and a part that can run in parallel. Only the parallel part gets faster; the serial part stays the same:

```text
new total time  = serial share + parallel share ÷ speedup of that part
overall speedup = 100% ÷ new total time
```

The example from the video: grading takes 90% of the time and recording the scores 10%, and grading becomes 1000 times faster.

```text
new total time  = 10% + 90% ÷ 1000 = 10.09%
overall speedup = 100% ÷ 10.09% ≈ 9.9
```

However fast the parallel part gets, its time can shrink at most to zero, so the total time is at least the serial part:

```text
upper limit of the overall speedup = 100% ÷ serial share
```

A 10% serial part limits it to 10 times; 1%, to 100 times; 0.1%, to 1000 times.
