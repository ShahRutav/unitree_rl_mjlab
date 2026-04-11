# kp Mapping — 29 DOF (G1)

Order: `left_leg(6)  right_leg(6)  waist(3)  left_arm(7)  right_arm(7)`

```
idx  joint               kp
---  -----------------   ---
# left leg (0–5)
 0   L_hip_pitch         100
 1   L_hip_roll          100
 2   L_hip_yaw           100
 3   L_knee              150
 4   L_ankle_pitch        40
 5   L_ankle_roll         40

# right leg (6–11)
 6   R_hip_pitch         100
 7   R_hip_roll          100
 8   R_hip_yaw           100
 9   R_knee              150
10   R_ankle_pitch        40
11   R_ankle_roll         40

# waist (12–14)
12   waist_yaw           200
13   waist_roll          200
14   waist_pitch         400

# left arm (15–21)
15   L_shoulder_pitch     40
16   L_shoulder_roll      40
17   L_shoulder_yaw       40
18   L_elbow              40
19   L_wrist_roll         40
20   L_wrist_pitch        40
21   L_wrist_yaw          40

# right arm (22–28)
22   R_shoulder_pitch     40
23   R_shoulder_roll      40
24   R_shoulder_yaw       40
25   R_elbow              80
26   R_wrist_roll         40
27   R_wrist_pitch        40
28   R_wrist_yaw          40
```

Source: `deploy/robots/g1/config/joint_cmd.yaml` (kp) + `deploy/include/FSM/State_JointCmd.h` (joint names).
