<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Residual-TD3-Robot-Navigation</h1>

<h2>Simulator</h2>
<ul>
    <li><strong>Environment</strong>: Continuous 2D action space for robot navigation.</li>
    <li><strong>Dynamics</strong>: Unknown non-linear environment dynamics.</li>
    <li><strong>Features</strong>:
        <ul>
            <li>Seeded start and end points.</li>
            <li>Ability to request demonstrations.</li>
            <li>Ability to request environment reset</li>
        </ul>
    </li>
    <li><strong>Costs</strong>:
        <ul>
            <li>Environment reset.</li>
            <li>Demonstrations.</li>
            <li>Taking a step.</li>
        </ul>
    </li>
</ul>

<h2>Algorithm</h2>
<ul>
    <li><strong>Baseline Policy</strong>: Utilizes imitation learning to establish a baseline policy from demonstrations.</li>
    <li><strong>Residual Learning</strong>: Applies residual learning to refine the baseline policy, accommodating non-linear dynamics and varied starting positions.</li>
    <li><strong>Reinforcement Learning</strong>: Implements Twin Delayed Deep Deterministic Policy Gradient (TD3) as the reinforcement learning algorithm of choice.</li>
</ul>

</body>
</html>
