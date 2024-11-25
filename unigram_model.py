import numpy as np
import matplotlib.pyplot as plt

observations = ["apple", "apple", "apple", "apple", "apple", "apple", "banana", "banana", "banana", "banana"]
n_apple = sum([_ == "apple" for _ in observations])
n_banana = len(observations) - n_apple

p_apple = np.linspace(0, 1, 100)
p_banana = 1 - p_apple
prob_observations = (p_apple ** n_apple) * (p_banana ** n_banana)

plt.figure()
plt.plot(p_apple, prob_observations)
plt.xlabel('$p_{apple}$')
plt.ylabel('probability of the observations')
plt.title('probability of the observations as a function of $p_{apple}$')
plt.show()
