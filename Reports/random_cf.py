import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
test_cfm = np.random.random((20,20))
fig, ax = plt.subplots(figsize=(20,20))
results = sns.heatmap(test_cfm , annot=True, cmap='coolwarm')
figure = results.get_figure()
figure.savefig('test_conf.png', dpi=1000) 