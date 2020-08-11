import matplotlib.pyplot as plt 
import numpy as np

text_file = open("Output.txt", "r")
content = eval(text_file.read())
print(content)
times = np.array(content)
plt.hist(times)
plt.show()