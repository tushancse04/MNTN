import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class tsne_plot:
	def fit_tsne(self,x,x1,x2,y):
		f = False
		xn = [x[i] for i in range(len(y)) if y[i] == f]
		yn = [y[i] for i in range(len(y)) if y[i] == f]
		nx1 = [x1[i] for i in range(len(y)) if y[i] == f]
		nx2 = [x2[i] for i in range(len(y)) if y[i] == f]
		xn = xn[:371]
		nx1 = nx1[:371]
		nx2 = nx1[:371]
		yn = yn[:371]
		n = len(yn)

		c = 0
		for i in range(len(y)):
			if c > n:
				break
			if y[i] == f:
				continue
			xn.append(x[i])
			yn.append(y[i])
			nx1.append(x1[i])
			nx2.append(x2[i])
			c += 1

		#print(yn)
		x = np.array(xn)
		y = np.array(yn)
		print(len(y),sum([0 if a == False else 1 for a in y]))
		'''
		x = TSNE(n_components=2).fit_transform(x)
		data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'R': y})
		plot = sns.scatterplot(data=data, x="x1", y="x2", hue="R").get_figure()
		plot.savefig("review/out/output.png")

		return
		'''
		nx = np.array(nx1) + np.array(nx2)
		x = TSNE(n_components=2).fit_transform(nx)
		data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'R': y})
		plot = sns.scatterplot(data=data, x="x1", y="x2", hue="R").get_figure()
		plot.savefig("review/out/output2.png")

