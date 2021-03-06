from DBManager import dbconfig
import random
from gensim.models import Word2Vec
import numpy as np
from NTN import NeuralTensorLayer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import time

class balancer:
	def __init__(self):
		self.TOPN = 10

	def get_w2v_model(self,sentences):
		model = Word2Vec(sentences, size=self.TOPN,window=1,min_count=0)
		return model

	def embed(self,pdm,pred_atoms):
		sentences = []
		for pname in pred_atoms:
			for objs in pred_atoms[pname]:
				s = []
				t = (pname,)
				for i,obj in enumerate(objs):
					s += [str(pdm[pname][i]) + '_' + str(obj)]
				s += [pname]
				t += tuple(objs)
				sentences += [s]
				
		print("sentences",len(sentences))
		self.model = self.get_w2v_model(sentences)


	def include_corrupt_atoms(self,p_atom_map,pdm):
		model = self.model
		wv = model.wv
		vocab = wv.vocab
		atoms = list(p_atom_map.keys())
		x,y = [],[]
		print('len : ',len(atoms))
		c = 0
		for atom in atoms:
			c += 1
			print(c)
			if len(atom) < 3:
				continue

			p = atom[0]

			if p not in vocab:
				p_atom_map.pop(atom)
				continue

			d1,d2 = pdm[p][0],pdm[p][1]
			obj1,obj2 = str(d1) + '_' + atom[1],str(d2) + '_' + atom[2]
			if obj1 not in vocab or obj2 not in vocab:
				continue

			pv = np.array(wv[p])
			xv = []
			for i in range(1,len(atom),1):
				d = pdm[p][i-1]
				obj = str(d) + '_' + atom[i]
				if obj not in vocab:
					p_atom_map.pop(atom)
					break
				xv.append((np.array(wv[obj])).tolist())



			#print(obj1,obj2)

			sim1,sim2 = model.most_similar(obj1,topn = self.TOPN*2),model.most_similar(obj2,topn = self.TOPN*2)
			img = []
			for i in range(len(sim1)):
				s1,s2 = sim1[i],sim2[i]
				v = model.wv[s1[0]].tolist()+model.wv[s2[0]].tolist()
				img.append((np.array(v) + np.array((pv.tolist()+pv.tolist()))).tolist())

			if atom in p_atom_map:
				x.append((atom,xv,img))
				y.append(p_atom_map[atom])
		print(len(atoms),len(y))
		return x,y


	def run_cnn(self,X_train, X_val, y_train, y_val):
		X_train,X_val = np.array([x[2] for x in X_train]),np.array([x[2] for x in X_val])
		X_train = np.reshape(X_train,(len(X_train),self.TOPN*2,self.TOPN*2,1))
		X_val = np.reshape(X_val,(len(X_val),self.TOPN*2,self.TOPN*2,1))
		y_train,y_val = to_categorical(y_train),[True if y == 0 else False for y in y_val]
		CVN = 4
		MPN = 2
		m = len(X_train[0])
		model = Sequential()
		model.add(Conv2D(m*2, kernel_size=CVN, activation='relu', input_shape=(m,m,1),padding='same'))
		model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding='same'))
		model.add(Conv2D(1, kernel_size=CVN, activation='relu',padding='same'))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(2, activation='softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(X_train, y_train, epochs=1)
		r = model.predict(X_val)
		r = [x[0]  for x in r]
		s = roc_auc_score(y_val,r)
		r = [False if r[i] < .5 else True for i in range(len(r))]
		print('cnn : ',s)
		ofile = open('review/out/out.txt','a')
		ofile.write('cnn : ' + str(s) + '\n')
		ofile.close()
		return r

	def re_gen_atoms(self,r,X_train,y_train,X_val,y_val):
		pred_atoms = {}
		c = 0
		for i in range(len(r)):

			if r[i] == False or y_val[i] == False:
				continue
			atom = X_val[i][0]
			p = atom[0]
			if p not in pred_atoms:
				pred_atoms[p] = []
			pred_atoms[p].append([atom[1],atom[2]])


		for i in range(len(X_train)):
			if y_train[i] == False:
				continue
			atom = X_train[i][0]
			p = atom[0]
			if p not in pred_atoms:
				pred_atoms[p] = []
			pred_atoms[p].append([atom[1],atom[2]])

		return pred_atoms		

	def get_pred_atoms(self,X_train, X_val,y_train,y_val):
		xtrain,xval,ytrain,yval = {},{},{},{}
		for i,x in enumerate(X_train):
			atom = x[0]
			pname = atom[0]
			if pname not in xtrain:
				xtrain[pname] = []
				ytrain[pname] = []
			xtrain[pname].append(x)
			ytrain[pname].append(y_train[i])

		for i,x in enumerate(X_val):
			atom = x[0]
			pname = atom[0]
			if pname not in xval:
				xval[pname] = []
				yval[pname] = []
			xval[pname].append(x)
			yval[pname].append(y_val[i])
		return xtrain,xval,ytrain,yval

	def run_ntn(self,X_train, X_val, y_train, y_val):
		xtrain,xval,ytrain,yval = self.get_pred_atoms(X_train,X_val,y_train,y_val)
		print('reconstruction done')
		ymap,x_val1,x_val2 = [],[],[]

		res = {}
		x_val1 = np.array([x[1][0] for x in X_val])
		x_val2 = np.array([x[1][1] for x in X_val])
		for pname in xtrain:
			x_train1,x_train2 = np.array([x[1][0] for x in xtrain[pname]]),np.array([x[1][1] for x in xtrain[pname]])
			ntn = NeuralTensorLayer(len(x_train1[0]),10)
			r = ntn.fit(x_train1,x_train2,np.array(ytrain[pname]),x_val1,x_val2,np.array(y_val))
			for i in range(len(r)):
				if y_val[i] == False:
					continue
				atom = X_val[i][0]
				atom = (atom[1],atom[2])
				if atom not in res:
					res[atom] = (pname,r[i])
				if res[atom][1] < r[i]:
					res[atom] = (pname,r[i])
		r = []
		for i in range(len(y_val)):
			atom = X_val[i][0]
			if (atom[1],atom[2]) not in res:
				r.append(False)
			elif res[atom[1],atom[2]][0] == atom[0]:
				r.append(True)
			else:
				r.append(False)
		tp,fp,fn = 0,0,0
		for i in range(len(r)):
			if y_val[i] == False:
				continue
			if r[i] == False:
				fn += 1
			else:
				tp += 1
		ofile = open('review/out/out.txt','a')
		ofile.write('ntn : ' + str(tp/(tp+fn)) + '\n')
		ofile.close()
		print('Accuracy: ',(tp/(tp+fn)))
		return r



		#print(len(p_atom_map))

	def run_cnn_ntn(self,p_atom_map,pdm,is_cnn):
		t1 = time.time()
		x,y = self.include_corrupt_atoms(p_atom_map,pdm)
		print('include_corrupt_atoms: ',time.time()-t1 )
		x,y = np.array(x),np.array(y)		
		X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42,shuffle=True)
		print('include_corrupt_atoms: ',time.time()-t1 )
		r = None
		if is_cnn:
			pass
			r = self.run_cnn(X_train, X_val, y_train, y_val)
		else:
			print('running ntn...')
			r = self.run_ntn(X_train, X_val, y_train, y_val)
		pred_atoms = self.re_gen_atoms(r,X_train,y_train,X_val,y_val)
		return pred_atoms







	def get_bal_atoms(self,pred_atoms,pdm,dom_size_map):
		p_atom_map = {}
		for p in pred_atoms:
			for atom in pred_atoms[p]:
				atom = (p,) + tuple(atom)
				p_atom_map[atom] = True

		#print(len(p_atom_map))

		for p in pred_atoms:
			if len(pred_atoms) == 0:
				continue
			n = len(pred_atoms[p][0])
			d1 = pdm[p][0]
			d1 = int(dom_size_map[d1])
			poss_atoms = d1
			d2 = None
			if n > 1:
				d2 = pdm[p][1]
				d2 = int(dom_size_map[d2])
				poss_atoms *= d2
			n = len(pred_atoms[p]) 
			if n > .4*poss_atoms:
				continue


			for atom in pred_atoms[p]:
				def get_rand_atom():
					n1 = str(random.randint(0,d1-1))
					t = (p,n1)
					if d2:
						n2 = str(random.randint(0,d2-1))
						t += (n2,)
					if t not in p_atom_map:
						return t
					return None

				atom = get_rand_atom()
				while not atom:
					atom = get_rand_atom()
				#print(atom)
				p_atom_map[atom] = False

		#print(len(p_atom_map))
		return p_atom_map




