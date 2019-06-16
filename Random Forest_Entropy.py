import csv
import random
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Implement your decision tree below
class DecisionTree():
	
	def learn(self, training_set,features):
        # implement this function
		
		def split_tree(attr,val,ipdata):
			tree_left = []
			tree_right = []
			for ix in ipdata:
				if float(ix[attr]) < float(val):
					tree_left.append(ix)
				else:
					tree_right.append(ix)
			return tree_left,tree_right
		
		def calc_entropy(ip_vec):
			ix0_count = 0
			ix1_count = 0
			for ix in ip_vec:
				if int(ix) == 0:
					ix0_count += 1
				else:
					ix1_count += 1
			if ix0_count == 0 or ix1_count == 0:
				return 0
			else:
				return (-1*(((ix0_count/(ix0_count+ix1_count))*(np.log2((ix0_count/(ix0_count+ix1_count)))))+((ix1_count/(ix0_count+ix1_count))*(np.log2(ix1_count/(ix0_count+ix1_count))))))		
	
		def calc_info_gain(parent_ent,i,tree_lt,tree_rt):
			lt_entropy = calc_entropy([ix[-1] for ix in tree_lt])
			rt_entropy = calc_entropy([ix[-1] for ix in tree_rt])
			tmp_gain = parent_ent - ((len(tree_lt)/(len(tree_lt)+len(tree_rt)))*lt_entropy) - ((len(tree_rt)/(len(tree_lt)+len(tree_rt)))*rt_entropy)
			return tmp_gain
		
		def chk_last_node(ipSet):
			class_val = [elem[-1] for elem in ipSet]
			response = max(set(class_val),key=class_val.count)
			return(response)
			
		
		def get_child(ipdata,features):
		
			def sortByIndex(item):
				return item[2]
				
			decision_cols = random.sample(list(range(len(ipdata[0])-1)),features)
			parent_entropy = calc_entropy([ip[-1] for ip in ipdata])
			info_gain = []
			
			for ix in decision_cols:
				ix_values = set(ix1[ix] for  ix1 in ipdata)
				for ix_val in ix_values:
					result_left, result_right = split_tree(ix,ix_val,ipdata)
					info_gain.append((ix,ix_val,calc_info_gain(parent_entropy,ix,result_left,result_right),result_left,result_right))
			
			info_gain=sorted(info_gain,key=sortByIndex,reverse=True)
			return {'attribute':info_gain[0][0],'chk_value':info_gain[0][1],'left_branch':info_gain[0][3],'right_branch':info_gain[0][4]}
		
		def splitTree(main_tree,features):
			left_child = main_tree['left_branch']
			right_child = main_tree['right_branch']

			del(main_tree['left_branch'])
			del(main_tree['right_branch'])
			
			if not left_child and right_child:
				main_tree['left_child'] = chk_last_node(right_child)
				main_tree['right_child'] = chk_last_node(right_child)
				return
			
			if not right_child and left_child:
				main_tree['left_child'] = chk_last_node(left_child)
				main_tree['right_child'] = chk_last_node(left_child)
				return

			main_tree['left_child'] = get_child(left_child,features)
			splitTree(main_tree['left_child'],features)
				
			main_tree['right_child'] = get_child(right_child,features)
			splitTree(main_tree['right_child'],features)
				
			return main_tree
		
		## Start of the functin
		self.tree = {}
		self.tree = get_child(training_set,features)
		splitTree(self.tree,features)
	

    # implement this function
	def classify(self, test_instance):
		result = 0 # baseline: always classifies as 0
		
		def get_class(tree_input,test_data):
			if float(test_data[tree_input['attribute']]) < float(tree_input['chk_value']):
				if isinstance(tree_input['left_child'], dict):
					return get_class(tree_input['left_child'],test_data)
				else:
					return tree_input['left_child']
			else:
				if isinstance(tree_input['right_child'], dict):
					return get_class(tree_input['right_child'],test_data)
				else:
					return tree_input['right_child']
		
		result = get_class(self.tree,test_instance)
		return result

# Load data set
with open("spam.data") as f:
    data = [tuple(line) for line in csv.reader(f, delimiter=" ")]
ip_size = len(data)
print("Number of records: %d" % ip_size)

## Split data into train and test
random.shuffle(data)
X_train = data[:round(len(data)*0.7)]
X_test = data[round(len(data)*0.7):]

print("Number of records in Training set: %d" % len(X_train))
print("Number of records in Test set: %d" %len(X_test))

def run_random_forest(num_of_trees,num_of_features):
	start_time = time.time()
	tree = [None]*num_of_trees

	for ix in list(range(num_of_trees)):
		sample_ix = np.random.choice(len(X_train)-1,len(X_train),replace=True)
		training_set = []
		for i in sample_ix:
			training_set.append(X_train[i])
		tree[ix] = DecisionTree()
		
		# Construct a tree using training set, set maximum depth to prevent looping
		output = tree[ix].learn(training_set,num_of_features)

	# Classify the test set using the tree we just constructed
	results = []
	
	for instance in X_test:
		tree_output = []
		for ix in list(range(num_of_trees)):
			result = tree[ix].classify( instance[:-1] )
			tree_output.append(result)
		spam_result = max(set(tree_output),key=tree_output.count)
		results.append( spam_result == instance[-1])

	# Accuracy
	accuracy = float(results.count(True))/float(len(results))
	end_time = time.time()
	comp_time = end_time-start_time
	print("Accuracy using Random Forest (Native code): " + str(accuracy))
	print("Computation Time using Random Forest (Native code): " + str(comp_time))
	
def run_random_forest_sklearn(num_of_trees,num_of_features):
	start_time = time.time()
	clf = RandomForestClassifier(n_estimators=num_of_trees,criterion="entropy",max_features=num_of_features)
	trainX = [x[:-1] for x in X_train]
	trainY = [x[-1] for x in X_train]
	testX = [x[:-1] for x in X_test]
	testY = [x[-1] for x in X_test]
	clf.fit(trainX,trainY)
	accuracy = clf.score(testX,testY)
	end_time = time.time()
	comp_time = end_time-start_time
	print("Accuracy using Random Forest (Scikit learn package): " + str(accuracy))
	print("Computation Time using Random Forest (Scikit learn package): " + str(comp_time))

if __name__ == "__main__":
	num_of_trees = 10
	n_features = [1,5,7,9,11,15]
	native_results = []
	sklearn_results = []
	for num_of_features in n_features:
		print(" ")
		print("Building Random Forest using Native code with " +str(num_of_trees) + " trees and " + str(num_of_features) + " feature(s)...")
		native_results.append(run_random_forest(num_of_trees,num_of_features))
		print(" ")
		print("Building Random Forest using Scikitlearn package with " +str(num_of_trees) + " trees and " + str(num_of_features) + " feature(s)...")
		sklearn_results.append(run_random_forest_sklearn(num_of_trees,num_of_features))